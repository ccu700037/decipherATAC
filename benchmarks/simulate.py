import numpy as np
import scanpy as sc


def simulate_modular_rna(
    n_cells: int = 5000,
    n_modules: int = 10,
    genes_per_module: int = 50,
    n_confounders: int = 500,
    module_overlap: int = 15,
    tf_correlation: float = 0.6,
    noise_scale: float = 0.5,
    seed: int = 42,
) -> sc.AnnData:
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. True trajectory: cells ordered along a figure-8
    # ------------------------------------------------------------------
    t = np.linspace(0, 2 * np.pi, n_cells)
    true_v = np.stack([np.sin(t), np.sin(t) * np.cos(t)], axis=1)
    true_v += rng.normal(0, 0.05, true_v.shape)

    # ------------------------------------------------------------------
    # 2. True TF activities with LARGE dynamic range
    #    Use raw linear combinations of t, then standardize so each
    #    module spans a wide range (not sigmoid-compressed near 0.5)
    # ------------------------------------------------------------------
    cov = tf_correlation * np.ones((n_modules, n_modules)) + \
          (1 - tf_correlation) * np.eye(n_modules)
    A = rng.multivariate_normal(np.zeros(n_modules), cov)
    B = rng.multivariate_normal(np.zeros(n_modules), cov)

    raw = np.outer(np.cos(t), A) + np.outer(np.sin(t), B)
    # Standardize each module to have mean=0, std=1, then shift to (0, 1)
    raw = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)
    # Now rescale: true_z in [0, 1] with full dynamic range
    true_z = (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0) + 1e-8)
    # true_z: (n_cells, n_modules), each column spans nearly full [0,1]

    # ------------------------------------------------------------------
    # 3 & 4. Gene expression using Splatter-inspired two-component model
    # Splatter separates: (a) whether a gene is expressed (dropout),
    # (b) how much it's expressed (NB with mean-dispersion relationship)
    # ------------------------------------------------------------------
    library_size = rng.lognormal(mean=np.log(3000), sigma=0.6, size=n_cells).astype(np.float32)

    # Gene-level baseline means: log-normal as in real data
    # Real scRNA: gene means span ~4 orders of magnitude
    gene_base_mean = rng.lognormal(mean=-2.0, sigma=1.5, size=(n_modules, genes_per_module))
    # This gives means ranging ~0.01 to ~10, with most genes lowly expressed

    # Dispersion: inversely related to mean (as observed empirically across 59 datasets)
    # High-mean genes: low dispersion (theta~10), low-mean genes: high dispersion (theta~0.5)
    def mean_to_dispersion(mu):
        # Empirical relationship from Hafemeister & Satija 2019
        return np.clip(100 / (mu + 1) + 0.5, 0.5, 50)

    counts_list = []
    module_labels = []

    for i in range(n_modules):
        # TF effect: multiplicative fold change on baseline
        # true_z=0 → baseline, true_z=1 → baseline * fold_change
        fold_change = rng.lognormal(mean=2.0, sigma=0.5, size=genes_per_module)
        # fold_change ~ 4-20x for TF targets (realistic)
        
        mu_base = gene_base_mean[i]  # (genes_per_module,)
        # TF activity multiplicatively scales expression
        mu = mu_base * (1 + true_z[:, i:i+1] * (fold_change - 1))
        # Cross-regulation: one other TF adds small effect
        other = rng.choice([j for j in range(n_modules) if j != i])
        mu = mu * (1 + true_z[:, other:other+1] * 0.2)
        
        # Scale by library size
        mu = mu * library_size[:, None] / 3000  # normalize to median library
        
        # Dispersion from empirical mean-dispersion relationship
        theta = mean_to_dispersion(mu)
        
        # Dropout: probability a true count becomes 0 (technical zeros)
        # Splatter uses logistic function of log mean
        p_dropout = 1 / (1 + np.exp(-((-2.0 - np.log1p(mu)) * 1.5)))
        
        p_nb = np.clip(theta / (theta + mu + 1e-8), 1e-6, 1 - 1e-6)
        raw_counts = rng.negative_binomial(theta, p_nb).astype(np.float32)
        
        # Apply dropout mask
        dropout_mask = rng.binomial(1, 1 - p_dropout).astype(np.float32)
        counts_list.append(raw_counts * dropout_mask)
        module_labels.extend([i] * genes_per_module)

    X_modules = np.hstack(counts_list)

    # ------------------------------------------------------------------
    # 5. Confounders: same generative process, orthogonal latent factors
    # ------------------------------------------------------------------
    confound_z = rng.normal(0, 1, (n_cells, 3))
    for k in range(3):
        for j in range(n_modules):
            confound_z[:, k] -= (
                np.dot(confound_z[:, k], true_z[:, j]) /
                (np.dot(true_z[:, j], true_z[:, j]) + 1e-8)
            ) * true_z[:, j]
    confound_z_norm = (confound_z - confound_z.min(0)) / (confound_z.max(0) - confound_z.min(0) + 1e-8)

    gene_base_mean_c = rng.lognormal(mean=-2.0, sigma=1.5, size=(3, n_confounders))
    fold_change_c = rng.lognormal(mean=2.0, sigma=0.5, size=(3, n_confounders))

    mu_c = gene_base_mean_c[0] * (1 + confound_z_norm[:, 0:1] * (fold_change_c[0] - 1))
    for k in range(1, 3):
        mu_c += gene_base_mean_c[k] * confound_z_norm[:, k:k+1] * 0.3
    mu_c = mu_c * library_size[:, None] / 3000

    theta_c = mean_to_dispersion(mu_c)
    p_dropout_c = 1 / (1 + np.exp(-((-2.0 - np.log1p(mu_c)) * 1.5)))
    p_nb_c = np.clip(theta_c / (theta_c + mu_c + 1e-8), 1e-6, 1 - 1e-6)
    X_confounders = rng.negative_binomial(theta_c, p_nb_c).astype(np.float32)
    X_confounders *= rng.binomial(1, 1 - p_dropout_c).astype(np.float32)
    module_labels.extend([-1] * n_confounders)
    # ------------------------------------------------------------------
    # 6. Combine
    # ------------------------------------------------------------------
    X = np.hstack([X_modules, X_confounders])

    # ------------------------------------------------------------------
    # 7. Gene modules with overlap
    # ------------------------------------------------------------------
    base_indices = [
        list(range(i * genes_per_module, (i + 1) * genes_per_module))
        for i in range(n_modules)
    ]
    gene_modules = []
    for i in range(n_modules):
        neighbor = (i + 1) % n_modules
        gene_modules.append(base_indices[i] + base_indices[neighbor][:module_overlap])

    # ------------------------------------------------------------------
    # 8. Package
    # ------------------------------------------------------------------
    adata = sc.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = (
        [f"gene_mod{i}_{j}" for i in range(n_modules) for j in range(genes_per_module)]
        + [f"confounder_{j}" for j in range(n_confounders)]
    )
    adata.obsm["true_v"] = true_v
    adata.obsm["true_z"] = true_z
    adata.obs["library_size"] = library_size
    adata.var["module"] = module_labels
    adata.uns["gene_modules"] = gene_modules

    n_mod = n_modules * genes_per_module
    print(f"Simulated {n_cells} cells x {X.shape[1]} genes")
    print(f"  {n_modules} modules x {genes_per_module} genes/module "
          f"(+{module_overlap} overlap) + {n_confounders} structured confounders")
    print(f"  TF correlation: {tf_correlation}, noise: {noise_scale}")
    print(f"\n--- Simulation diagnostics ---")
    print(f"true_z range:     {true_z.min():.3f} - {true_z.max():.3f}")
    print(f"true_z std/module: {true_z.std(axis=0).mean():.3f}")
    print(f"Library size:     median={np.median(library_size):.0f}, "
          f"range={library_size.min():.0f}-{library_size.max():.0f}")
    print(f"Module gene mean: {X[:, :n_mod].mean():.2f}")
    print(f"Confounder mean:  {X[:, n_mod:].mean():.2f}")
    print(f"Sparsity:         {(X == 0).mean():.1%} zeros")
    print(f"Max count:        {X.max():.0f}")
    nonzero = X[X > 0]
    print(f"Mean of non-zero counts: {nonzero.mean():.2f}")
    print(f"% cells with count > 5: {(X > 5).mean():.1%}")
    print(f"99th percentile count: {np.percentile(adata.X, 99):.0f}")
    print(f"% counts > 100: {(adata.X > 100).mean():.2%}")
    
    print(f"X_modules shape: {X_modules.shape}")
    print(f"X_confounders shape: {X_confounders.shape}")  
    print(f"Final X shape: {adata.X.shape}")
    print(f"gene_modules[0][:5]: {adata.uns['gene_modules'][0][:5]}")
    print(f"gene_modules max index: {max(max(m) for m in adata.uns['gene_modules'])}")
    return adata


if __name__ == "__main__":
    adata_sim = simulate_modular_rna()