import numpy as np
from decipher.tools._decipher import Decipher, DecipherConfig, DecipherATACConfig
from decipher.tools.decipher import decipher_train

def run_benchmark(adata_sim):
    from scipy.stats import spearmanr

    from scipy.optimize import linear_sum_assignment

    def best_module_correlation(pred_z, true_z):
        n = true_z.shape[1]
        corr_matrix = np.array([
            [abs(spearmanr(pred_z[:, i], true_z[:, j]).statistic)
            for j in range(n)]
            for i in range(n)
        ])
        # Hungarian algorithm: optimal assignment instead of greedy
        row_ind, col_ind = linear_sum_assignment(-corr_matrix)  # negate = maximize
        return corr_matrix[row_ind, col_ind].mean()

    true_gene_modules = adata_sim.uns['gene_modules']

    # --- Train standard Decipher ---
    config_std = DecipherConfig(dim_z=10, dim_v=2, n_epochs=500, early_stopping_patience=50)
    config_std.initialize_from_adata(adata_sim)
    decipher_std, _ = decipher_train(adata_sim, config_std)
    z_std = adata_sim.obsm['decipher_z'].copy()

    # --- Train DecipherATAC ---
    config_atac = DecipherATACConfig(dim_z=10, dim_v=2, n_epochs=1000, early_stopping_patience=100)
    config_atac.initialize_from_adata(adata_sim)
    config_atac.gene_modules = true_gene_modules
    print(f"Module 0 genes (first 5): {config_atac.gene_modules[0][:5]}")
    print(f"Gene index range: {min(min(m) for m in config_atac.gene_modules)} - {max(max(m) for m in config_atac.gene_modules)}")
    print(f"Total genes in adata: {adata_sim.shape[1]}")
    print(f"Any overlap between modules and confounder range: {any(g >= 500 for m in config_atac.gene_modules for g in m)}")

    decipher_atac, _ = decipher_atac_train(adata_sim, config_atac)
    z_atac = adata_sim.obsm['decipher_z'].copy()

    true_z = adata_sim.obsm['true_z']

    std_score = best_module_correlation(z_std, true_z)
    atac_score = best_module_correlation(z_atac, true_z)

    print("Z interpretability (Spearman, higher=better):")
    print(f"  Standard Decipher : {std_score:.3f}")
    print(f"  DecipherATAC      : {atac_score:.3f}")

    return std_score, atac_score

def decipher_atac_train(adata, config, **kwargs):
    from decipher.tools._decipher import DecipherATAC
    # Monkey-patch: swap Decipher for DecipherATAC inside decipher_train
    import decipher.tools.decipher as dt
    original = dt.Decipher
    dt.Decipher = DecipherATAC
    try:
        result = decipher_train(adata, config, **kwargs)
    finally:
        dt.Decipher = original  # always restore
    return result

# In benchmarks/run_benchmark.py

def run_mofa(adata_sim, n_factors=10):
    try:
        from mofapy2.run.entry_point import entry_point
    except ImportError:
        raise ImportError("pip install mofapy2")

    # Normalize: log1p of library-size normalized counts
    X = adata_sim.X.copy().astype(np.float64)
    lib = X.sum(axis=1, keepdims=True)
    X = np.log1p(X / lib * 1e4)

    ent = entry_point()
    ent.set_data_options(scale_groups=False, scale_views=False)

    # MOFA+ expects [views][groups] of numpy arrays, samples=rows, features=cols
    ent.set_data_matrix(
        [[X]],
        likelihoods=["gaussian"],
        views_names=["RNA"],
        groups_names=["all_cells"]
    )
    ent.set_model_options(factors=n_factors)
    ent.set_train_options(seed=42, convergence_mode="fast", verbose=False)
    ent.build()
    ent.run()

    # getExpectations()["E"] returns (n_cells, n_factors) per group
    z_mofa = ent.model.nodes["Z"].getExpectations()["E"]
    print(f"z_mofa shape: {z_mofa.shape}")  # should be (5000, 10)

    node = ent.model.nodes["Z"]
    print("node type:", type(node))
    print("getExpectations keys:", node.getExpectations().keys())
    print("E shape:", node.getExpectations()["E"].shape if hasattr(node.getExpectations()["E"], "shape") else [x.shape for x in node.getExpectations()["E"]])
    print("E2 shape:", node.getExpectations()["E2"].shape if hasattr(node.getExpectations()["E2"], "shape") else [x.shape for x in node.getExpectations()["E2"]])
    return z_mofa

if __name__ == "__main__":
    from benchmarks.simulate import simulate_modular_rna
    from benchmarks.save_results import save_results
    from scipy.stats import spearmanr
    from scipy.optimize import linear_sum_assignment
    import random

    # ── Sanity check: 0 confounders, easy signal ──────────────────────────────
    print("=== Quick sanity check ===")
    adata_easy = simulate_modular_rna(n_confounders=0, tf_correlation=0.3, n_cells=2000)
    sanity_std, sanity_atac = run_benchmark(adata_easy)

    # ── Default simulation ────────────────────────────────────────────────────
    print("\n=== Default simulation ===")
    adata_sim = simulate_modular_rna()
    default_std, default_atac = run_benchmark(adata_sim)

    # ── MOFA+ on default simulation (500 confounders) ─────────────────────────
    mofa_score = None
    mofa_n_conf = 500
    print("\n=== MOFA+ on default simulation ===")
    try:
        z_mofa = run_mofa(adata_sim, n_factors=10)
        true_z_default = adata_sim.obsm['true_z']

        def _best_corr(pred_z, true_z):
            n = true_z.shape[1]
            corr_matrix = np.array([
                [abs(spearmanr(pred_z[:, i], true_z[:, j]).statistic)
                 for j in range(n)]
                for i in range(n)
            ])
            row_ind, col_ind = linear_sum_assignment(-corr_matrix)
            return corr_matrix[row_ind, col_ind].mean()

        mofa_score = _best_corr(z_mofa, true_z_default)
        print(f"MOFA+ (linear, n_confounders={mofa_n_conf}): {mofa_score:.3f}")
    except Exception as e:
        print(f"MOFA+ failed: {e}")

    # ── Confounder sweep ──────────────────────────────────────────────────────
    print("\n=== Confounder sweep ===")
    sweep_results = []
    for n_conf in [0, 100, 250, 500]:
        for seed in [42, 123, 7]:
            print(f"\n-- n_confounders={n_conf}, seed={seed} --")
            adata = simulate_modular_rna(
                n_confounders=n_conf,
                tf_correlation=0.6,
                n_cells=3000,
                seed=seed
            )
            std_score, atac_score = run_benchmark(adata)
            sweep_results.append((n_conf, seed, std_score, atac_score))

    print("\n=== Sweep results ===")
    print("n_confounders | seed | Std Decipher | DecipherATAC")
    for r in sweep_results:
        print(f"{r[0]:13} | {r[1]:4} | {r[2]:12.3f} | {r[3]:13.3f}")

    # ── Save everything ───────────────────────────────────────────────────────
    save_results(
        sweep_results=sweep_results,
        mofa_score=mofa_score,
        mofa_n_confounders=mofa_n_conf,
        sanity_std=sanity_std,
        sanity_atac=sanity_atac,
        default_std=default_std,
        default_atac=default_atac,
    )