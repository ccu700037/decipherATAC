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
    from scipy.stats import spearmanr
    from scipy.optimize import linear_sum_assignment
    import random

    adata_sim = simulate_modular_rna()
    true_z = adata_sim.obsm['true_z']
    true_gene_modules = adata_sim.uns['gene_modules']

    all_genes = list(range(adata_sim.shape[1]))
    genes_copy = all_genes.copy()
    random.shuffle(genes_copy)
    module_size = len(true_gene_modules[0])
    shuffled_modules = [genes_copy[i*module_size:(i+1)*module_size]
                        for i in range(len(true_gene_modules))]

    config_shuffled = DecipherATACConfig(dim_z=10, dim_v=2, n_epochs=1000, early_stopping_patience=100)
    config_shuffled.initialize_from_adata(adata_sim)
    config_shuffled.gene_modules = shuffled_modules
    decipher_atac_train(adata_sim, config_shuffled)
    z_shuffled = adata_sim.obsm['decipher_z'].copy()

    def best_module_correlation(pred_z, true_z):
        n = true_z.shape[1]
        corr_matrix = np.array([
            [abs(spearmanr(pred_z[:, i], true_z[:, j]).statistic)
            for j in range(n)]
            for i in range(n)
        ])
        row_ind, col_ind = linear_sum_assignment(-corr_matrix)
        return corr_matrix[row_ind, col_ind].mean()

    shuffled_score = best_module_correlation(z_shuffled, true_z)
    print(f"\nDecipherATAC (shuffled modules): {shuffled_score:.3f}")
    print(f"DecipherATAC (true modules):     0.862  (from previous run)")
    print(f"Standard Decipher:               0.562  (from previous run)")

    # Small sanity check
    # print("=== Quick sanity check ===")
    # adata_easy = simulate_modular_rna(n_confounders=0, tf_correlation=0.3, n_cells=2000)
    # run_benchmark(adata_easy)

    # print("\n=== Default simulation ===")
    # adata_sim = simulate_modular_rna()
    # run_benchmark(adata_sim)

    # print("\n=== MOFA+ on default simulation ===")
    # try:
    #     z_mofa = run_mofa(adata_sim, n_factors=10)  # or config_std.dim_z
    #     from scipy.stats import spearmanr

    #     def best_module_correlation(pred_z, true_z):
    #         n = true_z.shape[1]
    #         corr_matrix = np.array([
    #             [abs(spearmanr(pred_z[:, i], true_z[:, j]).statistic)
    #              for j in range(n)]
    #             for i in range(n)
    #         ])
    #         matched = []
    #         for _ in range(n):
    #             i, j = np.unravel_index(corr_matrix.argmax(), corr_matrix.shape)
    #             matched.append(corr_matrix[i, j])
    #             corr_matrix[i, :] = -1
    #             corr_matrix[:, j] = -1
    #         return np.mean(matched)

    #     true_z = adata_sim.obsm['true_z']
    #     mofa_score = best_module_correlation(z_mofa, true_z)
    #     print(f"MOFA+ (linear) score: {mofa_score:.3f}")

    # except ImportError:
    #     print("MOFA+ not installed. Run: pip install mofapy2")

    # print("\n=== Confounder sweep ===")
    # results = []
    # for n_conf in [0, 100, 250, 500]:
    #     for seed in [42, 123, 7]:  # 3 seeds for error bars
    #         print(f"\n-- n_confounders={n_conf}, seed={seed} --")
    #         adata = simulate_modular_rna(
    #             n_confounders=n_conf,
    #             tf_correlation=0.6,
    #             n_cells=3000,
    #             seed=seed
    #         )
    #         std_score, atac_score = run_benchmark(adata)
    #         results.append((n_conf, seed, std_score, atac_score))

    # # Optional: print a simple table
    # print("\n=== Sweep results ===")
    # print("n_confounders | seed | Std Decipher | DecipherATAC")
    # for r in results:
    #     print(f"{r[0]:13} | {r[1]:4} | {r[2]:12.3f} | {r[3]:13.3f}")
