import json
import os
from datetime import datetime


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_results(sweep_results, mofa_score=None, mofa_n_confounders=None,
                 sanity_std=None, sanity_atac=None,
                 default_std=None, default_atac=None,
                 extra=None):
    """
    Save all benchmark results to a timestamped JSON file.

    Parameters
    ----------
    sweep_results : list of (n_conf, seed, std_score, atac_score)
    mofa_score : float or None
    mofa_n_confounders : int or None  — which n_confounders MOFA+ was run on
    sanity_std, sanity_atac : float or None  — quick sanity check scores
    default_std, default_atac : float or None  — default sim scores
    extra : dict or None  — any additional metadata
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "sweep": [
            {"n_confounders": r[0], "seed": r[1],
             "std_score": r[2], "atac_score": r[3]}
            for r in sweep_results
        ],
        "mofa": {
            "score": mofa_score,
            "n_confounders": mofa_n_confounders,
        },
        "sanity_check": {
            "std_score": sanity_std,
            "atac_score": sanity_atac,
            "n_confounders": 0,
            "tf_correlation": 0.3,
            "n_cells": 2000,
        },
        "default_sim": {
            "std_score": default_std,
            "atac_score": default_atac,
            "n_confounders": 500,
            "tf_correlation": 0.6,
            "n_cells": 5000,
        },
        "extra": extra or {},
    }

    fname = os.path.join(RESULTS_DIR, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)

    # Also overwrite a "latest" symlink for easy loading
    latest = os.path.join(RESULTS_DIR, "latest.json")
    with open(latest, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to {fname}")
    print(f"Latest results at {latest}")
    return fname


def load_results(path=None):
    """Load results from JSON. Defaults to latest.json."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "latest.json")
    with open(path) as f:
        return json.load(f)