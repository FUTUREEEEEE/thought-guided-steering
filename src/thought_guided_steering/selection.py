"""Fixed-candidate standardization and global deployment selection (Eqs. 19–21)."""

import numpy as np

from .directions import finite_array


def select_candidate(candidate_ids, statistics) -> dict:
    """Rows are candidates, columns are R_dir, R_tail, R_stab, in that order.

    Population standard deviations are used. Constant columns make Eq. 19
    undefined and raise an error. Exact ties retain the declared candidate order.
    """
    matrix = finite_array(statistics, 2)
    if matrix.shape != (len(candidate_ids), 3) or len(candidate_ids) < 2:
        raise ValueError("Expected at least two candidates with exactly three statistics")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate IDs must be unique")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    if (std <= 1e-12).any():
        raise ValueError("Cannot standardize a constant SPS component")
    standardized = (matrix - mean) / std
    scores = standardized.sum(axis=1)
    selected = int(np.argmax(scores))
    return {"candidate_ids": list(candidate_ids), "mean": mean.tolist(), "std": std.tolist(),
            "standardized": standardized.tolist(), "scores": scores.tolist(),
            "selected_index": selected, "selected_id": candidate_ids[selected]}
