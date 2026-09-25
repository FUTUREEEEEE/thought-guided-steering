"""Step-weighted contrastive directions and case-grouped resampling (Eqs. 6, 16)."""

import numpy as np


def finite_array(values, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise ValueError(f"Expected a nonempty finite {ndim}-dimensional array")
    return array


def unit(vector) -> np.ndarray:
    vector = finite_array(vector, 1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Direction has zero or invalid norm")
    return vector / norm


def paired_differences(safe_hidden, risky_hidden) -> np.ndarray:
    safe = finite_array(safe_hidden, 2)
    risky = finite_array(risky_hidden, 2)
    if safe.shape != risky.shape:
        raise ValueError("Matched hidden-state arrays must have identical shapes")
    return safe - risky


def mean_direction(differences) -> np.ndarray:
    return unit(finite_array(differences, 2).mean(axis=0))


def bootstrap_multiplicities(case_ids, count: int, seed: int) -> np.ndarray:
    """Return [replicate, step] weights; every step in a case has one multiplicity."""
    if len(case_ids) == 0 or not isinstance(count, int) or count < 1:
        raise ValueError("Nonempty case IDs and positive replicate count required")
    cases, inverse = np.unique(np.asarray(case_ids, dtype=str), return_inverse=True)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(cases), size=(count, len(cases)))
    return np.stack([np.bincount(row, minlength=len(cases))[inverse] for row in sampled])


def weighted_direction(differences, multiplicities) -> np.ndarray:
    differences = finite_array(differences, 2)
    weights = finite_array(multiplicities, 1)
    if len(weights) != len(differences) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid construction weights")
    if not np.equal(weights, np.floor(weights)).all():
        raise ValueError("Bootstrap multiplicities must be integers")
    return unit(np.average(differences, axis=0, weights=weights))
