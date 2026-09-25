"""Safety Preference Stability statistics, with explicit finite-sample conventions."""

from dataclasses import dataclass
import math

import numpy as np

from .directions import finite_array, mean_direction, unit


@dataclass(frozen=True)
class StabilityConfig:
    rho: float
    components: int
    bootstrap_count: int
    beta: float
    seed: int
    collateral_weight: float

    def __post_init__(self):
        if any(type(value) is not int for value in (self.components, self.bootstrap_count, self.seed)):
            raise ValueError("PCA rank, bootstrap count and seed must be integers")
        if not np.isfinite(self.rho) or self.rho <= 0 or self.components < 1:
            raise ValueError("rho and PCA component count must be positive")
        if self.bootstrap_count < 2 or not 0 < self.beta <= 1:
            raise ValueError("At least two bootstrap replicates and beta in (0, 1] required")
        if not np.isfinite(self.collateral_weight) or self.collateral_weight < 0:
            raise ValueError("Collateral weight must be finite and nonnegative")


def tangent_neighborhood(differences, components: int, rho: float) -> np.ndarray:
    """Eqs. 13–15: centered PCA, nominal plus both signs of each tangent axis.

    Reject degenerate axes rather than silently changing the candidate neighborhood.
    """
    differences = finite_array(differences, 2)
    if not 1 <= components <= min(differences.shape[0] - 1, differences.shape[1]):
        raise ValueError("PCA rank exceeds the available construction sample/dimension")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    nominal = mean_direction(differences)
    _, singular, axes = np.linalg.svd(differences - differences.mean(axis=0), full_matrices=False)
    if singular[components - 1] <= 1e-12:
        raise ValueError("Requested PCA subspace is rank deficient")
    variants = [nominal]
    for axis in axes[:components]:
        tangent = unit(axis - (axis @ nominal) * nominal)
        variants.extend([unit(nominal + rho * tangent), unit(nominal - rho * tangent)])
    return np.stack(variants)


def construction_statistics(utilities, beta: float) -> tuple[float, float]:
    """Eqs. 17–18; mean of worst ceil(beta * B) samples; population std (ddof=0)."""
    values = finite_array(utilities, 1)
    if len(values) < 2 or not 0 < beta <= 1:
        raise ValueError("At least two utilities and beta in (0, 1] required")
    tail = np.sort(values)[:math.ceil(beta * len(values))]
    return float(tail.mean()), -float(values.std(ddof=0))
