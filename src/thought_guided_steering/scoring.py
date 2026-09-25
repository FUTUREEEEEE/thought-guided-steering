"""Preference margin and collateral-adjusted utility (Eqs. 9–12)."""

import numpy as np

from .directions import finite_array


def preference_stability(safe_steered, risky_steered, safe_base, risky_base) -> np.ndarray:
    arrays = [finite_array(x, 1) for x in (safe_steered, risky_steered, safe_base, risky_base)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("Each likelihood array must cover exactly the same paired rows")
    return (arrays[0] - arrays[1]) - (arrays[2] - arrays[3])


def utility(preference_values, control_base, control_steered, collateral_weight: float) -> dict:
    preference = finite_array(preference_values, 1)
    base = finite_array(control_base, 1)
    steered = finite_array(control_steered, 1)
    if base.shape != steered.shape:
        raise ValueError("Safe-control likelihood arrays must match")
    if not np.isfinite(collateral_weight) or collateral_weight < 0:
        raise ValueError("Collateral weight must be finite and nonnegative")
    collateral = np.maximum(base - steered, 0).mean()
    return {"preference": float(preference.mean()), "collateral": float(collateral),
            "utility": float(preference.mean() - collateral_weight * collateral)}
