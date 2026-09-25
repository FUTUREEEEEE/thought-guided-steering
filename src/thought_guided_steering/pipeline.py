"""Construction → fixed validation scoring → SPS selection; no parameter search."""

from dataclasses import asdict

import numpy as np

from .directions import bootstrap_multiplicities, paired_differences, weighted_direction
from .intervention import Candidate
from .representations import validate_partitions
from .scoring import preference_stability, utility
from .selection import select_candidate
from .stability import StabilityConfig, construction_statistics, tangent_neighborhood


def select_sps(actor, build, validation, safe_controls, *, layers: list[int], strength: float,
               config: StabilityConfig) -> dict:
    """Each layer contributes its full-build nominal direction; scoring gate is forced on."""
    validate_partitions(build, validation, safe_controls)
    if len(layers) < 2 or len(set(layers)) != len(layers):
        raise ValueError("A fixed set of at least two unique candidate layers is required")
    weights = bootstrap_multiplicities([row.case_id for row in build],
                                       config.bootstrap_count, config.seed)
    base_safe = [actor.mean_logp(row.prefix_ids, row.safe.token_ids) for row in validation]
    base_risky = [actor.mean_logp(row.prefix_ids, row.risky.token_ids) for row in validation]
    base_control = [actor.mean_logp(row.prefix_ids, row.proposal_ids) for row in safe_controls]
    details, statistics = [], []
    for layer in layers:
        differences = paired_differences(
            [actor.thought_hidden(row.prefix_ids, row.safe, layer) for row in build],
            [actor.thought_hidden(row.prefix_ids, row.risky, layer) for row in build])
        neighborhood = tangent_neighborhood(differences, config.components, config.rho)

        def score(direction):
            candidate = Candidate(layer, tuple(direction), strength)
            safe = [actor.mean_logp(row.prefix_ids, row.safe.token_ids, candidate) for row in validation]
            risky = [actor.mean_logp(row.prefix_ids, row.risky.token_ids, candidate) for row in validation]
            control = [actor.mean_logp(row.prefix_ids, row.proposal_ids, candidate) for row in safe_controls]
            return utility(preference_stability(safe, risky, base_safe, base_risky),
                           base_control, control, config.collateral_weight)

        local = [score(direction) for direction in neighborhood]
        reconstructed = [weighted_direction(differences, row_weights) for row_weights in weights]
        bootstrap = [score(direction) for direction in reconstructed]
        r_dir = min(row["utility"] for row in local)
        r_tail, r_stab = construction_statistics([row["utility"] for row in bootstrap], config.beta)
        statistics.append([r_dir, r_tail, r_stab])
        details.append({"layer": layer, "direction": neighborhood[0].tolist(),
                        "neighborhood": neighborhood.tolist(), "local_utilities": local,
                        "bootstrap_directions": np.asarray(reconstructed).tolist(),
                        "bootstrap_utilities": bootstrap})
    selection = select_candidate([str(layer) for layer in layers], statistics)
    chosen = details[selection["selected_index"]]
    return {"implementation": "paper_constant_strength_v1", "config": asdict(config),
            "strength": strength, "statistics": statistics, "selection": selection,
            "candidates": details, "bootstrap_step_weights": weights.tolist(),
            "deployment": {"layer": chosen["layer"], "direction": chosen["direction"],
                           "strength": strength}}
