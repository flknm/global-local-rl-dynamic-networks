"""Configuration loading and strict separation of exact/experimental modes."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path


NETWORK_TYPES = {
    "core_periphery",
    "cycle",
    "star",
    "inverse_star",
    "asymmetric_stars",
    "pairs",
    "chain",
    "symmetric_stars",
    "mirrored_cycles",
    "random",
    "mixed_networks",
}

MIXED_NETWORK_TYPES = (
    "random",
    "core_periphery",
    "cycle",
    "star",
    "inverse_star",
    "asymmetric_stars",
    "pairs",
    "chain",
)

ONE_DECISION_NETWORK_TYPES = {
    "symmetric_stars",
    "mirrored_cycles",
}

BASE_CONFIG = {
    "mode": "exact",
    "discount_factor": 0.9,
    "rollout": {"scheme": "geometric", "horizon": 50},
    "transition_budget": 100_000,
    "device": "cpu",
    "initial_state_mode": "fixed",
    "seeds": {"model": 42, "environment": 43, "initial_state": 44},
    "environment": {
        "network_type": "core_periphery",
        "num_nodes": 10,
        "num_core_nodes": 3,
        "price_risky_asset": 1.0,
        "mu": 1.0,
        "sigma": 0.15,
        "theta": 0.2,
        "rho": 0.5,
        "delta": 0.1,
        "alpha": 0.8,
        "beta": 0.8,
        "inverse_demand_slope": 0.1,
        "factor_networth": 1.0,
        "factor_capital": 1.3333333333333333,
        "factor_penalty": 0.0,
        "upper_limit_capital_injections": 1.0,
        "num_capital_injection_actions": 11,
        "num_action_weights": 3,
        "action_space_architecture": "weighted_limit",
        "permute_on_reset": False,
        "tolerance": 1e-6,
        "max_iter": 10000,
    },
    "policy": {
        "family": "shared",
        "encoder": "bidirectional_mpnn",
        "num_layers": 2,
        "embedding_dim": 160,
        "aggregation": "sum",
        "readout": "sum",
        "activation": "tanh",
        "num_heads": 1,
        "budget_head_num_layers": 1,
        "node_global_context": False,
        "budget_bias_nudge": 0.2,
        "routed": {
            "budget": {"num_layers": 2, "embedding_dim": 160},
            "node": {"num_layers": 2, "embedding_dim": 160},
        },
    },
    "step_size": {
        "type": "polynomial",
        "initial": 0.001,
        "exponent": 0.6,
        "decay_factor": 0.99,
        "final_step_ratio": None,
    },
    "batch_size": {"type": "constant", "initial": 2, "exponent": 0.25, "maximum": None},
    "projection": {"enabled": True, "default_bound": 5.0, "group_bounds": {}},
    # Training options are stored together in both modes.
    "experimental": {
        "optimizer": "manual_sgd",
        "baseline": "none",
        "standardize_advantages": False,
        "gradient_clip_norm": None,
    },
    "checkpoint_interval_updates": 10,
    "metrics_interval_updates": 1,
}


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    """Load JSON settings and fill omitted values from the defaults."""
    with Path(path).open("r", encoding="utf-8") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError("The configuration file must contain one JSON object.")
    return validate_config(_deep_merge(BASE_CONFIG, overrides))


def expected_optimizer_updates(config: dict) -> float:
    """Return the expected update count for the configured transition budget."""
    batch_size = config["batch_size"]
    if str(batch_size["type"]).lower() != "constant":
        raise ValueError(
            "step_size.final_step_ratio requires a constant batch size."
        )

    if str(config["rollout"]["scheme"]).lower() == "fixed_horizon":
        transitions_per_sample = float(config["rollout"]["horizon"])
    else:
        gamma = float(config["discount_factor"])
        sqrt_gamma = math.sqrt(gamma)
        transitions_per_sample = (
            gamma / (1.0 - gamma)
            + 1.0
            + sqrt_gamma / (1.0 - sqrt_gamma)
        )

    return float(config["transition_budget"]) / (
        float(batch_size["initial"]) * transitions_per_sample
    )


def exponential_decay_factor(config: dict, final_step_ratio: float) -> float:
    """Choose the per-update decay factor for a requested final step ratio."""
    final_step_ratio = float(final_step_ratio)
    if not 0.0 < final_step_ratio < 1.0:
        raise ValueError(
            "step_size.final_step_ratio must lie strictly between zero and one."
        )
    return final_step_ratio ** (1.0 / expected_optimizer_updates(config))


def validate_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    mode = str(config["mode"]).lower()
    if mode not in {"exact", "experimental"}:
        raise ValueError("mode must be 'exact' or 'experimental'.")
    config["mode"] = mode

    gamma = float(config["discount_factor"])
    if not 0.0 < gamma < 1.0:
        raise ValueError("discount_factor must lie strictly between zero and one.")
    config["discount_factor"] = gamma
    if int(config["transition_budget"]) <= 0:
        raise ValueError("transition_budget must be positive.")

    rollout = config["rollout"]
    rollout["scheme"] = str(rollout["scheme"]).lower()
    if rollout["scheme"] not in {"geometric", "fixed_horizon"}:
        raise ValueError("rollout.scheme must be 'geometric' or 'fixed_horizon'.")
    rollout["horizon"] = int(rollout["horizon"])
    if rollout["horizon"] <= 0:
        raise ValueError("rollout.horizon must be positive.")
    network_type = str(config["environment"]["network_type"]).lower()
    if network_type not in NETWORK_TYPES:
        raise ValueError(
            f"Unknown network '{network_type}'. Choose from {sorted(NETWORK_TYPES)}."
        )
    config["environment"]["network_type"] = network_type
    if network_type == "mixed_networks" and int(config["environment"]["num_nodes"]) != 10:
        raise ValueError("mixed_networks requires environment.num_nodes=10.")
    action_space = str(config["environment"].get("action_space_architecture", "weighted_limit")).lower()
    if action_space != "weighted_limit":
        raise ValueError("Only weighted_limit is supported as a policy action space.")
    config["environment"]["action_space_architecture"] = action_space
    forbidden = {"max_steps", "network", "gamma"}
    present = forbidden.intersection(config["environment"])
    if present:
        raise ValueError(f"Removed or ambiguous environment keys are not accepted: {sorted(present)}")

    initial_state_mode = str(config["initial_state_mode"]).lower()
    if initial_state_mode not in {"fixed", "distribution"}:
        raise ValueError("initial_state_mode must be 'fixed' or 'distribution'.")
    config["initial_state_mode"] = initial_state_mode

    policy = config["policy"]
    policy.setdefault("activation", "tanh")
    if policy.get("node_global_context", False):
        raise ValueError("Only local node heads are supported.")
    policy["family"] = str(policy["family"]).lower()
    policy["encoder"] = str(policy["encoder"]).lower()
    if policy["family"] not in {"shared", "routed", "flat"}:
        raise ValueError("Unsupported policy family.")
    graph_encoders = {"bidirectional_mpnn", "bidirectional_gatv2"}
    flat_encoders = {"mlp"}
    if policy["family"] in {"shared", "routed"} and policy["encoder"] not in graph_encoders:
        raise ValueError("Graph policy families require a bidirectional MPNN or GATv2 encoder.")
    if policy["family"] == "flat" and policy["encoder"] not in flat_encoders:
        raise ValueError("The flat policy family requires the MLP encoder.")

    step_size = config["step_size"]
    step_size["type"] = str(step_size["type"]).lower()
    if step_size["type"] not in {"constant", "polynomial", "exponential"}:
        raise ValueError(
            "step_size.type must be constant, polynomial, or exponential."
        )
    if float(step_size["initial"]) <= 0:
        raise ValueError("step_size.initial must be positive.")
    step_size.setdefault("decay_factor", 0.99)
    step_size.setdefault("final_step_ratio", None)
    final_step_ratio = step_size["final_step_ratio"]
    if final_step_ratio is not None:
        final_step_ratio = float(final_step_ratio)
        if not 0.0 < final_step_ratio < 1.0:
            raise ValueError(
                "step_size.final_step_ratio must lie strictly between zero and one."
            )
        if step_size["type"] != "exponential":
            raise ValueError(
                "step_size.final_step_ratio is only supported by the exponential schedule."
            )
        step_size["final_step_ratio"] = final_step_ratio

    batch_size = config["batch_size"]
    batch_size["type"] = str(batch_size["type"]).lower()
    if batch_size["type"] not in {"constant", "polynomial"}:
        raise ValueError("batch_size.type must be constant or polynomial.")
    if float(batch_size["initial"]) <= 0:
        raise ValueError("batch_size.initial must be positive.")
    if int(config["batch_size"]["initial"]) < 1:
        raise ValueError("batch_size.initial must be at least one.")

    if step_size["type"] == "exponential":
        if final_step_ratio is not None:
            step_size["decay_factor"] = exponential_decay_factor(
                config, final_step_ratio
            )
        else:
            decay_factor = float(step_size["decay_factor"])
            if not 0.0 < decay_factor < 1.0:
                raise ValueError(
                    "step_size.decay_factor must lie strictly between zero and one."
                )
            step_size["decay_factor"] = decay_factor

    projection = config["projection"]
    if projection["enabled"] and float(projection["default_bound"]) <= 0:
        raise ValueError("projection.default_bound must be positive.")
    for name, bound in projection["group_bounds"].items():
        if float(bound) <= 0:
            raise ValueError(f"Projection bound for '{name}' must be positive.")

    supplied_extensions = config.get("experimental", {})
    if not isinstance(supplied_extensions, dict):
        raise ValueError("experimental must be a settings object.")
    defaults = BASE_CONFIG["experimental"]
    unsupported = set(supplied_extensions) - set(defaults)
    active_unsupported = {
        key for key in unsupported
        if not (supplied_extensions[key] is None or
                isinstance(supplied_extensions[key], (bool, int, float))
                and supplied_extensions[key] == 0)
    }
    if active_unsupported:
        raise ValueError(f"Unsupported training extensions: {sorted(active_unsupported)}")
    experimental = _deep_merge(defaults, {
        key: value for key, value in supplied_extensions.items() if key in defaults
    })
    config["experimental"] = experimental
    if experimental["optimizer"] not in {"manual_sgd", "sgd", "adam"}:
        raise ValueError("experimental.optimizer must be manual_sgd, sgd, or adam.")
    if experimental["baseline"] not in {"none", "leave_one_out", "batch_mean"}:
        raise ValueError("Unsupported experimental baseline.")

    checkpoint_interval = int(config.get("checkpoint_interval_updates", 10))
    metrics_interval = int(config.get("metrics_interval_updates", 1))
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval_updates must be non-negative.")
    if metrics_interval <= 0:
        raise ValueError("metrics_interval_updates must be positive.")
    config["checkpoint_interval_updates"] = checkpoint_interval
    config["metrics_interval_updates"] = metrics_interval

    if mode == "exact":
        exact_requirements = {
            "optimizer": "manual_sgd",
            "baseline": "none",
            "standardize_advantages": False,
            "gradient_clip_norm": None,
        }
        mismatches = {
            key: (experimental.get(key), expected)
            for key, expected in exact_requirements.items()
            if experimental.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Exact mode forbids experimental extensions: {mismatches}")
        if not projection["enabled"]:
            raise ValueError("Exact mode requires projection.")
        if projection["group_bounds"]:
            raise ValueError("Exact mode uses the common W_Theta bound, not group-specific bounds.")
        if initial_state_mode != "fixed":
            raise ValueError("Exact mode requires the fixed initial state s_0.")
        if rollout["scheme"] != "geometric":
            raise ValueError("Exact mode requires geometric rollouts.")

    return config
