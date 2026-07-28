"""Fixed-horizon evaluation for trained policies."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from heuristic_agents import (
    LargestDeficitAgent,
    UniformSolventAgent,
    ZeroInjectionAgent,
)

from .config import load_config, validate_config
from .environment import FinancialNetworkEnv
from .environment.absorbing import AbsorbingStateAdapter
from .policies import batch_observation, build_policy
from .trainer import _configured_network_types, _environment_kwargs


EVALUATION_MODES = ("deterministic", "stochastic")
HEURISTIC_NAMES = ("zero", "greedy", "uniform_solvent")
HEURISTIC_LABELS = {
    "zero": "Zero",
    "greedy": "Greedy",
    "uniform_solvent": "Uniform solvent",
}
HEURISTIC_CLASSES = {
    "zero": ZeroInjectionAgent,
    "greedy": LargestDeficitAgent,
    "uniform_solvent": UniformSolventAgent,
}


class _RawEnvironmentSlot:
    def __init__(self, env):
        self.env = env


class _SingleEnvironmentVectorAdapter:
    def __init__(self, env):
        self.envs = [_RawEnvironmentSlot(env)]


def _vectorized_observation(observation: dict) -> dict:
    return {
        key: np.expand_dims(np.asarray(value), axis=0)
        for key, value in observation.items()
    }


def _heuristic_action(agent, observation: dict, env) -> np.ndarray:
    direct_action, _ = agent.predict(_vectorized_observation(observation))
    direct_action = np.asarray(direct_action, dtype=np.int64).reshape(-1)
    if env.action_space_architecture != "weighted_limit":
        return direct_action

    action = np.zeros(env.action_space.shape, dtype=np.int64)
    selected = np.flatnonzero(direct_action > 0)
    if len(selected) == 0:
        return action
    if env.num_action_weights < 2:
        raise ValueError(
            "Heuristic evaluation in the weighted action space requires "
            "at least two node-weight levels."
        )
    action[0] = len(env.bins_action) - 1
    action[1 + selected] = 1
    return action




def _normalize_modes(modes: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(mode).lower() for mode in modes))
    if not normalized:
        raise ValueError("At least one evaluation mode is required.")
    unknown = set(normalized).difference(EVALUATION_MODES)
    if unknown:
        raise ValueError(
            f"Unknown evaluation modes {sorted(unknown)}; choose from {EVALUATION_MODES}."
        )
    return normalized


def _summary(episodes: list[dict]) -> dict:
    discounted_returns = np.asarray(
        [episode["discounted_return"] for episode in episodes], dtype=np.float64
    )
    undiscounted_returns = np.asarray(
        [episode["undiscounted_return"] for episode in episodes], dtype=np.float64
    )
    lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.float64)
    summary = {
        "mean_discounted_return": float(discounted_returns.mean()),
        "std_discounted_return": float(discounted_returns.std(ddof=0)),
        "mean_undiscounted_return": float(undiscounted_returns.mean()),
        "std_undiscounted_return": float(undiscounted_returns.std(ddof=0)),
        "mean_episode_length": float(lengths.mean()),
        "termination_rate": float(
            np.mean([episode["terminated"] for episode in episodes])
        ),
        "episodes": episodes,
    }
    optional_means = {
        "total_injected_capital": "mean_total_injected_capital",
        "final_default_count": "mean_final_default_count",
        "new_default_count": "mean_new_default_count",
        "final_price_risky_asset": "mean_final_price_risky_asset",
    }
    for episode_key, summary_key in optional_means.items():
        if all(episode_key in episode for episode in episodes):
            summary[summary_key] = float(
                np.mean([episode[episode_key] for episode in episodes])
            )
    return summary


def _default_count(env) -> int:
    return int(
        sum(
            values.get("solvency") == 1
            for values in env.node_features.values()
        )
    )


def _solvent_count(env) -> int:
    return int(
        sum(
            values.get("solvency") == 0
            for values in env.node_features.values()
        )
    )


def _capital_values(env, action_indices):
    if hasattr(env, "_capital_from_action_indices"):
        return env._capital_from_action_indices(action_indices)
    bins = np.asarray(getattr(env, "bins_action", [0.0]), dtype=np.float32)
    indices = np.clip(np.asarray(action_indices, dtype=int), 0, len(bins) - 1)
    return bins[indices].astype(np.float32)





def _estimated_injected_capital_from_action(env, action) -> float:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    observed_nodes = list(env.observed_nodes)
    num_observed = int(env.num_observed_nodes)

    if env.action_space_architecture == "weighted_limit":
        if len(action) == 0:
            return 0.0
        weights = action[1 : 1 + num_observed].copy()
        for observed_index, node in enumerate(observed_nodes):
            if env.node_features[node]["solvency"] > 0:
                weights[observed_index] = 0
        if float(np.sum(weights)) <= 0:
            return 0.0
        return float(_capital_values(env, [action[0]])[0])

    return 0.0


@torch.no_grad()
def evaluate_policy(
    policy,
    config: dict,
    *,
    seeds: Iterable[int],
    horizon: int = 100,
    modes: Iterable[str] = EVALUATION_MODES,
    initial_state_snapshot=None,
) -> dict:
    """Evaluate one policy deterministically, stochastically, or in both modes."""
    config = validate_config(config)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("Evaluation horizon must be positive.")
    modes = _normalize_modes(modes)

    device = torch.device(config["device"])
    policy = policy.to(device)
    policy.eval()
    results = {
        "horizon": horizon,
        "discount_factor": float(config["discount_factor"]),
        "seeds": list(seeds),
        "modes": {},
    }

    network_types = _configured_network_types(config)
    for mode in modes:
        raw_envs = {
            network_type: FinancialNetworkEnv(
                **_environment_kwargs(config, network_type=network_type)
            )
            for network_type in network_types
        }
        envs = {
            network_type: AbsorbingStateAdapter(raw_env)
            for network_type, raw_env in raw_envs.items()
        }
        episodes = []
        try:
            if len(network_types) == 1:
                snapshots = {network_types[0]: initial_state_snapshot}
            else:
                snapshots = dict(initial_state_snapshot or {})

            if config["initial_state_mode"] == "fixed":
                for network_index, network_type in enumerate(network_types):
                    if snapshots.get(network_type) is None:
                        envs[network_type].reset(
                            seed=int(config["seeds"]["initial_state"]) + network_index
                        )
                        snapshots[network_type] = raw_envs[
                            network_type
                        ].capture_initial_state()

            for episode_index, seed in enumerate(seeds):
                network_type = network_types[episode_index % len(network_types)]
                env = envs[network_type]
                if config["initial_state_mode"] == "fixed":
                    observation, _ = env.reset_to_initial_state(
                        snapshots[network_type], seed=seed
                    )
                else:
                    observation, _ = env.reset(seed=seed)

                generator_device = (
                    device if device.type == "cuda" else torch.device("cpu")
                )
                generator = torch.Generator(device=generator_device)
                generator.manual_seed(seed + 1_000_003)
                discounted_return = 0.0
                undiscounted_return = 0.0
                discount = 1.0
                budget_actions = []
                terminated = False
                raw_env = raw_envs[network_type]
                initial_default_count = _default_count(raw_env)
                initial_solvent_count = _solvent_count(raw_env)
                total_injected_capital = 0.0
                final_default_count = initial_default_count
                final_solvent_count = initial_solvent_count
                final_price_risky_asset = float(raw_env.price_risky)

                for step in range(horizon):
                    observation_batch = batch_observation(observation, device)
                    action = policy.sample_action(
                        observation_batch,
                        env.action_masks(),
                        generator=generator,
                        deterministic=mode == "deterministic",
                    )
                    injected_capital = _estimated_injected_capital_from_action(
                        raw_env, action
                    )
                    observation, reward, terminated, _, _ = env.step(action)
                    reward = float(reward)
                    discounted_return += discount * reward
                    undiscounted_return += reward
                    discount *= float(config["discount_factor"])
                    budget_actions.append(int(action[0]))
                    total_injected_capital += injected_capital
                    final_default_count = _default_count(raw_env)
                    final_solvent_count = _solvent_count(raw_env)
                    final_price_risky_asset = float(raw_env.price_risky)
                    if terminated:
                        break

                episodes.append(
                    {
                        "seed": seed,
                        "network_type": network_type,
                        "discounted_return": discounted_return,
                        "undiscounted_return": undiscounted_return,
                        "length": step + 1,
                        "terminated": bool(terminated),
                        "mean_budget_action": float(np.mean(budget_actions)),
                        "nonzero_budget_rate": float(np.mean(np.asarray(budget_actions) != 0)),
                        "total_injected_capital": total_injected_capital,
                        "mean_injected_capital_per_step": total_injected_capital
                        / (step + 1),
                        "budget_utilization": total_injected_capital
                        / (float(raw_env.upper_limit_capital_injections) * (step + 1)),
                        "initial_default_count": initial_default_count,
                        "final_default_count": final_default_count,
                        "new_default_count": final_default_count
                        - initial_default_count,
                        "initial_solvent_count": initial_solvent_count,
                        "final_solvent_count": final_solvent_count,
                        "final_price_risky_asset": final_price_risky_asset,
                    }
                )
        finally:
            for env in envs.values():
                env.close()

        results["modes"][mode] = _summary(episodes)

    return results


def evaluate_heuristic(
    config: dict,
    heuristic: str,
    *,
    seeds: Iterable[int],
    horizon: int = 100,
    initial_state_snapshot=None,
) -> dict:
    """Evaluate one heuristic with the same reset and horizon rules as a policy."""
    config = validate_config(config)
    heuristic = str(heuristic).lower()
    if heuristic not in HEURISTIC_NAMES:
        raise ValueError(
            f"Unknown heuristic '{heuristic}'. Choose from {HEURISTIC_NAMES}."
        )
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("Evaluation horizon must be positive.")

    network_types = _configured_network_types(config)
    raw_envs = {
        network_type: FinancialNetworkEnv(
            **_environment_kwargs(config, network_type=network_type)
        )
        for network_type in network_types
    }
    envs = {
        network_type: AbsorbingStateAdapter(raw_env)
        for network_type, raw_env in raw_envs.items()
    }
    agents = {
        network_type: HEURISTIC_CLASSES[heuristic](
            _SingleEnvironmentVectorAdapter(raw_env)
        )
        for network_type, raw_env in raw_envs.items()
    }
    episodes = []
    try:
        if len(network_types) == 1:
            snapshots = {network_types[0]: initial_state_snapshot}
        else:
            snapshots = dict(initial_state_snapshot or {})

        if config["initial_state_mode"] == "fixed":
            for network_index, network_type in enumerate(network_types):
                if snapshots.get(network_type) is None:
                    envs[network_type].reset(
                        seed=int(config["seeds"]["initial_state"])
                        + network_index
                    )
                    snapshots[network_type] = raw_envs[
                        network_type
                    ].capture_initial_state()

        for episode_index, seed in enumerate(seeds):
            network_type = network_types[episode_index % len(network_types)]
            env = envs[network_type]
            raw_env = raw_envs[network_type]
            if config["initial_state_mode"] == "fixed":
                observation, _ = env.reset_to_initial_state(
                    snapshots[network_type], seed=seed
                )
            else:
                observation, _ = env.reset(seed=seed)

            discounted_return = 0.0
            undiscounted_return = 0.0
            discount = 1.0
            budget_actions = []
            terminated = False
            initial_default_count = _default_count(raw_env)
            initial_solvent_count = _solvent_count(raw_env)
            final_default_count = initial_default_count
            final_solvent_count = initial_solvent_count
            final_price_risky_asset = float(raw_env.price_risky)
            total_injected_capital = 0.0

            for step in range(horizon):
                action = _heuristic_action(
                    agents[network_type], observation, raw_env
                )
                injected_capital = _estimated_injected_capital_from_action(
                    raw_env, action
                )
                observation, reward, terminated, _, _ = env.step(action)
                reward = float(reward)
                discounted_return += discount * reward
                undiscounted_return += reward
                discount *= float(config["discount_factor"])
                budget_actions.append(int(action[0]))
                total_injected_capital += injected_capital
                final_default_count = _default_count(raw_env)
                final_solvent_count = _solvent_count(raw_env)
                final_price_risky_asset = float(raw_env.price_risky)
                if terminated:
                    break

            episodes.append(
                {
                    "seed": seed,
                    "network_type": network_type,
                    "discounted_return": discounted_return,
                    "undiscounted_return": undiscounted_return,
                    "length": step + 1,
                    "terminated": bool(terminated),
                    "mean_budget_action": float(np.mean(budget_actions)),
                    "nonzero_budget_rate": float(
                        np.mean(np.asarray(budget_actions) != 0)
                    ),
                    "total_injected_capital": total_injected_capital,
                    "mean_injected_capital_per_step": total_injected_capital
                    / (step + 1),
                    "budget_utilization": total_injected_capital
                    / (
                        float(raw_env.upper_limit_capital_injections)
                        * (step + 1)
                    ),
                    "initial_default_count": initial_default_count,
                    "final_default_count": final_default_count,
                    "new_default_count": final_default_count
                    - initial_default_count,
                    "initial_solvent_count": initial_solvent_count,
                    "final_solvent_count": final_solvent_count,
                    "final_price_risky_asset": final_price_risky_asset,
                }
            )
    finally:
        for env in envs.values():
            env.close()

    return {
        "heuristic": heuristic,
        "heuristic_label": HEURISTIC_LABELS[heuristic],
        "horizon": horizon,
        "discount_factor": float(config["discount_factor"]),
        "seeds": list(seeds),
        "modes": {"deterministic": _summary(episodes)},
    }



def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    seeds: Iterable[int],
    horizon: int = 100,
    modes: Iterable[str] = EVALUATION_MODES,
    device: str | None = None,
) -> dict:
    """Load a checkpoint and evaluate its policy without resuming training."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    if device is not None:
        config["device"] = device
    config = validate_config(config)

    representative_network_type = _configured_network_types(config)[0]
    raw_env = FinancialNetworkEnv(
        **_environment_kwargs(config, network_type=representative_network_type)
    )
    try:
        policy = build_policy(config, raw_env.observation_space, raw_env.action_space)
    finally:
        raw_env.close()
    policy.load_state_dict(payload["policy_state"])

    results = evaluate_policy(
        policy,
        config,
        seeds=seeds,
        horizon=horizon,
        modes=modes,
        initial_state_snapshot=payload.get("initial_state_snapshot"),
    )
    results["checkpoint"] = str(checkpoint_path)
    return results


def evaluate_checkpoint_on_reference_environment(
    checkpoint_path: str | Path,
    reference_checkpoint_path: str | Path,
    *,
    seeds: Iterable[int],
    horizon: int = 100,
    modes: Iterable[str] = EVALUATION_MODES,
    device: str | None = None,
) -> dict:
    """Evaluate a policy using another checkpoint's environment and fixed state."""
    checkpoint_path = Path(checkpoint_path)
    reference_checkpoint_path = Path(reference_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference_payload = torch.load(
        reference_checkpoint_path, map_location="cpu", weights_only=False
    )

    policy_config = copy.deepcopy(payload["config"])
    evaluation_config = copy.deepcopy(reference_payload["config"])
    if device is not None:
        policy_config["device"] = device
        evaluation_config["device"] = device
    policy_config = validate_config(policy_config)
    evaluation_config = validate_config(evaluation_config)

    representative_network_type = _configured_network_types(policy_config)[0]
    raw_env = FinancialNetworkEnv(
        **_environment_kwargs(
            policy_config, network_type=representative_network_type
        )
    )
    try:
        policy = build_policy(
            policy_config, raw_env.observation_space, raw_env.action_space
        )
    finally:
        raw_env.close()
    policy.load_state_dict(payload["policy_state"])

    results = evaluate_policy(
        policy,
        evaluation_config,
        seeds=seeds,
        horizon=horizon,
        modes=modes,
        initial_state_snapshot=reference_payload.get(
            "initial_state_snapshot"
        ),
    )
    results["checkpoint"] = str(checkpoint_path)
    results["reference_checkpoint"] = str(reference_checkpoint_path)
    return results


def evaluate_checkpoint_on_environment_config(
    checkpoint_path: str | Path,
    evaluation_config: dict,
    *,
    seeds: Iterable[int],
    horizon: int = 100,
    modes: Iterable[str] = EVALUATION_MODES,
    device: str | None = None,
) -> dict:
    """Evaluate a checkpoint using an explicitly supplied environment configuration."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_config = copy.deepcopy(payload["config"])
    evaluation_config = copy.deepcopy(evaluation_config)
    if device is not None:
        policy_config["device"] = device
        evaluation_config["device"] = device
    policy_config = validate_config(policy_config)
    evaluation_config = validate_config(evaluation_config)

    representative_network_type = _configured_network_types(policy_config)[0]
    raw_env = FinancialNetworkEnv(
        **_environment_kwargs(
            policy_config, network_type=representative_network_type
        )
    )
    try:
        policy = build_policy(
            policy_config, raw_env.observation_space, raw_env.action_space
        )
    finally:
        raw_env.close()
    policy.load_state_dict(payload["policy_state"])

    results = evaluate_policy(
        policy,
        evaluation_config,
        seeds=seeds,
        horizon=horizon,
        modes=modes,
    )
    results["checkpoint"] = str(checkpoint_path)
    results["reference_environment"] = evaluation_config["environment"][
        "network_type"
    ]
    return results


def evaluate_heuristic_checkpoint(
    checkpoint_path: str | Path,
    heuristic: str,
    *,
    seeds: Iterable[int],
    horizon: int = 100,
    device: str | None = None,
) -> dict:
    """Use a checkpoint's environment and initial-state snapshot."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    if device is not None:
        config["device"] = device
    result = evaluate_heuristic(
        config,
        heuristic,
        seeds=seeds,
        horizon=horizon,
        initial_state_snapshot=payload.get("initial_state_snapshot"),
    )
    result["reference_checkpoint"] = str(checkpoint_path)
    return result


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated seed.")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--environment-config", help="Optional JSON configuration for the evaluation environment.")
    target.add_argument("--reference-checkpoint", help="Optional checkpoint supplying the target environment and fixed state.")
    parser.add_argument("--seeds", type=_parse_seeds, default=(2101, 2102, 2103, 2104, 2105))
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("both", *EVALUATION_MODES),
        default="both",
    )
    parser.add_argument("--device")
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    modes = EVALUATION_MODES if args.mode == "both" else (args.mode,)
    options = {"seeds": args.seeds, "horizon": args.horizon,
               "modes": modes, "device": args.device}
    if args.environment_config is not None:
        results = evaluate_checkpoint_on_environment_config(
            args.checkpoint, load_config(args.environment_config), **options)
    elif args.reference_checkpoint is not None:
        results = evaluate_checkpoint_on_reference_environment(
            args.checkpoint, args.reference_checkpoint, **options)
    else:
        results = evaluate_checkpoint(args.checkpoint, **options)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
