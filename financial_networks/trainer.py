"""Mini-batch projected REINFORCE with geometric or fixed-horizon rollouts."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch

from .config import (
    MIXED_NETWORK_TYPES,
    ONE_DECISION_NETWORK_TYPES,
    validate_config,
)
from .distributions import sample_geometric_n0
from .environment import FinancialNetworkEnv
from .environment.absorbing import AbsorbingStateAdapter
from .policies import batch_observation, build_policy


def _schedule_value(settings: dict, update: int, *, integer: bool):
    initial = float(settings["initial"])
    if settings["type"] == "constant":
        value = initial
    elif settings["type"] == "polynomial":
        exponent = float(settings["exponent"])
        value = initial * (update + 1) ** (exponent if integer else -exponent)
    elif settings["type"] == "exponential" and not integer:
        value = initial * float(settings["decay_factor"]) ** update
    else:
        raise ValueError(f"Unsupported schedule type '{settings['type']}'.")
    if integer:
        value = int(math.ceil(value))
        maximum = settings.get("maximum")
        if maximum is not None:
            value = min(value, int(maximum))
        return max(1, value)
    return float(value)


def _configured_network_types(config: dict) -> tuple[str, ...]:
    network_type = config["environment"]["network_type"]
    if network_type == "mixed_networks":
        return MIXED_NETWORK_TYPES
    return (network_type,)


def _environment_kwargs(config: dict, *, network_type: str | None = None) -> dict:
    settings = dict(config["environment"])
    configured_network_type = settings.pop("network_type")
    network_type = network_type or configured_network_type
    if network_type == "mixed_networks":
        raise ValueError("A concrete network type is required for mixed_networks.")
    inverse_demand_slope = settings.pop("inverse_demand_slope")
    return {
        **settings,
        "network": "random" if network_type == "random" else "fixed",
        "network_type": network_type,
        "gamma": inverse_demand_slope,
        "action_space_architecture": settings.get("action_space_architecture", "weighted_limit"),
        "render_mode": "human",
        "save_render": False,
    }


class ProjectedReinforceTrainer:

    def __init__(self, config: dict, output_dir: str | Path, *, resume=False):
        self.config = validate_config(config)
        if self.config["environment"]["action_space_architecture"] != "weighted_limit":
            raise ValueError(
                "Global-local training requires weighted_limit actions."
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        if self.metrics_path.exists() and not resume:
            raise FileExistsError(
                f"{self.metrics_path} already exists. Choose a new output directory or resume."
            )

        self.device = torch.device(self.config["device"])
        seeds = self.config["seeds"]
        torch.manual_seed(int(seeds["model"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seeds["model"]))

        seed_sequence = np.random.SeedSequence(
            [int(seeds["model"]), int(seeds["environment"])]
        )
        horizon_seed, environment_seed = seed_sequence.spawn(2)
        self.horizon_rng = np.random.default_rng(horizon_seed)
        self.environment_seed_rng = np.random.default_rng(environment_seed)
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.action_generator = torch.Generator(device=generator_device)
        self.action_generator.manual_seed(int(seeds["model"]))

        network_seed = np.random.SeedSequence(
            [int(seeds["environment"]), int(seeds["initial_state"]), 1]
        )
        self.network_rng = np.random.default_rng(network_seed)
        self.network_types = _configured_network_types(self.config)
        self.envs = {}
        self.initial_state_snapshots = {}
        representative_raw_env = None
        for network_index, network_type in enumerate(self.network_types):
            raw_env = FinancialNetworkEnv(
                **_environment_kwargs(self.config, network_type=network_type)
            )
            env = AbsorbingStateAdapter(raw_env)
            env.reset(seed=int(seeds["initial_state"]) + network_index)
            self.envs[network_type] = env
            self.initial_state_snapshots[network_type] = (
                raw_env.capture_initial_state()
                if self.config["initial_state_mode"] == "fixed"
                else None
            )
            if representative_raw_env is None:
                representative_raw_env = raw_env

        self.active_network_type = self.network_types[0]
        self.env = self.envs[self.active_network_type]
        self.initial_state_snapshot = self._checkpoint_initial_state_snapshot()
        self._warn_for_one_decision_network()
        self.policy = build_policy(
            self.config,
            representative_raw_env.observation_space,
            representative_raw_env.action_space,
        )

        self.optimizer = self._build_optimizer()
        self.update = 0
        self.transitions = 0
        self.environment_transitions = 0
        self._last_metrics = None
        self._last_logged_update = 0
        self._project_parameters()
        if not resume:
            (self.output_dir / "resolved_config.json").write_text(
                json.dumps(self.config, indent=2, sort_keys=True), encoding="utf-8"
            )

    def _warn_for_one_decision_network(self):
        network_type = self.config["environment"]["network_type"]
        rollout = self.config["rollout"]
        unsuitable_rollout = rollout["scheme"] == "geometric" or (
            rollout["scheme"] == "fixed_horizon" and int(rollout["horizon"]) > 10
        )
        if network_type in ONE_DECISION_NETWORK_TYPES and unsuitable_rollout:
            warnings.warn(
                f"Network '{network_type}' has meaningful transition dynamics only "
                "for the first decision. It should be trained with a short fixed "
                "horizon instead of geometric rollouts or a long fixed horizon.",
                UserWarning,
                stacklevel=2,
            )

    def _checkpoint_initial_state_snapshot(self):
        if len(self.network_types) == 1:
            return self.initial_state_snapshots[self.network_types[0]]
        return dict(self.initial_state_snapshots)

    def _build_optimizer(self):
        optimizer_name = self.config["experimental"]["optimizer"]
        initial_lr = float(self.config["step_size"]["initial"])
        if optimizer_name == "manual_sgd":
            return None
        if optimizer_name == "sgd":
            return torch.optim.SGD(self.policy.parameters(), lr=initial_lr)
        if optimizer_name == "adam":
            return torch.optim.Adam(self.policy.parameters(), lr=initial_lr)
        raise ValueError(f"Unknown optimizer '{optimizer_name}'.")

    def _reset_rollout(self):
        if len(self.network_types) > 1:
            network_index = int(self.network_rng.integers(0, len(self.network_types)))
            self.active_network_type = self.network_types[network_index]
            self.env = self.envs[self.active_network_type]
        seed = int(self.environment_seed_rng.integers(0, 2**31 - 1))
        snapshot = self.initial_state_snapshots[self.active_network_type]
        if snapshot is None:
            observation, _ = self.env.reset(seed=seed)
        else:
            observation, _ = self.env.reset_to_initial_state(
                snapshot, seed=seed
            )
        return observation

    def _sample_action(self, observation):
        observation_batch = batch_observation(observation, self.device)
        masks = self.env.action_masks()
        return self.policy.sample_action(
            observation_batch,
            masks,
            generator=self.action_generator,
            deterministic=False,
        )

    def _sample_and_evaluate_action(self, observation):
        observation_batch = batch_observation(observation, self.device)
        masks = self.env.action_masks()
        action = self.policy.sample_action(
            observation_batch,
            masks,
            generator=self.action_generator,
            deterministic=False,
        )
        log_prob = self.policy.evaluate_action(
            observation_batch, action, masks
        )
        return action, log_prob

    def _one_geometric_gradient_sample(self):
        gamma = self.config["discount_factor"]
        first_horizon = sample_geometric_n0(self.horizon_rng, 1.0 - gamma)
        observation = self._reset_rollout()
        environment_transitions = 0
        terminated = False

        for _ in range(first_horizon):
            action = self._sample_action(observation)
            observation, _, terminated, _, _ = self.env.step(action)
            environment_transitions += 1
            if terminated:
                break

        final_action, log_prob = self._sample_and_evaluate_action(observation)

        second_horizon = sample_geometric_n0(
            self.horizon_rng, 1.0 - math.sqrt(gamma)
        )
        formal_transitions = first_horizon + 1 + second_horizon
        if terminated:
            return {
                "log_prob": log_prob,
                "q_estimate": 0.0,
                "first_horizon": first_horizon,
                "second_horizon": second_horizon,
                "transitions": formal_transitions,
                "environment_transitions": environment_transitions,
                "network_type": self.active_network_type,
            }

        observation, reward, terminated, _, _ = self.env.step(final_action)
        environment_transitions += 1
        q_estimate = float(reward)
        weight = math.sqrt(gamma)

        for _ in range(second_horizon):
            if terminated:
                break
            action = self._sample_action(observation)
            observation, reward, terminated, _, _ = self.env.step(action)
            environment_transitions += 1
            q_estimate += weight * float(reward)
            weight *= math.sqrt(gamma)

        return {
            "log_prob": log_prob,
            "q_estimate": q_estimate,
            "first_horizon": first_horizon,
            "second_horizon": second_horizon,
            "transitions": formal_transitions,
            "environment_transitions": environment_transitions,
            "network_type": self.active_network_type,
        }

    def _one_fixed_horizon_trajectory(self):
        gamma = self.config["discount_factor"]
        horizon = int(self.config["rollout"]["horizon"])
        observation = self._reset_rollout()
        log_probs = []
        rewards = []

        for _ in range(horizon):
            action, log_prob = self._sample_and_evaluate_action(observation)
            observation, reward, terminated, _, _ = self.env.step(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            if terminated:
                break

        returns = []
        reward_to_go = 0.0
        for reward in reversed(rewards):
            reward_to_go = reward + gamma * reward_to_go
            returns.append(reward_to_go)
        returns.reverse()

        return {
            "log_probs": torch.stack(log_probs),
            "returns": torch.tensor(
                returns, dtype=torch.float32, device=self.device
            ),
            "outer_discounts": torch.tensor(
                [gamma**time for time in range(len(rewards))],
                dtype=torch.float32,
                device=self.device,
            ),
            "trajectory_length": len(rewards),
            "transitions": horizon,
            "environment_transitions": len(rewards),
            "network_type": self.active_network_type,
        }

    def _advantages(self, returns: torch.Tensor) -> torch.Tensor:
        settings = self.config["experimental"]
        baseline = settings["baseline"]
        if baseline == "none":
            advantages = returns
        elif baseline == "batch_mean":
            advantages = returns - returns.mean()
        elif baseline == "leave_one_out":
            if returns.numel() < 2:
                raise ValueError("leave_one_out baseline requires a batch size of at least two.")
            advantages = returns - (returns.sum() - returns) / (returns.numel() - 1)
        else:
            raise ValueError(f"Unknown baseline '{baseline}'.")

        if settings["standardize_advantages"]:
            standard_deviation = advantages.std(unbiased=False)
            advantages = (advantages - advantages.mean()) / standard_deviation.clamp_min(1e-8)
        return advantages

    def _fixed_horizon_advantages(self, trajectories: list[dict]) -> torch.Tensor:
        """Return fixed-horizon coefficients, using a trajectory-level baseline for LOO."""
        returns = torch.cat([trajectory["returns"] for trajectory in trajectories])
        settings = self.config["experimental"]
        if settings["baseline"] != "leave_one_out":
            return self._advantages(returns)

        if len(trajectories) < 2:
            raise ValueError(
                "leave_one_out baseline requires a batch size of at least two."
            )
        trajectory_returns = torch.stack(
            [trajectory["returns"][0] for trajectory in trajectories]
        )
        baselines = (
            trajectory_returns.sum() - trajectory_returns
        ) / (len(trajectories) - 1)
        advantages = torch.cat(
            [
                trajectory["returns"] - baselines[index]
                for index, trajectory in enumerate(trajectories)
            ]
        )
        if settings["standardize_advantages"]:
            standard_deviation = advantages.std(unbiased=False)
            advantages = (
                advantages - advantages.mean()
            ) / standard_deviation.clamp_min(1e-8)
        return advantages

    @staticmethod
    def _gradient_norm(parameters):
        squared_norm = torch.zeros((), dtype=torch.float64)
        for parameter in parameters:
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().double().pow(2).sum().cpu()
        return float(torch.sqrt(squared_norm))

    def _apply_update(self, objective: torch.Tensor, step_size: float):
        self.policy.zero_grad(set_to_none=True)
        if self.optimizer is None:
            objective.backward()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            (-objective).backward()

        clip_norm = self.config["experimental"]["gradient_clip_norm"]
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(clip_norm))
        gradient_norm = self._gradient_norm(self.policy.parameters())

        if self.optimizer is None:
            with torch.no_grad():
                for parameter in self.policy.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=step_size)
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = step_size
            self.optimizer.step()

        projection_hits = self._project_parameters()
        return gradient_norm, projection_hits

    def _projection_bound(self, parameter_name: str) -> float:
        projection = self.config["projection"]
        group = parameter_name.split(".", 1)[0]
        return float(projection["group_bounds"].get(group, projection["default_bound"]))

    def _project_parameters(self):
        if not self.config["projection"]["enabled"]:
            return 0
        hits = 0
        with torch.no_grad():
            for name, parameter in self.policy.named_parameters():
                bound = self._projection_bound(name)
                hits += int((parameter.abs() > bound).sum().item())
                parameter.clamp_(-bound, bound)
        return hits

    def _max_parameter_absolute_value(self):
        with torch.no_grad():
            return max(float(parameter.abs().max()) for parameter in self.policy.parameters())

    def train_one_update(self):
        batch_size = _schedule_value(
            self.config["batch_size"], self.update, integer=True
        )
        step_size = _schedule_value(
            self.config["step_size"], self.update, integer=False
        )
        rollout_scheme = self.config["rollout"]["scheme"]
        scheme_metrics = {}

        if rollout_scheme == "geometric":
            samples = [
                self._one_geometric_gradient_sample() for _ in range(batch_size)
            ]
            returns = torch.tensor(
                [sample["q_estimate"] for sample in samples],
                dtype=torch.float32,
                device=self.device,
            )
            advantages = self._advantages(returns).detach()
            log_probs = torch.stack([sample["log_prob"] for sample in samples])
            score_objective = (
                log_probs * advantages / (1.0 - self.config["discount_factor"])
            ).mean()
            transition_increment = sum(sample["transitions"] for sample in samples)
            environment_transition_increment = sum(
                sample["environment_transitions"] for sample in samples
            )
            scheme_metrics = {
                "mean_first_horizon": float(
                    np.mean([sample["first_horizon"] for sample in samples])
                ),
                "mean_second_horizon": float(
                    np.mean([sample["second_horizon"] for sample in samples])
                ),
                "network_counts": {
                    network_type: sum(
                        sample.get("network_type", self.active_network_type)
                        == network_type
                        for sample in samples
                    )
                    for network_type in self.network_types
                },
            }
        elif rollout_scheme == "fixed_horizon":
            trajectories = [
                self._one_fixed_horizon_trajectory() for _ in range(batch_size)
            ]
            returns = torch.cat(
                [trajectory["returns"] for trajectory in trajectories]
            )
            advantages = self._fixed_horizon_advantages(trajectories).detach()
            log_probs = torch.cat(
                [trajectory["log_probs"] for trajectory in trajectories]
            )
            outer_discounts = torch.cat(
                [trajectory["outer_discounts"] for trajectory in trajectories]
            )
            score_objective = (
                log_probs * outer_discounts * advantages
            ).sum() / batch_size
            transition_increment = sum(
                trajectory["transitions"] for trajectory in trajectories
            )
            environment_transition_increment = sum(
                trajectory["environment_transitions"] for trajectory in trajectories
            )
            scheme_metrics = {
                "fixed_horizon": int(self.config["rollout"]["horizon"]),
                "mean_trajectory_length": float(
                    np.mean(
                        [trajectory["trajectory_length"] for trajectory in trajectories]
                    )
                ),
                "network_counts": {
                    network_type: sum(
                        trajectory.get("network_type", self.active_network_type)
                        == network_type
                        for trajectory in trajectories
                    )
                    for network_type in self.network_types
                },
            }
        else:
            raise RuntimeError(f"Unsupported rollout scheme '{rollout_scheme}'.")

        self.transitions += transition_increment
        self.environment_transitions += environment_transition_increment
        objective = score_objective
        gradient_norm, projection_hits = self._apply_update(objective, step_size)

        self.update += 1
        metrics = {
            "update": self.update,
            "rollout_scheme": rollout_scheme,
            "transition_budget": int(self.config["transition_budget"]),
            "transitions": self.transitions,
            "transition_increment": transition_increment,
            "environment_transitions": self.environment_transitions,
            "environment_transition_increment": environment_transition_increment,
            "batch_size": batch_size,
            "step_size": step_size,
            "mean_q_estimate": float(returns.mean()),
            "mean_advantage": float(advantages.mean()),
            "objective": float(objective.detach()),
            "score_objective": float(score_objective.detach()),
            "gradient_norm": gradient_norm,
            "projection_hits": projection_hits,
            "max_parameter_abs": self._max_parameter_absolute_value(),
            **scheme_metrics,
        }
        self._last_metrics = metrics
        interval = int(self.config["metrics_interval_updates"])
        if self.update % interval == 0:
            self._write_metrics(metrics)
        return metrics

    def _write_metrics(self, metrics):
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        self._last_logged_update = int(metrics["update"])

    def train(self, target_transition_budget=None):
        configured_budget = int(self.config["transition_budget"])
        if target_transition_budget is None:
            target_budget = configured_budget
        else:
            target_budget = int(target_transition_budget)
            if target_budget <= 0:
                raise ValueError("target_transition_budget must be positive.")
            if target_budget > configured_budget:
                raise ValueError(
                    "target_transition_budget cannot exceed the configured transition budget."
                )

        while self.transitions < target_budget:
            metrics = self.train_one_update()
            interval = int(self.config["checkpoint_interval_updates"])
            if interval > 0 and self.update % interval == 0:
                self.save_checkpoint(self.output_dir / "latest_checkpoint.pt")

        if self._last_metrics is not None and self._last_logged_update != self.update:
            self._write_metrics(self._last_metrics)

        complete = self.transitions >= configured_budget
        checkpoint_name = "final_checkpoint.pt" if complete else "latest_checkpoint.pt"
        self.save_checkpoint(self.output_dir / checkpoint_name)
        summary = {
            "updates": self.update,
            "configured_transition_budget": configured_budget,
            "target_transition_budget": target_budget,
            "actual_transitions": self.transitions,
            "actual_environment_transitions": self.environment_transitions,
            "overshoot": self.transitions - target_budget,
            "complete": complete,
            "mode": self.config["mode"],
            "initial_state_mode": self.config["initial_state_mode"],
            "rollout_scheme": self.config["rollout"]["scheme"],
            "network_setting": self.config["environment"]["network_type"],
            "network_types": list(self.network_types),
        }
        summary_name = "summary.json" if complete else "progress_summary.json"
        (self.output_dir / summary_name).write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if complete:
            latest_checkpoint = self.output_dir / "latest_checkpoint.pt"
            latest_checkpoint.unlink(missing_ok=True)
            (self.output_dir / "progress_summary.json").unlink(missing_ok=True)
        return summary

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        payload = {
            "config": self.config,
            "policy_state": self.policy.state_dict(),
            "optimizer_state": None if self.optimizer is None else self.optimizer.state_dict(),
            "update": self.update,
            "transitions": self.transitions,
            "environment_transitions": self.environment_transitions,
            "horizon_rng_state": self.horizon_rng.bit_generator.state,
            "environment_seed_rng_state": self.environment_seed_rng.bit_generator.state,
            "network_rng_state": self.network_rng.bit_generator.state,
            "action_generator_state": self.action_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "initial_state_snapshot": self.initial_state_snapshot,
        }
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if validate_config(payload["config"]) != self.config:
            raise ValueError("Checkpoint configuration does not match the resolved configuration.")
        self.policy.load_state_dict(payload["policy_state"])
        if self.optimizer is not None and payload["optimizer_state"] is not None:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.update = int(payload["update"])
        self.transitions = int(payload["transitions"])
        self.environment_transitions = int(
            payload.get("environment_transitions", payload["transitions"])
        )
        self.horizon_rng.bit_generator.state = payload["horizon_rng_state"]
        self.environment_seed_rng.bit_generator.state = payload[
            "environment_seed_rng_state"
        ]
        if "network_rng_state" in payload:
            self.network_rng.bit_generator.state = payload["network_rng_state"]
        self.action_generator.set_state(payload["action_generator_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        self.initial_state_snapshot = payload["initial_state_snapshot"]
        self._last_logged_update = self.update
        if len(self.network_types) == 1:
            self.initial_state_snapshots = {
                self.network_types[0]: self.initial_state_snapshot
            }
        else:
            self.initial_state_snapshots = dict(self.initial_state_snapshot)

    def close(self):
        for env in self.envs.values():
            env.close()
