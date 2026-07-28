"""Global-local policies for bank capital allocation."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from .encoders import (
    BidirectionalGATv2Encoder,
    BidirectionalMPNNEncoder,
    MLPEncoder,
)
from .encoders.common import build_activation


def batch_observation(observation: Mapping[str, np.ndarray], device: torch.device):
    """Convert one Gymnasium dictionary observation to a one-item tensor batch."""
    return {
        key: torch.as_tensor(np.asarray(value), device=device).unsqueeze(0)
        for key, value in observation.items()
    }


def _readout(node_embeddings: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "sum":
        return node_embeddings.sum(dim=1)
    if kind == "mean":
        return node_embeddings.mean(dim=1)
    if kind == "min":
        return node_embeddings.min(dim=1).values
    if kind == "max":
        return node_embeddings.max(dim=1).values
    raise ValueError(f"Unknown readout '{kind}'.")


class BudgetHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_count: int,
        *,
        readout: str,
        num_layers: int,
        activation: str,
    ):
        super().__init__()
        self.readout = readout
        self.hidden_layers = nn.ModuleList(
            nn.Linear(input_dim, input_dim) for _ in range(max(0, int(num_layers) - 1))
        )
        self.activation = build_activation(activation)
        self.output = nn.Linear(input_dim, action_count)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        features = _readout(node_embeddings, self.readout)
        for layer in self.hidden_layers:
            features = self.activation(layer(features))
        return self.output(features)


class SharedNodeHead(nn.Module):
    def __init__(self, input_dim: int, action_count: int):
        super().__init__()
        self.output = nn.Linear(input_dim, action_count)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.output(node_embeddings)




class GlobalLocalPolicy(nn.Module):
    """Common masked distribution and budget-zero gradient gate."""

    def __init__(self, action_space):
        super().__init__()
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        if nvec.ndim != 1 or nvec.size < 2:
            raise ValueError("Expected MultiDiscrete([budget, node_1, ...]).")
        if not np.all(nvec[1:] == nvec[1]):
            raise ValueError("Every node-level action set must have the same cardinality.")
        self.num_budget_actions = int(nvec[0])
        self.num_nodes = int(nvec.size - 1)
        self.num_node_actions = int(nvec[1])
        self.total_logits = self.num_budget_actions + self.num_nodes * self.num_node_actions

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward_logits(self, observation_batch):
        raise NotImplementedError

    def _masked_logits(self, budget_logits, node_logits, action_masks):
        if action_masks is None:
            return budget_logits, node_logits
        mask = torch.as_tensor(action_masks, dtype=torch.bool, device=budget_logits.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        expected = self.total_logits
        if mask.ndim != 2 or mask.shape[1] != expected:
            raise ValueError(f"Action mask must have shape (batch, {expected}), got {tuple(mask.shape)}.")
        if mask.shape[0] == 1 and budget_logits.shape[0] > 1:
            mask = mask.expand(budget_logits.shape[0], -1)
        budget_mask = mask[:, : self.num_budget_actions]
        node_mask = mask[:, self.num_budget_actions :].view(
            -1, self.num_nodes, self.num_node_actions
        )
        if not budget_mask.any(dim=-1).all() or not node_mask.any(dim=-1).all():
            raise ValueError("Every categorical action dimension needs at least one valid action.")
        negative = torch.tensor(-1e9, dtype=budget_logits.dtype, device=budget_logits.device)
        return (
            budget_logits.masked_fill(~budget_mask, negative),
            node_logits.masked_fill(~node_mask, negative),
        )

    @staticmethod
    def _draw(logits, *, generator, deterministic):
        if deterministic:
            return logits.argmax(dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        samples = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
        return samples.view(probabilities.shape[:-1])

    @torch.no_grad()
    def sample_action(self, observation_batch, action_masks, *, generator, deterministic=False):
        """Draw from the policy or choose the masked maximum-probability action."""
        budget_logits, node_logits = self.forward_logits(observation_batch)
        budget_logits, node_logits = self._masked_logits(
            budget_logits, node_logits, action_masks
        )
        budget_action = self._draw(
            budget_logits, generator=generator, deterministic=deterministic
        )
        node_actions = self._draw(
            node_logits, generator=generator, deterministic=deterministic
        )
        active = budget_action.ne(0).unsqueeze(-1)
        node_actions = torch.where(active, node_actions, torch.zeros_like(node_actions))
        action = torch.cat([budget_action.unsqueeze(-1), node_actions], dim=-1)
        return action.squeeze(0).cpu().numpy().astype(np.int64, copy=False)

    def evaluate_action(self, observation_batch, action, action_masks):
        """Return the selected action's log probability for the score estimator."""
        budget_logits, node_logits = self.forward_logits(observation_batch)
        budget_logits, node_logits = self._masked_logits(
            budget_logits, node_logits, action_masks
        )
        action_tensor = torch.as_tensor(action, dtype=torch.long, device=budget_logits.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        budget_action = action_tensor[:, 0]
        node_actions = action_tensor[:, 1:]

        budget_log_probs = torch.log_softmax(budget_logits, dim=-1)
        node_log_probs = torch.log_softmax(node_logits, dim=-1)
        selected_budget = budget_log_probs.gather(1, budget_action.unsqueeze(-1)).squeeze(-1)
        selected_nodes = node_log_probs.gather(
            2, node_actions.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)
        gate = budget_action.ne(0).to(dtype=budget_logits.dtype)
        log_prob = selected_budget + gate * selected_nodes

        return log_prob.squeeze(0)


def _initialize_linear(layer: nn.Linear, gain: float):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _build_node_encoder(kind, observation_space, num_nodes, settings):
    common = dict(
        observation_space=observation_space,
        num_nodes=num_nodes,
        num_layers=int(settings["num_layers"]),
        embedding_dim=int(settings["embedding_dim"]),
        aggregation=settings.get("aggregation", "sum"),
        activation_function=settings.get("activation", "tanh"),
    )
    if kind == "bidirectional_mpnn":
        return BidirectionalMPNNEncoder(**common)
    if kind == "bidirectional_gatv2":
        common["num_heads"] = int(settings.get("num_heads", 1))
        return BidirectionalGATv2Encoder(**common)
    raise ValueError(f"Unknown graph encoder '{kind}'.")


class SharedGlobalLocalPolicy(GlobalLocalPolicy):
    def __init__(self, observation_space, action_space, settings):
        super().__init__(action_space)
        self.encoder = _build_node_encoder(
            settings["encoder"], observation_space, self.num_nodes, settings
        )
        self.budget_head = BudgetHead(
            self.encoder.output_dim,
            self.num_budget_actions,
            readout=settings["readout"],
            num_layers=settings["budget_head_num_layers"],
            activation=settings["activation"],
        )
        self.node_head = SharedNodeHead(self.encoder.output_dim, self.num_node_actions)
        self._initialize(settings)

    def _initialize(self, settings):
        for layer in self.budget_head.hidden_layers:
            _initialize_linear(layer, 1.0)
        _initialize_linear(self.budget_head.output, 0.01)
        _initialize_linear(self.node_head.output, 0.01)
        with torch.no_grad():
            if self.budget_head.output.bias is not None:
                self.budget_head.output.bias[1:] += float(settings["budget_bias_nudge"])

    def forward_logits(self, observation_batch):
        embeddings = self.encoder(observation_batch)
        return self.budget_head(embeddings), self.node_head(embeddings)


class RoutedGlobalLocalPolicy(GlobalLocalPolicy):
    def __init__(self, observation_space, action_space, settings):
        super().__init__(action_space)
        budget_settings = dict(settings)
        budget_settings.update(settings["routed"]["budget"])
        node_settings = dict(settings)
        node_settings.update(settings["routed"]["node"])
        self.budget_encoder = _build_node_encoder(
            settings["encoder"], observation_space, self.num_nodes, budget_settings
        )
        self.node_encoder = _build_node_encoder(
            settings["encoder"], observation_space, self.num_nodes, node_settings
        )
        self.budget_head = BudgetHead(
            self.budget_encoder.output_dim,
            self.num_budget_actions,
            readout=settings["readout"],
            num_layers=settings["budget_head_num_layers"],
            activation=settings["activation"],
        )
        self.node_head = SharedNodeHead(self.node_encoder.output_dim, self.num_node_actions)
        for layer in self.budget_head.hidden_layers:
            _initialize_linear(layer, 1.0)
        _initialize_linear(self.budget_head.output, 0.01)
        _initialize_linear(self.node_head.output, 0.01)
        with torch.no_grad():
            if self.budget_head.output.bias is not None:
                self.budget_head.output.bias[1:] += float(settings["budget_bias_nudge"])

    def forward_logits(self, observation_batch):
        budget_embeddings = self.budget_encoder(observation_batch)
        node_embeddings = self.node_encoder(observation_batch)
        return self.budget_head(budget_embeddings), self.node_head(node_embeddings)


class FlatGlobalLocalPolicy(GlobalLocalPolicy):
    def __init__(self, observation_space, action_space, settings):
        super().__init__(action_space)
        if settings["encoder"] == "mlp":
            self.encoder = MLPEncoder(
                observation_space,
                num_nodes=self.num_nodes,
                num_layers=settings["num_layers"],
                embedding_dim=settings["embedding_dim"],
                activation_function=settings["activation"],
            )
        else:
            raise ValueError(f"Unknown flat encoder '{settings['encoder']}'.")
        self.logit_head = nn.Linear(self.encoder.output_dim, self.total_logits)
        _initialize_linear(self.logit_head, 0.01)
        with torch.no_grad():
            if self.logit_head.bias is not None:
                self.logit_head.bias[1 : self.num_budget_actions] += float(
                    settings["budget_bias_nudge"]
                )

    def forward_logits(self, observation_batch):
        logits = self.logit_head(self.encoder(observation_batch))
        budget_logits = logits[:, : self.num_budget_actions]
        node_logits = logits[:, self.num_budget_actions :].view(
            -1, self.num_nodes, self.num_node_actions
        )
        return budget_logits, node_logits


def build_policy(config, observation_space, action_space):
    if config["environment"].get("action_space_architecture", "weighted_limit") != "weighted_limit":
        raise ValueError("Global-local policies require weighted_limit actions.")
    settings = config["policy"]
    family = settings["family"]
    if family == "shared":
        policy = SharedGlobalLocalPolicy(
            observation_space, action_space, settings
        )
    elif family == "routed":
        policy = RoutedGlobalLocalPolicy(observation_space, action_space, settings)
    elif family == "flat":
        policy = FlatGlobalLocalPolicy(observation_space, action_space, settings)
    else:
        raise ValueError(f"Unknown policy family '{family}'.")
    return policy.to(torch.device(config["device"]))
