"""Heuristic allocation rules used as baselines."""

# Each entry gives the position of a bank's chosen amount in env.bins_action.

import numpy as np
import torch


def _get_num_observed_nodes(env):
    return int(getattr(env, "num_observed_nodes", env.num_nodes))


def _extract_observed_state(observation, num_observed_nodes):
    """Convert a vectorized observation into node-wise arrays."""
    node_features = observation["node_features"][0]
    int_node_features = observation["int_node_features"][0]
    price = observation["price_risky_asset"][0]

    if isinstance(node_features, torch.Tensor):
        node_features = node_features.detach().cpu().numpy()
    if isinstance(int_node_features, torch.Tensor):
        int_node_features = int_node_features.detach().cpu().numpy()
    if isinstance(price, torch.Tensor):
        price = price.detach().cpu().numpy()

    node_features = np.asarray(node_features, dtype=float)
    int_node_features = np.asarray(int_node_features, dtype=int)
    price = float(np.asarray(price).reshape(-1)[0])

    num_observed_nodes = int(num_observed_nodes)
    if num_observed_nodes <= 0:
        raise ValueError("Observed state does not contain any nodes.")
    if node_features.shape[0] % num_observed_nodes != 0:
        raise ValueError("Observed node features cannot be reshaped consistently.")

    node_feature_dim = node_features.shape[0] // num_observed_nodes
    node_features = node_features.reshape(num_observed_nodes, node_feature_dim)
    int_feature_dim = int_node_features.shape[0] // num_observed_nodes
    int_node_features = int_node_features.reshape(num_observed_nodes, int_feature_dim)
    solvency = int_node_features[:, 0]
    return node_features, solvency, price


class UniformSolventAgent:
    """Split the capital limit uniformly across the currently solvent banks."""

    def __init__(self, vec_env):
        self.env = vec_env.envs[0].env
        self.num_nodes = _get_num_observed_nodes(self.env)
        self.limit = self.env.upper_limit_capital_injections
        self.bins_action = self.env.bins_action

    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_masks=None):
        _, solvency, _ = _extract_observed_state(observation, self.num_nodes)
        solvent_indices = np.where(solvency == 0)[0]
        if len(solvent_indices) == 0:
            return [np.zeros(self.num_nodes, dtype=int)], None

        amount_per_node = self.limit / len(solvent_indices)
        action_per_node = min(
            np.digitize(amount_per_node, self.bins_action, right=True),
            len(self.bins_action) - 1,
        )
        action = np.zeros(self.num_nodes, dtype=int)
        action[solvent_indices] = action_per_node
        return [action], None


class ZeroInjectionAgent:
    """Never inject capital."""

    def __init__(self, vec_env):
        self.env = vec_env.envs[0].env
        self.num_nodes = _get_num_observed_nodes(self.env)

    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_masks=None):
        return [np.zeros(self.num_nodes, dtype=int)], None


class LargestDeficitAgent:
    """Give the full budget to the solvent bank with the largest deficit."""

    def __init__(self, vec_env):
        self.env = vec_env.envs[0].env
        self.num_nodes = _get_num_observed_nodes(self.env)
        self.limit = self.env.upper_limit_capital_injections
        self.bins_action = self.env.bins_action

    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_masks=None):
        node_features, solvency, _ = _extract_observed_state(observation, self.num_nodes)
        risk_free_assets = node_features[:, 0]
        claims = node_features[:, 4]
        liabilities = node_features[:, 2] + node_features[:, 3]
        deficits = (liabilities - risk_free_assets - claims).astype(float)
        deficits[solvency != 0] = -np.inf

        if np.all(deficits <= 0):
            return [np.zeros(self.num_nodes, dtype=int)], None

        target = int(np.argmax(deficits))
        action_index = min(
            np.digitize(self.limit, self.bins_action, right=True),
            len(self.bins_action) - 1,
        )
        action = np.zeros(self.num_nodes, dtype=int)
        action[target] = action_index
        return [action], None
