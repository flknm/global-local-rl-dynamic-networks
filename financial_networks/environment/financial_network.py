import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from . import clearing_algorithm as ca
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import shutil
import warnings
from pathlib import Path
import copy
import pandas as pd
from IPython.display import display_html


class FinancialNetworkEnv(gym.Env):
    '''
    Gymnasium environment for fixed or randomly generated directed networks of banks.

    The environment models financial contagion and systemic risk through interactions of banks.
    '''
    metadata = {"render.modes": ["human"]}

    def __init__(self, network_type, network="fixed", num_nodes=10, num_core_nodes=8, price_risky_asset=1.0, mu=1.0, sigma=0.15, theta = 0.2,
                 delta=0.1, rho=0.5, upper_limit_capital_injections=30, num_capital_injection_actions=11, alpha=0.8, beta=0.8,
                 gamma=0.1, tolerance=1e-6, max_iter=10000, render_mode="human", factor_networth=1.0, factor_capital=4/3, factor_penalty=0, save_render=False,
                 permute_on_reset = False, action_space_architecture = "weighted_limit", num_action_weights = 3):
        super().__init__()
        
        '''
        Initializes the financial network environment.

        global Parameters:
          - upper_limit_capital_injections (float): Maximum allowable capital injections.
    	  - num_capital_injection_actions (int): Number of options for the injected capital amount.
          - mu (float): mean reversion level for Ornstein-Uhlenbeck process.
          - sigma (float): Volatility parameter for Ornstein-Uhlenbeck process.
          - theta (float): mean reversion speed Ornstein-Uhlenbeck process.
        
        Parameters for constructing randomly generated initial networks:
          - num_nodes (int): Number of observed banks; generated networks also contain one auxiliary node.
          - num_core_nodes (int): Number of core banks.
          - rho (float): Fraction of risky assets in the portfolio.
          - price_risky_asset (float): Starting price of the risky asset.
          - delta (float): Capital buffer multiplier.
          
        Parameters for clearing and fire sales:
          - alpha (float): Fraction of external asset value realized by defaulting banks.
          - beta (float): Fraction of interbank asset value realized by defaulting banks.
          - gamma (float): Parameter for the inverse demand function.
          - tolerance (float): Tolerance for iterative convergence.
          - max_iter (int): Maximum number of iterations for equilibrium computation.
        '''  

        upper_limit_capital_injections = float(np.round(float(upper_limit_capital_injections), 1))        
        
        # Bounds used to clip observations.
        self.max_price = 2.0
        self.min_price = - self.max_price
        self.max_adjacency = 5.0
        self.max_node_features = 7.0
        self.min_node_features = - self.max_node_features
        
        self.num_capital_injection_actions = int(num_capital_injection_actions)
        if self.num_capital_injection_actions < 2:
            raise ValueError("num_capital_injection_actions must be at least 2.")
        self.bins_action = np.round(
            np.linspace(0.0, upper_limit_capital_injections, self.num_capital_injection_actions),
            6,
        ).astype(np.float32)
        self.capital_action_step = float(self.bins_action[1] - self.bins_action[0])
        self.num_action_weights = int(num_action_weights)
        
        # Graph and environment parameters
        self.num_core_nodes = num_core_nodes
        self.price_risky = self.quantize_value(price_risky_asset,"price")
        self.true_price_risky = self.quantize_value(price_risky_asset,"price")
        self.starting_price_risky = self.quantize_value(price_risky_asset,"price")
        self.delta = delta
        self.rho = rho
        self.upper_limit_capital_injections = upper_limit_capital_injections
       
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_step = 0 
        self.tolerance = tolerance
        self.max_iter = max_iter
        
        # Define inverse demand function
        self.inverse_demand_function = lambda x: np.exp(-self.gamma * x)
        # Set num_nodes based on network type
        self.network = self._resolve_network_mode(network)
        self.network_type = network_type
        if (
            self.network == "fixed"
            and self.network_type in {"symmetric_stars", "mirrored_cycles"}
            and (self.alpha != 0.8 or self.beta != 0.8)
        ):
            warnings.warn(
                f"Fixed network '{self.network_type}' overrides alpha and beta to 0.8.",
                UserWarning,
                stacklevel=2,
            )
        network_nodes = {
            "core_periphery": 10,
            "cycle": 11,
            "star": 11,
            "chain": 10,
            "pairs": 11,
            "inverse_star": 10,
            "asymmetric_stars": 11,
            "mirrored_cycles": 11,
            "symmetric_stars": 11,
        }

        if self.network == "fixed" and self.network_type not in network_nodes:
            raise ValueError(
                f"Unknown fixed network '{self.network_type}'. "
                f"Choose from {sorted(network_nodes)}."
            )

        base_num_nodes = num_nodes if self.network == "random" else network_nodes[self.network_type]
        # num_nodes counts observed banks, generated networks also keep an auxiliary node.
        self.num_observed_target_nodes = base_num_nodes
        self.num_nodes = base_num_nodes + 1 if self.network == "random" else base_num_nodes

        # Only observed bank nodes enter the policy, auxiliary nodes remain internal.
        preview_node_features = self._preview_node_features()
        self.observed_nodes = self._get_observed_node_indices(preview_node_features)
        self.num_observed_nodes = len(self.observed_nodes)
        self.dummy_nodes = [node for node in range(self.num_nodes) if node not in self.observed_nodes]

        state_node_feature_dim, state_int_feature_dim = 5, 1
        
        # Define observation space consisting of an adjacency_matrix, node_features and the price of the risky asset
        self.observation_space = spaces.Dict({
            "adjacency_matrix": spaces.Box(
                low=0.0,
                high=self.max_adjacency,
                shape=(self.num_observed_nodes * self.num_observed_nodes,),
                dtype=np.float32,
            ),
            "node_features": spaces.Box(
                low=self.min_node_features,
                high=self.max_node_features,
                shape=(state_node_feature_dim * self.num_observed_nodes,),
                dtype=np.float32,
            ),
            "int_node_features": spaces.MultiDiscrete(
                [3] * (state_int_feature_dim * self.num_observed_nodes),
                dtype=np.int32,
            ),
            "price_risky_asset": spaces.Box(
                low=-self.max_price,
                high=self.max_price,
                shape=(1,),
                dtype=np.float32,
            ),
        })
        
        #Form of the observation space: Dic with 4 entries.
        #1: list with num_observed_nodes^2 entries, containing only interbank liabilities
        #   between the observed nodes.
        #2: list with num_observed_nodes * state_node_feature_dim entries:
        #   risk-free assets, risky assets, internal liabilities, external liabilities,
        #   total claims.
        #3: list with num_observed_nodes * state_int_feature_dim entries:
        #   solvency.
        #4: price of the risky asset

        self.action_space_architecture = action_space_architecture
        if self.action_space_architecture != "weighted_limit":
            raise ValueError("Only weighted_limit is supported as a policy action space.")
        if self.action_space_architecture == "weighted_limit":
            # The policy chooses one budget bin and one allocation weight per bank.
            self.action_space = spaces.MultiDiscrete([len(self.bins_action)] + [self.num_action_weights] * self.num_observed_nodes)

        self.np_random = None
        self.render_mode = render_mode
        self.permute_on_reset = permute_on_reset

        # Coefficients of the reward terms.
        self.factor_networth = factor_networth
        self.factor_capital = factor_capital
        self.factor_penalty =  factor_penalty

        # State retained for rendering the last transition.
        self.node_positions = None
        self.last_original_action = None
        self.last_rescaled_action = None
        self.last_reward = None
        self.last_price = self.quantize_value(price_risky_asset,"price")
        self.frame_count = 0
        self.save_render = save_render
        self.delta_fs = None
        self.delta_simulation = None
        self.graph_before_clearing = None
        self.node_features_before_clearing = None


    def _resolve_network_mode(self, network):
        """
        Normalise the network source configuration.
        """
        network_key = str(network).strip().lower()
        aliases = {
            "fixed": "fixed",
            "static": "fixed",
            "random": "random",
            "generated": "random",
        }
        if network_key not in aliases:
            raise ValueError("Unsupported network setting. Choose 'fixed' or 'random'.")
        return aliases[network_key]

    def _preview_node_features(self):
        """
        Return node-feature preview for action and observation space sizing.

        """
        if self.network == "fixed":
            _, preview_node_features = self._fixed_financial_network()
            return preview_node_features

        preview_node_features = {
            node: {"solvency": 0}
            for node in range(self.num_nodes)
        }
        preview_node_features[self.num_nodes - 1] = {"solvency": 2}
        return preview_node_features

    def _build_initial_network(self):
        """
        Build the network used at reset.
        """
        if self.network == "random":
            return self._generate_financial_network()
        return self._fixed_financial_network()

    def _get_observed_node_indices(self, node_features):
        """
        Return the bank indices included in policy observations.
        """
        return [node for node, data in node_features.items() if data.get("solvency") != 2]

    def _refresh_observed_nodes(self):
        """
        Update the observed-node view after permutations or state transitions.
        """
        self.observed_nodes = self._get_observed_node_indices(self.node_features)
        self.num_observed_nodes = len(self.observed_nodes)
        self.dummy_nodes = [node for node in range(self.num_nodes) if node not in self.observed_nodes]

    def _refresh_liability_splits(self):
        """
        Keep internal/external liabilities and interbank claims in sync with the
        current graph.
        """
        observed_index_set = set(self.observed_nodes)
        dummy_index_set = set(self.dummy_nodes)

        for node in range(self.num_nodes):
            internal_liabilities = float(np.sum([self.graph[node, j] for j in observed_index_set]))
            external_liabilities = float(np.sum([self.graph[node, j] for j in dummy_index_set]))
            interbank_claims = float(np.sum([self.graph[i, node] for i in observed_index_set]))

            self.node_features[node]["internal_liabilities"] = self.quantize_value(
                internal_liabilities, "node_features"
            )
            self.node_features[node]["external_liabilities"] = self.quantize_value(
                external_liabilities, "node_features"
            )
            self.node_features[node]["total_claims"] = self.quantize_value(
                interbank_claims, "node_features"
            )
            self.node_features[node]["total_liabilities"] = self.quantize_value(
                internal_liabilities + external_liabilities, "node_features"
            )

            if self.node_features[node]["solvency"] == 2:
                self.node_features[node]["internal_liabilities"] = np.float32(0.0)
                self.node_features[node]["external_liabilities"] = np.float32(0.0)

    def _simulate_price(self, current_price, dt=1.0):
        '''
        Simulates the price of the risky asset using an Ornstein-Uhlenbeck process.

        Parameters:
          - current_price (float): Current price of the risky asset.

        Returns:
          - new_price (float): Simulated price of the risky asset.
        '''  
        mean = self.mu + (current_price - self.mu) * np.exp(-self.theta * dt)
        variance = (self.sigma ** 2) * (1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta)
        std = np.sqrt(variance)

        # Clip the observed price while retaining the unbounded draw for the dynamics.
        self.true_price_risky = self.np_random.normal(mean, std)
        new_price = self.quantize_value(self.true_price_risky, "price")
        return new_price

    def _generate_financial_network(self):
        '''
        Create a random directed financial network and its initial bank balances.

        Returns:
          - adjacency_matrix (np.ndarray): Adjacency matrix of the graph.
          - node_features (dict): Dictionary with asset and liability information.
        '''
        if self.np_random is None:
            self.np_random = np.random.default_rng()

        max_external_liability_fraction = 0.20
        dummy_node = self.num_nodes - 1
        bank_nodes = range(dummy_node)
        num_bank_nodes = dummy_node
        n_c = min(self.num_core_nodes, num_bank_nodes)  # Number of core banks
        n_p = num_bank_nodes - n_c  # Number of periphery banks

        # Connection probabilities
        p_cc = 0.66  # Core-Core
        p_cp = 0.15  # Core-Periphery
        p_pc = 0.07  # Periphery-Core
        p_pp = 0.001  # Periphery-Periphery

        # Sample edges between bank nodes.
        adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))

        # Core-Core connections
        for i in range(n_c):
            for j in range(n_c):
                if i != j and self.np_random.random() < p_cc:
                    adjacency_matrix[i, j] = 1

        # Core-Periphery connections
        for i in range(n_c):
            for j in range(n_c, num_bank_nodes):
                if self.np_random.random() < p_cp:
                    adjacency_matrix[i, j] = 1

        # Periphery-Core connections
        for i in range(n_c, num_bank_nodes):
            for j in range(n_c):
                if self.np_random.random() < p_pc:
                    adjacency_matrix[i, j] = 1

        # Periphery-Periphery connections
        for i in range(n_c, num_bank_nodes):
            for j in range(n_c, num_bank_nodes):
                if i != j and self.np_random.random() < p_pp:
                    adjacency_matrix[i, j] = 1

        # Distribute weights
        total_liabilities = num_bank_nodes  # Total internal liabilities of all banks
        c_pi = 1  # Fraction allocated to interbank liabilities

        # Fractions of total liabilities for each block
        x_cc = 0.35
        x_cp = 0.16
        x_pc = 0.47
        x_pp = 0.02
        
        # Assign weights to the blocks, initialise total liability matrix.
        num_cc_edges = np.count_nonzero(adjacency_matrix[:n_c, :n_c])
        num_cp_edges = np.count_nonzero(adjacency_matrix[:n_c, n_c:num_bank_nodes])
        num_pc_edges = np.count_nonzero(adjacency_matrix[n_c:num_bank_nodes, :n_c])
        num_pp_edges = np.count_nonzero(adjacency_matrix[n_c:num_bank_nodes, n_c:num_bank_nodes])
        edge_rows, edge_cols = np.nonzero(adjacency_matrix[:num_bank_nodes, :num_bank_nodes])

        for i, j in zip(edge_rows, edge_cols):
            if i < n_c and j < n_c:
                weight = x_cc * total_liabilities * c_pi / num_cc_edges
            elif i < n_c and j >= n_c:
                weight = x_cp * total_liabilities * c_pi / num_cp_edges
            elif i >= n_c and j < n_c:
                weight = x_pc * total_liabilities * c_pi / num_pc_edges
            else:
                weight = x_pp * total_liabilities * c_pi / num_pp_edges
            adjacency_matrix[i, j] = self.quantize_value(weight,"adjacency")


        internal_liabilities = np.sum(adjacency_matrix[:, :num_bank_nodes], axis=1)
        external_liabilities = np.zeros(self.num_nodes, dtype=np.float32)
        for node in bank_nodes:
            external_liability_fraction = self.np_random.uniform(
                0.0,
                max_external_liability_fraction,
            )
            external_liabilities[node] = self.quantize_value(
                external_liability_fraction * internal_liabilities[node],
                "adjacency",
            )
            adjacency_matrix[node, dummy_node] = external_liabilities[node]

        node_features = {}
        
        # Determine initial node feature solvency
        for node in bank_nodes:
            node_features[node] = {"solvency": 0}
        node_features[dummy_node] = {"solvency": 2}

        # determine relative liability matrix and absolute liabilities
        rel_adjacency_matrix, liabilities, claims = self.relative_matrix(adjacency_matrix)
        
        # Determine asset holdings from total liabilities, including the sampled
        # external liabilities, which are at most 20% of internal liabilities.
        p = liabilities
        rel_adjacency_matrix_transpose = rel_adjacency_matrix.T
        h = np.maximum(p - np.dot(rel_adjacency_matrix_transpose, p), 0)  # h = max(p - rel_adjacency_matrix^T * p, 0)
        e = (1 + self.delta) * h
        risk_free_assets_vector = (1 - self.rho) * e
        risky_assets_value_vector = self.rho * e

        liabilities = self.quantize_value(p,"node_features")
        claims = self.quantize_value(claims, "node_features")
        for node in range(self.num_nodes):
            risk_free_assets = self.quantize_value(risk_free_assets_vector[node],"node_features")
            risky_assets = self.quantize_value(risky_assets_value_vector[node] / self.price_risky,"node_features")
            internal_liability = self.quantize_value(internal_liabilities[node], "node_features")
            external_liability = self.quantize_value(external_liabilities[node], "node_features")
            if node != dummy_node:
                # check if all nodes are solvent
                initial_networth = (
                    risk_free_assets
                    + risky_assets * self.price_risky
                    + claims[node]
                    - liabilities[node]
                )
                if initial_networth < -1e-6:
                    raise ValueError(
                        "Random financial network generated an initially "
                        f"insolvent bank {node} with networth={initial_networth:.6f}."
                    )
            node_features[node].update({
                "risk_free_assets": risk_free_assets,
                "risky_assets": risky_assets,
                "internal_liabilities": internal_liability,
                "external_liabilities": external_liability,
                "total_liabilities": liabilities[node],
                "total_claims": claims[node]
            })

        return adjacency_matrix, node_features


    def _fixed_financial_network(self):
        """Return a fixed benchmark network."""
        if self.network_type == "core_periphery":
            adjacency_matrix = np.array([
                [0.0, 0.75, 0.75, 0.0, 0.26, 0.0, 0.26, 0.26, 0.0, 0.0],
                [0.75, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.26, 0.26, 0.26],
                [0.75, 0.75, 0.0, 0.0, 0.0, 0.26, 0.0, 0.26, 0.0, 0.0],
                [0.0, 0.0, 0.32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.32, 0.32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.32, 0.32, 0.32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.32, 0.32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ])
            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.28, 'external_liabilities': 0.0, 'total_liabilities': 2.28, 'total_claims': 3.1},
                1: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.28, 'external_liabilities': 0.0, 'total_liabilities': 2.28, 'total_claims': 2.46},
                2: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.02, 'external_liabilities': 0.0, 'total_liabilities': 2.02, 'total_claims': 2.14},
                3: {'solvency': 0, 'risk_free_assets': 0.18, 'risky_assets': 0.18, 'internal_liabilities': 0.32, 'external_liabilities': 0.0, 'total_liabilities': 0.32, 'total_claims': 0.0},
                4: {'solvency': 0, 'risk_free_assets': 0.21, 'risky_assets': 0.21, 'internal_liabilities': 0.64, 'external_liabilities': 0.0, 'total_liabilities': 0.64, 'total_claims': 0.26},
                5: {'solvency': 0, 'risk_free_assets': 0.38, 'risky_assets': 0.38, 'internal_liabilities': 0.96, 'external_liabilities': 0.0, 'total_liabilities': 0.96, 'total_claims': 0.26},
                6: {'solvency': 0, 'risk_free_assets': 0.03, 'risky_assets': 0.03, 'internal_liabilities': 0.32, 'external_liabilities': 0.0, 'total_liabilities': 0.32, 'total_claims': 0.26},
                7: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 0.78},
                8: {'solvency': 0, 'risk_free_assets': 0.21, 'risky_assets': 0.21, 'internal_liabilities': 0.64, 'external_liabilities': 0.0, 'total_liabilities': 0.64, 'total_claims': 0.26},
                9: {'solvency': 0, 'risk_free_assets': 0.03, 'risky_assets': 0.03, 'internal_liabilities': 0.32, 'external_liabilities': 0.0, 'total_liabilities': 0.32, 'total_claims': 0.26},
            }
            
        elif self.network_type == "cycle":
            adjacency_matrix = np.array([
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])

            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 1.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                1: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                2: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                3: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                4: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                5: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                6: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                7: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                8: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                9: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 1.0, 'total_liabilities': 2.0, 'total_claims': 1.0},
                10: {'solvency': 2, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 10.0},
            }
        
        elif self.network_type == "star":
            spoke_weight = 4.5 / 9.0
            adjacency_matrix = np.array([
                [0.0, spoke_weight, spoke_weight, spoke_weight, spoke_weight, spoke_weight, spoke_weight, spoke_weight, spoke_weight, spoke_weight, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, spoke_weight],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            
            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 2*(1+self.delta), 'risky_assets': 2.5*(1+self.delta), 'internal_liabilities': 4.5, 'external_liabilities': 0.0, 'total_liabilities': 4.5, 'total_claims': 0.0},
                1: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                2: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                3: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                4: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                5: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                6: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                7: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                8: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                9: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.5, 'total_liabilities': 0.5, 'total_claims': 0.5},
                10: {'solvency': 2, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 4.5},
            }
            
        elif self.network_type == "chain":
        
            adjacency_matrix = np.array([
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ])
            
            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 0.5*(1+self.delta), 'risky_assets': 0.5*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                1: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                2: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                3: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                4: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                5: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                6: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                7: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                8: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 2.0},
                9: {'solvency': 0, 'risk_free_assets': 0.5*(1+self.delta), 'risky_assets': 0.5*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
            }
            
        elif self.network_type == "inverse_star":
            adjacency_matrix = np.array([
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            
            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 0.0*(1+self.delta), 'risky_assets': 1.0*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                1: {'solvency': 0, 'risk_free_assets': 0.1*(1+self.delta), 'risky_assets': 0.9*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                2: {'solvency': 0, 'risk_free_assets': 0.2*(1+self.delta), 'risky_assets': 0.8*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                3: {'solvency': 0, 'risk_free_assets': 0.3*(1+self.delta), 'risky_assets': 0.7*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                4: {'solvency': 0, 'risk_free_assets': 0.0*(1+self.delta), 'risky_assets': 0.0*(1+self.delta), 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 9.0},
                5: {'solvency': 0, 'risk_free_assets': 0.4*(1+self.delta), 'risky_assets': 0.6*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                6: {'solvency': 0, 'risk_free_assets': 0.5*(1+self.delta), 'risky_assets': 0.5*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                7: {'solvency': 0, 'risk_free_assets': 0.6*(1+self.delta), 'risky_assets': 0.4*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                8: {'solvency': 0, 'risk_free_assets': 0.7*(1+self.delta), 'risky_assets': 0.3*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                9: {'solvency': 0, 'risk_free_assets': 0.8*(1+self.delta), 'risky_assets': 0.2*(1+self.delta), 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
            }
            
        elif self.network_type == "pairs":
            adjacency_matrix = np.array([
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            ])
            
            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 0.3, 'risky_assets': 0.7, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                1: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                2: {'solvency': 0, 'risk_free_assets': 0.4, 'risky_assets': 0.6, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                3: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                4: {'solvency': 0, 'risk_free_assets': 0.5, 'risky_assets': 0.5, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                5: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                6: {'solvency': 0, 'risk_free_assets': 0.6, 'risky_assets': 0.4, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                7: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                8: {'solvency': 0, 'risk_free_assets': 0.7, 'risky_assets': 0.3, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 0.0},
                9: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                10: {'solvency': 2, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 5.0},
            }
            
        elif self.network_type == "symmetric_stars":
            # These benchmark balances use alpha = beta = 0.8.
            self.alpha = 0.8
            self.beta = 0.8
            adjacency_matrix = np.array([
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            
            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 0.0},
                1: {'solvency': 0, 'risk_free_assets': 1.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 0.0},
                2: {'solvency': 0, 'risk_free_assets': 3.3, 'risky_assets': 0.0, 'internal_liabilities': 6.0, 'external_liabilities': 0.0, 'total_liabilities': 6.0, 'total_claims': 4.0},
                3: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                4: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                5: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                6: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                7: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                8: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                9: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 6.0, 'total_liabilities': 6.0, 'total_claims': 6.0},
                10: {'solvency': 2, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 6.0},
            }

        elif self.network_type == "asymmetric_stars":
            adjacency_matrix = np.array([
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])

            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 3.0*(1+self.delta), 'risky_assets': 3.0*(1+self.delta), 'internal_liabilities': 6.0, 'external_liabilities': 0.0, 'total_liabilities': 6.0, 'total_claims': 0.0},
                1: {'solvency': 0, 'risk_free_assets': 1.0*(1+self.delta), 'risky_assets': 1.0*(1+self.delta), 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 0.0},
                2: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                3: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                4: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                5: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                6: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                7: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                8: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                9: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 1.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                10: {'solvency': 2, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 8.0},
            }
        
        elif self.network_type == "mirrored_cycles":
            # These benchmark balances use alpha = beta = 0.8.
            self.alpha = 0.8
            self.beta = 0.8
            adjacency_matrix = np.array([
                [0.0, 1.0, 0.0, 3.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
                [3.9, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])
            
            node_features = {
                0: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 4.9, 'external_liabilities': 0.0, 'total_liabilities': 4.9, 'total_claims': 4.4},
                1: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                2: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.5, 'external_liabilities': 0.5, 'total_liabilities': 2.0, 'total_claims': 2.0},
                3: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 4.9, 'external_liabilities': 0.0, 'total_liabilities': 4.9, 'total_claims': 4.4},
                4: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 2.0, 'external_liabilities': 0.0, 'total_liabilities': 2.0, 'total_claims': 2.0},
                5: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.5, 'external_liabilities': 0.5, 'total_liabilities': 2.0, 'total_claims': 2.0},
                6: {'solvency': 2, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 0.0, 'external_liabilities': 0.0, 'total_liabilities': 0.0, 'total_claims': 1.0},
                7: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                8: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                9: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
                10: {'solvency': 0, 'risk_free_assets': 0.0, 'risky_assets': 0.0, 'internal_liabilities': 1.0, 'external_liabilities': 0.0, 'total_liabilities': 1.0, 'total_claims': 1.0},
            }

 
        
        else:
            raise ValueError(f"Unsupported fixed network '{self.network_type}'.")

        return adjacency_matrix, node_features
    def reset(self, seed=None, options=None):
        '''
        Starts a new episode, rebuilding the configured network and bank balances.

        Parameters:
          - seed (int, optional): Random seed for reproducibility.
          - options (dict, optional): Accepted for Gymnasium compatibility, unused.

        Returns:
          - current_state (dict): Initial state of the environment.
          - info (dict): Empty reset information.
        '''
        super().reset(seed=seed) 

        # set np_random with gym utils
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.price_risky = self.starting_price_risky
        self.true_price_risky = self.starting_price_risky 
        self.last_price = self.starting_price_risky
        self.delta_simulation = None
        self.delta_fs = None
        self.node_positions = None
        self.graph, self.node_features = self._build_initial_network()

        if self.permute_on_reset:
            self.permutation = self.np_random.permutation(self.num_nodes)
            self.graph = self.graph[self.permutation][:, self.permutation]
            original_node_features = self.node_features
            self.node_features = {new_idx: original_node_features[old_idx] for new_idx, old_idx in enumerate(self.permutation)}

        self._refresh_observed_nodes()
   
        self.current_step = 0  # reset steps  
        
        if self.network == "random":
            self._refresh_liability_splits()
        self.current_state = self.matrix_to_state()

        if self.save_render:
            self.frame_dir = Path("frames")
            if self.frame_dir.exists():
                shutil.rmtree(self.frame_dir)
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            self.frame_count = 0
        
        self.last_original_action = None
        self.last_rescaled_action = None
        self.last_reward = None
        self.graph_before_clearing = self.graph.copy()
        self.node_features_before_clearing = copy.deepcopy(self.node_features)

        # Check that the observation satisfies the declared bounds.
        is_valid = self.observation_space.contains(self.current_state)
        if not is_valid:
            print(f"Observation of incorrect type in reset!")
            print(f"Current_state:{self.current_state}")
            print(f"adjacency:{self.graph}")
            print(f"node_features:{self.node_features}")
            print(f"price: {self.price_risky}")
        assert np.all(self.graph >= 0), "Adjacency matrix contains negative values."
        assert all(data["risky_assets"] >= 0 for data in self.node_features.values()), "Node features risky_assets contain negative values."
        for node, data in self.node_features.items():
            assert "risk_free_assets" in data, f"Missing 'risk_free_assets' for node {node}."
            assert "risky_assets" in data, f"Missing 'risky_assets' for node {node}."
            assert "internal_liabilities" in data, f"Missing 'internal_liabilities' for node {node}."
            assert "external_liabilities" in data, f"Missing 'external_liabilities' for node {node}."
            assert "total_liabilities" in data, f"Missing 'total_liabilities' for node {node}."
            assert "total_claims" in data, f"Missing 'total_claims' for node {node}."
            assert "solvency" in data, f"Missing 'solvency' for node {node}."
       
        return self.current_state, {}

    def capture_initial_state(self):
        """Copy the network, bank balances, prices, and node ordering after reset.

        The random generator is kept outside this snapshot. Rollouts can then
        share the same initial state while drawing independent future price noise.
        The trainer stores this snapshot when initial_state_mode is fixed.
        """
        snapshot = {
            "graph": self.graph,
            "node_features": self.node_features,
            "price_risky": self.price_risky,
            "true_price_risky": self.true_price_risky,
            "last_price": self.last_price,
        }
        if hasattr(self, "permutation"):
            snapshot["permutation"] = self.permutation
        return copy.deepcopy(snapshot)

    def restore_initial_state(self, snapshot, seed=None):
        """Restore a captured initial state and restart the episode counter.

        seed controls future price noise, it does not regenerate the network
        or overwrite the bank balances saved in the snapshot.
        """
        super().reset(seed=seed)
        restored = copy.deepcopy(snapshot)
        self.graph = restored["graph"]
        self.node_features = restored["node_features"]
        self.price_risky = restored["price_risky"]
        self.true_price_risky = restored["true_price_risky"]
        self.last_price = restored["last_price"]
        if "permutation" in restored:
            self.permutation = restored["permutation"]

        self._refresh_observed_nodes()
        self.current_step = 0
        self.current_state = self.matrix_to_state()
        self.last_original_action = None
        self.last_rescaled_action = None
        self.last_reward = None
        self.delta_simulation = None
        self.delta_fs = None
        self.graph_before_clearing = self.graph.copy()
        self.node_features_before_clearing = copy.deepcopy(self.node_features)
        return self.current_state, {}
    
    def _capital_from_action_indices(self, action_indices):
        """Look up capital amounts for discrete action indices.

        The policy's budget index selects an amount from bins_action.
        Bank weights are handled separately in step().
        """
        indices = np.asarray(action_indices, dtype=int)
        indices = np.clip(indices, 0, len(self.bins_action) - 1)
        return self.bins_action[indices].astype(np.float32)


    def _floor_to_capital_grid(self, values):
        """Round capital amounts down to the available grid."""
        if self.capital_action_step <= 0:
            return np.asarray(values, dtype=np.float32)
        floored = np.floor(np.asarray(values, dtype=np.float32) / self.capital_action_step) * self.capital_action_step
        return np.round(np.clip(floored, 0.0, self.upper_limit_capital_injections), 6).astype(np.float32)


    def step(self, action):
        '''
        Applies the chosen action and advances the environment by one step.

        Parameters:
          - action (np.ndarray): One index into bins_action for the global budget,
            followed by one allocation weight per observed bank. Insolvent banks
            receive no capital, the remaining weights divide the selected budget.
            If all eligible weights are zero, no capital is allocated.

        Returns:
          - current_state (dict): Updated state of the environment. Contains:
              - "adjacency_matrix": The updated graph structure with adjusted weights.
              - "node_features": Flattened numeric bank features.
              - "int_node_features": Solvency labels for the observed banks.
              - "price_risky_asset": Current observed price of the risky asset.
          - reward (float): Weighted equity change minus the selected budget cost
            and the penalty for newly insolvent banks.
          - terminated (bool): Whether all observed banks are insolvent.
          - truncated (bool): Always False; the trainer or evaluator sets the horizon.
          - info (dict): The reward under key r.
        '''
        # Update the time step
        self.current_step += 1
        
        observed_nodes = self.observed_nodes

        # The next part of the code transforms a given action into a vector of
        # capital injections on the full simulated graph.
        if self.action_space_architecture == "weighted_limit":
            action = np.array(action, dtype=np.float32).copy()
            budget_amount = float(self._capital_from_action_indices([action[0]])[0])
            original_action = np.zeros(self.num_nodes, dtype=np.float32)
            filtered_node_weights = action[1:].copy()
            for obs_idx, node in enumerate(observed_nodes):
                if self.node_features[node]["solvency"] > 0:
                    filtered_node_weights[obs_idx] = 0  # No capital injection allowed for insolvent banks
            weight_sum = np.sum(filtered_node_weights)
            if weight_sum > 0:
                distributed_action = budget_amount * filtered_node_weights / weight_sum
                for obs_idx, node in enumerate(observed_nodes):
                    original_action[node] = distributed_action[obs_idx]
            rescaled_action = original_action.copy()
            self.last_original_action = original_action.copy()
            self.last_rescaled_action = rescaled_action.copy()
            capital_cost = budget_amount
                
        self.last_price = self.price_risky
        self.delta_simulation = -self.true_price_risky

        # Simulate the new price of the risky asset
        self.price_risky = self._simulate_price(self.true_price_risky)
        self.delta_simulation += self.true_price_risky
        self.delta_fs = -self.true_price_risky

        rel_adjacency_matrix,_, _ = self.relative_matrix(self.graph)
        
        # Save old net worth (with old price) for reward calculation
        old_networth = (
            self.graph.sum(axis=0)
            + np.array([self.node_features[node]["risk_free_assets"] for node in range(self.num_nodes)]) 
            + np.array([self.node_features[node]["risky_assets"] for node in range(self.num_nodes)]) * self.last_price 
            - np.array([self.node_features[node]["total_liabilities"] for node in range(self.num_nodes)])
        )

        # Update risk-free assets based on the action and ensure node_features are in observation space
        for node in range(self.num_nodes):
            self.node_features[node]["risk_free_assets"] += rescaled_action[node]
            self.node_features[node]["risk_free_assets"] = self.quantize_value(self.node_features[node]["risk_free_assets"], "node_features")

        # save state and labels before clearing for render
        self.graph_before_clearing = self.graph.copy()
        self.node_features_before_clearing = copy.deepcopy(self.node_features)
        
        # Calculate payments and update solvency with FS:
        payment_vector, self.true_price_risky, networth_vector = ca.equilibrium_algorithm(
            np.array([self.node_features[node]["total_liabilities"] for node in range(self.num_nodes)]),
            rel_adjacency_matrix,
            np.array([self.node_features[node]["risk_free_assets"] for node in range(self.num_nodes)]),
            np.array([self.node_features[node]["risky_assets"] for node in range(self.num_nodes)]),
            self.true_price_risky,
            self.alpha,
            self.beta,
            self.inverse_demand_function,
            self.tolerance,
            self.max_iter
        )
        # sort new price into allowed prices
        self.price_risky = self.quantize_value(self.true_price_risky,"price")
        self.delta_fs += self.true_price_risky
        
        # test net worth of insolvent banks
        for i in range(self.num_nodes):
            if self.node_features[i]["solvency"] == 1:
                if not networth_vector[i] == 0:
                    networth_vector[i] = 0
                    print(f"Issue encountered while assessing networth of insolvent bank")
                

        counter_newly_insolvent = 0 
        # Update node solvency and liabilities
        for i in range(self.num_nodes):
            if networth_vector[i] < -1e-10:  # Node is newly insolvent
                self.node_features[i]["solvency"] = 1
                counter_newly_insolvent += 1
                total_liabilities = self.node_features[i]["total_liabilities"]
                if total_liabilities > 0:
                    for j in range(self.num_nodes):
                        payment = rel_adjacency_matrix[i, j] * payment_vector[i]
                        self.node_features[j]["risk_free_assets"] += payment
            elif self.node_features[i]["solvency"] == 0:  # Node is solvent
                for j in range(self.num_nodes):
                    if networth_vector[j] < -1e-10:  # Pay only to newly insolvent banks
                        payment = rel_adjacency_matrix[i, j] * payment_vector[i]
                        self.node_features[i]["risk_free_assets"] -= payment
                        

        for i in range(self.num_nodes):   
            if self.node_features[i]["solvency"] == 1:
                self.node_features[i]["risk_free_assets"] = 0
                self.node_features[i]["risky_assets"] = 0
                self.node_features[i]["total_liabilities"] = 0
                self.node_features[i]["total_claims"] = 0
                self.node_features[i]["internal_liabilities"] = 0
                self.node_features[i]["external_liabilities"] = 0
                self.graph[i,:] = 0
                self.graph[:,i] = 0
            else:
                self.node_features[i]["risk_free_assets"] = self.quantize_value(self.node_features[i]["risk_free_assets"], "node_features")

        self._refresh_liability_splits()
        
        reward = 0
        
        # Combine the equity change, budget cost, and newly insolvent-bank penalty.
        reward = self.factor_networth*np.sum(networth_vector - old_networth) - self.factor_capital* capital_cost - self.factor_penalty * counter_newly_insolvent
        
        self.last_reward = reward

        # Check termination conditions
        truncated = False
        terminated = all(self.node_features[node]['solvency'] > 0 for node in range(self.num_nodes))
        info = {}
        info["r"] = reward

        # Update current state
        self.current_state = self.matrix_to_state()

        is_valid = self.observation_space.contains(self.current_state)
        if not is_valid:
            print(f"Observation of incorrect type in step!")
            print(f"node_features:{self.node_features}")
            print(f"price: {self.price_risky}")
            print(f"adjacency:{self.graph}")
            print(f"Current_state:{self.current_state}")
        assert np.all(self.graph >= 0), "Adjacency matrix contains negative values."
        assert all(data["risky_assets"] >= 0 for data in self.node_features.values()), "Node features risky_assets contain negative values."
        for node, data in self.node_features.items():
            assert "risk_free_assets" in data, f"Missing 'risk_free_assets' for node {node}."
            assert "risky_assets" in data, f"Missing 'risky_assets' for node {node}."
            assert "total_claims" in data, f"Missing 'total_claims' for node {node}."
            assert "internal_liabilities" in data, f"Missing 'internal_liabilities' for node {node}."
            assert "external_liabilities" in data, f"Missing 'external_liabilities' for node {node}."
            assert "total_liabilities" in data, f"Missing 'total_liabilities' for node {node}."
            assert "solvency" in data, f"Missing 'solvency' for node {node}."

        return self.current_state, reward, terminated, truncated, info


    
    def render(self, action=None, reward=None, mode="human", price_influences = None, adjacency_matrix = None, node_features = None):
        """
        Visualizes the initial state or the last transition using matplotlib.
    
        - Before clearing, solvent-node colors show allocated capital.
        - After clearing, solvent nodes use the zero-injection color.
        - Insolvent nodes are gray.
        - Edge weights are shown as labels and control edge thickness.
        - Node labels show risk-free and risky asset values. Sizes reflect liabilities.
        - The risky-asset price is shown at the bottom left.
        """

        # Plot
        dpi = 160
        fig_width = 50*16 / dpi
        fig_height = 40*16 / dpi
        
        # Build graph and layout
        G_before_clearing = nx.from_numpy_array(self.graph_before_clearing, create_using=nx.DiGraph)
        dummy_nodes = [n for n in G_before_clearing.nodes if self.node_features_before_clearing[n].get("solvency")==2]
        G_before_clearing.remove_nodes_from(dummy_nodes)
        G = nx.from_numpy_array(self.graph, create_using=nx.DiGraph)
        dummy_nodes = [n for n in G.nodes if self.node_features[n].get("solvency") == 2]
        G.remove_nodes_from(dummy_nodes)

        remap_static_layout = False
        if self.node_positions:
            pos = self.node_positions
        else:
            positions_for_graphs = {
                'cycle': {node: (1.7 * np.cos(2 * np.pi * node / 10), 1.7 * np.sin(2 * np.pi * node / 10)) for node in range(10)},
                'star': {0: (0.0, 0.0), **{node: (1.6 * np.cos(2 * np.pi * (node - 1) / 9), 1.6 * np.sin(2 * np.pi * (node - 1) / 9)) for node in range(1, 10)}},
                'chain': {node: (1.2 * float(node), 0.0) for node in range(10)},
                'inverse_star': {4: (0.0, 0.0), **{node: (1.8 * np.cos(2 * np.pi * idx / 9), 1.8 * np.sin(2 * np.pi * idx / 9)) for idx, node in enumerate([0, 1, 2, 3, 5, 6, 7, 8, 9])}},
                'pairs': {node: (2 * float(node % 2), -float(node // 2)) for node in range(10)},
                'symmetric_stars': {0: (-1.5, 2.0), 1: (1.5, 2.0), 2: (0.0, 1.0), 3: (-2.0, 0.0), 4: (0.0, 0.0), 5: (2.0, 0.0), 6: (-2.0, -1.0), 7: (0.0, -1.0), 8: (2.0, -1.0), 9: (0.0, -2.0)},
                'asymmetric_stars': {0: (-1.2, 1.2), 1: (1.2, 1.2), 2: (-2.4, 0.0), 3: (-1.6, 0.0), 4: (-0.8, 0.0), 5: (-2.4, -1.0), 6: (-1.6, -1.0), 7: (-0.8, -1.0), 8: (0.8, 0.0), 9: (1.6, 0.0)},
                'mirrored_cycles': {0: (-1.0, 0.0), 1: (-2.0, 1.0), 2: (-2.0, -1.0), 3: (1.0, 0.0), 4: (2.0, 1.0), 5: (2.0, -1.0), 6: (0.0, -2.0), 7: (-3.0, 1.0), 8: (-3.0, -1.0), 9: (3.0, 1.0), 10: (3.0, -1.0)},
                'core_periphery': {0: (0.0, 2.0), 1: (-1.5, 1.5), 2: (1.5, 1.5), 3: (0.0, 0.5), 4: (-2.5, 0.5), 5: (2.5, 0.5), 6: (-1.5, -0.5), 7: (1.5, -0.5), 8: (-0.75, -1.5), 9: (0.75, -1.5)},
            }
            if self.network_type in positions_for_graphs:
                pos = positions_for_graphs[self.network_type]
                remap_static_layout = True
            else:
                pos = nx.spring_layout(G, seed=42)
                
        self.node_positions = pos
        
        if self.permute_on_reset and remap_static_layout:
            original_positions = pos
            pos = {new_id: original_positions[old_id] for new_id, old_id in enumerate(self.permutation) if old_id in original_positions}
            
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        
        last_reward = reward if reward is not None else self.last_reward
        initial_state = False
        # Handle action input
        if action is not None:
            last_original_action = self._capital_from_action_indices(action)
            sum_rescaled = np.sum(last_original_action)
            rescaled = np.zeros(self.num_nodes)
            if sum_rescaled > 0:
                rescaled = self._floor_to_capital_grid(
                    (last_original_action / sum_rescaled)
                    * min(sum_rescaled, self.upper_limit_capital_injections)
                )
            last_rescaled_action = rescaled
        else:
            if self.last_original_action is not None:
                last_original_action = self.last_original_action.copy()
                last_rescaled_action = self.last_rescaled_action.copy()
            else: # for initial state of environment, choose action "0"
                initial_state = True
                last_original_action = np.zeros(self.num_nodes, dtype=int)
                last_rescaled_action = np.zeros(self.num_nodes, dtype=int)
                last_reward = 0
                
    
        # Prepare node groups
        solvent_nodes_before_clearing, solvent_colors_before_clearing, solvent_sizes_before_clearing = [], [], []
        solvent_nodes, solvent_colors, solvent_sizes = [], [], []
        insolvent_nodes_before_clearing, insolvent_sizes_before_clearing = [], []
        insolvent_nodes, insolvent_sizes = [], []
        labels = {}
        labels_before_clearing={}
        node_font_size = 10
       
        for node in G_before_clearing.nodes():
            if self.node_features_before_clearing[node].get("solvency") == 2:
                continue  # Skip dummy nodes entirely
            
            L = self.node_features_before_clearing[node]["total_liabilities"]
            size = 500 + 300 * L
            # labels of node can be set
            if self.node_features_before_clearing[node]["solvency"] == 1: 
                insolvent_nodes_before_clearing.append(node)
                insolvent_sizes_before_clearing.append(size)
            elif self.node_features_before_clearing[node]["solvency"] == 0:     
                risk_free_before_clearing = self.node_features_before_clearing[node]["risk_free_assets"]
                risky_before_clearing = self.node_features_before_clearing[node]["risky_assets"] * self.last_price
                if risk_free_before_clearing + risky_before_clearing > 0:
                    labels_before_clearing[node] = f"{risk_free_before_clearing:.1f}\n{risky_before_clearing:.1f}"
                solvent_nodes_before_clearing.append(node)
                solvent_colors_before_clearing.append(last_rescaled_action[node])
                solvent_sizes_before_clearing.append(size)

        for node in G.nodes():
            if self.node_features[node].get("solvency") == 2:
                continue  # Skip dummy nodes entirely
            L = self.node_features[node]["total_liabilities"]
            
            size = 500 + 300 * L

            # labels of node can be set
            if self.node_features[node]["solvency"] == 1: 
                insolvent_nodes.append(node)
                insolvent_sizes.append(size)
            elif self.node_features[node]["solvency"] == 0:     
                risk_free_value = self.node_features[node]["risk_free_assets"]
                risky_value = self.node_features[node]["risky_assets"] * self.price_risky
                if risk_free_value + risky_value > 0:
                    labels[node] = f"{risk_free_value:.1f}\n{risky_value:.1f}"
                solvent_nodes.append(node)
                solvent_colors.append(np.zeros(self.num_nodes, dtype=int)[node])
                solvent_sizes.append(size)
            

        # Edge weights for line width
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        edge_widths = [1 + 1 * w for w in edge_weights]
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height),dpi=dpi)


        if initial_state:
            ax.set_title("Initial State", fontsize=14)
        
        if not initial_state:
            ax.set_title("Before Clearing", fontsize=14)
            edge_weights_before_clearing = [G_before_clearing[u][v].get("weight", 1) for u, v in G_before_clearing.edges()]
            edge_widths_before_clearing = [1 + 1 * w for w in edge_weights_before_clearing]
            
            # Edge weights for line width
            # Draw edges
            if edge_widths_before_clearing:
                nx.draw_networkx_edges(G_before_clearing, pos, ax=ax, edge_color="gray", width=edge_widths_before_clearing, min_source_margin=15, min_target_margin=15, arrows=True)
        
            # Draw edge labels
            edge_labels = {k: f"{v:.1f}" for k, v in nx.get_edge_attributes(G_before_clearing, "weight").items()}
            nx.draw_networkx_edge_labels(G_before_clearing, pos, edge_labels=edge_labels, font_size=8, ax=ax, label_pos=0.5, bbox=dict(facecolor='white', edgecolor='none', pad=0))
        
            # Draw solvent nodes with color
            sc = nx.draw_networkx_nodes(G_before_clearing, pos, ax=ax,
                                         nodelist=solvent_nodes_before_clearing,
                                         node_color=solvent_colors_before_clearing,
                                         node_size=solvent_sizes_before_clearing,
                                         cmap="coolwarm", # Node colors show allocated capital before clearing.
                                         vmin=0,
                                         vmax=self.upper_limit_capital_injections,
                                         edgecolors="black",
                                         linewidths=1)
        
            # Draw insolvent nodes
            nx.draw_networkx_nodes(G_before_clearing, pos, ax=ax,
                                   nodelist=insolvent_nodes_before_clearing,
                                   node_color="gray",
                                   node_size=insolvent_sizes_before_clearing,
                                   edgecolors="black",
                                   linewidths=1)
            
            nx.draw_networkx_labels(G_before_clearing, pos, labels=labels_before_clearing, font_color="white", font_size=node_font_size, ax=ax)
       
            # Colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Action") 
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            #visualizing the price:
            # place bar
            min_price = self.min_price
            max_price = self.max_price
            mid_price = (max_price + min_price) / 2
    
            norm_price = (self.last_price-min_price)/(max_price-min_price)
    
            bar_x =0.01
            bar_y = 0.01
            bar_width = 0.25
            bar_height = 0.02
            bar = plt.Rectangle((bar_x,bar_y), bar_width, bar_height, transform=ax.transAxes, color='lightgrey')
            ax.add_patch(bar)
    
            # mark current price
            marker_width = 0.005
            marker_height = bar_height * 2
            marker_x = bar_x + norm_price * bar_width - marker_width / 2
            marker_y = bar_y + bar_height / 2 - marker_height / 2
    
            # draw marker
            marker = plt.Rectangle((marker_x, marker_y), marker_width, marker_height, transform = ax.transAxes, color = 'red')
    
            ax.add_patch(marker)
    
            fontsize = 8
            # text over bar
            ax.text(bar_x + norm_price * bar_width, bar_y + marker_height + 0.01, f"{self.last_price:.1f}",
                transform=ax.transAxes, fontsize=fontsize, ha='center', va='bottom')
    
            # legend under bar
            legend_y = bar_y - 0.03
            ax.text(bar_x, legend_y, f"{min_price:.1f}", transform=ax.transAxes,
                    fontsize=fontsize, ha='left', va='top')
            ax.text(bar_x + bar_width / 2, legend_y, f"{mid_price:.1f}", transform=ax.transAxes,
                    fontsize=fontsize, ha='center', va='top')
            ax.text(bar_x + bar_width, legend_y, f"{max_price:.1f}", transform=ax.transAxes,
                    fontsize=fontsize, ha='right', va='top')       
            # add margins
            x_margin = 0.5
            y_margin = 0.5
            ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
            ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)
            
            ax.set_aspect('equal', adjustable='datalim')

            # Final layout
            ax.set_facecolor("lightgray")
            ax.axis("off")
            plt.tight_layout()
            if self.save_render:
                filename = self.frame_dir / f"frame_{self.frame_count:04d}.png"
                plt.savefig(filename, dpi=160, bbox_inches=None)
                self.frame_count += 1
            plt.show()
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

            #rendering state after clearing:
            ax.set_title("After Clearing", fontsize=14)
            
        # Draw edges
        if edge_widths:
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=edge_widths, min_source_margin=15, min_target_margin=15, arrows=True)
    
        # Draw edge labels
        edge_labels = {k: f"{v:.1f}" for k, v in nx.get_edge_attributes(G, "weight").items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax, label_pos=0.5, bbox=dict(facecolor='white', edgecolor='none', pad=0))
    
        # Draw solvent nodes with color
        sc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                     nodelist=solvent_nodes,
                                     node_color = solvent_colors,
                                     node_size=solvent_sizes,
                                     cmap="coolwarm", # Use the zero-injection color after clearing.
                                     vmin=0,
                                     vmax=self.upper_limit_capital_injections,
                                     edgecolors="black",
                                     linewidths=1)
    
        # Draw insolvent nodes
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               nodelist=insolvent_nodes,
                               node_color="gray",
                               node_size=insolvent_sizes,
                               edgecolors="black",
                               linewidths=1)
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_color="white", font_size=node_font_size, ax=ax)
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Action")
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #visualizing the price:
        # place bar
        min_price = self.min_price
        max_price = self.max_price
        mid_price = (max_price + min_price) / 2
        norm_price = (self.price_risky - min_price) / (max_price - min_price)
               
        bar_x =0.01
        bar_y = 0.01
        bar_width = 0.25
        bar_height = 0.02
        bar = plt.Rectangle((bar_x,bar_y), bar_width, bar_height, transform=ax.transAxes, color='lightgrey')
        ax.add_patch(bar)

        # mark current price
        marker_width = 0.005
        marker_height = bar_height * 2
        marker_x = bar_x + norm_price * bar_width - marker_width / 2
        marker_y = bar_y + bar_height / 2 - marker_height / 2

        # draw marker
        marker = plt.Rectangle((marker_x, marker_y), marker_width, marker_height, transform = ax.transAxes, color = 'red')

        ax.add_patch(marker)

        fontsize = 8
        # text over bar

        # show current price (not the one before the decision!)
        ax.text(bar_x + norm_price * bar_width, bar_y + marker_height + 0.01, f"{self.price_risky:.1f}", transform=ax.transAxes, fontsize=fontsize, ha='center', va='bottom')

        # legend under bar
        legend_y = bar_y - 0.03
        ax.text(bar_x, legend_y, f"{min_price:.1f}", transform=ax.transAxes,
                fontsize=fontsize, ha='left', va='top')
        ax.text(bar_x + bar_width / 2, legend_y, f"{mid_price:.1f}", transform=ax.transAxes,
                fontsize=fontsize, ha='center', va='top')
        ax.text(bar_x + bar_width, legend_y, f"{max_price:.1f}", transform=ax.transAxes,
                fontsize=fontsize, ha='right', va='top')
       
        # add margins
        x_margin = 0.5
        y_margin = 0.5
        ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)
        
        ax.set_aspect('equal', adjustable='datalim')

        # Final layout
        ax.set_facecolor("lightgray")
        ax.axis("off")
        plt.tight_layout()
        if self.save_render:
            filename = self.frame_dir / f"frame_{self.frame_count:04d}.png"
            plt.savefig(filename, dpi=160, bbox_inches=None)
            self.frame_count += 1
        plt.show()

        if price_influences and self.delta_simulation:
            print("Price differences due to simulation and FS effects:")
            print(f"  Δ Simulation: {self.delta_simulation:.4f}")
            print(f"  Δ FS Effects: {self.delta_fs:.4f}")

        if adjacency_matrix:
            if node_features:
                # Initialize node data list
                node_data = []
                # Append node features
                for node in self.node_features:
                    if self.node_features[node]['solvency'] < 2:
                        node_data.append({
                            "L": f"{self.node_features[node]['total_liabilities']:.1f}",
                            "risk_free": f"{self.node_features[node]['risk_free_assets']:.1f}",
                            "risky": f"{self.node_features[node]['risky_assets']:.1f}",
                            "S": self.node_features[node]['solvency']
                        })
                df_nodes = pd.DataFrame(node_data)
            
                # Display adjacency matrix as a table
                A = nx.to_numpy_array(G)
                adj=np.round(A, 1)
                df_adj = pd.DataFrame(adj)
            
                # Show tables side by side
                df_nodes_html = (df_nodes.style
                                 .set_caption("node features")
                                 .to_html(index=True))
                df_adj_html = (df_adj.style
                               .set_caption("adjacency matrix")
                               .format("{:.1f}")
                               .to_html(index=True))
            
                display_html(
                    "<table><tr>"
                    f"<td style='vertical-align:top'>{df_nodes_html}</td>"
                    f"<td style='vertical-align:top'>{df_adj_html}</td>"
                    "</tr></table>",
                    raw=True
                )    
            else:    
                print(f" adjacency matrix: {self.graph}")
                
        elif node_features:
            print(f" node features: {self.node_features}")

    def close(self):
        '''
        Mark the environment as closed.
        '''
        self._closed = True

    def quantize_value(self, value, value_type):
        '''
        Clip a value to the bounds of its observation component.
        '''
        if value_type == "price":
            return np.float32(np.clip(value, self.min_price, self.max_price))
        elif value_type == "adjacency":
            return np.float32(np.clip(value, 0.0, self.max_adjacency))
        elif value_type == "node_features":
            return np.float32(np.clip(value, self.min_node_features, self.max_node_features))
        else:
            raise ValueError("Invalid value_type. Choose from 'price', 'adjacency', 'node_features'.")
            return

    def relative_matrix(self,adjacency_matrix):
        '''
        converts a matrix of absolute liabilities to a matrix of relative liabilities and returns absolute liabilities as vector
        '''
        total_liabilities = np.zeros(self.num_nodes)
        total_claims = np.zeros(self.num_nodes)
        rel_adjacency_matrix = adjacency_matrix.copy()
        for i in range(self.num_nodes):
            total_liabilities[i] = np.sum(adjacency_matrix[i, :])
            total_claims[i] = np.sum(adjacency_matrix[:, i])
            if total_liabilities[i] > 0:
                rel_adjacency_matrix[i, :] /= total_liabilities[i] 

        return rel_adjacency_matrix, total_liabilities, total_claims

    def matrix_to_state(self):
        '''
        translates an adjacency matrix and a dict. of node_features to the underlying state
        '''
        observed_graph = self.graph[np.ix_(self.observed_nodes, self.observed_nodes)]
        matrix = self.quantize_value(observed_graph.flatten(), "adjacency")

        node_features_array = []
        int_node_features_array = []

        for node in self.observed_nodes:
            data = self.node_features[node]
            node_values = [
                data["risk_free_assets"],
                data["risky_assets"],
                data["internal_liabilities"],
                data["external_liabilities"],
                data["total_claims"],
            ]

            node_features_array.extend(node_values)
            int_node_features_array.append(data["solvency"])

        node_features_array = self.quantize_value(
            np.array(node_features_array, dtype=np.float32), "node_features"
        )
        int_node_features_array = np.array(int_node_features_array, dtype=np.int32)
        
        # Include the clipped risky-asset price in the observation.
        price_value = np.array([self.quantize_value(self.price_risky, "price")])

        # Construct the state dictionary
        state = {
            "adjacency_matrix": matrix,
            "node_features": node_features_array,
            "int_node_features": int_node_features_array,
            "price_risky_asset": price_value
        }

        return state

    def seed(self, seed=None):
        """Set the random generator without resetting the current network state.

        Retained for compatibility with callers of the original environment.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        print(f"environment seed set to: {seed}")
        return [seed]

    def action_masks(self):
        """Allow nonzero interventions only for solvent observed banks.

        The flat mask covers the budget dimension followed by bank weights.
        """

        if self.action_space_architecture == "weighted_limit":
            allowed_actions = [[True]* len(self.bins_action)]

            for node in self.observed_nodes:
                if self.node_features[node]['solvency'] != 0:
                    allowed_actions.append([True] + [False] * (self.num_action_weights-1))
                    continue

                allowed_actions.append([True] * self.num_action_weights)
            return np.concatenate([allowed_actions[i] for i in range(len(allowed_actions))]) #flatted return as it is not homogenous


