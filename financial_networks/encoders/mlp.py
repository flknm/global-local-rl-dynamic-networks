import torch
import torch.nn as nn
from .common import (
    build_activation,
    detect_observation_layout,
    encode_int_node_features,
    normalize_observation_components,
    to_tensor,
)

class MLPEncoder(nn.Module):
    def __init__(self, observation_space, num_nodes=10, num_layers = None, num_heads = None, embedding_dim = None, aggregation=None, activation_function="tanh"):
        self.observation_layout, self.node_feature_dim, self.encoded_int_dim = detect_observation_layout(
            observation_space, num_nodes
        )
        in_channels = self.node_feature_dim + self.encoded_int_dim
        total_features = num_nodes**2+num_nodes*in_channels+1
        self.num_layers = int(num_layers or 0)
        self.output_dim = total_features if self.num_layers == 0 else int(embedding_dim)
        super().__init__()
        self.num_nodes = num_nodes
        self.mlp = self._build_mlp(
            input_dim=total_features,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            activation_function=activation_function,
        )

    @staticmethod
    def _build_mlp(input_dim, output_dim, num_layers, activation_function):
        if num_layers == 0:
            return nn.Identity()

        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, output_dim))
            layers.append(build_activation(activation_function))
            current_dim = output_dim
        return nn.Sequential(*layers)

    def forward(self, observations):
        self.batch_size = len(observations["price_risky_asset"])

        node_features = to_tensor(observations["node_features"], dtype=torch.float32).view(
            self.batch_size, self.num_nodes, self.node_feature_dim
        )
        adjacency_matrix = to_tensor(
            observations["adjacency_matrix"], dtype=torch.float32, device=node_features.device
        )
        price_risky_asset = to_tensor(
            observations["price_risky_asset"], dtype=torch.float32, device=node_features.device
        )
        int_node_features = encode_int_node_features(
            observations["int_node_features"],
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            layout=self.observation_layout,
            device=node_features.device,
        )

        node_features_matrix = torch.cat([node_features, int_node_features], dim=2)
        node_features = node_features_matrix.view(
            self.batch_size,
            self.num_nodes * (self.node_feature_dim + self.encoded_int_dim),
        )

        adjacency_matrix = adjacency_matrix.view(self.batch_size, self.num_nodes*self.num_nodes)
        price_risky_asset = price_risky_asset.view(self.batch_size, 1)

        #centralize features (everything into [-1,1] except adjacency matrix into [0,1])
        node_features, adjacency_matrix, price_risky_asset = normalize_observation_components(
            node_features,
            adjacency_matrix,
            price_risky_asset,
        )
        
        flat_features = torch.cat([adjacency_matrix, node_features, price_risky_asset], dim=1)
        return self.mlp(flat_features)
