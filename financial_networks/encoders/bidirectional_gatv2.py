import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch, Data
from .common import (
    build_activation,
    detect_observation_layout,
    normalize_message_aggregation,
    prepare_observation_tensors,
)


class BidirectionalGATv2Layer(torch.nn.Module):
    """
    Bidirectional GATv2 layer.

    - One GATv2Conv aggregates information from incoming neighbors (edge_index as is).
    - Another GATv2Conv aggregates information from outgoing neighbors
      by using the reversed edge_index.

    Let:
        h_in  = GATv2Conv_in(x, edge_index, edge_attr)   # incoming
        h_out = GATv2Conv_out(x, edge_index_rev, edge_attr)  # outgoing

    The two outputs are combined as:
        h_cat = [h_in || h_out]
        out   = activation(Linear(h_cat))

    Incoming and outgoing features use separate attention scores,
    and a final linear layer to combine them.
    """

    def __init__(self, in_channels, out_channels, num_heads, concat=True, aggregation="sum", activation_function="tanh"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat

        # Two separate GATv2 convolutions for in- and out-directions
        self.conv_in = GATv2Conv(
            in_channels,
            out_channels,
            heads=num_heads,
            concat=concat,
            edge_dim=1,          # Edge weights are one-dimensional edge attributes.
            add_self_loops=True,  # same default as standard GATv2
            aggr=normalize_message_aggregation(aggregation),
        )

        self.conv_out = GATv2Conv(
            in_channels,
            out_channels,
            heads=num_heads,
            concat=concat,
            edge_dim=1,
            add_self_loops=True,
            aggr=normalize_message_aggregation(aggregation),
        )

        # Output dimension of each GAT branch
        conv_dim = out_channels * num_heads if concat else out_channels

        # Linear layer to combine [h_in || h_out] into out_channels
        self.lin = nn.Linear(2 * conv_dim, out_channels)
        self.act = build_activation(activation_function)

    def forward(self, x, edge_index, edge_weight):
        # edge_weight: (E,) -> (E,1) for edge_attr
        edge_attr = edge_weight.view(-1, 1)

        # Incoming neighborhood aggregation
        h_in = self.conv_in(x, edge_index, edge_attr)

        # Outgoing neighborhood aggregation:
        # use reversed edge_index but the same edge_weight (A_ij)
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_out = self.conv_out(x, rev_edge_index, edge_attr)

        # Concatenate in/out features and combine linearly
        h_cat = torch.cat([h_in, h_out], dim=-1)
        out = self.lin(h_cat)
        out = self.act(out)

        return out


class BidirectionalGATv2Stack(torch.nn.Module):
    """
    A stack of multiple bidirectional GATv2 layers.

    - First layer:   in_channels → hidden_channels
    - Middle layers: hidden_channels → hidden_channels
    - Final layer:   hidden_channels → out_channels
    - A single layer maps in_channels directly to out_channels.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, aggregation="sum", activation_function="tanh"):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        if num_layers > 1:
            # First layer: concat=True inside GAT
            self.layers.append(
                BidirectionalGATv2Layer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    num_heads=num_heads,
                    concat=True,
                    aggregation=aggregation,
                    activation_function=activation_function,
                )
            )
            # Middle layers: concat=True
            for _ in range(num_layers - 2):
                self.layers.append(
                    BidirectionalGATv2Layer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        num_heads=num_heads,
                        concat=True,
                        aggregation=aggregation,
                        activation_function=activation_function,
                    )
                )
            # Final layer: concat=False (no head concatenation at the end)
            self.layers.append(
                BidirectionalGATv2Layer(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    num_heads=num_heads,
                    concat=False,
                    aggregation=aggregation,
                    activation_function=activation_function,
                )
            )
        else:
            # Single layer: concat=False
            self.layers.append(
                BidirectionalGATv2Layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_heads=num_heads,
                    concat=False,
                    aggregation=aggregation,
                    activation_function=activation_function,
                )
            )

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x


class BidirectionalGATv2Encoder(torch.nn.Module):
    """
    Bidirectional GATv2 encoder.

    - Uses the shared observation parsing and normalization helpers.
    - Each bank has hidden_dim = embedding_dim / num_nodes features.
    - Returns embeddings with shape (batch_size, num_nodes, hidden_dim).
    """

    def __init__(self, observation_space, num_layers=2, embedding_dim=160, num_nodes=10, num_heads=3, aggregation="sum", activation_function="tanh"):
        super().__init__()

        self.num_nodes = num_nodes
        self.batch_size = None

        self.observation_layout, self.node_feature_dim, self.encoded_int_dim = detect_observation_layout(
            observation_space, num_nodes
        )
        in_channels = self.node_feature_dim + 1 + self.encoded_int_dim

        total_embedding_dim = int(embedding_dim)
        if total_embedding_dim % int(num_nodes) != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_nodes. "
                f"Got embedding_dim={total_embedding_dim}, num_nodes={num_nodes}."
            )
        hidden_dim = total_embedding_dim // int(num_nodes)
        self.output_dim = hidden_dim

        self.gnn = BidirectionalGATv2Stack(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            aggregation=aggregation,
            activation_function=activation_function,
        )

        # Observation normalization is handled by the shared preparation helper.

    def forward(self, observations):
        node_features_matrix, adjacency_matrix, self.batch_size = prepare_observation_tensors(
            observations,
            num_nodes=self.num_nodes,
            layout=self.observation_layout,
            node_feature_dim=self.node_feature_dim,
        )

        # ----------------------
        # Build PyG batch
        # ----------------------
        data_list = []
        for b in range(self.batch_size):
            A = adjacency_matrix[b].to(node_features_matrix.device).float()  # (N, N)
            edge_index = A.nonzero(as_tuple=False).t().contiguous()          # (2, E)
            edge_weight = A[edge_index[0], edge_index[1]]                    # (E,)

            data_list.append(
                Data(
                    x=node_features_matrix[b],
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            )

        batch = Batch.from_data_list(data_list)

        # ----------------------
        # Apply bidirectional GATv2 stack
        # ----------------------
        node_embeddings = self.gnn(batch.x, batch.edge_index, batch.edge_weight)
        # node_embeddings: (total_nodes_in_batch, hidden_dim)

        # Reshape to (batch_size, num_nodes, hidden_dim)
        node_embeddings = node_embeddings.view(self.batch_size, self.num_nodes, -1)

        return node_embeddings

