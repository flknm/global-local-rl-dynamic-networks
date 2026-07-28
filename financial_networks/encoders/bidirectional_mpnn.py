import torch
from torch_geometric.data import Batch, Data
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from .common import (
    build_activation,
    detect_observation_layout,
    normalize_message_aggregation,
    prepare_observation_tensors,
)

class _DirectionalMPN(MessagePassing):
    """
    A simple one-direction message passing layer.
    It only performs neighborhood aggregation in one specified direction.
    No learnable weights inside this layer.
    """

    def __init__(self, in_channels, aggr="sum", flow="source_to_target"):
        super().__init__(aggr=normalize_message_aggregation(aggr), flow=flow)
        self.in_channels = in_channels

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1) # multiplies label with edge weight

    def update(self, aggr_out, x):
        return aggr_out # only return neighborhood aggregation


class BidirectionalMPNNLayer(torch.nn.Module):
    """
    Bidirectional Message Passing Layer.

    Performs two message passing operations:
    - aggr_in: standard "source_to_target" flow  (updated node is creditor)
    - aggr_out: reversed "target_to_source" flow (updated node is debitor)

    The final output is computed as:
        h = concat([x, aggr_in, aggr_out])
        out = activation(Linear(h))
    """

    def __init__(self, in_channels, out_channels, aggr="sum", activation_function="tanh"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        # Aggregate incoming and outgoing neighbors separately, without learnable weights.
        self.mp_in = _DirectionalMPN(in_channels, aggr=self.aggr, flow="source_to_target")
        self.mp_out = _DirectionalMPN(in_channels, aggr=self.aggr, flow="target_to_source")

        # combine own features + both aggregations
        self.lin = Linear(3 * in_channels, out_channels)
        self.act = build_activation(activation_function)

    def forward(self, x, edge_index, edge_weight):
        # gather info from neighbors that point to me
        aggr_in = self.mp_in(x, edge_index, edge_weight)   # (N, in_channels)

        # gather info from neighbors I point to (reverse flow)
        aggr_out = self.mp_out(x, edge_index, edge_weight) # (N, in_channels)

        # combine everything
        h = torch.cat([x, aggr_in, aggr_out], dim=1)       # (N, 3*in_channels)

        # Apply the selected activation after the linear transform.
        return self.act(self.lin(h))                       # (N, out_channels)
        

class BidirectionalMPNNStack(torch.nn.Module):
    """
    A stack of multiple bidirectional message passing layers.

    - First layer:   in_channels  → hidden_channels
    - Middle layers: hidden_channels → hidden_channels
    - Final layer:   hidden_channels → out_channels
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, aggr="sum", activation_function="tanh"):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        if num_layers == 1:
            # single layer directly mapping to output
            self.layers.append(BidirectionalMPNNLayer(in_channels, out_channels, aggr=aggr, activation_function=activation_function))
        else:
            # first layer
            self.layers.append(BidirectionalMPNNLayer(in_channels, hidden_channels, aggr=aggr, activation_function=activation_function))

            # optional middle layers
            for _ in range(num_layers - 2):
                self.layers.append(BidirectionalMPNNLayer(hidden_channels, hidden_channels, aggr=aggr, activation_function=activation_function))

            # final layer
            self.layers.append(BidirectionalMPNNLayer(hidden_channels, out_channels, aggr=aggr, activation_function=activation_function))

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x



class BidirectionalMPNNEncoder(torch.nn.Module):
    """MPNN encoder returning structured per-node embeddings."""

    def __init__(self, observation_space, num_layers=2, embedding_dim=160, num_nodes=10, aggregation="sum", activation_function="tanh"):
        super().__init__()

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
        self.output_dim = total_embedding_dim // int(num_nodes)
        self.gnn = BidirectionalMPNNStack(in_channels=in_channels, hidden_channels=self.output_dim,
                                  out_channels=self.output_dim, num_layers=num_layers,
                                  aggr=aggregation, activation_function=activation_function)
        self.num_nodes = num_nodes
        self.batch_size = None

    def forward(self, observations):
        node_features_matrix, adjacency_matrix, self.batch_size = prepare_observation_tensors(
            observations,
            num_nodes=self.num_nodes,
            layout=self.observation_layout,
            node_feature_dim=self.node_feature_dim,
        )
        
        data_list = []
        for batch in range(self.batch_size):
            edge_index = adjacency_matrix[batch].nonzero(as_tuple=False).t().contiguous()
            edge_weight = adjacency_matrix[batch][edge_index[0], edge_index[1]]
            data_list.append(Data(x=node_features_matrix[batch], edge_index=edge_index, edge_weight=edge_weight))

        batch = Batch.from_data_list(data_list)

        return self.gnn(batch.x, batch.edge_index, batch.edge_weight).view(
            self.batch_size, self.num_nodes, self.output_dim
        )

