import torch
import torch.nn as nn
import torch.nn.functional as F


OBSERVATION_NORMALIZATION_BOUNDS = {
    "min_price": -2.0,
    "max_price": 2.0,
    "max_adjacency": 5.0,
    "min_node_features": -7.0,
    "max_node_features": 7.0,
}


def get_observation_normalization_bounds():
    """
    Return the shared normalization limits used by all feature extractors.
    """
    return dict(OBSERVATION_NORMALIZATION_BOUNDS)


def detect_observation_layout(observation_space, num_nodes):
    """
    Validate the environment layout of five numeric features and solvency.
    """
    node_feat_dim = observation_space["node_features"].shape[0]
    int_feat_dim = observation_space["int_node_features"].nvec.shape[0]

    if node_feat_dim == 5 * num_nodes and int_feat_dim == num_nodes:
        return "basic", 5, 3

    raise ValueError(
        f"Expected observation layout (5*{num_nodes}, {num_nodes}), but got "
        f"node_features={node_feat_dim}, int_node_features={int_feat_dim}."
    )


def get_flat_node_feature_dim(layout, node_feature_dim, encoded_int_dim):
    """
    Return the per-node feature dimension after concatenating all features that
    are fed into the encoder.
    """
    _ = layout
    return node_feature_dim + 1 + encoded_int_dim


def to_tensor(value, dtype, device=None):
    """
    Convert numpy arrays or tensors into tensors with the requested dtype.
    """
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def encode_int_node_features(raw_int_features, batch_size, num_nodes, layout, device=None):
    """
    Convert the integer node features into the encoded representation that is
    used by the encoders.
    """
    int_features = to_tensor(raw_int_features, dtype=torch.long, device=device)

    if layout != "basic":
        raise ValueError(f"Unsupported observation layout '{layout}'.")
    if int_features.numel() == batch_size * num_nodes:
        # raw integer labels
        int_features = int_features.long().view(batch_size, num_nodes)
        return F.one_hot(int_features, num_classes=3).float().view(
            batch_size, num_nodes, 3
        )

    if int_features.numel() == batch_size * num_nodes * 3:
        # already preprocessed categorical features
        return int_features.view(batch_size, num_nodes, 3)

    raise ValueError(
        f"Unexpected int_node_features shape: {tuple(int_features.shape)} "
        f"for layout={layout}, batch_size={batch_size}, num_nodes={num_nodes}"
    )    

def normalize_observation_components(node_features, adjacency_matrix, price_risky_asset, normalization_bounds=None):
    """
    Normalize node features, adjacency values and prices into stable ranges.
    """
    bounds = get_observation_normalization_bounds()
    if normalization_bounds is not None:
        bounds.update(normalization_bounds)

    node_features = 2 * (node_features - bounds["min_node_features"]) / (
        bounds["max_node_features"] - bounds["min_node_features"]
    ) - 1
    adjacency_matrix = adjacency_matrix / bounds["max_adjacency"]
    price_risky_asset = 2 * (price_risky_asset - bounds["min_price"]) / (
        bounds["max_price"] - bounds["min_price"]
    ) - 1
    return node_features, adjacency_matrix, price_risky_asset


def prepare_observation_tensors(
    observations,
    num_nodes,
    layout,
    node_feature_dim,
    *,
    normalization_bounds=None,
):
    """
    Parse the observation and return normalized per-node features, the adjacency
    matrix, and the batch size. Each node includes the price and one-hot solvency.
    """
    batch_size = len(observations["price_risky_asset"])
    node_features = to_tensor(observations["node_features"], dtype=torch.float32).view(
        batch_size, num_nodes, node_feature_dim
    )
    adjacency_matrix = to_tensor(
        observations["adjacency_matrix"], dtype=torch.float32, device=node_features.device
    ).view(batch_size, num_nodes, num_nodes)
    price_risky_asset = to_tensor(
        observations["price_risky_asset"], dtype=torch.float32, device=node_features.device
    ).view(batch_size, 1, 1)
    int_node_features = encode_int_node_features(
        observations["int_node_features"],
        batch_size=batch_size,
        num_nodes=num_nodes,
        layout=layout,
        device=node_features.device,
    )

    node_features, adjacency_matrix, price_risky_asset = normalize_observation_components(
        node_features,
        adjacency_matrix,
        price_risky_asset,
        normalization_bounds=normalization_bounds,
    )

    price_feature = price_risky_asset.expand(batch_size, num_nodes, 1)
    node_features_matrix = torch.cat([node_features, price_feature, int_node_features], dim=2)
    return node_features_matrix, adjacency_matrix, batch_size


def build_activation(activation_name):
    """
    Create an activation module from a user-facing name.
    """
    key = str(activation_name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "tanh":
        return nn.Tanh()
    if key == "elu":
        return nn.ELU()
    if key in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(
        f"Unsupported activation '{activation_name}'. Choose from "
        "'relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', or 'identity'."
    )


def normalize_message_aggregation(aggregation_name):
    """
    Map the user-facing aggregation names to the values expected by
    torch_geometric.
    """
    key = str(aggregation_name).strip().lower()
    aliases = {
        "sum": "add",
        "mean": "mean",
        "min": "min",
        "max": "max",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported aggregation '{aggregation_name}'. Choose 'sum', 'mean', 'min', or 'max'."
        )
    return aliases[key]
