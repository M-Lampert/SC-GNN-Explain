import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import identity
from scipy.spatial import Voronoi, distance
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from torch_geometric.data import Data

extra_metrics = {
    "sqeuclidean": distance.sqeuclidean,
}

__all__ = [
    "build_empty_graph",
    "build_graph",
    "build_delaunay_graph",
    "build_knn_graph",
    "build_radius_graph",
    "build_smooth_graph",
    "build_radius_delaunay_graph",
]


def build_empty_graph(
    features: list | np.ndarray = None,
    labels: list | np.ndarray = None,
    library: str = "nx",
) -> Data:
    """Builds an empty graph from with the given node features.

    Args:
        features: The node features given as List or numpy array of shape (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape (num_nodes, num_labels). Defaults to None.
        library: The library of the graph representation that should be returned. Can be one of the following: nx | networkx | dgl | deep_graph_library | pyg | torch_geometric. Defaults to "nx".

    Returns:
        A graph representation
    """
    n_nodes = len(features)
    A = identity(n_nodes)

    return build_graph(adj_list=A, features=features, labels=labels, directed=False, library=library)


def build_knn_graph(
    positions: list | np.ndarray,
    k: int,
    include_self_loops: bool = True,
    add_distance: bool = False,
    metric: str = "sqeuclidean",
    features: list | np.ndarray = None,
    labels: list | np.ndarray = None,
    library: str = "nx",
) -> Data:
    """Builds an undirected graph from a list of positions using the k-nearest neighbors.

    Args:
        positions: The coordinates of the nodes in space given as List or numpy array of shape (num_nodes, num_dims).
        k: The number of neighbors for each node.
        include_self_loops: Whether to include self-loops in the graph. Defaults to True.
        add_distance: Whether to add the distance between the nodes as edge feature. Defaults to False.
        metric: The metric to use for the distance computation. Defaults to "sqeuclidean" which is the squared Euclidean norm.
        features: The node features given as List or numpy array of shape (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape (num_nodes, num_labels). Defaults to None.
        library: The library of the graph representation that should be returned. Can be one of the following: nx | networkx | dgl | deep_graph_library | pyg | torch_geometric. Defaults to "nx".

    Returns:
        A graph representation
    """
    mode = "connectivity" if not add_distance else "distance"
    if include_self_loops:
        k += 1
    A = kneighbors_graph(X=positions, n_neighbors=k, include_self=include_self_loops, mode=mode, metric=metric)

    edge_features = A.data + 1 if add_distance else None

    return build_graph(
        adj_list=A,
        positions=positions,
        features=features,
        labels=labels,
        edge_features=edge_features,
        directed=False,  # Graph can be directed but most GNNs work only on undirected graphs!
        library=library,
    )


def build_radius_graph(
    positions: list | np.ndarray,
    radius: float,
    include_self_loops: bool = True,
    add_distance: bool = False,
    metric: str = "sqeuclidean",
    features: list | np.ndarray = None,
    labels: list | np.ndarray = None,
    library: str = "nx",
) -> Data:
    """Builds an undirected graph from a list of points by connecting all points that are inside a certain radius.

    Args:
        positions: The coordinates of the nodes in space given as List or numpy array of shape (num_nodes, num_dims).
        radius: The radius for each node.
        include_self_loops: Whether to include self-loops in the graph. Defaults to True.
        add_distance: Whether to add the distance between the nodes as edge feature. Defaults to False.
        metric: The metric to use for the distance computation. Defaults to "sqeuclidean" which is the squared Euclidean norm.
        features: The node features given as List or numpy array of shape (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape (num_nodes, num_labels). Defaults to None.
        library: The library of the graph representation that should be returned. Can be one of the following: nx | networkx | dgl | deep_graph_library | pyg | torch_geometric. Defaults to "nx".

    Returns:
        A graph representation
    """
    mode = "connectivity" if not add_distance else "distance"
    A = radius_neighbors_graph(X=positions, radius=radius, include_self=include_self_loops, mode=mode, metric=metric)

    edge_features = A.data + 1 if add_distance else None

    return build_graph(
        adj_list=A,
        positions=positions,
        features=features,
        labels=labels,
        edge_features=edge_features,
        directed=False,
        library=library,
    )


def build_delaunay_graph(
    positions: list | np.ndarray,
    include_self_loops: bool = True,
    add_distance: bool = False,
    metric: str = "sqeuclidean",
    features: list | np.ndarray = None,
    labels: list | np.ndarray = None,
    library: str = "nx",
) -> Data:
    """Builds an undirected graph from a list of positions using the Delaunay graph.

    Args:
        positions: The coordinates of the nodes in space given as List or numpy array of shape (num_nodes, num_dims).
        include_self_loops: Whether to include self-loops in the graph. Defaults to True.
        add_distance: Whether to add the distance between the nodes as edge feature. Defaults to False.
        metric: The metric to use for the distance computation of the edge weights. Specifying another metric than the Euclidean norm will not change the metric used for the Delaunay graph Construction. Defaults to "sqeuclidean" which is the squared Euclidean norm.
        features: The node features given as List or numpy array of shape (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape (num_nodes, num_labels). Defaults to None.
        library: The library of the graph representation that should be returned. Can be one of the following: nx | networkx | dgl | deep_graph_library | pyg | torch_geometric. Defaults to "nx".

    Returns:
        A graph representation
    """
    adj_list, edge_features = get_Voronoi_edges(positions, include_self_loops, add_distance, metric)

    return build_graph(
        adj_list=adj_list,
        positions=positions,
        features=features,
        labels=labels,
        edge_features=edge_features,
        directed=False,
        library=library,
    )


def get_Voronoi_edges(positions, include_self_loops, add_distance, metric) -> tuple[np.ndarray, np.ndarray]:
    """Gets the edges and if specified the edge weights for the Delaunay graph.

    Args:
        positions: The coordinates of the nodes in space given as List or numpy array of shape (num_nodes, num_dims).
        include_self_loops: Whether to include self-loops in the graph.
        add_distance: Whether to add the distance between the nodes as edge feature.
        metric: The metric to use for the distance computation of the edge weights. Specifying another metric than the Euclidean norm will not change the metric used for the Delaunay graph Construction. Defaults to "sqeuclidean" which is the squared Euclidean norm.

    Returns:
        The adjacency list and the corresponding edge weights
    """

    vor = Voronoi(points=positions)
    adj_list = vor.ridge_points

    if add_distance:
        src_positions, dest_positions = positions[adj_list[:, 0]], positions[adj_list[:, 1]]
        if metric.lower() in extra_metrics:
            metric = extra_metrics[metric]
        edge_features = paired_distances(src_positions, dest_positions, metric=metric) + 1
    else:
        edge_features = None

    if include_self_loops:
        # Add self-loops
        num_nodes = len(positions)
        self_loops = np.repeat(np.arange(num_nodes), 2).reshape(-1, 2)
        adj_list = np.concatenate([adj_list, self_loops], axis=0)
        if add_distance:
            edge_features = np.concatenate([edge_features, np.ones(num_nodes)], axis=0)

    return adj_list, edge_features


def build_radius_delaunay_graph(
    positions: list | np.ndarray,
    radius: float,
    include_self_loops: bool = True,
    add_distance: bool = False,
    metric: str = "sqeuclidean",
    features: list | np.ndarray = None,
    labels: list | np.ndarray = None,
    library: str = "nx",
) -> Data:
    """Builds an undirected graph from a list of positions using the intersection of Delaunay graph and radius graph.

    Args:
        positions: The coordinates of the nodes in space given as List or numpy array of shape (num_nodes, num_dims).
        radius: The radius for each node.
        include_self_loops: Whether to include self-loops in the graph. Defaults to True.
        add_distance: Whether to add the distance between the nodes as edge feature. Defaults to False.
        metric: The metric to use for the distance computation of the edge weights. Specifying another metric than the Euclidean norm will not change the metric used for the Delaunay graph Construction. Defaults to "sqeuclidean" which is the squared Euclidean norm.
        features: The node features given as List or numpy array of shape (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape (num_nodes, num_labels). Defaults to None.
        library: The library of the graph representation that should be returned. Can be one of the following: nx | networkx | dgl | deep_graph_library | pyg | torch_geometric. Defaults to "nx".

    Returns:
        A graph representation
    """
    mode = "connectivity" if not add_distance else "distance"
    voronoi_adj_list, voronoi_edge_features = get_Voronoi_edges(positions, include_self_loops, add_distance, metric)

    A = radius_neighbors_graph(
        X=positions, radius=radius, include_self=include_self_loops, mode=mode, metric=metric
    ).tocoo()
    radius_edge_features = A.data + 1 if add_distance else None
    src_indices, dest_indices = A.row, A.col

    # Since all edges are only contained in one direction, we sort to make sure they are in the same direction in both lists
    # Sorting is normally not really efficient but since the sorted arrays all contain only 2 elements each it should not be much of a problem.
    radius_adj_list = np.sort(np.array([src_indices, dest_indices]).T, axis=1)
    voronoi_adj_list = np.sort(voronoi_adj_list, axis=1)
    radius_adj_df = pd.DataFrame(
        data=radius_adj_list,
        columns=["src", "dest"],
    )
    voronoi_adj_df = pd.DataFrame(
        data=voronoi_adj_list,
        columns=["src", "dest"],
    )
    if add_distance:
        radius_adj_df["feat"] = radius_edge_features
        voronoi_adj_df["feat"] = voronoi_edge_features
    radius_adj_df = radius_adj_df.drop_duplicates()

    merged_df = voronoi_adj_df.merge(radius_adj_df, how="inner", on=["src", "dest"])

    adj_list = merged_df[["src", "dest"]].values
    edge_features = merged_df["feat_x"].values if add_distance else None

    return build_graph(
        adj_list=adj_list,
        positions=positions,
        features=features,
        labels=labels,
        edge_features=edge_features,
        directed=False,
        library=library,
    )


def build_graph(
    adj_list: list[list | tuple[int, int]] | np.ndarray | sp.spmatrix,
    positions: list | np.ndarray = None,
    features: list | np.ndarray = None,
    labels: list | np.ndarray = None,
    edge_features: list | np.ndarray = None,
    directed: bool = False,
    library: str = "nx",
) -> Data:
    """Builds a graph from an adjacency list and assigns optional node attributes.

    Args:
        adj_list: The edges given as either as a list of tuples (u, v) or lists where u and v are the node indices or as numpy array with shape (num_edges, 2). It can also be a sparse adjacency matrix.
        positions: The coordinates of the nodes in space given as List or numpy array of shape (num_nodes, num_dims). Defaults to None.
        features: The node features given as List or numpy array of shape (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape (num_nodes, 1). Defaults to None.
        edge_features: The edge features given as List or numpy array of shape (num_edges, num_features). Defaults to None.
        directed: Whether the graph should be directed or not. Defaults to False.
        library: The library of the graph representation that should be returned. Can be one of the following: nx | networkx | dgl | deep_graph_library | pyg | torch_geometric. Defaults to "nx".

    Returns:
        A graph representation for the given adjacency list.
    """

    return build_graph_pyg(adj_list, positions, features, labels, edge_features, directed)


def build_graph_pyg(
    adj_list: list[list | tuple[int, int]] | np.ndarray | sp.spmatrix,
    positions: list | np.ndarray = None,
    features: list | np.ndarray = None,
    labels: list | np.ndarray = None,
    edge_features: list | np.ndarray = None,
    directed: bool = False,
) -> Data:
    """Builds a graph from an adjacency list and assigns optional node attributes.

    Args:
        adj_list: The edges given as either as a list of tuples (u, v) or lists where u and v are the node indices or as numpy array with shape (num_edges, 2). It can also be a sparse adjacency matrix.
        positions: The coordinates of the nodes in space given as List or numpy array of shape (num_nodes, num_dims). Defaults to None.
        features: The node features given as List or numpy array of shape (num_nodes, num_features). Defaults to None.
        labels: The node labels given as List or numpy array of shape (num_nodes, 1). Defaults to None.
        edge_features: The edge features given as List or numpy array of shape (num_edges, num_features). Defaults to None.
        directed: Whether the graph should be directed or not. Defaults to False.

    Returns:
        The graph representation from PyG for the given adjacency list.
    """
    import torch_geometric as pyg

    if isinstance(adj_list, sp.spmatrix):
        edge_index, _ = pyg.utils.from_scipy_sparse_matrix(adj_list)
    else:
        # transpose edge list to get shape (2, num_edges)
        # and use contiguous to copy the tensor into the correct memory layout and not just move the strides and stuff
        edge_index = to_tensor(adj_list, dtype=torch.long).T.contiguous()
    # convert optional attributes to torch tensors
    positions = to_tensor(positions, dtype=torch.float) if positions is not None else None
    node_features = to_tensor(features, dtype=torch.float) if features is not None else None
    labels = to_tensor(labels, dtype=torch.long) if labels is not None else None
    edge_features = to_tensor(edge_features, dtype=torch.float) if edge_features is not None else None

    if not directed:
        if edge_features is None:
            edge_index = pyg.utils.to_undirected(edge_index)
        else:
            edge_index, edge_features = pyg.utils.to_undirected(edge_index, edge_attr=edge_features, reduce="mean")

    return pyg.data.Data(x=node_features, edge_index=edge_index, y=labels, pos=positions, edge_attr=edge_features)


def to_tensor(x: any, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Helper function to copy tensors the `right` way i.e. suppress the warning that would come if tensors are initialized using a tensor.

    Args:
        x: Array or tensor that should be converted to a tensor
        dtype: The datatype. Defaults to torch.float.

    Returns:
        The tensor
    """
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return torch.tensor(x, dtype=dtype)
