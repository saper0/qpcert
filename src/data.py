import logging
from pathlib import Path
from typing import Any, Dict, Union, Tuple

from jaxtyping import Float, Integer
import networkx as nx
import numpy as np
from numpy import ndarray
import torch
from torch_geometric.datasets.planetoid import Planetoid
from torch_sparse import SparseTensor
import scipy.sparse as sp

from src.graph_models.csbm import CSBM


sparse_graph_properties = [
    'adj_matrix', 'attr_matrix', 'labels', 'node_names', 'attr_names', 'class_names', 'metadata'
]


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    All properties are immutable so users don't mess up the
    data format's assumptions.
    Be careful when circumventing this and changing the internal matrices
    regardless (e.g. by exchanging the data array of a sparse matrix).

    Parameters
    ----------
    adj_matrix
        Adjacency matrix in CSR format. Shape [num_nodes, num_nodes]
    attr_matrix
        Attribute matrix in CSR or numpy format. Shape [num_nodes, num_attr]
    labels
        Array, where each entry represents respective node's label(s). Shape [num_nodes]
        Alternatively, CSR matrix with labels in one-hot format. Shape [num_nodes, num_classes]
    node_names
        Names of nodes (as strings). Shape [num_nodes]
    attr_names
        Names of the attributes (as strings). Shape [num_attr]
    class_names
        Names of the class labels (as strings). Shape [num_classes]
    metadata
        Additional metadata such as text.

    """

    def __init__(
            self, adj_matrix: sp.spmatrix,
            attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
            labels: Union[np.ndarray, sp.spmatrix] = None,
            node_names: np.ndarray = None,
            attr_names: np.ndarray = None,
            class_names: np.ndarray = None,
            metadata: Any = None):
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)."
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree.")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree.")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree.")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree.")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree.")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph.
        """
        return self.adj_matrix.shape[0]

    def num_edges(self, warn: bool = True) -> int:
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as _two_ edges.

        """
        if warn and not self.is_directed():
            warnings.warn("num_edges always returns the number of directed edges now.", FutureWarning)
        return self.adj_matrix.nnz

    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def get_edgeid_to_idx_array(self) -> np.ndarray:
        """Return a Numpy Array that maps edgeids to the indices in the adjacency matrix.

        Returns
        -------
        np.ndarray
            The i'th entry contains the x- and y-coordinates of edge i in the adjacency matrix.
            Shape [num_edges, 2]

        """
        return np.transpose(self.adj_matrix.nonzero())

    def get_idx_to_edgeid_matrix(self) -> sp.csr_matrix:
        """Return a sparse matrix that maps indices in the adjacency matrix to edgeids.

        Caution: This contains one explicit 0 (zero stored as a nonzero),
        which is the index of the first edge.

        Returns
        -------
        sp.csr_matrix
            The entry [x, y] contains the edgeid of the corresponding edge (or 0 for non-edges).
            Shape [num_nodes, num_nodes]

        """
        return sp.csr_matrix(
            (np.arange(self.adj_matrix.nnz), self.adj_matrix.indices, self.adj_matrix.indptr),
            shape=self.adj_matrix.shape)

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric).
        """
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self) -> 'SparseGraph':
        """Convert to an undirected graph (make adjacency matrix symmetric).
        """
        idx = self.get_edgeid_to_idx_array().T
        ridx = np.ravel_multi_index(idx, self.adj_matrix.shape)
        ridx_rev = np.ravel_multi_index(idx[::-1], self.adj_matrix.shape)

        # Get duplicate edges (self-loops and opposing edges)
        dup_ridx = ridx[np.isin(ridx, ridx_rev)]
        dup_idx = np.unravel_index(dup_ridx, self.adj_matrix.shape)

        # Check if the adjacency matrix weights are symmetric (if nonzero)
        if len(dup_ridx) > 0 and not np.allclose(self.adj_matrix[dup_idx], self.adj_matrix[dup_idx[::-1]]):
            raise ValueError("Adjacency matrix weights of opposing edges differ.")

        # Create symmetric matrix
        new_adj_matrix = self.adj_matrix + self.adj_matrix.T
        if len(dup_ridx) > 0:
            new_adj_matrix[dup_idx] = (new_adj_matrix[dup_idx] - self.adj_matrix[dup_idx]).A1

        self.adj_matrix = new_adj_matrix
        return self

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1).
        """
        return np.any(np.unique(self.adj_matrix[self.adj_matrix.nonzero()].A1) != 1)

    def to_unweighted(self) -> 'SparseGraph':
        """Convert to an unweighted graph (set all edge weights to 1).
        """
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    def is_connected(self) -> bool:
        """Check if the graph is connected.
        """
        return sp.csgraph.connected_components(self.adj_matrix, return_labels=False) == 1

    def has_self_loops(self) -> bool:
        """Check if the graph has self-loops.
        """
        return not np.allclose(self.adj_matrix.diagonal(), 0)

    def __repr__(self) -> str:

        dir_string = 'Directed' if self.is_directed() else 'Undirected'
        weight_string = 'weighted' if self.is_weighted() else 'unweighted'
        conn_string = 'connected' if self.is_connected() else 'disconnected'
        loop_string = 'has self-loops' if self.has_self_loops() else 'no self-loops'
        return ("<{}, {} and {} SparseGraph with {} edges ({})>"
                .format(dir_string, weight_string, conn_string,
                        self.num_edges(warn=False), loop_string))

    def _adopt_graph(self, graph: 'SparseGraph'):
        """Copy all properties from the given graph to this graph.
        """
        for prop in sparse_graph_properties:
            setattr(self, '_{}'.format(prop), getattr(graph, prop))

    # Quality of life (shortcuts)
    def standardize(
            self, make_unweighted: bool = True,
            make_undirected: bool = True,
            no_self_loops: bool = True,
            select_lcc: bool = True
    ) -> 'SparseGraph':
        """Perform common preprocessing steps: remove self-loops, make unweighted/undirected, select LCC.

        All changes are done inplace.

        Parameters
        ----------
        make_unweighted
            Whether to set all edge weights to 1.
        make_undirected
            Whether to make the adjacency matrix symmetric. Can only be used if make_unweighted is True.
        no_self_loops
            Whether to remove self loops.
        select_lcc
            Whether to select the largest connected component of the graph.

        """
        G = self
        if make_unweighted and G.is_weighted():
            G = G.to_unweighted()
        if make_undirected and G.is_directed():
            G = G.to_undirected()
        if no_self_loops and G.has_self_loops():
            G = remove_self_loops(G)
        if select_lcc and not G.is_connected():
            G = largest_connected_components(G, 1, make_undirected)
        self._adopt_graph(G)
        return G

    @staticmethod
    def from_flat_dict(data_dict: Dict[str, Any]) -> 'SparseGraph':
        """Initialize SparseGraph from a flat dictionary.
        """
        init_dict = {}
        del_entries = []

        # Construct sparse matrices
        for key in data_dict.keys():
            if key.endswith('_data') or key.endswith('.data'):
                if key.endswith('_data'):
                    sep = '_'
                else:
                    sep = '.'
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = '{}{}indices'.format(matrix_name, sep)
                mat_indptr = '{}{}indptr'.format(matrix_name, sep)
                mat_shape = '{}{}shape'.format(matrix_name, sep)
                if matrix_name == 'adj' or matrix_name == 'attr':
                    matrix_name += '_matrix'
                init_dict[matrix_name] = sp.csr_matrix(
                    (data_dict[mat_data],
                     data_dict[mat_indices],
                     data_dict[mat_indptr]),
                    shape=data_dict[mat_shape])
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

        # Delete sparse matrix entries
        for del_entry in del_entries:
            del data_dict[del_entry]

        # Load everything else
        for key, val in data_dict.items():
            if ((val is not None) and (None not in val)):
                init_dict[key] = val

        return SparseGraph(**init_dict)


def largest_connected_components(sparse_graph: SparseGraph, n_components: int = 1, make_undirected=True) -> SparseGraph:
    """Select the largest connected components in the graph.

    Changes are returned in a partially new SparseGraph.

    Parameters
    ----------
    sparse_graph
        Input graph.
    n_components
        Number of largest connected components to keep.

    Returns
    -------
    SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix, directed=make_undirected)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def remove_self_loops(sparse_graph: SparseGraph) -> SparseGraph:
    """Remove self loops (diagonal entries in the adjacency matrix).

    Changes are returned in a partially new SparseGraph.

    """
    num_self_loops = (~np.isclose(sparse_graph.adj_matrix.diagonal(), 0)).sum()
    if num_self_loops > 0:
        adj_matrix = sparse_graph.adj_matrix.copy().tolil()
        adj_matrix.setdiag(0)
        adj_matrix = adj_matrix.tocsr()
        return SparseGraph(
            adj_matrix, sparse_graph.attr_matrix, sparse_graph.labels, sparse_graph.node_names,
            sparse_graph.attr_names, sparse_graph.class_names, sparse_graph.metadata)
    else:
        return sparse_graph


def create_subgraph(
        sparse_graph: SparseGraph,
        _sentinel: None = None,
        nodes_to_remove: np.ndarray = None,
        nodes_to_keep: np.ndarray = None
) -> SparseGraph:
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    The subgraph partially points to the old graph's data.

    Parameters
    ----------
    sparse_graph
        Input graph.
    _sentinel
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove
        Indices of nodes that have to removed.
    nodes_to_keep
        Indices of nodes that have to be kept.

    Returns
    -------
    SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...).")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is None:
        attr_matrix = None
    else:
        attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is None:
        labels = None
    else:
        labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is None:
        node_names = None
    else:
        node_names = sparse_graph.node_names[nodes_to_keep]
    return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, sparse_graph.attr_names,
        sparse_graph.class_names, sparse_graph.metadata)


def get_csbm(
        specification: Dict[str, Any]
) -> Tuple[Float[ndarray, "n n"], Integer[ndarray, "n n"], Integer[ndarray, "n"]]:
    n = specification["n_trn_labeled"] + specification["n_trn_unlabeled"] \
        + specification["n_val"] + specification["n_test"]
    csbm = CSBM(n=n, **specification)
    logging.info(f"CSBM(p={csbm.p}, q={csbm.q})")
    X, A, y = csbm.sample(n, specification["seed"])
    return X, A, y


def get_planetoid(dataset: str, specification: Dict[str, Any]):
    '''Loads Planetoid datasets from 
    https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid
    '''
    make_undirected = specification["make_undirected"]
    dataset_root = specification["data_dir"]
    assert make_undirected == True , "undirected not implemented for cora"
    cora = Planetoid(root = dataset_root, name=dataset)
    X = cora.data.x
    y = cora.data.y.numpy()
    edge_index = cora.data.edge_index
    edge_weight = torch.ones(edge_index.shape[1])
    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(edge_index.max()+1, edge_index.max()+1))
    return X, A, y


def get_cora_citeseer_pubmed(name: str,
                             dataset_root: str,
                             make_undirected: bool = True):
    """Prepares and normalizes the desired dataset

    Parameters
    ----------
    name : str
        Name of the data set. One of: `cora_ml`, `citeseer`, `pubmed`
    dataset_root : str
        Path where to find/store the dataset.
    make_undirected : bool, optional
        Normalize adjacency matrix with symmetric degree normalization (non-scalable implementation!), by default False

    Returns
    -------
    Tuple[torch.Tensor, SparseTensor, torch.Tensor]
        dense attribute tensor, sparse adjacency matrix (normalized) and labels tensor
    """
    graph = load_dataset(name, dataset_root).standardize(
        make_unweighted=True,
        make_undirected=make_undirected,
        no_self_loops=True,
        select_lcc=True
    )

    X = torch.FloatTensor(graph.attr_matrix.toarray())
    A = graph.adj_matrix.tocoo()
    A = SparseTensor.from_scipy(A).coalesce().to_dense()
    y = torch.LongTensor(graph.labels)
    print(A.shape)
    print(y.shape)
    print(A)
    return X, A, y


def load_dataset(name: str,
                 directory: Union[Path, str] = './data'
                 ) -> SparseGraph:
    """Load a dataset.

    Parameters
    ----------
    name
        Name of the dataset to load.
    directory
        Path to the directory where the datasets are stored.

    Returns
    -------
    SparseGraph
        The requested dataset in sparse format.

    """
    if isinstance(directory, str):
        directory = Path(directory)
    path_to_file = directory / (name + ".npz")
    if path_to_file.exists():
        with np.load(path_to_file, allow_pickle=True) as loader:
            loader = dict(loader)
            del loader['type']
            del loader['edge_attr_matrix']
            del loader['edge_attr_names']
            dataset = SparseGraph.from_flat_dict(loader)
    else:
        raise ValueError("{} doesn't exist.".format(path_to_file))
    return dataset


def get_graph(
        data_params: Dict[str, Any], sort: bool=True
) -> Tuple[Float[ndarray, "n n"], Integer[ndarray, "n n"], Integer[ndarray, "n"]]:
    """Return graph sampled from a CSBM.

    If sort is true, X, A and y are sorted for class.
    
    Returns X, A, y."""

    if data_params["dataset"] == "csbm":
        X, A, y = get_csbm(data_params["specification"])
    elif data_params["dataset"] in ["cora"]:
        X, A, y = get_planetoid(data_params["dataset"], data_params["specification"])
        A = A.to_dense()
        #if data_params["dataset"] == "citeseer":
        #    G = nx.from_numpy_array(A.detach().cpu().numpy())
        #    print(G.adj)
        #    G = max(nx.connected_components(G), key=len)
        #    #A = 
        #print("nodes without connections")
        #print((A.sum(dim=1)==0).sum())
    elif data_params["dataset"] in ['cora_ml', "citeseer", "pubmed"]:
        assert False, "Datasets currently not supported."
        X, A, y = get_cora_citeseer_pubmed(data_params["dataset"],
                                           data_params["specification"]["data_dir"],
                                           data_params["specification"]["make_undirected"])
    if sort:
        idx = np.argsort(y)
        y = y[idx]
        X = X[idx, :]
        A = A[idx, :]
        A = A[:, idx]
    return X, A, y


def split_csbm(
        data_params: Dict[str, Any], y: Integer[ndarray, "n"]
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Split nodes into training, validation and test indices.

    Nodes are split in a class balanced fashion for training and validation
    set. The remaining nodes constitute the test set. 

    TODO: Implement a semi-supervised (compared to fully labeled) setting.

    Returns:
        A tuple (idx_trn, idx_unlabeled, idx_val, idx_test).
    """
    assert data_params["n_trn_unlabeled"] == 0, \
        "Only fully labeled setting implemented so far."
    rng = np.random.Generator(np.random.PCG64(data_params["seed"]))
    n_cls0 = sum(y == 0)
    n = len(y)
    idx_cls0 = rng.permutation(np.arange(n_cls0))
    idx_cls1 = rng.permutation(np.arange(n_cls0, n))
    n_cls = data_params["classes"]
    assert data_params["n_trn_labeled"] % n_cls == 0 \
           and data_params["n_val"] % n_cls == 0, \
           "Unable to create class balanced training and validation split."
    n_labeled = int(data_params["n_trn_labeled"] / n_cls)
    n_val = int(data_params["n_val"] / n_cls)
    idx_trn = np.concatenate((idx_cls0[:n_labeled], idx_cls1[:n_labeled]))
    start_test_id = n_labeled + n_val
    idx_val = np.concatenate((idx_cls0[n_labeled:start_test_id], 
                              idx_cls1[n_labeled:start_test_id]))
    idx_test = np.concatenate((idx_cls0[start_test_id:], 
                               idx_cls1[start_test_id:]))
    return idx_trn, np.array([]), idx_val, idx_test


def split_inductive(labels, n_per_class=20, fraction_test=0.1, seed=None, 
                    balance_test = True):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    fraction_test : float
        How much % of nodes are test nodes.
    balance_test: bool
        wether to balance the classes in the test set; if true, take 10% of all nodes as test set
    seed: int
        Seed

    Returns
    -------
    split_labeled: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test: array-like [n_per_class * nc]
        The indices of the test nodes
    split_unlabeled: array-like [num_nodes - 3*n_per_class * nc]
        The indices of the unlabeled nodes
    """
    if seed is not None:
        np.random.seed(seed)
    nc = labels.max() + 1
    if balance_test:
    # compute n_per_class
        bins = np.bincount(labels)
        n_test_per_class = np.ceil(fraction_test*bins)
    else: 
        n_test_per_class = np.ones(nc)*n_per_class

    split_labeled, split_val, split_test = [], [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_labeled.append(perm[:n_per_class])
        split_val.append(perm[n_per_class: 2 * n_per_class])
        split_test.append(perm[2*n_per_class: 2 * n_per_class + n_test_per_class[label].astype(int)])

    split_labeled = np.random.permutation(np.concatenate(split_labeled))
    split_val = np.random.permutation(np.concatenate(split_val))
    split_test = np.random.permutation(np.concatenate(split_test))
    

    assert split_labeled.shape[0] == split_val.shape[0] == n_per_class * nc

    split_unlabeled = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_labeled, split_val, split_test)))

    print(f'number of samples\n - labeled: {n_per_class * nc} \n - val: {n_per_class * nc} \n - test: {split_test.shape[0]} \n - unlabeled: {split_unlabeled.shape[0]}')

    return split_labeled, split_unlabeled, split_val, split_test


def split(
        data_params: Dict[str, Any], y: Integer[ndarray, "n"]
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Split nodes into training, validation and test indices.

    Returns:
        A tuple (idx_trn, idx_unlabeled, idx_val, idx_test).
    """
    if data_params["dataset"] == "csbm":
        idx_trn, idx_unlabeled, idx_val, idx_test = split_csbm(
            data_params["specification"], y
        )
    else:
        spec = data_params["specification"]
        seed = 0
        if "seed" in spec:
            seed = spec["seed"]
        idx_trn, idx_unlabeled, idx_val, idx_test = split_inductive(
            y, n_per_class=spec["n_per_class"], balance_test=spec["balance_test"], 
            seed=seed
        )

    return idx_trn, idx_unlabeled, idx_val, idx_test
