import numpy as np
import scipy.sparse as sp


def sparse_flip_lr(sp_mat):
    """
    Takes a sparse matrix and flips it left-right (like numpy.fliplr())
    """
    sp_mat = sp_mat.tocsr()
    sp_mat.indices = -sp_mat.indices + sp_mat.shape[1] - 1
    return sp_mat


# def defunct_spatial_udlr_edges(spatial_shape):
#     if not len(spatial_shape) == 2:
#         raise ValueError(
#             f"spatial_shape must have two dimensions (i.e. both spatial dimension) but had shape: {spatial_shape}")
#     left_right_conns = sum(2 * sparse_flip_lr(
#         sp.eye(spatial_shape[0] * spatial_shape[1], spatial_shape[0] * spatial_shape[1],
#                spatial_shape[0] * spatial_shape[1] - row * spatial_shape[1] * 2)) for row in range(spatial_shape[0]))
#     left_right_conns += sp.eye(spatial_shape[0] * spatial_shape[1], spatial_shape[0] * spatial_shape[1], 1)
#     left_right_conns += sp.eye(spatial_shape[0] * spatial_shape[1], spatial_shape[0] * spatial_shape[1], -1)
#     left_right_conns[left_right_conns > 1] = 0  # removes the connections across spatial rows
#     up_down_conns = sp.eye(spatial_shape[0] * spatial_shape[1], spatial_shape[0] * spatial_shape[1], spatial_shape[1])
#     up_down_conns += sp.eye(spatial_shape[0] * spatial_shape[1], spatial_shape[0] * spatial_shape[1], -spatial_shape[1])
#     return left_right_conns + up_down_conns


def udlr(spatial_shape):
    """
    An adjacency matrix (in scipy.sparse.csr_matrix form) representing edges between pixels and their immediate
    neighbours above, below, left and right.
    """
    if not len(spatial_shape) == 2:
        raise ValueError(
            f"spatial_shape must have two dimensions (i.e. both spatial dimension) but had shape: {spatial_shape}")
    rows, cols = zip(*[(i, i + addition) for i in range(spatial_shape[0] * spatial_shape[1]) for part, addition in
                       zip([i >= (spatial_shape[0] - 1) * spatial_shape[1],
                            (i % spatial_shape[1] == spatial_shape[1] - 1)], [spatial_shape[1], 1]) if not part])
    upper = sp.csr_matrix(([1] * len(rows), (rows, cols)),
                          (spatial_shape[0] * spatial_shape[1], spatial_shape[0] * spatial_shape[1]), dtype=np.uint16)
    return upper + upper.T


def never_observed_check(adj, x, with_reversing_mask=True):
    if adj.shape[0] != x.shape[0]:
        raise ValueError("Shapes incompatible")
    degree = adj.sum(axis=1)
    if np.any(degree == 0):
        ever_seen = np.where(degree != 0)[0]
        if with_reversing_mask:
            return adj[ever_seen][:, ever_seen], x[ever_seen, ...], np.array(degree != 0).squeeze()
        else:
            return adj[ever_seen][:, ever_seen], x[ever_seen, ...]
    else:
        if with_reversing_mask:
            return adj, x, np.arange(x.shape[0])
        else:
            return adj, x
