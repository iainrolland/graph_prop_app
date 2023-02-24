import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import im_utils
import adj_utils


def graph_prop(adj, gappy_tens, omega, thresh=0.001, iterative=True):
    """
    Given an adjacency matrix, a matrix of entries and a mask (omega) denoting which entries have been observed, use
    graph-based diffusion to propagate and return a completed matrix/tensor.
    Can be solved either iteratively or analytically when `iterative' is True/False respectively.

    - If `gappy_tens' is one-dimensional, we flatten into a vector (i.e. we consider data as matrix with a single band)
    - If `gappy_tens' is two-dimensional, we flatten into a vector (i.e. we consider data as matrix with a single band)
    - If `gappy_tens' is three-dimensional, we flatten the first two dimensions and use the last as the bands

    adj: adjacency matrix (unweighted and undirected), scipy.sparse matrix
    gappy_tens: array, np.ndarray (two-dimensional or three-dimensional array of entries)
    omega: array, np.ndarray (1 if observed, 0 if missing)
    thresh: float (only used if `iterative' is True)
    iterative: bool, if True then we solve using diffusion equations if False we solve for steady state using analytical
    solution (not necessarily always faster)
    """
    if omega.shape != gappy_tens.shape[:2]:
        raise ValueError(
            f"Shape of `omega' {omega.shape} must match first two dimensions of `gappy_tens' {gappy_tens.shape}"
        )

    omega = (omega == 1).flatten()  # flattened bool array
    if gappy_tens.ndim == 1:
        observed = gappy_tens.copy().reshape(-1, 1)
    elif gappy_tens.ndim == 2:
        observed = gappy_tens.copy().reshape(-1, 1)
    elif gappy_tens.ndim == 3:
        observed = gappy_tens.copy().reshape(-1, gappy_tens.shape[2])

    # compute degree/laplacian matrix from adjacency
    degree = sp.diags(np.squeeze(np.asarray(adj.sum(axis=1))), 0, adj.shape, dtype=np.int16)
    laplacian = degree - adj

    # complete gappy tensor
    if iterative:
        completed = _iterative(observed,
                               diffuser=(-laplacian.astype(np.int32)) / np.max(degree.data),
                               omega=omega,
                               thresh=thresh)
    else:
        completed = _analytical(observed, laplacian, omega)
    return completed.reshape(gappy_tens.shape)  # return completed tensor in same shape as gappy tensor was provided


def _iterative(observed, diffuser, omega, thresh=0.001):
    completed = np.zeros_like(observed)
    completed[omega == 1] = observed[omega == 1]
    count = 0
    magn_break = None
    print("Solving iteratively...")
    while True:
        try:
            delta = diffuser.dot(completed)
            magn = np.mean(np.array(delta[~omega]) ** 2) ** 0.5
            if magn_break is None:
                magn_break = magn * thresh  # set the break threshold in first iteration
            completed[~omega] += delta[~omega]
            if (count + 1) % 100 == 0:
                print(f"Count: {count + 1}, magn: {magn}")
            if magn < magn_break:
                break
            if np.isnan(magn):
                raise RuntimeError("Magnitude of update is nan")
            count += 1
        except KeyboardInterrupt:
            print("Exiting before convergence...")
            exit(0)
    print(f"Solved! ({count + 1} iterations)")
    return completed


def _analytical(observed, laplacian, omega):
    completed = np.zeros_like(observed)
    completed[omega == 1] = observed[omega == 1]
    print("Solving analytically...")
    completed[omega == 0] = spsolve(laplacian[omega == 0][:, omega == 0],
                                    -laplacian[omega == 0][:, omega == 1].dot(observed[omega == 1]))
    print("Solved!")
    return completed


def demo():
    gt = np.random.normal(0, 1, (50, 50, 3))
    mask = im_utils.rolling_stripes_mask((50, 50, 3), gap_npixels=2, repeat_npixels=10, roll_npixels=4)
    print(gt.shape, mask.shape)
    adj = sp.eye(7500, 7500) + sp.block_diag([adj_utils.udlr((50, 50))] * 3)
    op = graph_prop(adj, gt.flatten(), mask.flatten())
    print(op.shape)


if __name__ == '__main__':
    demo()
