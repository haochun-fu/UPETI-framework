import numpy as np
from joblib import Parallel, delayed


def get_row(S, i):
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return S.data[lo:hi], S.indices[lo:hi]


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in range(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])
    return X_stack


def solve_batch(b, S, Y, WX, YTYpR, batch_size, beta, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)
    for ib, k in enumerate(range(lo, hi)):
        s_u, i_u = get_row(S, k)
        Y_u = Y[i_u]  # exploit sparsity
        A = (s_u + beta).dot(Y_u)

        if WX is not None:
            A += WX[:, k]

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B
    X_stack = solve_sequential(A_stack, B_stack)
    return X_stack


def recompute_factors_batched(Y, S, lambda_reg, W=None, X=None, beta=0.01,
                              dtype='float32', batch_size=10000,  n_jobs=4, **kwargs):
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors

    ids = np.where(np.sum(S, 0) > 0)[1]
    YTY = np.dot(Y[ids].T, Y[ids])
    # YTY = np.dot(Y.T, Y)

    YTYpR = beta * YTY + lambda_reg * np.eye(f)
    if X is not None:
        if kwargs['computeW']:
            WX = lambda_reg * (X.dot(W)).T
        else:
            WX = lambda_reg * X.T
    else:
        WX = None
    X_new = np.zeros((m, f), dtype=dtype)

    num_batches = int(np.ceil(m / float(batch_size)))
    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, WX, YTYpR, batch_size, beta, m, f, dtype)
                                  for b in range(num_batches))
    return np.concatenate(res, axis=0)
