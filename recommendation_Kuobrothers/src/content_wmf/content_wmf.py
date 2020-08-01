#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Content WMF"""
import sys
import time

import numpy as np

from .batched_inv_joblib import recompute_factors_batched

def log_surplus_confidence_matrix(B, alpha, epsilon):
    S = B.copy()
    S.data = alpha * np.log(1 + S.data / epsilon)
    return S


def factorize(S,
              num_factors,
              U=None,
              V=None,
              beta=0.01,
              X=None,
              W=None,
              vad=None,
              num_iters=10,
              init_std=0.01,
              lambda_U_reg=1e-2,
              lambda_V_reg=100,
              lambda_W_reg=1e-2,
              computeW=False,
              batch_size=10000,
              dtype='float32',
              random_state=98765,
              verbose=False,
              n_jobs=1,
              *args,
              **kwargs):

    num_users, num_items = S.shape
    if X is not None:
        assert X.shape[0] == num_items

    if verbose:
        print("Precompute S^T and X^TX (if necessary)")
        start_time = time.time()

    ST = S.T.tocsr()
    if computeW:
        n_feats = X.shape[1]
        R = np.eye(n_feats)
        R[n_feats - 1, n_feats - 1] = 0
        XTXpR = X.T.dot(X) + lambda_W_reg * R

    if verbose:
        print("  took %.3f seconds" % (time.time() - start_time))
        start_time = time.time()

    if type(random_state) is int:
        np.random.seed(random_state)
    elif random_state is not None:
        np.random.setstate(random_state)
    if V is None:
        V = np.random.randn(num_items, num_factors).astype(dtype) * init_std
    if X is not None:
        W = np.random.randn(X.shape[1], num_factors).astype(dtype) * init_std
    else:
        W = None

    for i in range(num_iters):
        if verbose:
            print("Iteration %d:" % i)
            start_t = _write_and_time('\tUpdating item factors...')
        U = recompute_factors_batched(
            V, S, lambda_U_reg, beta=beta, batch_size=batch_size, dtype=dtype, n_jobs=n_jobs)
        if verbose:
            print('\r\tUpdating item factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _write_and_time('\tUpdating user factors...')
        V = recompute_factors_batched(
            U, ST, lambda_V_reg, W=W, X=X, beta=beta, batch_size=batch_size, dtype=dtype, computeW=computeW)
        if verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            if vad is not None:
                pred_ll = _pred_loglikeli(U, V, dtype, **vad)
                print("\tPred likeli: %.5f" % pred_ll)
            sys.stdout.flush()
        if computeW:
            if verbose:
                start_t = _write_and_time(
                    '\tUpdating projection matrix...')
            W = np.linalg.solve(XTXpR, X.T.dot(V))
            if verbose:
                print('\r\tUpdating projection matrix: time=%.2f'
                      % (time.time() - start_t))

    return U, V, W


def _pred_loglikeli(U, V, dtype, X_new=None, rows_new=None, cols_new=None):
    X_pred = _inner(U, V, rows_new, cols_new, dtype)
    pred_ll = np.mean((X_new - X_pred)**2)
    return pred_ll


def _write_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def _inner(U, V, rows, cols, dtype):
    n_ratings = rows.size
    n_components = U.shape[1]
    assert V.shape[1] == n_components
    data = np.empty(n_ratings, dtype=dtype)
    code = r"""
    for (int i = 0; i < n_ratings; i++) {
       data[i] = 0.0;
       for (int j = 0; j < n_components; j++) {
           data[i] += U[rows[i] * n_components + j] * V[cols[i] * n_components + j];
       }
    }
    """
    weave.inline(code, ['data', 'U', 'V', 'rows', 'cols', 'n_ratings',
                        'n_components'])
    return data

