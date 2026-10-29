# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)

    n_tr, n_va = int(round(n * train_frac)), int(round(n * val_frac))
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr + n_va]
    test_idx = perm[n_tr + n_va:]

    # Add intercept column (bias term)
    X_tr = np.c_[np.ones(len(train_idx)), X[train_idx]]
    X_va = np.c_[np.ones(len(val_idx)), X[val_idx]]
    X_te = np.c_[np.ones(len(test_idx)), X[test_idx]]

    # Fit least-squares regression on the training fold
    w = np.linalg.lstsq(X_tr, y[train_idx], rcond=None)[0]
    base = float(np.mean(y[train_idx]))

    # Compare linear model vs. training mean baseline on the validation fold
    if len(val_idx) and np.mean(np.abs(y[val_idx] - X_va @ w)) < np.mean(np.abs(y[val_idx] - base)):
        test_pred = X_te @ w
    else:
        test_pred = np.full(len(test_idx), base)

    return test_pred, train_idx, val_idx, test_idx