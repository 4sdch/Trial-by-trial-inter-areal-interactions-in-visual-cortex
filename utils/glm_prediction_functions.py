
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import copy
from scipy import stats as st
from sklearn.linear_model import PoissonRegressor
# will's functions
def pearsonr2(a, b):
    """Calculate Pearson correlation coefficient.

    Parameters:
        a (numpy array): Input array a.
        b (numpy array): Input array b.

    Returns:
        float: Pearson correlation coefficient.
    """
    for x in (a, b):
        if x.std()==0:
            return np.nan
        mu = x.mean()
        if np.linalg.norm(x - mu) < 1e-13 * np.abs(mu):
            return np.nan
        if np.isnan(x).any():
            return np.nan
        if np.isinf(x).any():
            return np.nan
    return st.pearsonr(a, b)[0]

import numpy as np

def drop_constant_cols(X, tol=0.0):
    sd = np.nanstd(X, axis=0)
    keep = np.isfinite(sd) & (sd > tol)
    return X[:, keep], keep

def train_one_glm_split_full(X, y, train_idc, test_idc, count, n_splits=10,
                             frames_reduced=None, control_shuffle=False, make_nonnegative=True,
                             alpha=1e-4):
    if frames_reduced is not None:
        beginning_index = test_idc[0]
        if count == 0:
            train_idc_reduced = copy.deepcopy(train_idc[frames_reduced:])
        elif count == n_splits - 1:
            train_idc_reduced = copy.deepcopy(train_idc[:beginning_index - frames_reduced])
        else:
            train_idc_reduced = np.delete(train_idc, np.arange(beginning_index - frames_reduced, beginning_index + frames_reduced, 1))
    else:
        train_idc_reduced = train_idc.copy()

    test_idc_x = copy.deepcopy(test_idc)
    test_idc_y = copy.deepcopy(test_idc)
    
    train_idc_reduced_x = train_idc_reduced.copy()
    train_idc_reduced_y = train_idc_reduced.copy()

    if control_shuffle:
        np.random.shuffle(train_idc_reduced_x)
        np.random.shuffle(test_idc_x)

    Xtrain = X[train_idc_reduced_x]
    ytrain = y[train_idc_reduced_y]
    Xval = X[test_idc_x]
    yval = y[test_idc_y]
    
    if make_nonnegative:
        # per-neuron case handled outside; here y is 1D for one neuron
        eps = np.finfo(ytrain.dtype if np.issubdtype(ytrain.dtype, np.floating) else np.float64).tiny
        c = np.nanmin(ytrain)
        ytrain_pos = ytrain - c + eps
        yval_pos   = yval   - c + eps
    else:
        if np.nanmin(ytrain) < 0:
            raise ValueError("y contains negatives but make_nonnegative=False. Set make_nonnegative=True for Poisson GLM.")
        ytrain_pos = ytrain
        yval_pos   = yval
        c = 0
        eps = 0
    try:
        # no constant column needed; sklearn fits an intercept by default
        pr = PoissonRegressor(alpha=alpha, max_iter=1000, fit_intercept=True)
        pr.fit(Xtrain, ytrain_pos)
        ypred_pos = pr.predict(Xval)
        ypred     = ypred_pos + c - eps   # back to original scale for EV (optional but fine)

        coef = pr.coef_.copy()
        r    = pearsonr2(yval, ypred) # can only do one neuron at a time with this poisson regressor
        ev   = r**2
        
    except Exception:
        ypred = np.full_like(yval, fill_value=np.nan)
        coef = np.full(X.shape[1], fill_value=np.nan)
        ev = np.nan

    return ypred, test_idc, coef, ev

def get_glm_predictions_evars_parallel(layer_used, layer_to_predict, n_splits=10, frames_reduced=None,
                                       standarize_X=True, control_shuffle=False, mean_split_evars=True,
                                       save_weights=False, verbose=False, make_nonnegative=True, alpha=1e-4,
                                       use_only_1_split=False):
    X = layer_used
    y = layer_to_predict
    
    X, keep_X = drop_constant_cols(X)
    n_dropped = int((~keep_X).sum())
    if n_dropped:
        print(f"[dropped {n_dropped} constant predictors from X; kept {keep_X.sum()}.")

    ## change because of nans
    if standarize_X:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        bad = ~np.isfinite(sd) | (sd == 0)
        sd = np.where(bad, 1.0, sd)
        mu = np.where(~np.isfinite(mu), 0.0, mu)
        X = (X - mu) / (sd + 1e-12)
        X[~np.isfinite(X)] = 0.0


        

    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    kf = KFold(n_splits=n_splits, shuffle=False)
    n_neurons = y.shape[1]
    if verbose:
        print(f"Number of neurons: {n_neurons}, Number of splits: {n_splits}")
    y_preds = []
    f_indices = []
    split_evars = []
    all_coefs = np.zeros((n_splits, n_neurons, X.shape[1]))

    for count, (train_idx, test_idx) in enumerate(kf.split(X)):
        if use_only_1_split and count > 0:
            break
        if verbose:
            print(f"Processing split {count + 1}/{n_splits}...")
        results = Parallel(n_jobs=-1)(
            delayed(train_one_glm_split_full)(
                X, y[:, neuron], train_idx, test_idx, count,
                n_splits=n_splits, frames_reduced=frames_reduced, control_shuffle=control_shuffle,
                make_nonnegative=make_nonnegative,alpha=alpha
            ) for neuron in range(n_neurons)
        )

        split_pred = np.stack([res[0] for res in results], axis=1)
        y_preds.append(split_pred)
        f_indices.append(test_idx)
        split_evars.append([res[3] for res in results])
        if save_weights:
            for neuron in range(n_neurons):
                all_coefs[count, neuron, :] = results[neuron][2]

    y_preds = np.concatenate(y_preds, axis=0)
    f_indices = np.concatenate(f_indices, axis=0)
    sorted_preds = y_preds[np.argsort(f_indices)]
    if verbose:
        print(f"Shape of sorted predictions: {sorted_preds.shape}")
    corr = np.zeros(n_neurons)
    # corr = np.array([pearsonr2(y[:, i], sorted_preds[:, i]) for i in range(n_neurons)])
    print('y_preds shape:', y_preds.shape)
    print(f'y_shape_1_split', np.array(split_evars).shape)
    if not use_only_1_split:
        # EV per neuron, but skip targets with zero variance
        std_y = np.nanstd(y, axis=0)
        valid_targets = np.isfinite(std_y) & (std_y > 0)

        evars = np.full(y.shape[1], np.nan, dtype=float)  # preserve indexing!
        for i in np.flatnonzero(valid_targets):
            evars[i] = pearsonr2(y[:, i], sorted_preds[:, i]) ** 2

        # when you need a summary:
        mean_ev = np.nanmean(evars)
        if verbose:
            print(f"Mean EV (ignoring zero-var targets): {mean_ev:.4f}")

    if save_weights:
        return sorted_preds, evars, all_coefs, split_evars
    elif mean_split_evars:
        return sorted_preds, np.nanmean(np.array(split_evars), axis=0)
    else:
        return sorted_preds, evars
