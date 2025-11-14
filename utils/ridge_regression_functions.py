
import numpy as np
from scipy import stats
import copy


main_dir = ''
func_dir = main_dir + 'utils/'

import sys
sys.path.insert(0,func_dir)


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from scipy import stats as st

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
    return st.pearsonr(a, b)[0]

def worker_function(X, y, alpha, train_idc, test_idc, count, n_splits=10, frames_reduced=None, control_shuffle=False):
    """Worker function for parallel computation of Ridge regression.

    Parameters:
        X (numpy array): Input features.
        y (numpy array): Target variable.
        alpha (float): Regularization parameter.
        train_idc (numpy array): Training indices.
        test_idc (numpy array): Test indices.
        frames_reduced (int): Number of frames to reduce.
        control_shuffle (bool): Flag for shuffling indices.

    Returns:
        tuple: Tuple containing predicted values, test indices, coefficients, and split explained variances.
    """
    if frames_reduced is not None: # Check if frames_reduced is provided
        #Reduces the number of frames in the training set for time series data 
        # to avoid correlation spillover between the end of the training sample 
        # and the beginning of the validation sample
        beginning_index = test_idc[0]
        # print(f'len of training:{len(train_idc)}, len of testing:{len(test_idc)}')
        if count == 0:
            train_idc_reduced_x = copy.deepcopy(train_idc[frames_reduced:])
            train_idc_reduced_y = train_idc[frames_reduced:]
            # print(count, f'train_idc[{frames_reduced}:]')
        elif count == n_splits -1:
            train_idc_reduced_x = copy.deepcopy(train_idc[:beginning_index - frames_reduced])
            train_idc_reduced_y = train_idc[:beginning_index - frames_reduced]
            # print(count, f'train_idc[:{beginning_index - frames_reduced}]')
        else:
            train_idc_reduced_x = np.delete(train_idc, np.arange(beginning_index - frames_reduced, beginning_index + frames_reduced, 1))
            train_idc_reduced_y = np.delete(train_idc, np.arange(beginning_index - frames_reduced, beginning_index + frames_reduced, 1))
            # print(count, f'frames deleted: np.arange({beginning_index - frames_reduced}, {beginning_index + frames_reduced}, 1)')
    else:
        train_idc_reduced_x = train_idc.copy()
        train_idc_reduced_y = train_idc.copy()  
    
    test_idc_x = copy.deepcopy(test_idc)
    test_idc_y = copy.deepcopy(test_idc)
    
    if control_shuffle is True:
        np.random.shuffle(train_idc_reduced_x)
        np.random.shuffle(test_idc_x)

    # Subset data based on indices
    Xtrain = X[train_idc_reduced_x]
    ytrain = y[train_idc_reduced_y]
    Xval = X[test_idc_x]
    yval=y[test_idc_y]
    
    
    # Fit Ridge regression model
    model4fit = Ridge(alpha=alpha, max_iter=10000)
    model4fit.fit(Xtrain, ytrain)

    ypred=model4fit.predict(Xval)
    #reshape both yval and ypred if they are 1D
    if yval.ndim == 1:
        yval = yval[:, np.newaxis]
    if ypred.ndim == 1:
        ypred = ypred[:, np.newaxis]
    
    # Calculate Pearson correlation and squared explained variance
    corr = np.array([pearsonr2(*vs) for vs in zip(yval.T, ypred.T)])
    split_evars = np.square(corr / 1)    
    return ypred, test_idc, model4fit.coef_,split_evars


import numpy as np
from scipy.stats import pearsonr

def max_instant_corr(X, Y):
    """
    X: array (T, F) of predictors
    Y: array (T, M) of targets

    Returns:
      per_target: length-M array of |r| for each Y[:, j] vs. best single X[:, i]
      overall_max: maximum absolute r over all predictors and targets
    """
    T, F = X.shape
    _, M = Y.shape
    per_target = np.zeros(M)
    for j in range(M):
        # compute r of Y[:,j] with each X[:,i]
        rs = [pearsonr(X[:, i], Y[:, j])[0] for i in range(F)]
        per_target[j] = np.nanmax(np.abs(rs))
    overall_max = np.nanmax(per_target)
    return per_target, overall_max
# Parallel execution for getting predictions and explained variances
def get_predictions_evars_parallel(layer_used, layer_to_predict, alpha, n_splits=10, frames_reduced=None, 
                                verbose=None, save_weights=False, standarize_X=True, mean_subtract_y=False, 
                                control_shuffle=False, mean_split_evars=True):
    """Get predictions and explained variances in parallel using Ridge regression.

    Parameters:
        layer_used (numpy array): Features used for prediction.
        layer_to_predict (numpy array): Target variable to predict.
        alpha (float): Regularization parameter.
        n_splits (int): Number of splits for cross-validation.
        frames_reduced (int): Number of frames to reduce.
        verbose (int): Verbosity level.
        save_weights (bool): Flag to save coefficients.
        standarize_X (bool): Flag to standardize input features.
        mean_subtract_y (bool): Flag to subtract mean from target variable.
        control_shuffle (bool): Flag for shuffling indices.
        mean_split_evars (bool): Flag to compute mean split explained variances.

    Returns:
        tuple: Tuple containing predicted values and explained variances.
    """
    y = layer_to_predict
    if mean_subtract_y is True:
        y = y - np.nanmean(y, axis=0)
    if standarize_X is True:
        scaler = StandardScaler()
        X = scaler.fit_transform(layer_used)
    else:
        X=layer_used     
    if verbose == 1:
        print('dataset shape', X.shape, y.shape)
        
    # per_target_corrs, best_corr = max_instant_corr(X, y)
    # print("Best instantaneous corr per neuron:", per_target_corrs)
    # print("Overall best |r|:", best_corr)
    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=False)
    # If target data is 1D, add an axis
    if len(y.shape) == 1:
        # print('y is 1D, adding new axis')
        y = y[:, np.newaxis]
        # print('y shape after adding new axis:', y.shape)

    y_preds, f_indices, split_evars = [], [],[]
    all_coefs = np.zeros([n_splits, y.shape[1], X.shape[1]])
    
    # Parallel execution of worker function for each fold
    results = Parallel(n_jobs=-1)(delayed(worker_function)(X, y, alpha, train_idc, test_idc, count, n_splits, frames_reduced, control_shuffle=control_shuffle) for count, (train_idc, test_idc) in enumerate(kf.split(X)))
    for t, (ypred, test_idc, coef, split_evar) in enumerate(results):
        y_preds.append(ypred)
        f_indices.append(test_idc)
        split_evars.append(split_evar)
        if save_weights:
            all_coefs[t] = coef
    
    # Concatenate predictions and indices
    y_preds = np.concatenate(y_preds, axis=0)
    f_indices = np.concatenate(f_indices, axis=None)
    sorted_preds= y_preds[np.argsort(f_indices)]
    # add axis to sorted_preds if y is 1D
    if sorted_preds.ndim == 1:
        sorted_preds = sorted_preds[:, np.newaxis]
    corr = np.array([pearsonr2(*vs) for vs in zip(y.T, sorted_preds.T)])
    evars = np.square(corr / 1)
    
    if verbose == 1:
        print(f'mean_alpha: {np.nanmean(evars)}')

    if save_weights:
        return sorted_preds, evars, all_coefs, split_evars
    elif mean_split_evars is True:
        return sorted_preds, np.nanmean(np.array(split_evars), axis=0)
    else:
        return sorted_preds, evars

import numpy as np

from utils.glm_prediction_functions import get_glm_predictions_evars_parallel

def get_best_alpha_evars(layer_to_use,
                         layer_to_predict,
                         n_splits=10,
                         frames_reduced=5,
                         alphas=None,
                         silence=None,
                         standardize_X=True,
                         control_shuffle=False,
                         mean_split_evars=True,
                         prediction_type='ridge',
                         patience=2, 
                         min_delta=1e-4, 
                         do_fine=False):
    """
    Get the best alpha value based on explained variances, with an adaptive alpha grid
    that scales by the number of features and does a two-stage coarse→fine search.
    """

    if prediction_type == 'poisson_glm':
        pred_func = get_glm_predictions_evars_parallel
        if alphas is None:
            alphas = np.logspace(1, 3, 15) # (0, 2, 15)  originally
    else:
        pred_func = get_predictions_evars_parallel
        if alphas is None:
            # keep your old ridge convention if you like:
            alphas = np.logspace(1, 4, 15)
    # determine feature count
    # layer_to_use has shape (timepoints, features)
    n_features = layer_to_use.shape[1]
    print(f"Number of features: {n_features}")

    all_alpha_evars = []
    best_ev = -np.inf
    best_alpha = None
    downs_in_row = 0

    for k, alpha in enumerate(alphas):
        print(f"Testing alpha {k+1}/{len(alphas)}: {alpha:.2e}")
        if prediction_type=='poisson_glm':
            _, evars = pred_func(layer_to_use, layer_to_predict,
                                n_splits=n_splits,
                                alpha=alpha,
                                frames_reduced=frames_reduced,
                                standarize_X=standardize_X,
                                control_shuffle=control_shuffle,
                                mean_split_evars=mean_split_evars,
                                verbose=1,
                                use_only_1_split=True) # i have to loop through all neurons so ill only use 1 split
        elif prediction_type=='gamma_glm':
            _, evars = pred_func(layer_to_use, layer_to_predict,
                                n_splits=n_splits,
                                alpha=alpha,
                                frames_reduced=frames_reduced,
                                standarize_X=standardize_X,
                                control_shuffle=control_shuffle,
                                mean_split_evars=mean_split_evars,
                                use_only_1_split=True, # i have to loop through all neurons so ill only use 1 split
                                verbose=1)
        else:
            _, evars = pred_func(layer_to_use, layer_to_predict,
                                n_splits=n_splits,
                                alpha=alpha,
                                frames_reduced=frames_reduced,
                                standarize_X=standardize_X,
                                control_shuffle=control_shuffle,
                                mean_split_evars=mean_split_evars,
                                verbose=1)
        mean_ev = np.nanmean(evars)
        if not silence:
            print(f"α={alpha:.2e} → EV={mean_ev:.4f}")
        if np.isnan(mean_ev):
            print(f"Warning: EV is NaN for alpha={alpha:.2e}. Skipping this alpha.")
            print(evars)
            continue
        all_alpha_evars.append(mean_ev)

        # update best
        if mean_ev > best_ev + min_delta:
            best_ev = mean_ev
            best_alpha = alpha
            downs_in_row = 0
        else:
            # not improved enough
            downs_in_row += 1
            if downs_in_row >= patience:
                if not silence:
                    print(f"Early stop at α={alpha:.2e} after {patience} consecutive non-improvements.")
                break

    all_alpha_evars = np.array(all_alpha_evars)
    if all_alpha_evars.size == 0 or np.all(np.isnan(all_alpha_evars)):
        return alphas[0], all_alpha_evars  # or np.nan, all_alpha_evars

    if best_alpha is None:  # fallback
        best_idx = int(np.nanargmax(all_alpha_evars))
        best_alpha = alphas[best_idx]

    if not silence:
        print(f"** Best α={best_alpha:.2e}, EV≈{best_ev:.4f}")

    if do_fine and 0 < np.where(alphas == best_alpha)[0][0] < len(alphas)-1:
        i = int(np.where(alphas == best_alpha)[0][0])
        low, mid, high = alphas[i-1], alphas[i], alphas[i+1]
        alphas_fine = np.logspace(np.log10(low), np.log10(high), 9)
        ev_fine = []
        for a in alphas_fine:
            _, evars = pred_func(layer_to_use, layer_to_predict,
                                 n_splits=n_splits,
                                 alpha=a,
                                 frames_reduced=frames_reduced,
                                 standarize_X=standardize_X,
                                 control_shuffle=control_shuffle,
                                 mean_split_evars=mean_split_evars,
                                 verbose=0)
            ev_fine.append(np.nanmean(evars))
        j = int(np.nanargmax(ev_fine))
        best_alpha = alphas_fine[j]
        if not silence:
            print(f"** Fine best α={best_alpha:.2e}, EV≈{ev_fine[j]:.4f}")
    return best_alpha, all_alpha_evars




