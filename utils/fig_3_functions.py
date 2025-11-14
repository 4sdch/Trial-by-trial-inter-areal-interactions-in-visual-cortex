## get seeds for permutations
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mouse_data_functions as cs
from joblib import Parallel, delayed
import time
import pandas as pd
from scipy.stats import gaussian_kde
from utils.stats_functions import perm_test_paired  
from ridge_regression_functions import get_best_alpha_evars, get_predictions_evars_parallel
from stats_functions import get_comparison_test_stars, get_comparison_test_stars
from macaque_data_functions import get_resps,get_get_condition_type
from sklearn.linear_model import LinearRegression, Ridge

num_seeds = 10
random.seed(17)
seeds = [random.randint(1, 10000) for _ in range(num_seeds)]
# create another list of seeds
random.seed(18)
seeds2 = [random.randint(1, 10000) for _ in range(num_seeds)]

def get_simil_reli_indices(reli1, reli1_indices, reli2, reli2_indices, seed=None, verbose=False):
    """This function subsamples the indices of the first group (`reli1_indices`) 
    based on the reliability values of two groups (`reli1` and `reli2`) to ensure 
    similar reliability distributions between the two groups. The subsampling is 
    performed by selecting indices from `reli1_indices` that correspond to reliability 
    values in `reli1` that are close to the reliability values in `reli2`.

    Args:
        reli1 (numpy.ndarray): Array containing reliability values for the first group.
        reli1_indices (numpy.ndarray): Indices corresponding to the first group.
        reli2 (numpy.ndarray): Array containing reliability values for the second group.
        reli2_indices (numpy.ndarray): Indices corresponding to the second group.
        seed (int): Seed value for reproducible random sampling.

    Returns:
        numpy.ndarray: Subsampled indices from the first group.
        numpy.ndarray: Indices from the second group (unchanged).

    Raises:
        None

    """

    new_array1_indices = []
    # Extract reliability values for the first and second groups
    array1= reli1[reli1_indices]
    array2=reli2[reli2_indices]
    np.random.seed(seed)
    # Iterate over each reliability value in the second group
    new_array2_indices = []
    for r, reli_val2 in enumerate(array2):
        array1_vals = []
        # Define a tolerance for comparing reliability values
        tolerance = 0.001
        count=0
        
        # Iterate over each reliability value in the first group
        for a1, reli_val1 in enumerate(array1):
            # Check if the reliability value in the first group is close to the one in the second group
            if np.isclose(reli_val2, reli_val1, atol=tolerance) and a1 not in new_array1_indices:
                # Increment the count and add the index to the list if it meets the condition
                count =+1
                array1_vals.append(a1)  
        # If no similar reliability values are found, increase the tolerance and try again
        while count==0:
            if verbose is True:
                print(f'{tolerance} didnt work')
            tolerance *= 2
            for a1, reli_val1 in enumerate(array1):
                if np.isclose(reli_val2, reli_val1, atol=tolerance) and a1 not in new_array1_indices:
                    count =+1
                    array1_vals.append(a1)
            # if tolerance is greater than 0.005, break the loop
            if tolerance > 0.01:
                print(f'tolerance {tolerance} is too high, breaking the loop')
                break
                
        # Randomly select an index from the list of similar reliability values and add it to the new indices list
        if len(array1_vals) > 0:
            new_array1_indices.append(np.random.choice(array1_vals))
            new_array2_indices.append(r)
        else:
            print(f'no similar reliability values found for {reli_val2}, omitting reli2 index {r}')
    return reli1_indices[new_array1_indices], reli2_indices[new_array2_indices]


def store_L23_indices(mouse_stats_, condition_types = ['ori32','natimg32'], nonvisual_neurons=False, predictor_min=50):
    """Store indices for creating subpopulations in layer L23 based on reliability.

    This function iterates over each condition type and mouse in the provided `mouse_stats_`.
    For each mouse and condition type, it retrieves reliability and SNR values for layers L23 and L4.
    Then, it filters the indices based on a reliability threshold of 0.8 and an SNR threshold of 2.
    After filtering, it generates similar subpopulations in L23 and L4 using the `get_simil_reli_indices` function.
    Finally, it stores the generated L23 indices in the `mouse_stats_` dictionary.

    Args:
        mouse_stats_ (dict): Dictionary containing mouse statistics data.
        condition_types (list, optional): List of condition types to consider. Defaults to ['ori32', 'natim32'].
    """
    area='L23'
    area2='L4'
    nonvis_con = ''
    if nonvisual_neurons is True:
        nonvis_con = '_nonvisual'   
    for condition_type in condition_types:
        for mouse in mouse_stats_[condition_type]:
            reli = mouse_stats_[condition_type][mouse][area]['split_half_r']
            snr = mouse_stats_[condition_type][mouse][area]['SNR_meanspont']
            reli2 = mouse_stats_[condition_type][mouse][area2]['split_half_r']
            snr2 = mouse_stats_[condition_type][mouse][area2]['SNR_meanspont']
            if nonvisual_neurons is True:
                L23_filtered_indices = np.argwhere((reli < 0.8)&(snr<2))[:,0]
                L4_filtered_indices = np.argwhere((reli2 < 0.8)&(snr2<2))[:,0]
            else:
                L23_filtered_indices = np.argwhere((reli > 0.8)&(snr>=2))[:,0]
                L4_filtered_indices = np.argwhere((reli2 > 0.8)&(snr2>=2))[:,0]
            n_neurons = len(L4_filtered_indices)
            
            big_chosen_indices_ = []
            
            # for s, seed in enumerate(seeds+seeds2):
            for s, seed in enumerate(seeds):
                big_chosen_indices_seed, new_small_filtered_indices= get_simil_reli_indices(reli, L23_filtered_indices, reli2, L4_filtered_indices, seed)
                if len(big_chosen_indices_seed) < predictor_min or len(new_small_filtered_indices) < predictor_min:
                    print(f'not enough indices for {condition_type} {mouse}. Found {len(big_chosen_indices_seed)} big and {len(new_small_filtered_indices)} small indices')
                    break
                big_chosen_indices_.append(big_chosen_indices_seed)
            
            
            big_chosen_indices = []
            # find min len of big_chosen_indices_seed in list
            unique_lengths = list(set([len(el) for el in big_chosen_indices_]))
                        
            if unique_lengths[0] < predictor_min or len(new_small_filtered_indices) < predictor_min:
                continue
            
            if len(unique_lengths)>1:
                # subsample so that length of each array in list is the size of the unique_length
                min_length = min(unique_lengths)
                for array in big_chosen_indices_:
                    if len(array) > min_length:
                        array = array[:min_length]
                    big_chosen_indices.append(array)
            else:
                big_chosen_indices = big_chosen_indices_

            mouse_stats_[condition_type][mouse][area][f'L23_chosen_indices{nonvis_con}']=big_chosen_indices
            


def extract_mouse_name(input_string):
    index_of_MP = input_string.find('MP')
    return input_string[index_of_MP:index_of_MP + 5] if index_of_MP != -1 and index_of_MP + 5 <= len(input_string) else None

def store_mouse_directionality_alphas(main_dir, mouse_stats, activity_type='resp',n_splits=5, 
                                    frames_to_reduce=5, sample_size=500, verbose=False, nonvisual_neurons=False):
    alpha_unique_options = [1e1,5e1,1e2,5e2,1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7]
    area = 'L23'
    area2='L4'
    dataset_types=['ori32','natimg32']
    
    spont_con = ''
    if activity_type =='spont':
        spont_con = '_spont'
    
    nonvis_con = ''
    if nonvisual_neurons is True:
        nonvis_con = '_nonvisual'
    #get alpha per mouse 
    if f'mouse_directionality_alphas{nonvis_con}' not in list(mouse_stats.keys()):
        mouse_stats[f'mouse_directionality_alphas{nonvis_con}']={sample_size:{}}
        mouse_alphas = {sample_size:{}}
    else:
        mouse_alphas = mouse_stats[f'mouse_directionality_alphas{nonvis_con}']
        if sample_size not in list(mouse_alphas.keys()):
            mouse_alphas[sample_size] = {}
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type) #retrieves neural activity of a certain dataset type stored in data
        for mouse in mouse_stats[dataset_type + spont_con]:
            if extract_mouse_name(mouse) in list(mouse_alphas[sample_size].keys()):
                mouse_stats[dataset_type + spont_con][mouse][area]['alpha' + '_' + f'{sample_size}_sample_size']=mouse_alphas[sample_size][(extract_mouse_name(mouse))][area]
                mouse_stats[dataset_type + spont_con][mouse][area2]['alpha' + '_' + f'{sample_size}_sample_size']=mouse_alphas[sample_size][(extract_mouse_name(mouse))][area2]
                
                if verbose:
                    print(f'alpha already stored using for {extract_mouse_name(mouse)} in dataset_type: {mouse_alphas[sample_size][extract_mouse_name(mouse)]["dataset_type_used"]}')
            else:
                reli2 = mouse_stats[dataset_type][mouse][area2]['split_half_r']
                snr2 = mouse_stats[dataset_type][mouse][area2]['SNR_meanspont']
                
                L4_filtered_indices = np.argwhere((reli2 > 0.8)&(snr2>=2))[:,0]
                if nonvisual_neurons is True:
                    L4_filtered_indices = np.argwhere((reli2 < 0.8)&(snr2<2))[:,0]
                all_perm_indices_L23 = mouse_stats[dataset_type][mouse][area][f'L23_chosen_indices{nonvis_con}']
                
                resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse)
                if resp_L1.shape[0]<1000:
                    # there are some gray screen activity datasets that are too small to fit
                    continue      
                resp_L23 = resp_L23[:sample_size]
                resp_L4 = resp_L4[:sample_size]
                alpha, evars = get_best_alpha_evars(resp_L4[:,L4_filtered_indices], resp_L23[:,all_perm_indices_L23[0]], n_splits=n_splits, 
                                                    frames_reduced=frames_to_reduce, 
                                                    alphas=alpha_unique_options)
                alpha2, evars2 = get_best_alpha_evars(resp_L23[:,all_perm_indices_L23[0]], resp_L4[:,L4_filtered_indices], n_splits=n_splits, 
                                                    frames_reduced=frames_to_reduce, 
                                                    alphas=alpha_unique_options)
                mouse_alphas[sample_size][(extract_mouse_name(mouse))]={area:alpha, area2:alpha2, 'dataset_type_used':dataset_type + spont_con}
                if verbose:
                    print(f'directionality alpha for {mouse} {activity_type} calculated and stored. Will be used in other datasets of the same mouse')
    mouse_stats[f'mouse_directionality_alphas{nonvis_con}'] = mouse_alphas


def get_directionality_max_corr_vals(input_resp, pred_resp):
    connx_matrix = np.corrcoef(pred_resp.T, input_resp.T)
    pred_input_connx = connx_matrix[:pred_resp.shape[1], pred_resp.shape[1]:]
    pred_maxcorrval = np.nanmax(np.abs(pred_input_connx), axis=1)
    input_maxcorrval = np.nanmax(np.abs(pred_input_connx), axis=0)
    return pred_maxcorrval, input_maxcorrval



def get_directionality_max_corr_vals_mice(main_dir, mouse_stats, activity_type='resp',
                    sample_size=500, nonvisual_neurons=False):
    start_time = time.time()
    area = 'L23'
    area2='L4'
    dataset_types=['ori32','natimg32']
    
    spont_con = ''
    if activity_type =='spont':
        spont_con = '_spont'
    nonvis_con = ''
    if nonvisual_neurons is True:
        nonvis_con = '_nonvisual'
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type) #retrieves neural activity stored in data
        mouse_names = mt.filenames
        for mouse in mouse_names:
            reli2 = mouse_stats[dataset_type][mouse][area2]['split_half_r']
            snr2 = mouse_stats[dataset_type][mouse][area2]['SNR_meanspont']
            
            L4_filtered_indices = np.argwhere((reli2 > 0.8)&(snr2>=2))[:,0]
            all_perm_indices_L23 = mouse_stats[dataset_type][mouse][area][f'L23_chosen_indices{nonvis_con}']
            if nonvisual_neurons is True: 
                L4_filtered_indices = np.argwhere((reli2 < 0.8)&(snr2<2))[:,0]
            resp_L1, resp_L23, _, _, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse)
            if resp_L1.shape[0]<1000:
                continue
            if mouse_stats['mouse_directionality_alphas'+nonvis_con][sample_size][(extract_mouse_name(mouse))]['dataset_type_used']==dataset_type + spont_con:
                resp_L23=resp_L23[sample_size:]
                resp_L4=resp_L4[sample_size:]
            results = Parallel(n_jobs=-1)(delayed(get_directionality_max_corr_vals)(resp_L4[:,L4_filtered_indices], resp_L23[:,all_perm_indices_L23[s]]) for s in range(len(seeds)))
            
            mouse_stats[dataset_type + spont_con][mouse][area]['directionality_maxcorrvals'+nonvis_con]=np.array([e for e,_ in results])
            mouse_stats[dataset_type + spont_con][mouse][area2]['directionality_maxcorrvals'+nonvis_con]=np.array([e for _,e in results])

def get_directionality_evars_mice(main_dir, mouse_stats, activity_type='resp',n_splits=10, frames_to_reduce=5,
                    control_shuffle=False, sample_size=500, nonvisual_neurons=False):
    """Compute directionality-related metrics for mice neural activity.

    This function calculates directionality-related metrics for mouse neural activity.
    It iterates over each mouse and dataset type, retrieves neural activity from layers L23 and L4,
    filters the indices based on reliability and SNR thresholds for layer L4, and computes metrics
    using ridge regression for directionality estimation between layers L23 and L4.

    Args:
        main_dir (str): Directory containing data.
        mouse_stats (dict): Dictionary containing mouse statistics data.
        activity_type (str, optional): Type of neural activity to consider. Defaults to 'resp'.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        frames_to_reduce (int, optional): Number of frames to reduce during validation. Defaults to 5.
        control_shuffle (bool, optiona): Whether to control shuffle during computation. Defaults to False.

    Returns:
        None
    """
    start_time = time.time()
    area = 'L23'
    area2='L4'
    dataset_types=['ori32','natimg32']
    
    control_con = ''
    if control_shuffle is True:
        control_con = '_null'
    
    spont_con = ''
    if activity_type =='spont':
        spont_con = '_spont'
    nonvis_con = ''
    if nonvisual_neurons is True:
        nonvis_con = '_nonvisual'
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type) #retrieves neural activity stored in data
        mouse_names = mt.filenames
        for mouse in mouse_names:
            reli2 = mouse_stats[dataset_type][mouse][area2]['split_half_r']
            snr2 = mouse_stats[dataset_type][mouse][area2]['SNR_meanspont']
                
            L4_filtered_indices = np.argwhere((reli2 > 0.8)&(snr2>=2))[:,0]
            all_perm_indices_L23 = mouse_stats[dataset_type][mouse][area][f'L23_chosen_indices{nonvis_con}']
            if nonvisual_neurons is True:
                L4_filtered_indices = np.argwhere((reli2 < 0.8)&(snr2<2))[:,0]
            
            alpha = mouse_stats[f'mouse_directionality_alphas{nonvis_con}'][sample_size][(extract_mouse_name(mouse))][area]
            alpha2 = mouse_stats[f'mouse_directionality_alphas{nonvis_con}'][sample_size][(extract_mouse_name(mouse))][area2]
            
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse)
            if resp_L1.shape[0]<1000:
                continue
            if mouse_stats[f'mouse_directionality_alphas{nonvis_con}'][sample_size][(extract_mouse_name(mouse))]['dataset_type_used']==dataset_type + spont_con:
                resp_L23=resp_L23[sample_size:]
                resp_L4=resp_L4[sample_size:]
            results = Parallel(n_jobs=-1)(delayed(get_predictions_evars_parallel)(resp_L4[:,L4_filtered_indices], resp_L23[:,all_perm_indices_L23[s]], 
                                                    n_splits=n_splits, frames_reduced = frames_to_reduce, alpha=alpha, control_shuffle=control_shuffle) for s in range(len(seeds)))
        
            results2 = Parallel(n_jobs=-1)(delayed(get_predictions_evars_parallel)(resp_L23[:,all_perm_indices_L23[s]], resp_L4[:,L4_filtered_indices], 
                                                        n_splits=n_splits, frames_reduced = frames_to_reduce, alpha=alpha2, control_shuffle=control_shuffle) for s in range(len(seeds)))
            
            mouse_stats[dataset_type + spont_con][mouse][area][f'directionality_evars{nonvis_con}' + control_con]=np.array([e for _,e in results])
            mouse_stats[dataset_type + spont_con][mouse][area2][f'directionality_evars{nonvis_con}' + control_con]=np.array([e for _,e in results2])
            
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'Took {elapsed_time:.4f} minutes to complete')  


all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200, 'RF_large_spont':200}


def get_reli_condition(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string

def store_V1_indices(monkey_stats, condition_types = ['SNR','RF_thin','RF_large'], verbose=False,
                     reli_threshold=0.8, snr_threshold=2, n_of_permutations=10, predictor_min = 7, return_conditions_not_used=False):

    """Store V1 indices in monkey statistics.

    This function computes and stores V1 indices in the monkey statistics data. It iterates over each condition type
    and date, retrieves reliability values for both V4 and V1 areas, filters the indices based on reliability
    thresholds for V4, and computes similar reliability indices for V1 using the `get_simil_reli_indices` function.
    The resulting indices are stored in the monkey statistics data under the 'V1_chosen_indices' key.

    Args:
        monkey_stats (dict): Dictionary containing monkey statistics data.
        condition_types (list, optional): List of condition types. Defaults to ['SNR', 'RF_thin', 'RF_large'].

    Returns:
        None
    """
    area='V4'
    area2='V1'
    sessions_conditions_not_used = []
    for condition_type in condition_types:
        for date in monkey_stats[condition_type]:
            if date in ['140819','150819','160819']:
                continue
            print(
                f'Computing V1 indices for {condition_type} {date}')
            reli = monkey_stats[condition_type][date][area]['split_half_r']
            reli2 = monkey_stats[condition_type][date][area2]['split_half_r']
            snr = monkey_stats[condition_type][date][area]['SNR_meanspont']
            snr2 = monkey_stats[condition_type][date][area2]['SNR_meanspont']
            
            V4_filtered_indices = np.argwhere((reli >= reli_threshold)&(snr >= snr_threshold))[:,0]
            V1_filtered_indices = np.argwhere((reli2 >= reli_threshold)&(snr2 >= snr_threshold))[:,0]
            
            if len(V1_filtered_indices) < predictor_min or len(V4_filtered_indices) < predictor_min:
                print(f'not enough V1 or V4 indices for {condition_type} {date}. Found {len(V1_filtered_indices)} V1 and {len(V4_filtered_indices)} V4 indices')
                sessions_conditions_not_used.append((condition_type, date))
                continue
            
            if len(V4_filtered_indices)>len(V1_filtered_indices):
                big_filtered_indices = V4_filtered_indices
                big_relis = reli
                big_area = area
                small_filtered_indices = V1_filtered_indices
                small_relis = reli2
                small_area = area2
            else:
                big_filtered_indices = V1_filtered_indices
                big_relis = reli2
                big_area = area2
                small_filtered_indices = V4_filtered_indices
                small_relis = reli
                small_area = area
            big_chosen_indices_ = []
            
            for s, seed in enumerate(seeds):
                big_chosen_indices_seed, new_small_filtered_indices= get_simil_reli_indices(big_relis, big_filtered_indices, small_relis, small_filtered_indices, seed, verbose=verbose)
                if len(big_chosen_indices_seed) < predictor_min or len(new_small_filtered_indices) < predictor_min:
                    print(f'not enough indices for {condition_type} {date}. Found {len(big_chosen_indices_seed)} big and {len(new_small_filtered_indices)} small indices')
                    break
                big_chosen_indices_.append(big_chosen_indices_seed)
            
            
            big_chosen_indices = []
            # find min len of big_chosen_indices_seed in list
            unique_lengths = list(set([len(el) for el in big_chosen_indices_]))

            if len(new_small_filtered_indices) < predictor_min:
                print(f'not enough small indices for {condition_type} {date}. Found only {len(new_small_filtered_indices)} small indices')
                sessions_conditions_not_used.append((condition_type, date))
                continue
            elif unique_lengths[0] < predictor_min:
                continue
            
            if len(unique_lengths)>1:
                # subsample so that length of each array in list is the size of the unique_length
                min_length = min(unique_lengths)
                for array in big_chosen_indices_:
                    if len(array) > min_length:
                        array = array[:min_length]
                    big_chosen_indices.append(array)
            else:
                big_chosen_indices = big_chosen_indices_
            monkey_stats[condition_type][date][big_area]['big_chosen_indices']=big_chosen_indices
            monkey_stats[condition_type][date][small_area]['small_chosen_indices']=new_small_filtered_indices
    if return_conditions_not_used is True:
        return sessions_conditions_not_used

def store_macaque_directionality_alphas(monkey_stats,n_splits=5, w_size=25, 
                        sample_size=500, verbose=False, condition_type_used='RS', date_used = '090817', monkey='L', subsampled_indices = False):
    if subsampled_indices is True:
        assert monkey_stats[condition_type_used][date_used]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
        assert monkey_stats[condition_type_used][date_used]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
        subsampled_indices_dict = {'V1':monkey_stats[condition_type_used][date_used]['V1'][f'monkey_{monkey}_subsample_indices'],
                                    'V4':monkey_stats[condition_type_used][date_used]['V4'][f'monkey_{monkey}_subsample_indices']}
    alpha_unique_options = None
    area='V4'
    area2='V1'
    
    #get alpha per mouse 
    if 'monkey_directionality_alphas' not in list(monkey_stats.keys()):
        monkey_stats['monkey_directionality_alphas']={sample_size:{}}
        monkey_alphas = {sample_size:{}}
    else:
        monkey_alphas = monkey_stats['monkey_directionality_alphas']
        if sample_size not in list(monkey_alphas.keys()):
            monkey_alphas[sample_size] = {}
    
    if 'V4' in list(monkey_alphas[sample_size].keys()):
        if verbose:
            print('alpha already stored')
    else:
        if 'big_chosen_indices' in list(monkey_stats[get_reli_condition(condition_type_used)][date_used][area].keys()):
            V4_filtered_indices = monkey_stats[get_reli_condition(condition_type_used)][date_used][area]['big_chosen_indices'][0]
            V1_filtered_indices = monkey_stats[get_reli_condition(condition_type_used)][date_used][area2]['small_chosen_indices']
        elif 'small_chosen_indices' in list(monkey_stats[get_reli_condition(condition_type_used)][date_used][area].keys()):
            V4_filtered_indices = monkey_stats[get_reli_condition(condition_type_used)][date_used][area]['small_chosen_indices']
            V1_filtered_indices = monkey_stats[get_reli_condition(condition_type_used)][date_used][area2]['big_chosen_indices'][0]
        else:
            raise RuntimeError(f'no V4 or V1 indices found for {condition_type_used} {date_used} {monkey}. Use another condition type or date')

        if len(V1_filtered_indices)==0 or len(V4_filtered_indices)==0:
            raise RuntimeError(f"no electrodes to calculate alpha. choose another date")
        
        get_condition_type = get_get_condition_type(condition_type_used)
        resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date_used, 
                                        w_size=w_size,stim_off=all_ini_stim_offs[condition_type_used], monkey=monkey)
        
        if subsampled_indices is True and subsampled_indices_dict is not None: # we must first subsample before indexing the v4 and v4 indices
            resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
            resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
            
        resp_V4=resp_V4[:sample_size]
        resp_V1=resp_V1[:sample_size]
        
        alpha, evars = get_best_alpha_evars(resp_V1[:,V1_filtered_indices], resp_V4[:,V4_filtered_indices], n_splits=n_splits, alphas=alpha_unique_options,
                                                frames_reduced=all_frames_reduced[condition_type_used])
        alpha2, evars2 = get_best_alpha_evars(resp_V4[:,V4_filtered_indices], resp_V1[:,V1_filtered_indices], n_splits=n_splits, alphas=alpha_unique_options,
                                                frames_reduced=all_frames_reduced[condition_type_used])
        
        monkey_alphas[sample_size]= {area:alpha, area2:alpha2, 'condition_type_used':condition_type_used, 'date_used':date_used}
        if verbose:
            print(f'alpha for macaque calculated and stored. Will be used in other datasets of the same monkey')
    monkey_stats['monkey_directionality_alphas'] = monkey_alphas


def get_directionality_maxcorrvals_monkey(monkey_stats, frame_size=500, w_size=25, monkey='L', dataset_types=['SNR','RF_thin','RF_large'], subsampled_indices=False):
    """Compute directionality-related metrics for monkey neural activity.

    This function calculates directionality-related metrics for monkey neural activity.
    It iterates over each condition type and date, retrieves neural activity from layers V1 and V4,
    filters the indices based on reliability thresholds for layer V4, and computes metrics
    using ridge regression for directionality estimation between layers V1 and V4.

    Args:
        monkey_stats (dict): Dictionary containing monkey statistics data.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        w_size (int, optional): Window size. Defaults to 25.
        control_shuffle (bool, optional): Whether to control shuffle during computation. Defaults to False.

    Returns:
        None
    """
    start_time = time.time()
    area='V4'
    area2='V1'
    
    monkey_alphas=monkey_stats['monkey_directionality_alphas'][frame_size]

    for condition_type in dataset_types:
        if 'monkey' in condition_type:
            continue
        for date in monkey_stats[condition_type]:
            if date in ['140819','150819','160819']:
                continue
            if subsampled_indices is True:
                assert monkey_stats[condition_type][date]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                assert monkey_stats[condition_type][date]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                subsampled_indices_dict = {'V1':monkey_stats[condition_type][date]['V1'][f'monkey_{monkey}_subsample_indices'],
                                            'V4':monkey_stats[condition_type][date]['V4'][f'monkey_{monkey}_subsample_indices']}
        
            resp_V4, resp_V1 = get_resps(condition_type = get_get_condition_type(condition_type), date=date, w_size=w_size, stim_on=0, 
                                        stim_off=all_ini_stim_offs[condition_type], monkey=monkey)
            if subsampled_indices is True and subsampled_indices_dict is not None: # we must first subsample before indexing the v4 and v4 indices
                resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
                
            if condition_type ==monkey_alphas['condition_type_used'] and date==monkey_alphas['date_used']:
                resp_V4=resp_V4[frame_size:]
                resp_V1=resp_V1[frame_size:]
                
            if 'big_chosen_indices' in list(monkey_stats[get_reli_condition(condition_type)][date][area].keys()):
                print('there are more V4 sites than V1 sites')
                V4_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area]['big_chosen_indices']
                V1_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area2]['small_chosen_indices']
                results = Parallel(n_jobs=-1)(delayed(get_directionality_max_corr_vals)(resp_V1[:,V1_filtered_indices], 
                                                                        resp_V4[:, V4_filtered_indices[s]], 
                                                                        ) for s in range(len(V4_filtered_indices)))
            elif 'small_chosen_indices' in list(monkey_stats[get_reli_condition(condition_type)][date][area].keys()):
                print(f'there are more V1 sites than V4 sites for {monkey} {condition_type} {date}')
                V4_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area]['small_chosen_indices']
                V1_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area2]['big_chosen_indices']
                results = Parallel(n_jobs=-1)(delayed(get_directionality_max_corr_vals)(resp_V1[:,V1_filtered_indices[s]], 
                                                                        resp_V4[:, V4_filtered_indices], 
                                                                        ) for s in range(len(V1_filtered_indices)))
            else:
                print(f'no V4 or V1 indices found for {condition_type} {date} {monkey}. skipping this condition type and date')
                continue
                # print(f'V1 filtered indices: {len(V1_filtered_indices)}, V4 filtered indices: {list(len(V4_filtered_indices[s]) for s in range(len(V4_filtered_indices)))}')
            
            for _,e in results:
                print(f'found {len(e)} maxcorrvals for {condition_type} {date} {area}')
            monkey_stats[condition_type][date][area]['directionality_maxcorrvals']=np.array([e for e,_ in results])
            monkey_stats[condition_type][date][area2]['directionality_maxcorrvals']=np.array([e for _,e in results])
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'Took {elapsed_time:.4f} minutes to complete')   

import math
def frames_reduced_for(condition: str, bin_ms: int) -> int:
    base = all_frames_reduced[condition]
    gap_ms = base * 25  # because dict is defined at 25 ms/bin
    return max(1, math.ceil(gap_ms / bin_ms))


def get_directionality_evars_monkey(monkey_stats, n_splits=10, w_size=25, control_shuffle=False,
                                    frame_size=500, monkey='L', dataset_types = ['SNR','RF_thin','RF_large'], 
                                    subsampled_indices=False):
    """Compute directionality-related metrics for monkey neural activity.

    This function calculates directionality-related metrics for monkey neural activity.
    It iterates over each condition type and date, retrieves neural activity from layers V1 and V4,
    filters the indices based on reliability thresholds for layer V4, and computes metrics
    using ridge regression for directionality estimation between layers V1 and V4.

    Args:
        monkey_stats (dict): Dictionary containing monkey statistics data.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        w_size (int, optional): Window size. Defaults to 25.
        control_shuffle (bool, optional): Whether to control shuffle during computation. Defaults to False.

    Returns:
        None
    """
    start_time = time.time()
    area='V4'
    area2='V1'
    
    monkey_alphas=monkey_stats['monkey_directionality_alphas'][frame_size]
    alpha = monkey_alphas[area]
    alpha2=monkey_alphas[area2]    
    
    
    if control_shuffle is True:
        control_con = '_null'
    else:
        control_con = ''
    w_size_con = f'_{w_size}' if w_size != 25 else ''
    
    for condition_type in dataset_types:
        if 'monkey' in condition_type:
            continue
        for date in monkey_stats[condition_type]:
            if date in ['140819','150819','160819']:
                continue
            if subsampled_indices is True:
                assert monkey_stats[condition_type][date]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                assert monkey_stats[condition_type][date]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                subsampled_indices_dict = {'V1':monkey_stats[condition_type][date]['V1'][f'monkey_{monkey}_subsample_indices'],
                                            'V4':monkey_stats[condition_type][date]['V4'][f'monkey_{monkey}_subsample_indices']}
            resp_V4, resp_V1 = get_resps(condition_type = get_get_condition_type(condition_type), date=date, w_size=w_size, stim_on=0, 
                                        stim_off=all_ini_stim_offs[condition_type], monkey=monkey)
            if subsampled_indices is True and subsampled_indices_dict is not None: # we must first subsample before indexing the v4 and v4 indices
                print('subsampling')
                resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
            if condition_type ==monkey_alphas['condition_type_used'] and date==monkey_alphas['date_used']:
                resp_V4=resp_V4[int(frame_size*25/w_size):]
                resp_V1=resp_V1[int(frame_size*25/w_size):]
            
            if 'big_chosen_indices' in list(monkey_stats[get_reli_condition(condition_type)][date][area].keys()):
                V4_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area]['big_chosen_indices']
                V1_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area2]['small_chosen_indices']
                if len(V1_filtered_indices)==0 or len(V4_filtered_indices)==0:
                    print(f'not enough V1 or V4 indices for {condition_type} {date}. Found {len(V1_filtered_indices)} V1 and {len(V4_filtered_indices)} V4 indices')
                    continue
                results = Parallel(n_jobs=-1)(delayed(get_predictions_evars_parallel)(resp_V1[:,V1_filtered_indices], 
                                                                        resp_V4[:, V4_filtered_indices[s]], 
                                                                        n_splits=n_splits,alpha=alpha,
                                                                        frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size),
                                                                        control_shuffle=control_shuffle) for s in range(len(V4_filtered_indices))) 
                results2 = Parallel(n_jobs=-1)(delayed(get_predictions_evars_parallel)(resp_V4[:, V4_filtered_indices[s]], 
                                                                        resp_V1[:,V1_filtered_indices], 
                                                                        n_splits=n_splits,alpha=alpha2,
                                                                        frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size),
                                                                        control_shuffle=control_shuffle) for s in range(len(V4_filtered_indices)))
            elif 'small_chosen_indices' in list(monkey_stats[get_reli_condition(condition_type)][date][area].keys()):
                V4_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area]['small_chosen_indices']
                V1_filtered_indices = monkey_stats[get_reli_condition(condition_type)][date][area2]['big_chosen_indices']
                if len(V1_filtered_indices)==0 or len(V4_filtered_indices)==0:
                    print(f'not enough V1 or V4 indices for {condition_type} {date}. Found {len(V1_filtered_indices)} V1 and {len(V4_filtered_indices)} V4 indices')
                    continue
                results = Parallel(n_jobs=-1)(delayed(get_predictions_evars_parallel)(resp_V1[:,V1_filtered_indices[s]], 
                                                                        resp_V4[:, V4_filtered_indices], 
                                                                        n_splits=n_splits,alpha=alpha,
                                                                        frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size),
                                                                        control_shuffle=control_shuffle) for s in range(len(V1_filtered_indices))) 
                results2 = Parallel(n_jobs=-1)(delayed(get_predictions_evars_parallel)(resp_V4[:, V4_filtered_indices], 
                                                                        resp_V1[:,V1_filtered_indices[s]], 
                                                                        n_splits=n_splits,alpha=alpha2,
                                                                        frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size),
                                                                        control_shuffle=control_shuffle) for s in range(len(V1_filtered_indices)))
            else:
                print(f'no V4 or V1 indices found for {condition_type} {date} {monkey}. skipping this condition type and date')
                continue
            monkey_stats[condition_type][date][area]['directionality_evars' + w_size_con + control_con]=np.array([e for _,e in results])
            monkey_stats[condition_type][date][area2]['directionality_evars' + w_size_con + control_con]=np.array([e for _,e in results2])
            
            print(area,np.nanmean(monkey_stats[condition_type][date][area]['directionality_evars' + w_size_con + control_con]) )
            print(area2, np.nanmean(monkey_stats[condition_type][date][area2]['directionality_evars' + w_size_con + control_con]))
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'Took {elapsed_time:.4f} minutes to complete')   

## plotting functions


def extract_mouse_name(input_string):
    index_of_MP = input_string.find('MP')
    return input_string[index_of_MP:index_of_MP + 5] if index_of_MP != -1 and index_of_MP + 5 <= len(input_string) else None
def get_property_dataset_type(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    else:
        return input_string 
    

def get_3_substrings(label):
    string1= label.split('→')[0]
    string3 = label.split('→')[1]
    return string1+' ', '→', ' '+ string3


def color_label(ax, palette, fontsize, y_offset= -0.13, x_offset=0, fontsize_mod_factor =1, predictor_color=None):
    label = ax.get_xticklabels()[0].get_text()
    pos = ax.get_xticklabels()[0].get_unitless_position()
    new_label = get_3_substrings(label)
    predictor_color1= palette[1]
    predictor_color2= palette[0]
    if predictor_color is not None:
        predictor_color1 = predictor_color
        predictor_color2 = predictor_color
    ax.text(pos[0] + x_offset, pos[1]+ y_offset, new_label[0], color=predictor_color1, ha='right', fontdict={'fontsize':fontsize*fontsize_mod_factor})
    ax.text(pos[0]+ x_offset, pos[1] + y_offset, new_label[1], color='black', ha='center', fontdict={'fontsize':fontsize*fontsize_mod_factor})
    ax.text(pos[0]+ x_offset, pos[1] + y_offset, new_label[2], color=palette[0], ha='left', fontdict={'fontsize':fontsize*fontsize_mod_factor})
    
    label = ax.get_xticklabels()[1].get_text()
    pos = ax.get_xticklabels()[1].get_unitless_position()
    new_label = get_3_substrings(label)
    
    ax.set_xticklabels([])
    if predictor_color is None:
        predictor_color_ = palette[0]
    ax.text(pos[0]-x_offset, pos[1]+ y_offset, new_label[0], color=predictor_color2, ha='right', fontdict={'fontsize':fontsize*fontsize_mod_factor})
    ax.text(pos[0]-x_offset, pos[1] + y_offset, new_label[1], color='black', ha='center', fontdict={'fontsize':fontsize*fontsize_mod_factor})
    ax.text(pos[0]-x_offset, pos[1] + y_offset, new_label[2], color=palette[1], ha='left', fontdict={'fontsize':fontsize*fontsize_mod_factor})

def plot_directionalities(animal_stats, x, neuron_property, neuron_property_label, fontsize=6, 
                        fig_size=(1.3,1.3), height=1.05,plot_type='violin',
                        plot_control_line=True,linewidth=0,central_tendency='median',
                        impose_y_lim = True,animal='mouse', mouse_or_date=None,num_permutations=10000,
                        palette = ['#72BEB7','#EDAEAE'],y_offset=-0.13,x_offset=-0.1,print_pval=False,perm_type='ind',return_ax=False,
                        verbose=False,ax=None,
                        **args):
    """
    Plot directionalities using seaborn's violin plot.

    Parameters:
        animal_stats (DataFrame): DataFrame containing directionalities data.
        x (str): Variable to plot on the x-axis.
        neuron_property (str): Variable to plot on the y-axis.
        neuron_property_label (str): Label for the y-axis.
        fontsize (int, optional): Font size for labels and text. Default is 7.
        fig_size (tuple, optional): Size of the figure (width, height). Default is (1.3, 1.3).
        height (float, optional): Height of text above the plot. Default is 1.05.
        plot_control_line (bool, optional): Whether to plot a line representing the control. Default is True.
        linewidth (int, optional): Width of the line. Default is 0.
        impose_y_lim (bool, optional): Whether to impose a limit on the y-axis. Default is True.
        animal (str, optional): Type of animal. Default is 'mouse'.
        **args: Additional keyword arguments to pass to seaborn's violinplot.

    Returns:
        None
    """
    if animal.lower() != 'mouse':
        hierarchical=False
        mouse_or_date='Date'
    else:
        hierarchical=True
        if mouse_or_date is None:
            mouse_or_date = 'Mouse Name'
    if verbose:
        print('hierarchical:', hierarchical)
        print('mouse_or_date:', mouse_or_date)
        # print(hierarchical, mouse_or_date)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    if x == 'Direction':
        if animal == 'mouse':
            order=['L4→L2/3','L2/3→L4']
        else:
            order =['V1→V4','V4→V1']
    else:
        if animal.lower() == 'mouse':
            order=['L2/3','L4']
        else:
            order = ['V4','V1']
    if plot_type =='violin':
        sns.violinplot(x=x, y=neuron_property, 
                            data=animal_stats[animal_stats['control_shuffle']==False], hue=x,
                            ax=ax,
                            order=order,
                            hue_order=order,
                            palette=palette, saturation=1,
                            inner_kws={'box_width':2, 'whis_width':0.5,
                                    'marker':'_', 'markersize':3,
                                    'markeredgewidth':0.8,
                                    },linewidth=linewidth,cut=0,
                            **args,
                                        )
    elif plot_type =='stripplot':
        sns.stripplot(x=x, y=neuron_property, 
        data=animal_stats[(animal_stats['control_shuffle']==False)&(animal_stats['Permutation']==0)], hue=x,
        hue_order=order, order=order,
        palette=palette,
        **args)
    elif plot_type =='swarmplot':
        sns.swarmplot(x=x, y=neuron_property, 
        data=animal_stats[(animal_stats['control_shuffle']==False)&(animal_stats['Permutation']==0)], hue=x,
        hue_order=order, order=order,
        palette=palette,
        **args)
    sns.despine()
    pval, stars = get_comparison_test_stars(animal_stats[animal_stats['control_shuffle']==False], 'Direction', 
                            neuron_property, hierarchical=hierarchical, mouse_or_date=mouse_or_date, 
                            num_permutations=num_permutations, central_tendency=central_tendency, print_pval=print_pval,
                            perm_type=perm_type, return_pval=True)
    if verbose:
        print('p-value for {} {}: {:.4f}'.format(neuron_property, x, pval))
    if stars=='n.s.':
        height_ = height +0.02
        color='gray'
    else:
        height_=height
        color = 'black'
        
        
    ax.text(0.5, height_, stars, ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, color=color)
    
    ax.tick_params(axis='y', labelsize=fontsize, width=0.5, length=2, pad=0, )
    ax.tick_params(axis='x', labelsize=fontsize, width=0.5, length=2, pad=1, )
    
    ax.spines[:].set_linewidth(0.3)
    
    ax.set(xlabel=None)
    ax.set_ylabel(neuron_property_label, fontsize=fontsize, labelpad=1)

    if plot_control_line is True:
        data = animal_stats[animal_stats['control_shuffle']==True][neuron_property]
        per_25 = np.percentile(data.values, 25)
        per_75 = np.percentile(data.values, 75)
        ax.axhspan(per_25, per_75, alpha=0.3, color='blue', label='shuffle\ncontrol IQR',
                linewidth=0,
                )
    if impose_y_lim is True:
        # Get the y-axis ticks
        y_ticks = plt.gca().get_yticks()
        # Check if 1 is among the ticks
        if 1 in y_ticks:
            ax.set_ylim(top=1)
    if x == 'Direction':
        color_label(ax, palette, fontsize, x_offset=x_offset, y_offset=y_offset, predictor_color='black')
    if return_ax:
        return ax
    
### supplemental plotting functions

def extract_mouse_name(input_string):
    index_of_MP = input_string.find('MP')
    return input_string[index_of_MP:index_of_MP + 5] if index_of_MP != -1 and index_of_MP + 5 <= len(input_string) else None
def get_property_dataset_type(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    else:
        return input_string 
def make_mouse_df(mouse_stats_, dataset_types=['ori32','natimg32']):
    data = []
    for dataset_type in dataset_types:
        if 'spont' in dataset_type:
            act_type = 'gray screen'
        else:
            act_type = 'stimulus'
        for mouse, areas_data in mouse_stats_[dataset_type].items():
            mouse_name = extract_mouse_name(mouse)
            for area, values in areas_data.items():
                if area=='L23':
                    direction = 'L4→L2/3'
                    area_ = 'L2/3'
                else:
                    direction = 'L2/3→L4'
                    area_=area
                split_half_rs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['split_half_r']
                SNRs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['SNR_meanspont']
                one_vs_rest = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['1_vs_rest_r']
                variance_across_stim = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['var_across_stimuli']
                variance_within_stim_across_trials = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['var_within_stimulus_across_trials']
                for n, (split_half_r, snr,max_corr_val, evar, null_evar) in enumerate(zip(split_half_rs, SNRs,values['max_corr_val'],values['evars'],values['evars_null'])):
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area_,
                        'Direction':direction,
                        'EV': evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'max corr. val':max_corr_val,
                        'control_shuffle':False, 
                        '1-vs-rest r²': one_vs_rest[n] if one_vs_rest is not None else None,
                        'Variance across stimuli': variance_across_stim[n],
                        'Variance w/in stimulus\nacross trials': variance_within_stim_across_trials[n],
                    })
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area_,
                        'Direction':direction,
                        'EV': null_evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'max corr. val':max_corr_val,
                        'control_shuffle':True, 
                        '1-vs-rest r²': one_vs_rest[n] if one_vs_rest is not None else None,
                        'Variance across stimuli': variance_across_stim[n],
                        'Variance w/in stimulus\nacross trials': variance_within_stim_across_trials[n],
                    })
    # Create a DataFrame from the flattened data
    df_mouse_all = pd.DataFrame(data)
    return df_mouse_all

def get_property_dataset_type_monkey(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string 

def make_monkey_df(monkey_stats_, dataset_types=['SNR', 'RF_thin', 'RF_large']):
    data = []
    for dataset_type in dataset_types:
        if 'spont' in dataset_type:
            act_type = 'gray screen'
        elif 'RS' in dataset_type:
            act_type = 'lights off'
        else:
            act_type = 'stimulus'
        for date, areas_data in monkey_stats_[dataset_type].items():
            if date in ['140819', '150819', '160819']:
                continue
            for area, values in areas_data.items():
                if area=='V4':
                    direction = 'V1→V4'
                else:
                    direction = 'V4→V1'
                split_half_rs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['split_half_r']
                SNRs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['SNR_meanspont']
                for split_half_r, snr,max_corr_val, evar, null_evar in zip(split_half_rs, SNRs,values['max_corr_val'],values['evars'],values['evars_null']):
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date':date,
                        'Area': area,
                        'Direction':direction,
                        'EV': evar,
                        'SNR': snr,
                        'max corr. val':max_corr_val,
                        'Split-half r': split_half_r,
                        'control_shuffle':False, 
                    })
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date': date,
                        'Area': area,
                        'Direction':direction,
                        'EV': null_evar,
                        'SNR': snr,
                        'max corr. val':max_corr_val,
                        'Split-half r': split_half_r,
                        'control_shuffle':True, 
                    })
    # Create a DataFrame from the flattened data
    df_monkey_all = pd.DataFrame(data)
    return df_monkey_all




EPS = 1e-12

def _kde_bandwidth_std(x, bw="scott"):
    """Return the effective kernel std h used by gaussian_kde (1D)."""
    kde = gaussian_kde(x, bw_method=bw)
    return float(np.sqrt(kde.covariance.squeeze()))

def _mixture_sample_from_runs(runs, size, bw="scott", mode="auto", rng=None, weights=None):
    """
    Draw samples from the averaged density over runs by:
    run ~ weights; x ~ KDE(run, bw).
    Implemented via bootstrap + Gaussian jitter with KDE h per run.
    Supports: 'plain', 'reflect' (>=0), 'log' (positive, skew).
    """
    rng = np.random.default_rng(rng)
    runs = [np.asarray(r, float) for r in runs]

    # auto-detect support
    if mode == "auto":
        mode_eff = "reflect" if min(np.min(r) for r in runs) >= 0 else "plain"
    else:
        mode_eff = mode

    n = len(runs)
    w = np.ones(n)/n if weights is None else np.asarray(weights)/np.sum(weights)
    idx = rng.choice(n, size=size, p=w)

    out = np.empty(size, float)
    # precompute per-run kernel stds
    if mode_eff == "log":
        logs = [np.log(r + EPS) for r in runs]
        hs   = [ _kde_bandwidth_std(z, bw) for z in logs ]
        for i in range(n):
            sel = np.where(idx==i)[0]
            if sel.size:
                base = rng.choice(logs[i], size=sel.size, replace=True)
                z = base + rng.normal(0, hs[i], size=sel.size)
                out[sel] = np.exp(z)
    else:
        hs = [ _kde_bandwidth_std(r, bw) for r in runs ]
        for i in range(n):
            sel = np.where(idx==i)[0]
            if sel.size:
                base = rng.choice(runs[i], size=sel.size, replace=True)
                x = base + rng.normal(0, hs[i], size=sel.size)
                if mode_eff == "reflect":
                    x = np.abs(x)  # reflect negative mass
                out[sel] = x

    return out, mode_eff

def mixture_violinplot(A_runs, B_runs, draws_per_group=5000, bw="scott", mode="auto", rng=0):
    A_samp, modeA = _mixture_sample_from_runs(A_runs, draws_per_group, bw=bw, mode=mode, rng=rng)
    B_samp, modeB = _mixture_sample_from_runs(B_runs, draws_per_group, bw=bw, mode=mode, rng=rng+1)
    # For illustration violins; set the same bw you use elsewhere
    data = [A_samp, B_samp]
    ax = sns.violinplot(data=data, bw=bw, cut=0, inner="quartile")
    ax.set_xticklabels(["A (mixture)", "B (mixture)"])
    ax.set_ylabel("value")
    ax.set_title(f"Mixture violins (mode A={modeA}, B={modeB}, bw={bw})")
    return ax



# ---------- internal helpers ----------
def _kde_plain(x, bw):
    return gaussian_kde(x, bw_method=bw)

def _kde_reflect_nonneg(x, bw):
    x = np.asarray(x)
    xr = np.concatenate([x, -x])  # reflect about 0
    kde = gaussian_kde(xr, bw_method=bw)
    def pdf(t):
        t = np.asarray(t)
        return kde.evaluate(t) + kde.evaluate(-t)
    return pdf  # plain callable

def _fit_kdes(runs, bw, mode):
    kdes = []
    if mode == 'plain':
        for r in runs: kdes.append(_kde_plain(np.asarray(r), bw))
    elif mode == 'reflect':
        for r in runs: kdes.append(_kde_reflect_nonneg(np.asarray(r), bw))
    elif mode == 'log':
        for r in runs:
            r = np.asarray(r)
            r = np.log(r + 1e-12)  # keep >0
            kdes.append(_kde_plain(r, bw))
    else:
        raise ValueError("mode must be 'plain', 'reflect', or 'log'")
    return kdes

def _eval_kdes_on_grid(kdes, xgrid, mode):
    """
    Return matrix D with shape (n_runs, len(xgrid)) of densities in ORIGINAL space.
    Handles Jacobian if mode='log'. Works for callables (reflection) or gaussian_kde.
    """
    x = np.asarray(xgrid)
    n = len(kdes)
    D = np.empty((n, x.size), dtype=float)
    if mode == 'log':
        z = np.log(x + 1e-12)
        jac = 1.0 / (x + 1e-12)  # |dz/dx|
        for i, kde in enumerate(kdes):
            D[i] = kde.evaluate(z) * jac
    else:
        for i, kde in enumerate(kdes):
            f = getattr(kde, "evaluate", kde)  # .evaluate or callable
            D[i] = f(x)
    return D

def _l1_from_matrices(DA, DB, xgrid):
    """Mean across runs then L1 distance (trapz)."""
    pA = DA.mean(axis=0)
    pB = DB.mean(axis=0)
    return np.trapz(np.abs(pA - pB), xgrid)

# ---------- public API ----------
def compare_mixture_densities_verbose(
    A_runs, B_runs, bw="scott", n_perm=2000, seed=0,
    mode="auto", grid_points=1024, qpad=(0.001, 0.999),
    progress_every=100, verbose=True
):
    """
    Like before, but faster and chatty.

    * Precomputes each RUN's density on a grid (original space).
    * Permutes run labels and recomputes L1 diff using row means only.

    Returns dict(T_obs, pval, xgrid, mixA, mixB, mode)
    """
    t0 = time.time()
    # choose mode
    if mode == "auto":
        all_min = min(np.min(r) for r in (A_runs + B_runs))
        mode_eff = "reflect" if all_min >= 0 else "plain"
    else:
        mode_eff = mode

    # grid based on quantiles + padding (robust to skew/tails)
    all_concat = np.concatenate(A_runs + B_runs)
    lo_q, hi_q = np.quantile(all_concat, qpad)
    lo = min(all_concat.min(), lo_q)
    hi = max(all_concat.max(), hi_q)
    if mode_eff == "log":
        lo = max(lo, 1e-9)
    xgrid = np.linspace(lo, hi, grid_points)

    if verbose:
        print(f"[setup] runs: A={len(A_runs)}, B={len(B_runs)} | bw={bw} | mode={mode_eff}")
        print(f"[grid] range=({lo:.4g}, {hi:.4g}) points={grid_points}")
        print("[fit] fitting KDEs...")

    # fit KDEs
    kdesA = _fit_kdes(A_runs, bw, mode_eff)
    kdesB = _fit_kdes(B_runs, bw, mode_eff)

    # precompute densities on grid
    if verbose: print("[precompute] evaluating run-level densities on grid...")
    DA = _eval_kdes_on_grid(kdesA, xgrid, mode_eff)  # shape: (nA, G)
    DB = _eval_kdes_on_grid(kdesB, xgrid, mode_eff)  # shape: (nB, G)

    # observed stat
    T_obs = _l1_from_matrices(DA, DB, xgrid)

    # permutation setup
    nA, nB = DA.shape[0], DB.shape[0]
    M = np.vstack([DA, DB])  # all runs stacked (nA+nB, G)
    idx = np.arange(nA + nB)
    rng = np.random.default_rng(seed)
    Ts = np.empty(n_perm, dtype=float)

    if verbose:
        t1 = time.time()
        print(f"[obs] L1 diff = {T_obs:.6g}")
        print(f"[perm] starting {n_perm} permutations ...")
        print(f"[timing] prep took {(t1 - t0):.2f}s")

    # permutation loop (cheap: just mean rows)
    tperm0 = time.time()
    for t in range(n_perm):
        rng.shuffle(idx)
        iA = idx[:nA]
        iB = idx[nA:]
        Ts[t] = _l1_from_matrices(M[iA], M[iB], xgrid)

        if verbose and ((t + 1) % progress_every == 0 or (t + 1) == n_perm):
            elapsed = time.time() - tperm0
            rate = (t + 1) / max(elapsed, 1e-9)
            remaining = (n_perm - (t + 1)) / rate
            print(f"  perm {t + 1}/{n_perm}  "
                  f"elapsed={elapsed:.1f}s  rate≈{rate:.1f}/s  ETA≈{remaining:.1f}s")

    pval = (np.sum(Ts >= T_obs) + 1) / (n_perm + 1)

    # averaged curves for plotting
    mixA = DA.mean(axis=0)
    mixB = DB.mean(axis=0)

    if verbose:
        t2 = time.time()
        print(f"[done] p ≈ {pval:.4g}  total={(t2 - t0):.2f}s")

    return dict(T_obs=T_obs, pval=pval, xgrid=xgrid, mixA=mixA, mixB=mixB, mode=mode_eff)

def compare_mixture_densities_stratified(
    A_by_group, B_by_group, bw="scott", n_perm=2000, seed=0,
    mode="auto", grid_points=1024, qpad=(0.001, 0.999),
    combine="equal",  # "equal" or "runs" (weights by #runs per group)
    progress_every=200, verbose=True
):
    """
    Stratified version: A_by_group/B_by_group are dicts mapping group -> list of runs.
      Example: A_by_group['natimg32'] = [np.array([...]), np.array([...]), ...]
    We:
      1) fit KDEs per run within each group,
      2) compute per-group observed L1 (between mixA and mixB),
      3) permute *within* each group, recompute per-group L1, then
      4) combine per-group stats with chosen weights and get a global p.

    Returns: dict with pval, T_obs, per_group details.
    """

    rng = np.random.default_rng(seed)
    groups = sorted(set(A_by_group) | set(B_by_group))
    assert all(g in A_by_group and g in B_by_group for g in groups), "Mismatched groups."

    # choose mode from pooled minima
    if mode == "auto":
        all_min = min(np.min(r) for g in groups for r in (A_by_group[g] + B_by_group[g]))
        mode_eff = "reflect" if all_min >= 0 else "plain"
    else:
        mode_eff = mode

    # global grid based on all groups
    all_concat = np.concatenate([np.concatenate(A_by_group[g] + B_by_group[g]) for g in groups])
    lo_q, hi_q = np.quantile(all_concat, qpad)
    lo = min(all_concat.min(), lo_q)
    hi = max(all_concat.max(), hi_q)
    if mode_eff == "log":
        lo = max(lo, 1e-9)
    xgrid = np.linspace(lo, hi, grid_points)

    # per-group precompute
    per_group = {}
    for g in groups:
        kA = _fit_kdes(A_by_group[g], bw, mode_eff)
        kB = _fit_kdes(B_by_group[g], bw, mode_eff)
        DA = _eval_kdes_on_grid(kA, xgrid, mode_eff)
        DB = _eval_kdes_on_grid(kB, xgrid, mode_eff)
        T_obs = _l1_from_matrices(DA, DB, xgrid)
        per_group[g] = dict(DA=DA, DB=DB, T_obs=T_obs,
                            nA=DA.shape[0], nB=DB.shape[0],
                            M=np.vstack([DA, DB]))

    # weights
    if combine == "equal":
        w = {g: 1.0/len(groups) for g in groups}
    elif combine == "runs":
        tot = sum(per_group[g]['nA'] + per_group[g]['nB'] for g in groups)
        w = {g: (per_group[g]['nA'] + per_group[g]['nB'])/tot for g in groups}
    else:
        raise ValueError("combine must be 'equal' or 'runs'")

    # observed combined stat
    T_obs = sum(w[g] * per_group[g]['T_obs'] for g in groups)

    # permutations (within-group)
    Ts = np.empty(n_perm, dtype=float)
    if verbose:
        t0 = time.time()
        print(f"[stratified] groups={groups} | bw={bw} | mode={mode_eff} | combine={combine}")
        print(f"[obs] combined L1 = {T_obs:.6g}")

    for t in range(n_perm):
        T_sum = 0.0
        for g in groups:
            block = per_group[g]
            nA, nB = block['nA'], block['nB']
            idx = np.arange(nA + nB)
            rng.shuffle(idx)
            iA = idx[:nA]
            iB = idx[nA:]
            Tg = _l1_from_matrices(block['M'][iA], block['M'][iB], xgrid)
            T_sum += w[g] * Tg
        Ts[t] = T_sum
        if verbose and ((t+1) % progress_every == 0 or (t+1) == n_perm):
            print(f"  perm {t+1}/{n_perm}")

    pval = (np.sum(Ts >= T_obs) + 1) / (n_perm + 1)
    if verbose:
        print(f"[done] p ≈ {pval:.4g}  total={(time.time()-t0):.2f}s")

    return dict(pval=pval, T_obs=T_obs, Ts=Ts, xgrid=xgrid, mode=mode_eff,
                per_group={g: {'T_obs': per_group[g]['T_obs'],
                               'nA': per_group[g]['nA'], 'nB': per_group[g]['nB'],
                               'weight': w[g]} for g in groups})
    
def sample_stratified_mixture(runs_by_group, draws_total, bw="scott", mode="auto",
                              rng=0, weights=None):
    """
    runs_by_group: dict[str, list[np.ndarray]]  # per-dataset list of runs
    draws_total: int  # total samples you want overall
    weights: dict[str, float] or None
      - If None: equal weight per group
      - Else: must contain a weight for each group, will be normalized
    Returns: samples (np.ndarray), mode_eff (str)
    """
    groups = sorted(runs_by_group.keys())

    # decide mode like your compare_* (auto reflect if all >=0)
    if mode == "auto":
        all_min = min(np.min(r) for g in groups for r in runs_by_group[g])
        mode_eff = "reflect" if all_min >= 0 else "plain"
    else:
        mode_eff = mode

    # weights
    if weights is None:
        probs = np.ones(len(groups), dtype=float) / len(groups)
    else:
        probs = np.array([weights[g] for g in groups], dtype=float)
        probs = probs / probs.sum()

    # how many draws per group (multinomial, deterministic by expected value if you prefer)
    rng_np = np.random.default_rng(rng)
    draws_per_group = rng_np.multinomial(draws_total, probs)

    # sample per group using your existing sampler, then concat
    pieces = []
    seed = rng
    for g, n_draw in zip(groups, draws_per_group):
        if n_draw <= 0:
            continue
        samp_g, _mode_g = _mixture_sample_from_runs(
            runs_by_group[g], n_draw, bw=bw, mode=mode_eff, rng=seed
        )
        pieces.append(samp_g)
        seed += 1  # nudge rng so A/B differ

    if not pieces:
        return np.array([]), mode_eff
    return np.concatenate(pieces, axis=0), mode_eff


def plot_directionalities_densities(animal_stats, x, neuron_property, neuron_property_label, fontsize=6, 
                        fig_size=(1.3,1.3), height=1.05,plot_type='violin',
                        plot_control_line=True,linewidth=0,central_tendency='median',
                        impose_y_lim = True,animal='mouse', mouse_or_date=None,num_permutations=10000,
                        palette = ['#72BEB7','#EDAEAE'],y_offset=-0.13,x_offset=-0.1,print_pval=False,
                        perm_type='ind',draws_per_group=5000, bw="scott", mode="auto", rng=0,stratify_by=None,
                        collapse_across_iterations=True,ax=None,
                        **args):
    """
    Plot directionalities using seaborn's violin plot.

    Parameters:
        animal_stats (DataFrame): DataFrame containing directionalities data.
        x (str): Variable to plot on the x-axis.
        neuron_property (str): Variable to plot on the y-axis.
        neuron_property_label (str): Label for the y-axis.
        fontsize (int, optional): Font size for labels and text. Default is 7.
        fig_size (tuple, optional): Size of the figure (width, height). Default is (1.3, 1.3).
        height (float, optional): Height of text above the plot. Default is 1.05.
        plot_control_line (bool, optional): Whether to plot a line representing the control. Default is True.
        linewidth (int, optional): Width of the line. Default is 0.
        impose_y_lim (bool, optional): Whether to impose a limit on the y-axis. Default is True.
        animal (str, optional): Type of animal. Default is 'mouse'.
        **args: Additional keyword arguments to pass to seaborn's violinplot.

    Returns:
        None
    """
    if animal.lower() != 'mouse':
        hierarchical=False
        mouse_or_date='Date'
    else:
        hierarchical=True
        if mouse_or_date is None:
            mouse_or_date = 'Mouse Name'
        # print(hierarchical, mouse_or_date)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    if x == 'Direction':
        if animal == 'mouse':
            order=['L4→L2/3','L2/3→L4']
        else:
            order =['V1→V4','V4→V1']
    else:
        if animal.lower() == 'mouse':
            order=['L2/3','L4']
        else:
            order = ['V4','V1']
    
    if plot_type =='violin':
        if stratify_by is not None:
        # Build per-dataset runs (same loops you used for the stratified test)
            A_by_group, B_by_group = {}, {}
            for dt in animal_stats[stratify_by].unique():
                perms_dt = animal_stats.loc[animal_stats['Dataset Type']==dt, 'Permutation'].unique()
                A_by_group[dt] = [animal_stats[(animal_stats[x]==order[0]) & (~animal_stats['control_shuffle'])
                                            & (animal_stats['Dataset Type']==dt) & (animal_stats['Permutation']==n)][neuron_property].values
                                for n in perms_dt]
                B_by_group[dt] = [animal_stats[(animal_stats[x]==order[1]) & (~animal_stats['control_shuffle'])
                                            & (animal_stats['Dataset Type']==dt) & (animal_stats['Permutation']==n)][neuron_property].values
                                for n in perms_dt]

            weights = None

            A_samp, modeA = sample_stratified_mixture(A_by_group, draws_per_group, bw=bw, mode=mode, rng=rng, weights=weights)
            B_samp, modeB = sample_stratified_mixture(B_by_group, draws_per_group, bw=bw, mode=mode, rng=rng+1, weights=weights)
            
        else:
            A_runs = [animal_stats[(animal_stats[x]==order[0])&(animal_stats['control_shuffle']==False)&(animal_stats['Permutation']==n)][neuron_property].values for n in animal_stats['Permutation'].unique()]
            B_runs = [animal_stats[(animal_stats[x]==order[1])&(animal_stats['control_shuffle']==False)&(animal_stats['Permutation']==n)][neuron_property].values for n in animal_stats['Permutation'].unique()]
            A_samp, modeA = _mixture_sample_from_runs(A_runs, draws_per_group, bw=bw, mode=mode, rng=rng)
            B_samp, modeB = _mixture_sample_from_runs(B_runs, draws_per_group, bw=bw, mode=mode, rng=rng+1)

        if collapse_across_iterations is True:
            animal_stats_median = animal_stats.groupby(['control_shuffle','Dataset Type', mouse_or_date, 'Area', 'Direction','Permutation'])[neuron_property].median().reset_index()
            stars = get_comparison_test_stars(animal_stats_median[animal_stats_median['control_shuffle']==False], 'Direction', 
                                neuron_property, hierarchical=hierarchical, mouse_or_date=mouse_or_date, 
                                num_permutations=num_permutations, central_tendency=central_tendency, print_pval=print_pval,
                                perm_type='paired')
        else:
            stars = get_comparison_test_stars(animal_stats[animal_stats['control_shuffle']==False], 'Direction', 
                                neuron_property, hierarchical=hierarchical, mouse_or_date=mouse_or_date, 
                                num_permutations=num_permutations, central_tendency=central_tendency, print_pval=print_pval,
                                perm_type=perm_type)

                
        #create new dataframe with A_samp and B_samp
        data = pd.DataFrame({neuron_property: np.concatenate([A_samp, B_samp]),
                            x: np.concatenate([[order[0]]*len(A_samp), [order[1]]*len(B_samp)])})
        sns.violinplot(x=x, y=neuron_property, 
                            data=data, hue=x,
                            ax=ax,
                            order=order,
                            hue_order=order,
                            palette=palette, saturation=1,
                            inner_kws={'box_width':2, 'whis_width':0.5,
                                    'marker':'_', 'markersize':3,
                                    'markeredgewidth':0.8,
                                    },linewidth=linewidth,cut=0,
                            bw_method = bw,
                            **args,
                                        )
    elif plot_type =='stripplot':
        sns.stripplot(x=x, y=neuron_property, 
        data=animal_stats[(animal_stats['control_shuffle']==False)&(animal_stats['Permutation']==0)], hue=x,
        hue_order=order, order=order,
        palette=palette,
        **args)
    elif plot_type =='swarmplot':
        sns.swarmplot(x=x, y=neuron_property, 
        data=animal_stats[(animal_stats['control_shuffle']==False)&(animal_stats['Permutation']==0)], hue=x,
        hue_order=order, order=order,
        palette=palette,
        **args)
    sns.despine()


    if plot_type !='violin':
        stars = get_comparison_test_stars(animal_stats[animal_stats['control_shuffle']==False], 'Direction', 
                            neuron_property, hierarchical=hierarchical, mouse_or_date=mouse_or_date, 
                            num_permutations=num_permutations, central_tendency=central_tendency, print_pval=print_pval,
                            perm_type=perm_type)


    if stars=='n.s.':
        height_ = height +0.02
        color='gray'
    else:
        height_=height
        color = 'black'
        
        
    ax.text(0.5, height_, stars, ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, color=color)

    ax.tick_params(axis='y', labelsize=fontsize, width=0.5, length=2, pad=1, )
    ax.tick_params(axis='x', labelsize=fontsize, width=0.5, length=2, pad=1, )

    ax.spines[:].set_linewidth(0.3)

    ax.set(xlabel=None)
    ax.set_ylabel(neuron_property_label, fontsize=fontsize, labelpad=1)

    if plot_control_line is True:
        data = animal_stats[animal_stats['control_shuffle']==True][neuron_property]
        per_25 = np.percentile(data.values, 25)
        per_75 = np.percentile(data.values, 75)
        ax.axhspan(per_25, per_75, alpha=0.3, color='blue', label='shuffle\ncontrol IQR',
                linewidth=0,
                )
    if impose_y_lim is True:
        # Get the y-axis ticks
        y_ticks = ax.get_yticks()
        # Check if 1 is among the ticks
        if 1 in y_ticks:
            ax.set_ylim(top=1)
    if x == 'Direction':
        color_label(ax, palette, fontsize, x_offset=x_offset, y_offset=y_offset, predictor_color='black')
        
def p_to_stars(p):
    if p is None:
        return ""
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))
def plot_activitytype_by_direction_densities(animal_stats,neuron_property="EV",x="Activity Type",
	hue="Direction",neuron_property_label="EV fraction",*, animal="monkey",                 # 'mouse' or 'monkey' toggles default hue order
	order=None,                      # list of activity types in desired order
	hue_order=None,                  # directions order; defaults based on animal
	fig_size=(2.6, 1.3),fontsize=6,palette=("#72BEB7", "#EDAEAE"),draws_per_group=6000,
	bw="scott",mode="auto",rng=0,num_permutations=10000,grid_points=100,
	plot_control_line=True,          # shaded IQR from control rows (global across all)
	control_shade_color="blue",control_alpha=0.25,impose_y_lim=True, print_verbose=True,linewidth=0,
	stratify =False,stratify_by_dict= None,# e.g. 'Cortical Area' to plot separate figures per area
	show_pvals=True,rel_pval_pos = 99, subsampled_indices=False, omit_figs=False,
	**sns_kwargs
	):
    """
    Grouped mixture violins: x = Activity Type, hue = Direction.
    For each activity type, average densities across permutations (runs) per direction.
    Then compute a per-activity run-level permutation p-value and annotate stars.

    Expects columns: [x, hue, neuron_property, 'Permutation', 'control_shuffle'].
    """


    if 'mouse' in animal.lower():
        mouse_or_date = 'Mouse Name'
    else:
        mouse_or_date = 'Date'
    # --- defaults / ordering ---
    base_cols = [c for c in [mouse_or_date,'Activity Type','Dataset Type'] if c in animal_stats.columns]
    df = animal_stats.copy()
    df['SessionKey'] = df[base_cols].astype(str).agg(' | '.join, axis=1)
    # median across neurons per (SessionKey, Permutation, Direction, Activity)
    df_med = (df[df['control_shuffle']==False]
            .groupby(['SessionKey','Permutation',hue,x])[ [neuron_property] ]
            .median()
            .reset_index())


    if order is None:
        order = [c for c in df[x].unique()]
    # ensure hue_order is exactly two levels
    if hue_order is None:
        levels = list(df[hue].dropna().unique())
        if len(levels) != 2:
            raise ValueError(f"{hue} must have exactly 2 levels; got {levels}")
        hue_order = levels
    else:
        if len(hue_order) != 2:
            raise ValueError(f"hue_order must have length 2; got {hue_order}")


    # --- build mixture samples per (activity, direction) ---
    plot_rows = []
    pvals = {}  # activity -> p-value
    for i, act in enumerate(order):
        if stratify is True:
            if stratify_by_dict is None or act not in stratify_by_dict:
                raise ValueError("If stratify is True, stratify_by_dict must be provided and contain an entry for each order.")
            A_by_group, B_by_group = {}, {}
            stratify_by = stratify_by_dict[act]
            for dt in df[stratify_by].unique():
                perms_dt = df.loc[(df[x]==act) & (df[stratify_by]==dt), 'Permutation'].unique()
                runsA = [df[(df[x]==act) &(df[hue]==hue_order[0]) & (~df['control_shuffle'])
                                            & (df[stratify_by]==dt) & (df['Permutation']==n)][neuron_property].values
                                for n in perms_dt]
                runsB = [df[(df[x]==act) &(df[hue]==hue_order[1]) & (~df['control_shuffle'])
                                            & (df[stratify_by]==dt) & (df['Permutation']==n)][neuron_property].values
                                for n in perms_dt]
                # keep only non-empty per-run arrays
                runsA = [r for r in runsA if r.size]
                runsB = [r for r in runsB if r.size]
                if len(runsA) and len(runsB):
                    A_by_group[dt] = runsA
                    B_by_group[dt] = runsB
            if not A_by_group or not B_by_group:
                if print_verbose:
                    print(f"[warn] skipping activity '{act}' (no valid stratified groups).")
                continue

            weights = None
            #   or weight by number of runs (to mirror combine="runs"):
            # weights = {dt: len(A_by_group[dt]) + len(B_by_group[dt]) for dt in A_by_group}

            A_samp, modeA = sample_stratified_mixture(A_by_group, draws_per_group, bw=bw, mode=mode, rng=rng, weights=weights)
            B_samp, modeB = sample_stratified_mixture(B_by_group, draws_per_group, bw=bw, mode=mode, rng=rng+1, weights=weights)

            # guard empty samples
            if (A_samp is None) or (B_samp is None) or (len(A_samp)==0) or (len(B_samp)==0):
                if print_verbose:
                    print(f"[warn] skipping activity '{act}' (empty mixture samples).")
                continue

        else:
            # collect runs per direction, keeping perms that exist for BOTH sides
            perms = np.sort(df.loc[(df[x]==act) & (df['control_shuffle']==False), 'Permutation'].unique())
            
            runs_by_dir = {}
            valid_perms = []

            for n in perms:
                groupA = df[(df[x]==act) & (df[hue]==hue_order[0]) & (df['control_shuffle']==False) & (df['Permutation']==n)][neuron_property].values
                groupB = df[(df[x]==act) & (df[hue]==hue_order[1]) & (df['control_shuffle']==False) & (df['Permutation']==n)][neuron_property].values
                if groupA.size and groupB.size:
                    runs_by_dir.setdefault(hue_order[0], []).append(groupA)
                    runs_by_dir.setdefault(hue_order[1], []).append(groupB)
                    valid_perms.append(n)

            # if one side missing entirely, skip gracefully
            if (hue_order[0] not in runs_by_dir) or (hue_order[1] not in runs_by_dir):
                if print_verbose:
                    print(f"[warn] skipping activity '{act}' (one or both directions missing).")
                continue

            # draw mixture samples (same bw/mode as your stats)
            A_samp, _ = _mixture_sample_from_runs(runs_by_dir[hue_order[0]], draws_per_group, bw=bw, mode=mode, rng=rng + i*2)
            B_samp, _ = _mixture_sample_from_runs(runs_by_dir[hue_order[1]], draws_per_group, bw=bw, mode=mode, rng=rng + i*2 + 1)

        plot_rows.append(pd.DataFrame({
            x: act,
            hue: np.concatenate([[hue_order[0]]*len(A_samp), [hue_order[1]]*len(B_samp)]),
            neuron_property: np.concatenate([A_samp, B_samp])
        }))
                
        if subsampled_indices is True:
            # subsample 6 permutations per dataset type to make monkey A
            unique_permutations = df_med[df_med[x]==act]['Permutation'].unique()
            if len(unique_permutations)>6:
                subsampled_permutations = np.random.choice(unique_permutations, size=6, replace=False)
            else:
                subsampled_permutations = unique_permutations
            df_med = df_med[df_med['Permutation'].isin(subsampled_permutations)]
        wide = (df_med[df_med[x]==act]
            .pivot(index=['SessionKey','Permutation'], columns=hue, values=neuron_property))
        # print(act, wide.shape)
        if len(wide)<2:
            if print_verbose: print(f"[warn] no only 1 permutation for '{act}'")
            pvals[act] = None
            continue
        missing = [lvl for lvl in hue_order if lvl not in wide.columns]
        if missing:
            if print_verbose:
                print(f"[warn] skipping '{act}' (missing hue(s): {missing})")
            continue
        wide = wide.dropna(subset=hue_order)
        gA = wide[hue_order[0]].to_numpy()  # corresponds to direction “from the other area to this one”
        gB = wide[hue_order[1]].to_numpy()

        if gA.size == 0:
            if print_verbose: print(f"[warn] no paired rows for '{act}'")
            continue

        pvals[act] = perm_test_paired(gA, gB)
        # pvals[act] = res['pval']

    if not plot_rows:
        raise ValueError("No valid (activity, direction) pairs found to plot.")

    plot_df = pd.concat(plot_rows, ignore_index=True)
    if not omit_figs:
        # --- plot ---
        fig, ax = plt.subplots(figsize=fig_size)
        # seaborn 0.13 uses bw_method/bw_adjust; older uses bw
        try:
            v = sns.violinplot(
                data=plot_df, x=x, y=neuron_property, hue=hue,
                order=order, hue_order=hue_order,
                palette=palette, cut=0,
                bw_method=bw, bw_adjust=1, 
                ax=ax,
                inner_kws={'box_width':2, 'whis_width':0.5,
                                        'marker':'_', 'markersize':3,
                                        'markeredgewidth':0.8,
                                        },
                linewidth=linewidth,saturation=1,
                **sns_kwargs
            )
        except TypeError:
            v = sns.violinplot(
                data=plot_df, x=x, y=neuron_property, hue=hue,
                order=order, hue_order=hue_order,
                palette=palette, cut=0, 
                bw=bw,
                ax=ax, **sns_kwargs
            )

        # optional control IQR shading (global, across all control rows)
        if plot_control_line:
            ctrl = df[df['control_shuffle']==True][neuron_property].to_numpy()
            ctrl = ctrl[np.isfinite(ctrl)]
            if ctrl.size:
                q25, q75 = np.percentile(ctrl, [25, 75])
                ax.axhspan(q25, q75, color=control_shade_color, alpha=control_alpha, linewidth=0, zorder=0)

        # stars per activity (centered at category)
        # map p->stars
        if show_pvals is True:
            # compute y for stars a bit above the taller of the pair
            ymins, ymaxs = ax.get_ylim()
            yrange = ymaxs - ymins

            for i, act in enumerate(order):
                if act not in pvals:
                    continue
                p = pvals[act]
                stars = p_to_stars(p)	
                # find max y at this category from plotted data to place stars above
                sub = plot_df[plot_df[x]==act][neuron_property].to_numpy()
                sub = sub[np.isfinite(sub)]
                if sub.size == 0:
                    print(f"[warn] no finite data for '{act}' when placing stars")
                y_top = (np.percentile(sub, rel_pval_pos) if sub.size else ymaxs) + 0.03*yrange
                if stars =="n.s.":
                    y_top += 0.01*yrange
                color = "gray" if stars=="n.s." else "black"
                ax.text(i, y_top, stars, ha="center", va="bottom", fontsize=fontsize, color=color)

        # cosmetics
        sns.despine()
        ax.set_xlabel(None)
        ax.set_ylabel(neuron_property_label, fontsize=fontsize, labelpad=-0.5)
        ax.tick_params(axis='x', labelsize=fontsize, width=0.5, length=2, pad=1)
        ax.tick_params(axis='y', labelsize=fontsize, width=0.5, length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

        if impose_y_lim:
            # cap at 1 if relevant to EV
            yt = ax.get_yticks()
            if np.any(np.isclose(yt, 1.0)):
                ax.set_ylim(top=1.0)

        # optional: tidy legend
        if ax.legend_:
            ax.legend(title=None, fontsize=fontsize*0.8, frameon=True, labelspacing=0.2, handletextpad=0.3, borderpad=0.3)
            ax.legend_.get_frame().set_linewidth(0.3)
    else:
        fig, ax = None, None  
    return fig, ax, pvals


def make_mouse_df_directionality_nonvisual(mouse_stats_, dataset_types=['ori32','natimg32'], nonvisual_neurons=False, verbose=False):
	"""Creates a DataFrame for mouse directionality data.

	This function iterates over the provided mouse statistics data and constructs a DataFrame containing directionality
	information for each mouse, area, and permutation. It extracts relevant information such as mouse name, area,
	direction, reliability, SNR, and maximum correlation value.

	Args:
	mouse_stats_ (dict): Dictionary containing mouse statistics data.
	dataset_types (list, optional): List of dataset types. Defaults to ['ori32', 'natimg32'].

	Returns:
	pandas.DataFrame: DataFrame containing directionality data.
	"""
	nonvis_con = ''
	if nonvisual_neurons:
		nonvis_con = '_nonvisual'
	data = []
	for dataset_type in dataset_types:
		if 'spont' in dataset_type:
			act_type = 'gray screen'
		else:
			act_type = 'stimulus'
		for mouse, areas_data in mouse_stats_[dataset_type].items():
			mouse_name = extract_mouse_name(mouse)
			# loop throgh L23 indices to make sure we have unique combination
			list_of_unique_indices = []
			seeds_to_ommit = []
			for s in range(10):
				ordered_indices = np.sort(mouse_stats_[get_property_dataset_type(dataset_type)][mouse]['L23'][f'L23_chosen_indices{nonvis_con}'][s])
				if any(np.sum(ordered_indices == x) == len(ordered_indices) for x in list_of_unique_indices):
					if verbose:
						print(f'skipping {dataset_type} {mouse_name} seed {s}, indices already used')
					seeds_to_ommit.append(s)
				else:
					# print([np.sum(ordered_indices == x)/ len(ordered_indices) for x in list_of_unique_indices])
					list_of_unique_indices.append(ordered_indices)			
			for area, values in areas_data.items():
				split_half_rs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['split_half_r']
				SNRs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['SNR_meanspont']
				max_corr_vals = mouse_stats_[dataset_type][mouse][area]['max_corr_val']
				
				n_seeds = len(values[f'directionality_evars{nonvis_con}'])
				for s in range(n_seeds):
					if s in seeds_to_ommit:
						continue
					direction_evars = values[f'directionality_evars{nonvis_con}'][s]
					direction_maxcorrvals = values[f'directionality_maxcorrvals{nonvis_con}'][s]
					direction_evars_null = values[f'directionality_evars{nonvis_con}_null'][s]
					
					if area =='L23':
						direction = 'L4→L2/3'
						area_ ='L2/3'
						l23_indices = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area][f'L23_chosen_indices{nonvis_con}'][s]
						chosen_split_half_rs = split_half_rs[l23_indices]
						chosen_SNRs= SNRs[l23_indices]
						chosen_max_corr_vals = max_corr_vals[l23_indices]
						chosen_indices = l23_indices
					else:
						direction = 'L2/3→L4'
						area_=area
						if nonvisual_neurons:
							l4_indices = np.argwhere((split_half_rs < 0.8)&(SNRs<2))[:,0]
						else:
							l4_indices = np.argwhere((split_half_rs > 0.8)&(SNRs>=2))[:,0]
						chosen_split_half_rs = split_half_rs[l4_indices]
						chosen_SNRs = SNRs[l4_indices]
						chosen_max_corr_vals = max_corr_vals[l4_indices]
						chosen_indices = l4_indices
					
					for n, (split_half_r, snr, max_corr_val, direction_evar, direction_evar_null, dir_maxcorrval) in enumerate(zip(chosen_split_half_rs, 
																												chosen_SNRs, chosen_max_corr_vals, 
																												direction_evars, 
																												direction_evars_null,
																												direction_maxcorrvals)):
						data.append({
							'Dataset Type': dataset_type,
							'Activity Type': act_type,
							'Mouse': mouse,
							'Mouse Name':mouse_name,
							'Area': area_,
							'Direction':direction,
							'EV': direction_evar,
							'SNR': snr,
							'Split-half r': split_half_r,
							'max corr. val': max_corr_val,
							'max corr. val\npop. controlled':dir_maxcorrval,
							'control_shuffle':False,
							'Permutation':s, 
							'Neuron index':chosen_indices[n]
						})
						data.append({
							'Dataset Type': dataset_type,
							'Activity Type': act_type,
							'Mouse': mouse,
							'Mouse Name':mouse_name,
							'Area': area_,
							'Direction':direction,
							'EV': direction_evar_null,
							'SNR': snr,
							'Split-half r': split_half_r,
							'max corr. val': max_corr_val,
							'max corr. val\npop. controlled':dir_maxcorrval,
							'control_shuffle':True, 
							'Permutation':s,
							'Neuron index':chosen_indices[n]
						})
	# Create a DataFrame from the flattened data
	df_mouse_all = pd.DataFrame(data)
	return df_mouse_all

def make_mouse_df_directionality(mouse_stats_, dataset_types=['ori32','natimg32'], nonvisual_neurons=False):
	"""Creates a DataFrame for mouse directionality data.

	This function iterates over the provided mouse statistics data and constructs a DataFrame containing directionality
	information for each mouse, area, and permutation. It extracts relevant information such as mouse name, area,
	direction, reliability, SNR, and maximum correlation value.

	Args:
	mouse_stats_ (dict): Dictionary containing mouse statistics data.
	dataset_types (list, optional): List of dataset types. Defaults to ['ori32', 'natimg32'].

	Returns:
	pandas.DataFrame: DataFrame containing directionality data.
	"""
	nonvis_con = ''
	if nonvisual_neurons:
		nonvis_con = '_nonvisual'
	data = []
	for dataset_type in dataset_types:
		if 'spont' in dataset_type:
			act_type = 'gray screen'
		else:
			act_type = 'stimulus'
		for mouse, areas_data in mouse_stats_[dataset_type].items():
			mouse_name = extract_mouse_name(mouse)
			# loop throgh L23 indices to make sure we have unique combination
			list_of_unique_indices = []
			seeds_to_ommit = []
			for s in range(10):
				ordered_indices = np.sort(mouse_stats_[get_property_dataset_type(dataset_type)][mouse]['L23'][f'L23_chosen_indices{nonvis_con}'][s])
				if any(np.sum(ordered_indices == x) == len(ordered_indices) for x in list_of_unique_indices):
					print(f'skipping {dataset_type} {mouse_name} seed {s}, indices already used')
					seeds_to_ommit.append(s)
				else:
					# print([np.sum(ordered_indices == x)/ len(ordered_indices) for x in list_of_unique_indices])
					list_of_unique_indices.append(ordered_indices)			
			for area, values in areas_data.items():
				split_half_rs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['split_half_r']
				SNRs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['SNR_meanspont']
				max_corr_vals = mouse_stats_[dataset_type][mouse][area]['max_corr_val']
				variance_across_stim = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['var_across_stimuli']
				variance_within_stim_across_trials = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['var_within_stimulus_across_trials']
				n_seeds = len(values[f'directionality_evars{nonvis_con}'])
				for s in range(n_seeds):
					if s in seeds_to_ommit:
						continue
					direction_evars = values[f'directionality_evars{nonvis_con}'][s]
					direction_maxcorrvals = values[f'directionality_maxcorrvals{nonvis_con}'][s]
					direction_evars_null = values[f'directionality_evars{nonvis_con}_null'][s]
					
					if area =='L23':
						direction = 'L4→L2/3'
						area_ ='L2/3'
						l23_indices = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area][f'L23_chosen_indices{nonvis_con}'][s]
						chosen_split_half_rs = split_half_rs[l23_indices]
						chosen_SNRs= SNRs[l23_indices]
						chosen_max_corr_vals = max_corr_vals[l23_indices]
						chosen_indices = l23_indices
					else:
						direction = 'L2/3→L4'
						area_=area
						if nonvisual_neurons:
							l4_indices = np.argwhere((split_half_rs < 0.8)&(SNRs<2))[:,0]
						else:
							l4_indices = np.argwhere((split_half_rs > 0.8)&(SNRs>=2))[:,0]
						chosen_split_half_rs = split_half_rs[l4_indices]
						chosen_SNRs = SNRs[l4_indices]
						chosen_max_corr_vals = max_corr_vals[l4_indices]
						chosen_indices = l4_indices
						
					
					for n, (split_half_r, snr, max_corr_val, direction_evar, direction_evar_null, dir_maxcorrval) in enumerate(zip(chosen_split_half_rs, 
																												chosen_SNRs, chosen_max_corr_vals, 
																												direction_evars, 
																												direction_evars_null,
																												direction_maxcorrvals)):
						data.append({
							'Dataset Type': dataset_type,
							'Activity Type': act_type,
							'Mouse': mouse,
							'Mouse Name':mouse_name,
							'Area': area_,
							'Direction':direction,
							'EV': direction_evar,
							'SNR': snr,
							'Split-half r': split_half_r,
							'max corr. val': max_corr_val,
							'max corr. val\npop. controlled':dir_maxcorrval,
							'control_shuffle':False,
							'Permutation':s, 
							'Neuron index':chosen_indices[n],
							'Variance across stimuli': variance_across_stim[n],
                        	'Variance w/in stimulus\nacross trials': variance_within_stim_across_trials[n],
							})
						data.append({
							'Dataset Type': dataset_type,
							'Activity Type': act_type,
							'Mouse': mouse,
							'Mouse Name':mouse_name,
							'Area': area_,
							'Direction':direction,
							'EV': direction_evar_null,
							'SNR': snr,
							'Split-half r': split_half_r,
							'max corr. val': max_corr_val,
							'max corr. val\npop. controlled':dir_maxcorrval,
							'control_shuffle':True, 
							'Permutation':s,
							'Neuron index':chosen_indices[n],
							'Variance across stimuli': variance_across_stim[n],
                        'Variance w/in stimulus\nacross trials': variance_within_stim_across_trials[n],
						})
	# Create a DataFrame from the flattened data
	df_mouse_all = pd.DataFrame(data)
	return df_mouse_all


def make_monkey_df_directionality(monkey_stats_, dataset_types=['SNR','RF_thin','RF_large'], verbose=False):
	"""Creates a DataFrame for monkey directionality data.

	This function iterates over the provided mouse statistics data and constructs a DataFrame containing directionality
	information for each mouse, area, and permutation. It extracts relevant information such as date, area,
	direction, reliability, SNR, and maximum correlation value.

	Args:
		mouse_stats_ (dict): Dictionary containing monkey statistics data.
		dataset_types (list, optional): List of dataset types. Defaults to ['SNR','RF_thin','RF_large'].

	Returns:
		pandas.DataFrame: DataFrame containing directionality data.
	"""
	data = []

	for dataset_type in dataset_types:
		if 'spont' in dataset_type:
			act_type = 'gray screen'
		elif 'RS' in dataset_type:
			act_type = 'lights off'
		else:
			act_type = 'stimulus'
		for date, areas_data in monkey_stats_[dataset_type].items():
			if date in ['140819', '150819', '160819']:
				continue
			list_of_unique_indices = []
			seeds_to_ommit = []
			for s in range(10):
				for area_ in ['V1','V4']:
					if 'big_chosen_indices' in list(monkey_stats_[get_reli_condition(dataset_type)][date][area_].keys()):
						ordered_indices = np.sort(monkey_stats_[get_reli_condition(dataset_type)][date][area_]['big_chosen_indices'][s])
						if any(np.sum(ordered_indices == x) == len(ordered_indices) for x in list_of_unique_indices):
							if verbose:
								print(f'skipping {dataset_type} {date} {area_} seed {s}, indices already used')
							seeds_to_ommit.append(s)
						else:
							# print([np.sum(ordered_indices == x)/ len(ordered_indices) for x in list_of_unique_indices])
							list_of_unique_indices.append(ordered_indices)
								

			for area, values in areas_data.items():
				if 'directionality_evars' not in list(values.keys()):
					print(f'skipping {dataset_type} {date} {area}, no directionality evars found')
					continue
				if len(values['directionality_evars'])==1:
					if np.isnan(values['directionality_evars'][0].any()):
						print('skipping directionality, none was recorded')
						continue
				split_half_rs = monkey_stats_[get_reli_condition(dataset_type)][date][area]['split_half_r']
				SNRs = monkey_stats_[get_reli_condition(dataset_type)][date][area]['SNR_meanspont']
				if verbose:
					print(dataset_type, date, area)
				max_corr_vals = monkey_stats_[dataset_type][date][area]['max_corr_val']
				if 'RF' in dataset_type:
					rf_variance_across_stimuli = monkey_stats_[get_reli_condition(dataset_type)][date][area]['var_across_stimuli']
					var_within_trial_across_timepoints = np.mean(monkey_stats_[get_reli_condition(dataset_type)][date][area]['var_within_trial_across_timepoints'],axis=0)
					var_within_timepoint_across_trials = np.mean(monkey_stats_[get_reli_condition(dataset_type)][date][area]['var_across_trials_within_timepoints'],axis=0)
				else:
					rf_variance_across_stimuli = [np.nan]*len(split_half_rs)
					var_within_trial_across_timepoints = monkey_stats_[get_reli_condition(dataset_type)][date][area]['var_within_trial_across_timepoints']
					var_within_timepoint_across_trials = monkey_stats_[get_reli_condition(dataset_type)][date][area]['var_across_trials_within_timepoints']
				n_seeds = len(values['directionality_evars'])
				# print(values['directionality_evars'].shape)
				# go through each seed, find the big_chosen_indices, and find the s indices that are unique across all seeds
				for s in range(n_seeds):
					if s in seeds_to_ommit:
						continue
					direction_evars = values['directionality_evars'][s]
					direction_evars_null = values['directionality_evars_null'][s]
					direction_maxcorrvals = values['directionality_maxcorrvals'][s]
					
					if area =='V1':
						direction = 'V4→V1'
						if 'big_chosen_indices' in list(monkey_stats_[get_reli_condition(dataset_type)][date][area].keys()):
							v1_indices = list(monkey_stats_[get_reli_condition(dataset_type)][date][area]['big_chosen_indices'][s])
						elif 'small_chosen_indices' in list(monkey_stats_[get_reli_condition(dataset_type)][date][area].keys()):
							v1_indices = monkey_stats_[get_reli_condition(dataset_type)][date][area]['small_chosen_indices']
						else:
							print(f'no V1 indices found for {dataset_type} {date} {area}. skipping this condition type and date')
							break
						chosen_split_half_rs = split_half_rs[v1_indices]
						chosen_SNRs= SNRs[v1_indices]
						chosen_max_corr_vals = max_corr_vals[v1_indices]
						neuron_indices = v1_indices
					else:
						direction = 'V1→V4'
						if 'big_chosen_indices' in list(monkey_stats_[get_reli_condition(dataset_type)][date][area].keys()):
							v4_indices = monkey_stats_[get_reli_condition(dataset_type)][date][area]['big_chosen_indices'][s]

						elif 'small_chosen_indices' in list(monkey_stats_[get_reli_condition(dataset_type)][date][area].keys()):
							v4_indices = monkey_stats_[get_reli_condition(dataset_type)][date][area]['small_chosen_indices']
						else:
							print(f'no V1 indices found for {dataset_type} {date} {area}. skipping this condition type and date')
							break
						chosen_split_half_rs = split_half_rs[v4_indices]
						chosen_SNRs = SNRs[v4_indices]
						chosen_max_corr_vals = max_corr_vals[v4_indices]
						neuron_indices = v4_indices
					# print(f'found {len(neuron_indices)} neurons for {dataset_type} {date} {area} seed {s}')
					for n, (split_half_r, snr, max_corr_val, direction_evar, direction_evar_null, dir_maxcorrval, neuron_index) in enumerate(zip(chosen_split_half_rs, 
							chosen_SNRs, chosen_max_corr_vals, direction_evars, direction_evars_null,direction_maxcorrvals,neuron_indices)):
						data.append({
							'Dataset Type': dataset_type,
							'Activity Type': act_type,
							'Date': date,
							'Area': area,
							'Direction':direction,
							'EV': direction_evar,
							'SNR': snr,
							'Split-half r': split_half_r,
							'max corr. val': max_corr_val,
							'max corr. val\npop. controlled':dir_maxcorrval,
							'control_shuffle':False,
							'Permutation':s, 
							'Neuron index': neuron_index,
							'Neuron distribution ID':n,
							'Variance across stimuli': rf_variance_across_stimuli[n],
                        	'Variance w/in trial\nacross timepoints': var_within_trial_across_timepoints[n],
                        	'Variance w/in timepoint\nacross trials': var_within_timepoint_across_trials[n],
       
						})
						data.append({
							'Dataset Type': dataset_type,
							'Activity Type': act_type,
							'Date': date,
							'Area': area,
							'Direction':direction,
							'EV': direction_evar_null,
							'SNR': snr,
							'Split-half r': split_half_r,
							'max corr. val': max_corr_val,
							'max corr. val\npop. controlled':dir_maxcorrval,
							'control_shuffle':True, 
							'Permutation':s,
							'Neuron index': neuron_index,
							'Neuron distribution ID':n,
    						'Variance across stimuli': rf_variance_across_stimuli[n],
                        	'Variance w/in trial\nacross timepoints': var_within_trial_across_timepoints[n],
                        	'Variance w/in timepoint\nacross trials': var_within_timepoint_across_trials[n],
						})
	# Create a DataFrame from the flattened data
	df_monkey_all = pd.DataFrame(data)
	return df_monkey_all


def residualize_within_condition_multi(
    df,
    y='EV',
    xs=None,                     # list/tuple of covariate column names, 'auto', or a callable(g_df)->DataFrame
    cond='Dataset Type',
    flag_col='control_shuffle',
    flag_keep=False,             # compute residuals only where df[flag_col] == flag_keep
    categorical=None,            # list of covariate names to one-hot encode (if present in xs)
    standardize=False,           # z-score covariates within (condition × keep) before fit
    rank_covariates=False,       # replace covariates with within-(condition × keep) ranks
    rank_y=False,                # rank the response before fit (partial Spearman-style)
    model='ols',                 # 'ols' or 'ridge'
    ridge_alpha=1.0,             # alpha for ridge if model='ridge'
    include_intercept=True,      # keep intercept in regression
    add_predictions=False,       # also return EV_hat column
    exclude_cols=None            # only used when xs='auto': columns to exclude from auto selection
):
    """
    Residualize y against many covariates within each condition, using only rows
    where df[flag_col] == flag_keep. Others get EV_resid = NaN.

    Returns a copy of df with EV_resid (and optionally EV_hat).
    """
    out = df.copy()
    out['EV_resid'] = np.nan
    if add_predictions and 'EV_hat' not in out.columns:
        out['EV_hat'] = np.nan

    # helper: build covariate matrix for a group's "kept" rows
    def build_X(g_keep):
        # choose covariate columns
        if callable(xs):
            Xraw = xs(g_keep)  # must return a DataFrame
        elif xs == 'auto':
            # auto-pick numeric covariates
            excl = set(exclude_cols or []) | {y, cond, flag_col, 'EV_resid', 'EV_hat'}
            num_cols = g_keep.select_dtypes(include=[np.number]).columns
            chosen = [c for c in num_cols if c not in excl]
            Xraw = g_keep.loc[:, chosen]
        else:
            cols = list(xs) if isinstance(xs, (list, tuple, pd.Index)) else [xs]
            Xraw = g_keep.loc[:, cols]

        # one-hot for categoricals (if any)
        if categorical:
            cats_present = [c for c in categorical if c in Xraw.columns]
            if len(cats_present):
                Xcat = pd.get_dummies(Xraw[cats_present], drop_first=True)
                Xnum = Xraw.drop(columns=cats_present)
                Xraw = pd.concat([Xnum, Xcat], axis=1)

        # convert to float array
        X = Xraw.to_numpy(dtype=float, copy=True)

        return X, Xraw.columns  # return names for debugging if needed

    for c, g in out.groupby(cond):
        # subset to rows we want to compute residuals for
        if flag_col in g.columns:
            keep = (g[flag_col] == flag_keep).values
        else:
            keep = np.ones(len(g), dtype=bool)

        if not np.any(keep):
            continue

        g_keep = g.loc[g.index[keep]]
        if g_keep.shape[0] < 2:
            continue

        # build X and y
        X, _ = build_X(g_keep)
        yv = g_keep[y].to_numpy(dtype=float)

        # finite mask
        mask = np.isfinite(yv) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < 2:
            continue

        Xm = X[mask].copy()
        ym = yv[mask].copy()

        # optional rank transforms
        if rank_covariates:
            # rank each column separately within this (condition × keep × finite) subset
            for j in range(Xm.shape[1]):
                r = pd.Series(Xm[:, j]).rank(method='average').to_numpy()
                Xm[:, j] = r
        if rank_y:
            ym = pd.Series(ym).rank(method='average').to_numpy()

        # optional standardization of covariates
        if standardize and Xm.shape[0] > 1:
            mu = np.nanmean(Xm, axis=0, keepdims=True)
            sd = np.nanstd(Xm, axis=0, ddof=1, keepdims=True)
            sd[sd == 0] = 1.0
            Xm = (Xm - mu) / sd

        # fit model
        if model == 'ridge':
            reg = Ridge(alpha=ridge_alpha, fit_intercept=include_intercept, max_iter=10000)
        else:
            reg = LinearRegression(fit_intercept=include_intercept)
        reg.fit(Xm, ym)

        # predictions & residuals on the finite subset
        yhat = reg.predict(Xm)
        resid = ym - yhat

        # write back
        idx_write = g_keep.index[mask]
        out.loc[idx_write, 'EV_resid'] = resid
        if add_predictions:
            out.loc[idx_write, 'EV_hat'] = yhat

    return out