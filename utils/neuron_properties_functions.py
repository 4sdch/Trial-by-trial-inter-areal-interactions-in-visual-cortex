# Description: This file contains functions for analyzing neural data, specifically for mouse and monkey datasets
from joblib import Parallel, delayed
import numpy as np
import copy
import time
import scipy
from scipy import stats
main_dir = ''
func_dir = main_dir + 'utils/'

import sys
sys.path.insert(0,func_dir)
from macaque_data_functions import get_img_resp_avg_sem, get_resps, get_get_condition_type
import mouse_data_functions as cs
from ridge_regression_functions import get_best_alpha_evars, get_predictions_evars_parallel
from glm_prediction_functions import get_glm_predictions_evars_parallel

def create_empty_mouse_stats_dict(main_dir):
    """
    Create an empty dictionary to store mouse statistics.

    Args:
    - main_dir (str): Path to the main directory containing data.

    Returns:
    - mouse_stats (dict): Empty dictionary to store mouse statistics.
    """
    mouse_stats={}
    for dataset_type in ['natimg32','ori32']:
        mouse_stats[dataset_type]={}
        mouse_stats[f'{dataset_type}_spont']={}
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = mt.filenames
        for mouse in mousenames:
            mouse_stats[dataset_type][mouse]={}
            mouse_stats[dataset_type][mouse]['L23']={}
            mouse_stats[dataset_type][mouse]['L4']={}
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('spont', mouse)
            if len(resp_L1)<1000:
                continue
            mouse_stats[f'{dataset_type}_spont'][mouse]={}
            mouse_stats[f'{dataset_type}_spont'][mouse]['L23']={}
            mouse_stats[f'{dataset_type}_spont'][mouse]['L4']={}
    return mouse_stats

def get_SNR_all_mice(main_dir, mouse_stats, dataset_types = ['natimg32','ori32']):
    """
    Compute Signal to Noise Ratio (SNR) for all mice.
    signal to noise ratio is calculated using the average 
    activity in response to stimuli over the average activity
    in response to a gray screen presentation.
    
    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.

    Returns:
    - None
    """
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        for mouse in mousenames:
            mt.mt = mt.mts[mouse]
            resp, spont = mt.add_preprocessing() #retrieve raw activity of all neurons. 
            L1indices, L23indices, L2indices, L3indices, L4indices=mt.get_L_indices() # gets the neuron indices that belong to specific layers
            SNR_mean_over_spont= np.mean(resp, axis=0)/np.mean(spont, axis=0) # do simple resp over spont
            mouse_stats[dataset_type][mouse]['L23']['SNR_meanspont'] = SNR_mean_over_spont[L23indices]
            mouse_stats[dataset_type][mouse]['L4']['SNR_meanspont'] = SNR_mean_over_spont[L4indices]
                        
            
def get_split_half_mean_mouse_seed(s_idx, unique_istims, resp, istim,seed=None):
    """
    Compute split-half reliability for a given seed and stimulus index.

    Args:
    - s_idx (int): Index of the stimulus.
    - unique_istims (numpy.ndarray): Array containing unique stimulus indices.
    - resp (numpy.ndarray): Array containing neural responses.
    - istim (numpy.ndarray): Array containing stimulus indices.
    - seed (int): Seed for random number generation.

    Returns:
    - means_half1 (numpy.ndarray): Mean of responses for the first half.
    - means_half2 (numpy.ndarray): Mean of responses for the second half.
    """
    s = unique_istims[s_idx]
    loc = np.where(istim == s)[0]
    if len(loc) > 1:
        # Randomly split the loc array into two halves
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(loc)
        half_size = len(loc) // 2
        loc_half1 = loc[:half_size]
        loc_half2 = loc[half_size:half_size*2]

        # Compute means for both halves and all neurons at once
        means_half1 = np.nanmean(resp[loc_half1], axis=0)
        means_half2 = np.nanmean(resp[loc_half2], axis=0)
    return means_half1, means_half2

def get_split_half_r_mouse(istim, resp, seed=None):
    """
    Compute split-half reliability for all neurons.

    Args:
    - istim (numpy.ndarray): Array containing stimulus indices.
    - resp (numpy.ndarray): Array containing neural responses.
    - seed (int): Seed for random number generation.

    Returns:
    - scsb (numpy.ndarray): Split-half reliability values for each neuron.
    """
    unique_istims = np.unique(istim)
    num_unique_istims = len(unique_istims)
    num_neurons = resp.shape[1]

    scsb = np.zeros(num_neurons)  # Initialize the results array
    x = np.empty((0, num_neurons))  # Initialize x as an empty 2D array
    y = np.empty((0, num_neurons))  # Initialize y as an empty 2D array

    x, y=[],[]
    results = Parallel(n_jobs=-1)(delayed(get_split_half_mean_mouse_seed)(s_idx, unique_istims,resp, istim, seed) for s_idx in range(num_unique_istims))

    for x_, y_ in results:
        x.append(x_)
        y.append(y_)
    
    x = np.array(x)
    y=np.array(y)

    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    scsb = correlations*2/(1+correlations)

    return scsb

def get_max_corr_vals_all_mice(main_dir, mouse_stats,remove_pcs=False):
    """
    Compute maximum correlation values for all mice.

    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.
    - remove_pcs (bool): Flag indicating whether to remove principal components.

    Returns:
    - None
    """
    for dataset_type in ['natimg32','ori32']:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        rem_pc = ''
        if remove_pcs is True:
            rem_pc = '_removed_32_pcs'
        for mouse in mousenames:
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('resp', mouse, removed_pc=remove_pcs)
            connx_matrix = np.corrcoef(resp_L23.T, resp_L4.T)
            l23_l4_connx = connx_matrix[:resp_L23.shape[1], resp_L23.shape[1]:]
            mouse_stats[dataset_type][mouse]['L23']['max_corr_val' + rem_pc]=np.nanmax(np.abs(l23_l4_connx), axis=1)
            mouse_stats[dataset_type][mouse]['L4']['max_corr_val' + rem_pc]=np.nanmax(np.abs(l23_l4_connx), axis=0)
    for dataset_type in ['natimg32','ori32']:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        for mouse in mousenames:
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('spont', mouse)
            if resp_L1.shape[0]<1000:
                continue
            connx_matrix = np.corrcoef(resp_L23.T, resp_L4.T)
            l23_l4_connx = connx_matrix[:resp_L23.shape[1], resp_L23.shape[1]:]
            mouse_stats[dataset_type + '_spont'][mouse]['L23']['max_corr_val']=np.nanmax(np.abs(l23_l4_connx), axis=1)
            mouse_stats[dataset_type + '_spont'][mouse]['L4']['max_corr_val']=np.nanmax(np.abs(l23_l4_connx), axis=0)

def get_split_half_r_all_mice(main_dir, mouse_stats, remove_pcs=False, dataset_types = ['natimg32','ori32']):
    """
    Compute split-half reliability for all mice.

    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.
    - remove_pcs (bool): Flag indicating whether to remove principal components.

    Returns:
    - None
    """
    rem_pc=''
    if remove_pcs is True:
        rem_pc='_removed_32_pcs'
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        for mouse in mousenames:
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('resp', mouse, removed_pc=remove_pcs)
            istim = mt.istim
            mouse_stats[dataset_type][mouse]['L23']['split_half_r'+ rem_pc]=get_split_half_r_mouse(istim, resp_L23)
            mouse_stats[dataset_type][mouse]['L4']['split_half_r'+ rem_pc]=get_split_half_r_mouse(istim, resp_L4)

def extract_mouse_name(input_string):
    index_of_MP = input_string.find('MP')
    return input_string[index_of_MP:index_of_MP + 5] if index_of_MP != -1 and index_of_MP + 5 <= len(input_string) else None

def store_mouse_alphas(main_dir, mouse_stats, activity_type='resp',n_splits=5, frames_to_reduce=5, sample_size=500, 
                       verbose=False, prediction_type='ridge'):
    # alpha_unique_options = [1e1,5e1,1e2,5e2,1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7]
    area = 'L23'
    area2='L4'
    dataset_types=['ori32','natimg32']
    
    spont_con = ''
    if activity_type =='spont':
        spont_con = '_spont'
    glm_con = ''
    if prediction_type=='poisson_glm':
        glm_con='_glm'
    #get alpha per mouse 
    if f'mouse_alphas{glm_con}' not in list(mouse_stats.keys()):
        mouse_stats[f'mouse_alphas{glm_con}']={sample_size:{}}
        mouse_alphas = {sample_size:{}}
    else:
        mouse_alphas = mouse_stats[f'mouse_alphas{glm_con}']
        if sample_size not in list(mouse_alphas.keys()):
            mouse_alphas[sample_size] = {}
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type) #retrieves neural activity of a certain dataset type stored in data
        for mouse in mouse_stats[dataset_type + spont_con]:
            if extract_mouse_name(mouse) in list(mouse_alphas[sample_size].keys()):
                if verbose:
                    print(f'alpha already stored for {extract_mouse_name(mouse)} in dataset_type: {mouse_alphas[sample_size][extract_mouse_name(mouse)]["dataset_type_used"]}')
            else:
                resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse)
                if resp_L1.shape[0]<1000:
                    # there are some gray screen activity datasets that are too small to fit
                    continue      
                resp_L23 = resp_L23[:sample_size]
                resp_L4 = resp_L4[:sample_size]
                alpha, evars = get_best_alpha_evars(resp_L4, resp_L23, n_splits=n_splits, 
                                                    frames_reduced=frames_to_reduce, 
                                                    alphas=None, prediction_type=prediction_type)
                alpha2, evars2 = get_best_alpha_evars(resp_L23, resp_L4, n_splits=n_splits, 
                                                    frames_reduced=frames_to_reduce, 
                                                    alphas=None, prediction_type=prediction_type)
                mouse_alphas[sample_size][(extract_mouse_name(mouse))]={area:alpha, area2:alpha2, 'dataset_type_used':dataset_type + spont_con}
                if verbose:
                    print(f'alpha{glm_con} for {mouse} {activity_type} calculated and stored. Will be used in other datasets of the same mouse')
    mouse_stats[f'mouse_alphas{glm_con}'] = mouse_alphas

def get_evars_all_mice(main_dir, mouse_stats, activity_type='resp',n_splits=10, frames_to_reduce=5,
                        control_shuffle=False, remove_pcs=False,sample_size=500, prediction_type='ridge'):
    """
    Compute explained variance for all mice.

    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.
    - activity_type (str): Type of neural activity ('resp' or 'spont').
    - n_splits (int): Number of splits for cross-validation.
    - frames_to_reduce (int): Number of frames to reduce.
    - control_shuffle (bool): Flag indicating whether to shuffle for control.
    - remove_pcs (bool): Flag indicating whether to remove principal components.

    Returns:
    - None
    """
    start_time = time.time()
    area = 'L23'
    area2='L4'
    dataset_types=['ori32','natimg32']
    control_con = ''
    rem_pc = ''
    spont_con = ''
    glm_con = ''
    pred_func = get_predictions_evars_parallel
    if control_shuffle is True:
        control_con = '_null'
    if remove_pcs is True:
        rem_pc = '_removed_32_pcs'
    if activity_type =='spont':
        spont_con = '_spont'
    if prediction_type=='poisson_glm':
        glm_con='_glm'
        pred_func = get_glm_predictions_evars_parallel
    elif prediction_type!='ridge':
        raise ValueError('prediction type not recognized. use ridge or poisson_glm')
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type) #retrieves neural activity stored in data
        for mouse in mouse_stats[dataset_type + spont_con]:
            
            alpha = mouse_stats[f'mouse_alphas{glm_con}'][sample_size][(extract_mouse_name(mouse))][area]
            alpha2 = mouse_stats[f'mouse_alphas{glm_con}'][sample_size][(extract_mouse_name(mouse))][area2]
            
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse,removed_pc=remove_pcs)
            if resp_L1.shape[0]<1000:
                # there are some gray screen activity datasets that are too small to fit
                continue      
            if mouse_stats[f'mouse_alphas{glm_con}'][sample_size][(extract_mouse_name(mouse))]['dataset_type_used']==dataset_type + spont_con:
                resp_L23=resp_L23[sample_size:]
                resp_L4=resp_L4[sample_size:]
                
            _, evars = pred_func(resp_L4, resp_L23, alpha=alpha,n_splits=n_splits, 
                                                frames_reduced=frames_to_reduce, control_shuffle=control_shuffle,
                                                )
            _, evars2 = pred_func(resp_L23, resp_L4, alpha=alpha2, n_splits=n_splits, 
                                                frames_reduced=frames_to_reduce,control_shuffle=control_shuffle,
                                                )
            mouse_stats[dataset_type + spont_con][mouse][area]['evars' + control_con + rem_pc + glm_con]=evars
            mouse_stats[dataset_type + spont_con][mouse][area2]['evars' + control_con + rem_pc + glm_con]=evars2
        print(dataset_type, 'done')
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'yay! it took {elapsed_time:.2f} minutes to finish all dataset types!')


condition_types =['SNR', 'SNR_spont', 'RS', 'RS_open', 'RS_closed', 
                'RF_thin', 'RF_large', 'RF_thin_spont', 'RF_large_spont']
def get_dates(condition_type, monkey='L'):
    if monkey=='L':
        if 'SNR' in condition_type or 'RS' in condition_type:
            return ['090817', '100817', '250717']
        elif 'large' in condition_type:
            return ['260617']
        else:
            return ['280617']
    elif monkey=='A':
        if 'SNR' in condition_type:
            return ['041018','140819', '150819', '160819']
        elif 'RS' in condition_type:
            return ['140819', '150819', '160819']
        elif 'large' in condition_type:
            return ['280818']
        else:
            return ['290818']
    elif monkey=='D':
        return ['250225','260225']
    else:
        raise ValueError('monkey not found')
    
def create_empty_monkey_stats_dict(monkey='L'):
    """
    Create an empty dictionary for monkey statistics.

    Returns:
    - monkey_stats (dict): Empty dictionary for monkey statistics.
    """
    monkey_stats={}
    for dataset_type in ['SNR', 'SNR_spont','RS','RS_open','RS_closed','RF_thin','RF_thin_spont','RF_large','RF_large_spont']:
        monkey_stats[dataset_type]={}
        dates = get_dates(dataset_type,monkey)
        for date in dates:
            monkey_stats[dataset_type][date]={}
            monkey_stats[dataset_type][date]['V4']={}
            monkey_stats[dataset_type][date]['V1']={}
    return monkey_stats


def get_split_half_shape_monkey_seed(resp_array, condition_type, subsample_size=20, step_size=None, rebin=True):
    if rebin is True:
        binned_epochs = get_img_resp_avg_sem(resp_array, condition_type=condition_type, get_chunks=True, step_size=step_size)
    else:
        binned_epochs = resp_array
    all_epoch_indices = np.arange(len(binned_epochs))
    epoch_indices = np.random.choice(all_epoch_indices, subsample_size)
    half_size = len(epoch_indices) // 2
    x=binned_epochs[epoch_indices[:half_size]].mean(axis=0)
    y=binned_epochs[epoch_indices[half_size:half_size*2]].mean(axis=0)
    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    return correlations*2/(1+correlations)

def get_split_half_r_monkey(resp_array, condition_type, n_perms=1000, step_size=None, rebin=True):
    
    results = Parallel(n_jobs=-1)(delayed(get_split_half_shape_monkey_seed)(resp_array, condition_type, step_size=step_size, rebin=rebin) for p in range(n_perms))
    v_elec_mean_rs = np.array(results).mean(axis=0)
    return v_elec_mean_rs

def get_split_half_shape_monkey_RF_seed(resp_array, cond_labels, condition_type, step_size=None):
    all_x = []
    all_y = []

    binned_epochs = get_img_resp_avg_sem(resp_array, condition_type=condition_type, get_chunks=True, step_size=step_size)
    binned_labels = cond_labels[:,0,0]

    for cond_num in range(len(np.unique(binned_labels))):
        stim_epochs = binned_epochs[np.argwhere(binned_labels==cond_num)[:, 0]]
        epoch_indices = np.arange(len(stim_epochs))
        np.random.shuffle(epoch_indices)
        half_size = len(epoch_indices) // 2
        all_x.append(stim_epochs[epoch_indices[:half_size]].mean(axis=0))
        all_y.append(stim_epochs[epoch_indices[half_size:half_size*2]].mean(axis=0))
    
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # We now have x and y being shaped [40 timepoints_per_epoch * 4 labels, n_neurons]
    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    return correlations*2/(1+correlations)

def get_split_half_r_monkey_RF(resp_array, cond_labels, condition_type, n_perms=100, step_size=None):
    v_elec_rs = []
    results = Parallel(n_jobs=-1)(delayed(get_split_half_shape_monkey_RF_seed)(resp_array, cond_labels, condition_type, step_size=step_size) for p in range(n_perms))
    for corr in results:
        v_elec_rs.append(corr)
    return np.array(v_elec_rs).mean(axis=0)

#depending on the dataset type, there are different times of autocorrelation to mitigate
all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
#different stimulus presentaion types have different durations
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 300, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':300, 
                    'RF_large_spont':300}

def get_split_half_r_monkey_all_dates(monkey_stats, w_size=25, monkey='L', specific_dataset_types=['SNR', 'RF_thin','RF_large'], step_size=None, ponce_step_size=None):
    """
    Compute split-half reliability for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    area='V4'
    area2='V1'
    if monkey=='D':
        step_size=ponce_step_size
    for dataset_type in specific_dataset_types:
        dates = get_dates(dataset_type, monkey)
        print(f'all dates for {dataset_type} are: {dates}')
        # print(dates)
        for date in dates:
            if 'RF' in dataset_type:
                resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], 
                                                        get_RF_labels=True, monkey=monkey)
                monkey_stats[dataset_type][date][area]['split_half_r']=get_split_half_r_monkey_RF(resp_V4, cond_labels,dataset_type, step_size=step_size)
                monkey_stats[dataset_type][date][area2]['split_half_r']=get_split_half_r_monkey_RF(resp_V1, cond_labels, dataset_type, step_size=step_size)
            else:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], 
                                            monkey=monkey)
                monkey_stats[dataset_type][date][area]['split_half_r']=get_split_half_r_monkey(resp_V4, dataset_type, step_size=step_size)
                monkey_stats[dataset_type][date][area2]['split_half_r']=get_split_half_r_monkey(resp_V1, dataset_type, step_size=step_size)
            print(date, 'split-half r done')

def get_SNR_monkey(binned_resp, binned_spont):
    baseline_stack_avg = np.mean(binned_spont, axis=0)
    baseline_avg = np.mean(baseline_stack_avg, axis=0)
    baseline_std = np.std(baseline_stack_avg, axis=0)
    print("Channels with zero std in baseline:", np.where(baseline_std == 0)[0])
    print("Number of such channels:", np.sum(baseline_std == 0))

    MUA_avg = np.mean(binned_resp, axis=0)
    window = 20  # hard-coded
    mask = np.ones((window)) / window
    MUA_sm = scipy.ndimage.convolve1d(MUA_avg, mask, axis=0)

    # get max of absolute value of smoothed MUA
    MUA_max = np.max(np.abs(MUA_sm), axis=0)
    
    # Calculate channel Signal to Noise Ratio (SNR)
    SNR = (MUA_max - baseline_avg) / baseline_std
    return SNR

def get_SNR_monkey_RF(binned_resp_, binned_spont_, cond_labels):

    binned_labels = cond_labels[:,0,0]
    resp_list, spont_list = [],[]
    for cond_num in range(len(np.unique(binned_labels))):
        stim_epochs = binned_resp_[np.argwhere(binned_labels==cond_num)[:, 0]]
        stim_spont_epochs = binned_spont_[np.argwhere(binned_labels==cond_num)[:, 0]]
        resp_list.append(np.mean(stim_epochs, axis=0))
        spont_list.append(np.mean(stim_spont_epochs, axis=0))
        

    baseline_stack_avg = np.concatenate(spont_list, axis=0)
    baseline_avg = np.mean(baseline_stack_avg, axis=0) #overall mean of all periods 
    baseline_std = np.std(baseline_stack_avg, axis=0)
    
    MUA_avg = np.concatenate(resp_list, axis=0) #instead of averaging all directions, concatenate them all into a 4 second average
    window = 20  # hard-coded
    mask = np.ones((window)) / window
    MUA_sm = scipy.ndimage.convolve1d(MUA_avg, mask, axis=0)
    MUA_max = np.max(MUA_sm, axis=0)
    
    # Calculate channel Signal to Noise Ratio (SNR)
    SNR = (MUA_max - baseline_avg) / baseline_std
    return SNR

def get_SNR_monkey_all_dates(monkey_stats, w_size=1, specific_dataset_types = ['SNR','RF_large','RF_thin'], monkey='L'):
    """
    Compute SNR for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    area='V4'
    area2='V1'
    
    donot_smooth = True if monkey=='D' else False
    for dataset_type in specific_dataset_types:
        dates = get_dates(dataset_type, monkey=monkey)
        print(f'all dates for {dataset_type} are: {dates}')
        for date in dates:
            if 'RF' in dataset_type:
                resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, 
                                                        stim_off=all_ini_stim_offs[dataset_type], raw_resp=True,return_binned=True, 
                                                        get_RF_labels=True, monkey=monkey,donot_smooth=donot_smooth)
                spont_V4, spont_V1 =get_resps(condition_type=get_get_condition_type(dataset_type)+'_spont', date=date, w_size=w_size, 
                                            stim_off=all_ini_stim_offs[dataset_type], raw_resp=True, return_binned=True, monkey=monkey,
                                            donot_smooth=donot_smooth)
                monkey_stats[dataset_type][date][area]['SNR_meanspont']=get_SNR_monkey_RF(resp_V4, spont_V4, cond_labels)
                monkey_stats[dataset_type][date][area2]['SNR_meanspont']=get_SNR_monkey_RF(resp_V1, spont_V1, cond_labels)
            else:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, 
                                            stim_off=all_ini_stim_offs[dataset_type], raw_resp=True,return_binned=True, monkey=monkey,
                                            donot_smooth=donot_smooth)
                spont_V4, spont_V1 =get_resps(condition_type=get_get_condition_type(dataset_type)+'_spont', date=date, 
                                            w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], raw_resp=True,return_binned=True, monkey=monkey,
                                            donot_smooth=donot_smooth)
                monkey_stats[dataset_type][date][area]['SNR_meanspont']=get_SNR_monkey(resp_V4, spont_V4)
                monkey_stats[dataset_type][date][area2]['SNR_meanspont']=get_SNR_monkey(resp_V1, spont_V1)
            print(date, 'SNR calculation done')
def get_max_corr_vals_monkey_all_dates(monkey_stats, w_size=25, monkey='L', dataset_types=['SNR','RF_thin','RF_large'], subsampled_indices = False):
    area='V4'
    area2='V1'
    for dataset_type in dataset_types:
        if 'monkey' in dataset_type:
            continue
        for date in monkey_stats[dataset_type]:
            if date in ['140819','150819','160819']:
                continue
            if subsampled_indices is True:
                assert monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                assert monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                subsampled_indices_dict = {'V1':monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'],
                                            'V4':monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices']}
            resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], monkey=monkey)
            if subsampled_indices is True:
                resp_V4 = resp_V4[:, subsampled_indices_dict['V4']]
                resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]
            
            connx_matrix = np.corrcoef(resp_V4.T, resp_V1.T)
            v4_v1_connx = connx_matrix[:resp_V4.shape[1], resp_V4.shape[1]:]
            monkey_stats[dataset_type][date][area]['max_corr_val'] = np.nanmax(np.abs(v4_v1_connx), axis=1)
            monkey_stats[dataset_type][date][area2]['max_corr_val'] = np.nanmax(np.abs(v4_v1_connx), axis=0)

### monkey

def get_1_vs_all_scsb_monkey_1trial(trial_no, binned_epochs):
    # Compute means for both halves and all neurons at once
    x= binned_epochs[trial_no]
    bulk_half = np.delete(binned_epochs, trial_no, axis=0)
    y = np.nanmean(bulk_half, axis=0)
    
    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    # no corrections

    return correlations



def get_1_vs_rest_r_monkey(binned_epochs):
    n_trials = len(binned_epochs)
    results = Parallel(n_jobs=-1)(delayed(get_1_vs_all_scsb_monkey_1trial)(trial_no, binned_epochs) for trial_no in range(n_trials))
    scsbs = []
    for sc in results:
        scsbs.append(sc)
    scsb = np.mean(np.array(scsbs), axis=0)

    return scsb

def get_1_vs_all_scsb_monkey_RF_1trial(binned_labels, binned_epochs, trial_no, trial_avg=False):
    x, y = [],[]
    for cond_num in range(len(np.unique(binned_labels))):
        loc = np.argwhere(binned_labels==cond_num)[:, 0]
        x.append(binned_epochs[loc[trial_no]])
        y.append(np.nanmean(binned_epochs[np.delete(loc, trial_no)],axis=0))

    if trial_avg is True:
        x = np.array(x).mean(axis=1)
        y = np.array(y).mean(axis=1)
    else:
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    # no correction 
    return correlations

def get_min_trials(binned_labels):
    trial_nos=[]
    for cond_num in range(len(np.unique(binned_labels))):
        loc = np.argwhere(binned_labels==cond_num)[:, 0]
        trial_nos.append(len(loc))
    return min(trial_nos)


def get_1_vs_rest_r_monkey_RF(resp_array, cond_labels, date, condition_type, trial_avg=False):
    scsbs = []

    binned_epochs = get_img_resp_avg_sem(resp_array, condition_type=condition_type, get_chunks=True)
    binned_labels = cond_labels[:,0,0]

    n_trials = get_min_trials(binned_labels)
    results = Parallel(n_jobs=-1)(delayed(get_1_vs_all_scsb_monkey_RF_1trial)(binned_labels, binned_epochs, trial_no, trial_avg) for trial_no in range(n_trials))

    for sc in results:
        scsbs.append(sc)
    
    return np.mean(np.array(scsbs), axis=0)

def get_property_dataset_type_monkey(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string 

def get_one_vs_rest_r_monkey_all_dates(monkey_stats, w_size=25, monkey='L'):
    """
    Compute 1 vs. rest reliability for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    area='V4'
    area2='V1'
    for dataset_type in ['RF_large','SNR','RF_thin']:
        dates = get_dates(dataset_type, monkey=monkey)
        for date in dates:
            if date in ['140819','150819','160819']:
                continue
            if 'RF' in dataset_type:
                resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], get_RF_labels=True, monkey=monkey)
                monkey_stats[dataset_type][date][area]['1_vs_rest_r']=get_1_vs_rest_r_monkey_RF(resp_V4, cond_labels, date, dataset_type)
                monkey_stats[dataset_type][date][area2]['1_vs_rest_r']=get_1_vs_rest_r_monkey_RF(resp_V1, cond_labels, date, dataset_type)
            else:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], monkey=monkey)
                binned_epochs = get_img_resp_avg_sem(resp_V4, condition_type=dataset_type, get_chunks=True)  
                monkey_stats[dataset_type][date][area]['1_vs_rest_r']=get_1_vs_rest_r_monkey(binned_epochs)
                
                binned_epochs = get_img_resp_avg_sem(resp_V1, condition_type=dataset_type, get_chunks=True)  # accidentally used resp_V4, changed 21May2025
                monkey_stats[dataset_type][date][area2]['1_vs_rest_r']=get_1_vs_rest_r_monkey(binned_epochs)

def store_macaque_alphas(main_dir, monkey_stats,n_splits=5, w_size=25, 
                        sample_size=500, verbose=False, condition_type_used='RS', 
                        date_used = '090817', monkey = 'L', silence=None, prediction_type='ridge',
                        subsampled_indices = False, optimize_visually_responses=False):
    if subsampled_indices is True:
        assert monkey_stats[condition_type_used][date_used]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
        assert monkey_stats[condition_type_used][date_used]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
        subsampled_indices_dict = {'V1':monkey_stats[condition_type_used][date_used]['V1'][f'monkey_{monkey}_subsample_indices'],
                                    'V4':monkey_stats[condition_type_used][date_used]['V4'][f'monkey_{monkey}_subsample_indices']}
    start_time = time.time()
    # alpha_unique_options = [1e1,5e1,1e2,5e2,1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7
    alpha_unique_options = None
    alpha_unique_options = np.logspace(-1, 4, 25)# from 10^0 to 10^3, 15 values
    area='V4'
    area2='V1'
    
    glm_con = ''
    if prediction_type=='poisson_glm':
        glm_con='_glm'
    elif prediction_type =='gamma_glm':
        glm_con='_gamma_glm'
    #get alpha per monkey 
    if f'monkey_alphas{glm_con}' not in list(monkey_stats.keys()):
        monkey_stats[f'monkey_alphas{glm_con}']={sample_size:{}}
        monkey_alphas = {sample_size:{}}
    else:
        monkey_alphas = monkey_stats[f'monkey_alphas{glm_con}']
        if sample_size not in list(monkey_alphas.keys()):
            monkey_alphas[sample_size] = {}
    if optimize_visually_responses is True:
        SNRs_V4 = monkey_stats[get_property_dataset_type_monkey(condition_type_used)][date_used]['V4']['SNR_meanspont']
        SNRs_V1 = monkey_stats[get_property_dataset_type_monkey(condition_type_used)][date_used]['V1']['SNR_meanspont']
        relis_V4 = monkey_stats[get_property_dataset_type_monkey(condition_type_used)][date_used]['V4']['split_half_r']
        relis_V1 = monkey_stats[get_property_dataset_type_monkey(condition_type_used)][date_used]['V1']['split_half_r']
        pred_V4_indices = np.where((SNRs_V4>2) & (relis_V4>0.6))[0]
        pred_V1_indices = np.where((SNRs_V1>2) & (relis_V1>0.6))[0]
    if 'V4' in list(monkey_alphas[sample_size].keys()):
        if verbose:
            print('alpha already stored')
    else:
        get_condition_type = get_get_condition_type(condition_type_used)
        if prediction_type == 'ridge':
            resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date_used, 
                                        w_size=w_size,stim_off=all_ini_stim_offs[condition_type_used],monkey=monkey)
            if subsampled_indices is True and subsampled_indices_dict is not None:
                resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
            
        elif prediction_type == 'poisson_glm' or prediction_type == 'gamma_glm':
            resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date_used, 
                                        w_size=w_size,stim_off=all_ini_stim_offs[condition_type_used],monkey=monkey,
                                        subtract_spont_resp=False)
            if subsampled_indices is True and subsampled_indices_dict is not None:
                resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
        resp_V4=resp_V4[:sample_size]
        resp_V1=resp_V1[:sample_size]
        if optimize_visually_responses is True:
            assert len(pred_V4_indices)>1, "not enough V4 visually responsive neurons to optimize alphas"
            assert len(pred_V1_indices)>1, "not enough V1 visually responsive neurons to optimize alphas"
            alpha, evars = get_best_alpha_evars(resp_V1, resp_V4[:,pred_V4_indices], n_splits=n_splits, alphas=alpha_unique_options,
                                                frames_reduced=all_frames_reduced[condition_type_used],silence=silence,
                                                prediction_type=prediction_type)
            alpha2, evars2 = get_best_alpha_evars(resp_V4, resp_V1[:,pred_V1_indices], n_splits=n_splits, alphas=alpha_unique_options,
                                                frames_reduced=all_frames_reduced[condition_type_used], silence=silence,prediction_type=prediction_type)
        
        else:
            alpha, evars = get_best_alpha_evars(resp_V1, resp_V4, n_splits=n_splits, alphas=alpha_unique_options,
                                                    frames_reduced=all_frames_reduced[condition_type_used],silence=silence,
                                                    prediction_type=prediction_type)
            alpha2, evars2 = get_best_alpha_evars(resp_V4, resp_V1, n_splits=n_splits, alphas=alpha_unique_options,
                                                    frames_reduced=all_frames_reduced[condition_type_used], silence=silence,prediction_type=prediction_type)
        
        monkey_alphas[sample_size]= {area:alpha, area2:alpha2, 'condition_type_used':condition_type_used, 'date_used':date_used}
        if verbose:
            print(f'alpha for macaque calculated and stored. Will be used in other datasets of the same monkey')
    monkey_stats[f'monkey_alphas{glm_con}'] = monkey_alphas

import math
def frames_reduced_for(condition: str, bin_ms: int) -> int:
    base = all_frames_reduced[condition]
    gap_ms = base * 25  # because dict is defined at 25 ms/bin
    return max(1, math.ceil(gap_ms / bin_ms))



def get_evar_monkey_all_dates(monkey_stats, w_size=25,n_splits=10, control_shuffle=False, 
                            frame_size=500, monkey='L', dataset_types = ['SNR','RF_thin','RF_large'],
                            prediction_type='ridge', subsampled_indices = False, recording_type = '', frames_reduced=None):
    """
    Compute explained variance for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    
    start_time = time.time()
    area='V4'
    area2='V1'
    
    glm_con = ''
    if prediction_type=='poisson_glm':
        glm_con='_glm'
    elif prediction_type =='gamma_glm':
        glm_con='_gamma_glm'
        
    monkey_alphas=monkey_stats[f'monkey_alphas{glm_con}'][frame_size]
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
            # skip certain dates for now
            if date == '041018' and 'RS' in condition_type:
                continue
            if date in ['140819','150819','160819']:
                continue
            if subsampled_indices is True:
                assert monkey_stats[condition_type][date]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                assert monkey_stats[condition_type][date]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                subsampled_indices_dict = {'V1':monkey_stats[condition_type][date]['V1'][f'monkey_{monkey}_subsample_indices'],
                                            'V4':monkey_stats[condition_type][date]['V4'][f'monkey_{monkey}_subsample_indices']}
        
            get_condition_type = get_get_condition_type(condition_type)
            resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date, 
                                        w_size=w_size,stim_off=all_ini_stim_offs[condition_type], monkey=monkey)
            
            if subsampled_indices is True and subsampled_indices_dict is not None:
                resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
            if condition_type ==monkey_alphas['condition_type_used'] and date==monkey_alphas['date_used']:
                resp_V4=resp_V4[int(frame_size*25/w_size):]
                resp_V1=resp_V1[int(frame_size*25/w_size):]
            if prediction_type == 'ridge':
                _, evars = get_predictions_evars_parallel(resp_V1, resp_V4, n_splits=n_splits, alpha=alpha,
                                                    frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size), control_shuffle=control_shuffle)
                _, evars2 = get_predictions_evars_parallel(resp_V4, resp_V1, n_splits=n_splits, alpha=alpha2,
                                                    frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size), control_shuffle=control_shuffle)
                monkey_stats[condition_type][date][area]['evars'+ w_size_con + control_con]=evars
                monkey_stats[condition_type][date][area2]['evars'+ w_size_con + control_con]=evars2
            elif prediction_type == 'poisson_glm':
                prediction_function = get_glm_predictions_evars_parallel
                resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date, 
                                        w_size=w_size,stim_off=all_ini_stim_offs[condition_type], 
                                        monkey=monkey, subtract_spont_resp=False)
                if subsampled_indices is True and subsampled_indices_dict is not None:
                    resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                    resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
                if condition_type ==monkey_alphas['condition_type_used'] and date==monkey_alphas['date_used']:
                    resp_V4=resp_V4[frame_size:]
                    resp_V1=resp_V1[frame_size:]
                _, evars = prediction_function(resp_V1, resp_V4, n_splits=n_splits,
                                                    frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size), 
                                                    control_shuffle=control_shuffle, verbose=True, alpha=alpha)
                _, evars2 = prediction_function(resp_V4, resp_V1, n_splits=n_splits, 
                                                    frames_reduced=frames_reduced_for(condition_type, bin_ms=w_size), 
                                                    control_shuffle=control_shuffle, verbose=True, alpha=alpha2)
                monkey_stats[condition_type][date][area]['evars' + w_size_con + control_con + glm_con]=evars
                monkey_stats[condition_type][date][area2]['evars' + w_size_con + control_con + glm_con]=evars2
            else:
                raise ValueError('prediction type not recognized')
            print(date,'evar done')
        print(condition_type, 'done')
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'yay! it took {elapsed_time:.2f} minutes to finish all dataset types!')
    
    
    #depending on the dataset type, there are different times of autocorrelation to mitigate
all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
#different stimulus presentaion types have different durations
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 300, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':300, 
                    'RF_large_spont':300}

def get_variance_within_trial_across_timepoints(monkey_stats, w_size=25, 
                                                specific_dataset_types = ['SNR','RF_large','RF_thin'], 
                                                monkey='L', subtract_spont_resp=True, verbose=False,
                                                subsampled_indices = False):
    """
    Compute SNR for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """

    donot_smooth = True if monkey=='D' else False
    for dataset_type in specific_dataset_types:
        dates = get_dates(dataset_type, monkey=monkey)
        print(f'all dates for {dataset_type} are: {dates}')
        for date in dates:
            if date in ['140819','150819','160819']:
                continue
            if subsampled_indices is True:
                assert monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                assert monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                subsampled_indices_dict = {'V1':monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'],
                                            'V4':monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices']}
            if 'RF' in dataset_type:
                resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, 
                                                        stim_off=all_ini_stim_offs[dataset_type], return_binned=True, 
                                                        get_RF_labels=True, monkey=monkey, subtract_spont_resp=subtract_spont_resp, z_score=True)
                if subsampled_indices is True and subsampled_indices_dict is not None:
                    resp_V4 = resp_V4[:, subsampled_indices_dict['V4']]
                    resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]
                unique_labels = np.unique(cond_labels)
                #compute variance across timepoints within a trial, then average across trials of the same condition, then average across conditions
                variances_trials_sites_all_areas = {}
                for area,  resp_area in zip(['V4','V1'],[resp_V4, resp_V1]):
                    variances_wihin_trials = []
                    for label in unique_labels:
                        label_inds = np.where(cond_labels[:,0]==label)[0]
                        var_within_trials = np.nanvar(resp_area[label_inds], axis=1) # n_trials x n_sites
                        mean_var_within_trial = np.nanmean(var_within_trials, axis=0) # n_sites
                        variances_wihin_trials.append(mean_var_within_trial)
                    variances_trials_sites_all_areas[area] = np.array(variances_wihin_trials) # n_stimuli x n_sites
                
            else:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, 
                                                                stim_off=all_ini_stim_offs[dataset_type], return_binned=True, 
                                                                monkey=monkey, subtract_spont_resp=subtract_spont_resp,z_score=True)
                if subsampled_indices is True and subsampled_indices_dict is not None:
                    resp_V4 = resp_V4[:, subsampled_indices_dict['V4']]
                    resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]
                # only one condition
                print('resp_V4 shape..',resp_V4.shape)
                variances_trials_sites_all_areas = {}
                for area,  resp_area in zip(['V4','V1'],[resp_V4, resp_V1]):
                    var_within_trials = np.nanvar(resp_area, axis=1)
                    print('var_within_trials shape:',var_within_trials.shape)
                    mean_var_within_trial = np.nanmean(var_within_trials, axis=0) # n_sites
                    print('number of sites ...',len(mean_var_within_trial))
                    variances_trials_sites_all_areas[area] = mean_var_within_trial
            monkey_stats[dataset_type][date]['V4']['var_within_trial_across_timepoints'] = variances_trials_sites_all_areas['V4']
            monkey_stats[dataset_type][date]['V1']['var_within_trial_across_timepoints'] = variances_trials_sites_all_areas['V1']
            if verbose:
                print('V4',variances_trials_sites_all_areas['V4'])
                print('V1', variances_trials_sites_all_areas['V1'])
        print(f'Finished computing variance within trial across timepoints for {monkey} {dataset_type}')

def get_variance_within_timepoints_across_trials(monkey_stats, w_size=25, specific_dataset_types = ['SNR','RF_large','RF_thin'], 
                                                 monkey='L', subtract_spont_resp=True, verbose=False, subsampled_indices=True):
    """
    Compute variance across trials for each timepoint, then average across timepoints.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    donot_smooth = True if monkey=='D' else False
    for dataset_type in specific_dataset_types:
        dates = get_dates(dataset_type, monkey=monkey)
        print(f'all dates for {dataset_type} are: {dates}')
        for date in dates:
            if date in ['140819','150819','160819']:
                continue
            if subsampled_indices is True:
                assert monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                assert monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                subsampled_indices_dict = {'V1':monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'],
                                            'V4':monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices']}
            if 'RF' in dataset_type:
                resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, 
                                                        stim_off=all_ini_stim_offs[dataset_type], return_binned=True, 
                                                        get_RF_labels=True, monkey=monkey,donot_smooth=donot_smooth, 
                                                        subtract_spont_resp=subtract_spont_resp,z_score=True)
                if subsampled_indices is True and subsampled_indices_dict is not None:
                    resp_V4 = resp_V4[:, subsampled_indices_dict['V4']]
                    resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]
                unique_labels = np.unique(cond_labels)
                #compute variance across trials for each timepoint, then average across timepoints
                variances_timepoints_sites_all_areas = {}
                for area,  resp_area in zip(['V4','V1'],[resp_V4, resp_V1]):
                    variances_across_timepoints = []
                    for label in unique_labels:
                        label_inds = np.where(cond_labels[:,0]==label)[0]
                        var_across_trials = np.var(resp_area[label_inds], axis=0) # n_timepoints x n_sites
                        mean_var_across_timepoints = np.mean(var_across_trials, axis=0) # n_sites
                        variances_across_timepoints.append(mean_var_across_timepoints)
                    variances_timepoints_sites_all_areas[area] = np.array(variances_across_timepoints) # n_stimuli x n_sites
            else:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, 
                                                                stim_off=all_ini_stim_offs[dataset_type], return_binned=True,
                                                                monkey=monkey,donot_smooth=donot_smooth, subtract_spont_resp=subtract_spont_resp,
                                                                z_score=True)
                if subsampled_indices is True and subsampled_indices_dict is not None:
                    resp_V4 = resp_V4[:, subsampled_indices_dict['V4']]
                    resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]
                # only one condition
                variances_timepoints_sites_all_areas = {}
                for area,  resp_area in zip(['V4','V1'],[resp_V4, resp_V1]):
                    var_across_trials = np.var(resp_area, axis=0)# n_timepoints x n_sites
                    mean_var_across_timepoints = np.mean(var_across_trials, axis=0) # n_sites
                    variances_timepoints_sites_all_areas[area] = mean_var_across_timepoints
            monkey_stats[dataset_type][date]['V4']['var_across_trials_within_timepoints'] = variances_timepoints_sites_all_areas['V4']
            monkey_stats[dataset_type][date]['V1']['var_across_trials_within_timepoints'] = variances_timepoints_sites_all_areas['V1']
        print(f'Finished computing variance across trials within timepoints for {monkey} {dataset_type}')
        if verbose:
                print('V4',variances_timepoints_sites_all_areas['V4'])
                print('V1', variances_timepoints_sites_all_areas['V1'])


def get_RF_variance_across_stimuli(monkey_stats, w_size=25, specific_dataset_types = ['RF_large','RF_thin'], monkey='L',
                                    subsampled_indices = False):
    """
    Compute variance across stimuli for RF datasets.Take average response across trials and time for each stimulus, 
    then compute variance across stimuli.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    assert all(dataset_type in ['RF_large','RF_thin'] for dataset_type in specific_dataset_types), "This function is only for RF datasets"
    donot_smooth = True if monkey=='D' else False
    for dataset_type in specific_dataset_types:
        dates = get_dates(dataset_type, monkey=monkey)
        print(f'all dates for {dataset_type} are: {dates}')
        for date in dates:
            
            resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, 
                                                    stim_off=all_ini_stim_offs[dataset_type], return_binned=True, 
                                                    get_RF_labels=True, monkey=monkey,donot_smooth=donot_smooth,
                                                    z_score=True)
            if subsampled_indices is True:
                assert monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                assert monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                subsampled_indices_dict = {'V1':monkey_stats[dataset_type][date]['V1'][f'monkey_{monkey}_subsample_indices'],
                                            'V4':monkey_stats[dataset_type][date]['V4'][f'monkey_{monkey}_subsample_indices']}
                resp_V4 = resp_V4[:, subsampled_indices_dict['V4']]
                resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]
            # cond_label gives the stimulus identity
            unique_labels = np.unique(cond_labels)
            #take average response across trials for each stimulus, then compute variance across stimuli for each timepoint, then average across timepoints
            for area,  resp_area in zip(['V4','V1'],[resp_V4, resp_V1]):
                mean_resps_per_stim = []
                for label in unique_labels:
                    label_inds = np.where(cond_labels[:,0]==label)[0]
                    #average across timepoints then trials
                    stimulus_trials = np.nanmean(resp_area[label_inds],axis=(1,0)) # n_sites
                    mean_resps_per_stim.append(stimulus_trials)	
                mean_resps_per_stim = np.array(mean_resps_per_stim) # n_stimuli x n_sites
                var_across_stimuli = np.nanvar(mean_resps_per_stim, axis=0, ddof=1) # n_sites
                monkey_stats[dataset_type][date][area]['var_across_stimuli'] =var_across_stimuli
        print(f'Finished computing variance across stimuli for {monkey} {dataset_type}')


def get_variance_across_stimuli_all_mice(main_dir, mouse_stats):
	"""
	Compute the variance across stimuli for each neuron across all mice in the dataset.

	Parameters:
	- main_dir (str): The main directory containing the dataset.
	- mouse_stats (dict): A dictionary containing statistics for each mouse in the dataset.
	"""
	for dataset_type in ['natimg32','ori32']:
		mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
		mousenames = list(mouse_stats[dataset_type].keys())
		for mouse in mousenames:
			mt.mt = mt.mts[mouse] 
			resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('resp', mouse)
			istim = mt.istim
			unique_istims = np.unique(istim)
			for area, resp_area in zip(['L23','L4'], [resp_L23, resp_L4]):
				organized_data = []
				for stimulus in unique_istims:
					stimulus_indices = np.where(istim == stimulus)[0] 
					stimulus_trials = resp_area[stimulus_indices] # trials x neurons
					organized_data.append(stimulus_trials)
				
				organized_data = np.array(organized_data) # n_stimuli x n_trials x n_neurons
				
				# Calculate variance of responses for each stimulus and neuron
				mean_responses = np.mean(organized_data, axis=1) # n_stimuli x n_neurons
				variance_across_stimuli = np.var(mean_responses, axis=0) # n_neurons
				mouse_stats[dataset_type][mouse][area]['var_across_stimuli'] = variance_across_stimuli

def get_variance_within_stimulus_across_trials_all_mice(main_dir, mouse_stats):
	"""
	Compute the variance within stimulus across trials for each neuron across all mice in the dataset.

	Parameters:
	- main_dir (str): The main directory containing the dataset.
	- mouse_stats (dict): A dictionary containing statistics for each mouse in the dataset.
	"""
	for dataset_type in ['natimg32','ori32']:
		mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
		mousenames = list(mouse_stats[dataset_type].keys())
		for mouse in mousenames:
			mt.mt = mt.mts[mouse] 
			resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('resp', mouse)
			istim = mt.istim
			unique_istims = np.unique(istim)
			for area, resp_area in zip(['L23','L4'], [resp_L23, resp_L4]):
				organized_data = []
				for stimulus in unique_istims:
					stimulus_indices = np.where(istim == stimulus)[0] 
					stimulus_trials = resp_area[stimulus_indices] # trials x neurons
					organized_data.append(stimulus_trials)
				
				organized_data = np.array(organized_data) # n_stimuli x n_trials x n_neurons
				
				# Calculate variance of responses for each stimulus and neuron
				variance_within_stimuli = []
				for i, stimulus_trials in enumerate(organized_data):
					var_within_stimulus = np.var(stimulus_trials, axis=0) # n_neurons
					variance_within_stimuli.append(var_within_stimulus)
				variance_within_stimuli = np.array(variance_within_stimuli) # n_stimuli x n_neurons
				mean_variance_within_stimuli = np.mean(variance_within_stimuli, axis=0) # n_neurons
				mouse_stats[dataset_type][mouse][area]['var_within_stimulus_across_trials'] = mean_variance_within_stimuli

all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
#different stimulus presentaion types have different durations
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 300, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':300, 
                    'RF_large_spont':300}

dataset_type_dict = {'A': ['SNR','RF_thin', 'RF_large'],
                    'L': ['SNR','RF_thin', 'RF_large'],
                    'D': ['SNR']}
    