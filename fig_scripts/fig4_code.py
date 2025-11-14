from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import sys
import os 

previous_results_dir = os.path.join(project_root,'results/fig_3/')
results_dir = os.path.join(project_root,'results/fig_4/')


# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)

import sys
import json
import time
sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)

import utils.mouse_data_functions as cs
from utils.fig_4_functions import get_control_evars
from utils.neuron_properties_functions import create_empty_monkey_stats_dict, get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates, get_max_corr_vals_monkey_all_dates, store_macaque_alphas
from utils.neuron_properties_functions import create_empty_mouse_stats_dict, get_split_half_r_all_mice, get_SNR_all_mice, get_max_corr_vals_all_mice
from utils.macaque_data_functions import get_resps, get_get_condition_type



################################## MOUSE PREDICTIONS ####################################

## store the lengths of the neurons and frames


if not os.path.exists(os.path.join(previous_results_dir, 'mouse_stats.pkl')):
	mouse_stats= create_empty_mouse_stats_dict(main_dir)
	get_SNR_all_mice(main_dir, mouse_stats)
	get_split_half_r_all_mice(main_dir, mouse_stats)
	get_max_corr_vals_all_mice(main_dir, mouse_stats)
else:
	with open(os.path.join(previous_results_dir, 'mouse_stats.pkl'), 'rb') as handle:
		mouse_stats = pickle.load(handle)

seed = 17
dataset_types = ['ori32','natimg32']
area='L23'
area2='L4'
activity_type = 'resp'
L23_lengths = []
L4_lengths = []
trial_lengths = []
for dataset_type in dataset_types:
    mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
    mousenames= sorted(mt.filenames)
    for mouse in mousenames:
        resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse)
        L23_lengths.append(len(mouse_stats[dataset_type][mouse]['L23']['SNR_meanspont']))
        L4_lengths.append(len(mouse_stats[dataset_type][mouse]['L4']['SNR_meanspont']))
        trial_lengths.append(resp_L23.shape[0])

L23_control_lengths = L23_lengths[np.argsort(np.array(L23_lengths))[1]]
L4_control_lengths = L4_lengths[np.argsort(np.array(L4_lengths))[1]]
control_trial_length = np.min(trial_lengths)


get_control_evars(mouse_stats,input_control_lengths=L4_control_lengths,pred_control_lengths= L23_control_lengths,
                control_trial_length=control_trial_length,animal='mouse', area=area,area2=area2,dataset_types=dataset_types,
                num_seeds=10, shuffle_frames=False, seed=20)

get_control_evars(mouse_stats,input_control_lengths=L4_control_lengths,pred_control_lengths= L23_control_lengths,
                control_trial_length=control_trial_length,animal='mouse', area=area,area2=area2,dataset_types=dataset_types,
                num_seeds=10, shuffle_frames=True, seed=20)

# save the mouse stats
with open(os.path.join(results_dir, 'mouse_stats.pkl'), 'wb') as handle:
    pickle.dump(mouse_stats, handle)

################################## MACAQUE PREDICTIONS ####################################


all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200, 'RF_large_spont':200}


V4_lengths = []
V1_lengths = []
trial_lengths = []

area='V4'
area2='V1'
dataset_types = ['SNR', 'RF_thin', 'RF_large']
w_size=25

monkey_names = ['A','L']
date_used_dict = {'L': '090817', 'A': '290818'}
condition_type_used_dict = {'L':'RS','A':'RF_thin'}

for monkey_name in monkey_names:
    print(f'Processing monkey {monkey_name}')
    if not os.path.exists(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')):
        monkey_stats= create_empty_monkey_stats_dict(monkey=monkey_name)
        get_SNR_monkey_all_dates(monkey_stats, monkey=monkey_name)
        get_split_half_r_monkey_all_dates(monkey_stats, monkey=monkey_name)
        get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name)
        store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name])

    else:
        with open(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
            monkey_stats = pickle.load(handle)

    for dataset_type in dataset_types:
        for date in monkey_stats[dataset_type]:
            if date in ['140819', '150819', '160819']:
                continue
            V4_lengths.append(len(monkey_stats[dataset_type][date][area]['SNR_meanspont']))
            V1_lengths.append(len(monkey_stats[dataset_type][date][area2]['SNR_meanspont']))
            resp_V4, resp_V1= get_resps(condition_type=get_get_condition_type(dataset_type), 
                            date=date, w_size=w_size, stim_on=0, 
                            stim_off=all_ini_stim_offs[dataset_type], monkey=monkey_name)
            trial_lengths.append(resp_V4.shape[0])
    V4_control_lengths = V4_lengths[np.argsort(np.array(V4_lengths))[0]] #control for number of neurons. then the next smallest sample?
    V1_control_lengths = V1_lengths[np.argsort(np.array(V1_lengths))[0]]
    print(f'V4 control lengths: {V4_control_lengths}, V1 control lengths: {V1_control_lengths}')
    control_trial_length = np.min(trial_lengths)
    print(f'Control trial length: {control_trial_length}')

    get_control_evars(monkey_stats,input_control_lengths=V1_control_lengths,pred_control_lengths= V4_control_lengths,
                    control_trial_length=control_trial_length,animal='monkey', area=area,area2=area2,dataset_types=dataset_types,
                    num_seeds=10, shuffle_frames=False, seed=20, monkey=monkey_name)


    get_control_evars(monkey_stats,input_control_lengths=V1_control_lengths,pred_control_lengths= V4_control_lengths,
                    control_trial_length=control_trial_length,animal='monkey', area=area,area2=area2,dataset_types=dataset_types,
                    num_seeds=10, shuffle_frames=True, seed=20, monkey=monkey_name)
    # save the monkey stats
    with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
        pickle.dump(monkey_stats, handle)
        
############################### subsample monkey L to match monkey A ###############################



with open(os.path.join(results_dir, 'subsample_seeds.json'), 'r') as f:
    subsample_seeds = json.load(f)

main_monkey_name = 'L'
subsample_monkey_name= 'A'
start_time = time.time()
condition_type_used = 'SNR'
subsampled_indices = True

print('Subsampling monkey L to match monkey ', subsample_monkey_name)
for seed in subsample_seeds:
    subsampled_monkey_stats_path = os.path.join(project_root,f'results/fig_4/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
    with open (subsampled_monkey_stats_path, 'rb') as handle:
        subsampled_monkey_stats = pickle.load(handle)
    date_used  = list(subsampled_monkey_stats['SNR'].keys())[0]
    stim_datataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and 'spont' not in k and 'RS' not in k]
    spont_dataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and ('spont' in k or 'RS' in k)]
    
    for dataset_type in dataset_types:
        for date in subsampled_monkey_stats[dataset_type]:
            V4_lengths.append(len(subsampled_monkey_stats[dataset_type][date][area]['SNR_meanspont']))
            V1_lengths.append(len(subsampled_monkey_stats[dataset_type][date][area2]['SNR_meanspont']))
            resp_V4, resp_V1= get_resps(condition_type=get_get_condition_type(dataset_type), 
                            date=date, w_size=w_size, stim_on=0, 
                            stim_off=all_ini_stim_offs[dataset_type], monkey=main_monkey_name) # no need to subsammple neurons since we are just checking trial lengths
            trial_lengths.append(resp_V4.shape[0])
    V4_control_lengths = V4_lengths[np.argsort(np.array(V4_lengths))[0]] #control for number of neurons. then the next smallest sample?
    V1_control_lengths = V1_lengths[np.argsort(np.array(V1_lengths))[0]]
    print(f'V4 control lengths: {V4_control_lengths}, V1 control lengths: {V1_control_lengths}')
    control_trial_length = np.min(trial_lengths)
    print(f'Control trial length: {control_trial_length}')

    get_control_evars(subsampled_monkey_stats,input_control_lengths=V1_control_lengths,pred_control_lengths= V4_control_lengths,
                    control_trial_length=control_trial_length,animal='monkey', area=area,area2=area2,dataset_types=dataset_types,
                    num_seeds=10, shuffle_frames=False, seed=20, monkey=main_monkey_name, subsampled_indices=subsampled_indices)


    get_control_evars(subsampled_monkey_stats,input_control_lengths=V1_control_lengths,pred_control_lengths= V4_control_lengths,
                    control_trial_length=control_trial_length,animal='monkey', area=area,area2=area2,dataset_types=dataset_types,
                    num_seeds=10, shuffle_frames=True, seed=20, monkey=main_monkey_name, subsampled_indices=subsampled_indices)
    # save the monkey stats
    with open(subsampled_monkey_stats_path, 'wb') as handle:
        pickle.dump(subsampled_monkey_stats, handle)
    print(f'Seed {seed} done')
end_time = time.time()
# print time taken in minutes 
print(f'Time taken: {(end_time - start_time)/60:.2f} minutes')