
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
import scipy.stats as stats
import os
import random
results_dir = os.path.join(project_root,'results/fig_2/')

previous_results_dir = os.path.join(project_root,'results/fig_2/')

# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)

import sys
sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)
from utils.neuron_properties_functions import store_mouse_alphas, get_evars_all_mice,create_empty_mouse_stats_dict, get_split_half_r_all_mice, get_SNR_all_mice
from utils.neuron_properties_functions import  get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates, create_empty_monkey_stats_dict, get_evar_monkey_all_dates, store_macaque_alphas




#################################### MOUSE PREDICTIONS ####################################

# mouse predictions 
#create empty mouse stat directory and store SNR, split-half r val, and inter-area prediction performance


mouse_stats= create_empty_mouse_stats_dict(main_dir)
get_SNR_all_mice(main_dir, mouse_stats)
get_split_half_r_all_mice(main_dir, mouse_stats)

store_mouse_alphas(main_dir, mouse_stats, activity_type='resp', verbose=True)
get_evars_all_mice(main_dir, mouse_stats, activity_type='resp') #stimulus activity
#shuffle frames for comparison EV to null
get_evars_all_mice(main_dir, mouse_stats, activity_type='resp', control_shuffle=True) 


with open(results_dir+ f'mouse_stats.pkl', 'wb') as handle:
	pickle.dump(mouse_stats, handle)

################################# MACAQUE PREDICTIONS ####################################

#depending on the dataset type, there are different times of autocorrelation to mitigate
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
                    'D': ['SNR'],}
date_used_dict = {'L': '090817', 'A': '041018', 'D': '250225'}
condition_type_used_dict = {'L':'RS','A':'SNR','D':'RS'}

monkey_names = ['L','A','D' ]
for monkey_name in monkey_names:
    previous_monkey_stats_path = os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')
    if not os.path.exists(previous_monkey_stats_path):
        monkey_stats=create_empty_monkey_stats_dict(monkey=monkey_name)
        get_split_half_r_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=dataset_type_dict[monkey_name])
        get_SNR_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=dataset_type_dict[monkey_name])
        store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], 
                        condition_type_used=condition_type_used_dict[monkey_name])
    else:
        print('retrieved info from fig 2')
        with open(previous_monkey_stats_path, 'rb') as handle:
            monkey_stats = pickle.load(handle)
    get_evar_monkey_all_dates(monkey_stats, control_shuffle=False, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])
    get_evar_monkey_all_dates(monkey_stats, control_shuffle=True, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])
    with open(os.path.join(results_dir + f'monkey_{monkey_name}_stats.pkl'), 'wb') as file:
        pickle.dump(monkey_stats, file)

######################## poisson glm predictions for figure 2 all monkeys ##########################

prediction_type = 'poisson_glm'
monkey_names = ['L','A','D' ]

for monkey_name in monkey_names:
    previous_monkey_stats_path = os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')
    with open(previous_monkey_stats_path, 'rb') as handle:
        monkey_stats = pickle.load(handle)
    if f'monkey_alphas_{prediction_type}' in monkey_stats.keys():
        del monkey_stats[f'monkey_alphas_{prediction_type}']
    store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name],
                        prediction_type=prediction_type)
    with open(os.path.join(results_dir + f'monkey_{monkey_name}_stats.pkl'), 'wb') as file:
        pickle.dump(monkey_stats, file)
    print(f'stored monkey alphas {prediction_type} for monkey ', monkey_name)
    dataset_types = dataset_type_dict[monkey_name]
    for dt in dataset_types:
        get_evar_monkey_all_dates(monkey_stats, control_shuffle=False, monkey=monkey_name, 
                            dataset_types=[dt], prediction_type=prediction_type)
        get_evar_monkey_all_dates(monkey_stats, control_shuffle=True, monkey=monkey_name, 
                            dataset_types=[dt], prediction_type=prediction_type)
        with open(os.path.join(results_dir + f'monkey_{monkey_name}_stats.pkl'), 'wb') as file:
            pickle.dump(monkey_stats, file)
        print(f'stored monkey evars {prediction_type} for monkey ', monkey_name, ' dataset type ', dt)
        print(f'saved monkey stats evars {prediction_type} control_shuffle False to fig 2 results dir for monkey {monkey_name} dataset type {dt}')
    
    get_evar_monkey_all_dates(monkey_stats, control_shuffle=False, monkey=monkey_name, 
                            dataset_types=['SNR_spont'])
    get_evar_monkey_all_dates(monkey_stats, control_shuffle=True, monkey=monkey_name, 
                            dataset_types=['SNR_spont'])
    get_evar_monkey_all_dates(monkey_stats, control_shuffle=False, monkey=monkey_name, 
                            dataset_types=['SNR_spont'], prediction_type=prediction_type)
    get_evar_monkey_all_dates(monkey_stats, control_shuffle=True, monkey=monkey_name, 
                        dataset_types=['SNR_spont'], prediction_type=prediction_type)
    with open(os.path.join(results_dir + f'monkey_{monkey_name}_stats.pkl'), 'wb') as file:
        pickle.dump(monkey_stats, file)


# # ############################ subsampling monkey L to match monkey A and D ############################

results_dir = os.path.join(project_root,'results/fig_2/')
import json
import time
with open(os.path.join(results_dir, 'subsample_seeds.json'), 'r') as f:
    subsample_seeds = json.load(f)

main_monkey_name = 'L'
subsample_monkey_names = ['A', 'D']

start_time = time.time()


for subsample_monkey_name in subsample_monkey_names:
    print('Subsampling monkey L to match monkey ', subsample_monkey_name)
    for seed in subsample_seeds:
        subsampled_monkey_stats_path = os.path.join(project_root,f'results/fig_2/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
        with open (subsampled_monkey_stats_path, 'rb') as handle:
            subsampled_monkey_stats = pickle.load(handle)
        subsampled_indices = True
        date_used  = list(subsampled_monkey_stats['SNR'].keys())[0]
        datataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm']]
        store_macaque_alphas(main_dir, subsampled_monkey_stats, verbose=True, monkey=main_monkey_name, date_used=date_used, 
                            condition_type_used='SNR',subsampled_indices=subsampled_indices)
        get_evar_monkey_all_dates(subsampled_monkey_stats, control_shuffle=False, 
                                monkey=main_monkey_name, dataset_types=datataset_types_, subsampled_indices=subsampled_indices)
        get_evar_monkey_all_dates(subsampled_monkey_stats, control_shuffle=True, 
                                monkey=main_monkey_name, dataset_types=datataset_types_, subsampled_indices=subsampled_indices)
        with open(subsampled_monkey_stats_path, 'wb') as file:
            pickle.dump(subsampled_monkey_stats, file)

        print('saved subsampled monkey stats evars for monkey ', main_monkey_name, ' subsampled to ', subsample_monkey_name, ' seed ', seed)
    print('Finished subsampling monkey L to match monkey ', subsample_monkey_name)
end_time = time.time()
#print time spent in minutes
print('Time spent subsampling monkey L to match monkeys A and D: ', (end_time - start_time)/60, ' minutes')

