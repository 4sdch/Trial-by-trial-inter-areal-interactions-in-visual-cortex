from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)

import os
import sys
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

previous_results_dir = os.path.join(project_root,'results/fig_5/')
results_dir = os.path.join(project_root,'results/fig_6/')

# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)


sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)

from utils.neuron_properties_functions import create_empty_mouse_stats_dict, get_split_half_r_all_mice, get_SNR_all_mice, get_max_corr_vals_all_mice, get_evars_all_mice, store_mouse_alphas
from utils.fig_6_functions import get_norm_variance_all_mice, get_1_vs_rest_all_mice, get_one_vs_rest_r_monkey_all_dates
from utils.neuron_properties_functions import create_empty_monkey_stats_dict, get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates,get_max_corr_vals_monkey_all_dates,get_evar_monkey_all_dates, store_macaque_alphas
import random
from utils.fig_6_functions import get_predictor_indices_elec_ids,get_xtarget_predictor_indices_elecs,get_x_target_overlap_evars, get_electrode_ids_all_dates
from utils.neuron_properties_functions import get_variance_within_trial_across_timepoints, get_variance_within_timepoints_across_trials, get_RF_variance_across_stimuli 
from utils.macaque_data_functions import get_get_condition_type

#################################### MOUSE PREDICTIONS######################################
start_time = time.time()
if not os.path.exists(os.path.join(previous_results_dir, 'mouse_stats.pkl')):
	print('Creating mouse stats')
	mouse_stats= create_empty_mouse_stats_dict(main_dir)
	get_SNR_all_mice(main_dir, mouse_stats)
	get_split_half_r_all_mice(main_dir, mouse_stats)
	get_max_corr_vals_all_mice(main_dir, mouse_stats)
	store_mouse_alphas(main_dir, mouse_stats, activity_type='resp')
	store_mouse_alphas(main_dir, mouse_stats, activity_type='spont')

	#get inter-area predictability 
	get_evars_all_mice(main_dir, mouse_stats)
	get_evars_all_mice(main_dir, mouse_stats, control_shuffle=True)
	get_evars_all_mice(main_dir, mouse_stats, activity_type='spont')
	get_evars_all_mice(main_dir, mouse_stats, activity_type='spont', control_shuffle=True)
else:
    print('Using previous mouse stats')
    with open(os.path.join(previous_results_dir, 'mouse_stats.pkl'), 'rb') as handle:
        mouse_stats = pickle.load(handle)
        
get_norm_variance_all_mice(main_dir, mouse_stats)
get_1_vs_rest_all_mice(main_dir, mouse_stats)

#neuron properties once removing 32 dimensions of gray screen activity
get_split_half_r_all_mice(main_dir, mouse_stats, remove_pcs=True)
get_max_corr_vals_all_mice(main_dir, mouse_stats, remove_pcs=True)
get_1_vs_rest_all_mice(main_dir, mouse_stats, remove_pcs=True)

#removing 32 pcs of gray screen activity
get_evars_all_mice(main_dir, mouse_stats, remove_pcs=True)

end_time = time.time()
elapsed_time = (end_time - start_time)/60

print(f'Yay! work for all mice is completed. Took {elapsed_time:.4f} minutes to complete')
# SAVE MOUSE STATS
with open(os.path.join(results_dir, 'mouse_stats.pkl'), 'wb') as handle:
	pickle.dump(mouse_stats, handle)


    ######################################### MACAQUE NEURAL PROPERTIES ######################################
dataset_type_dict = {'A': ['SNR','RF_thin', 'RF_large'],
                    'L': ['SNR','RF_thin', 'RF_large'],
                    'D': ['SNR']}
spont_dataset_type_dict = {'A': ['SNR_spont','RF_thin_spont', 'RF_large_spont', 'RS','RS_open', 'RS_closed'],
					'L': ['SNR_spont','RF_thin_spont', 'RF_large_spont', 'RS','RS_open', 'RS_closed'],
					'D': ['SNR_spont','RS_open']
					}

monkey_names = ['D']
date_used_dict = {'L': '090817', 'A': '041018', 'D': '250225'}
condition_type_used_dict = {'L':'RS','A':'SNR','D':'RS'}

monkey_names = ['L','A','D']

for monkey_name in monkey_names:
	start_time = time.time()
	print(f'Processing monkey {monkey_name}')
	if not os.path.exists(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')):
		print('Creating monkey stats')
		monkey_stats= create_empty_monkey_stats_dict(monkey=monkey_name)
		get_split_half_r_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=dataset_type_dict[monkey_name])
		get_SNR_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=dataset_type_dict[monkey_name])
		store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name])
		get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name], control_shuffle=False)
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name], control_shuffle=True)
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=spont_dataset_type_dict[monkey_name], control_shuffle=False)
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=spont_dataset_type_dict[monkey_name], control_shuffle=True)
		get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=spont_dataset_type_dict[monkey_name])
	else:
		print(f'Using previous monkey stats')
		with open(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
			monkey_stats = pickle.load(handle)
	get_one_vs_rest_r_monkey_all_dates(monkey_stats, w_size=25, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])
	get_variance_within_trial_across_timepoints(monkey_stats, w_size=25, specific_dataset_types = dataset_type_dict[monkey_name], 
                                             monkey=monkey_name, subtract_spont_resp=True)
	get_variance_within_timepoints_across_trials(monkey_stats, w_size=25, specific_dataset_types = dataset_type_dict[monkey_name], 
                                              monkey=monkey_name, subtract_spont_resp=True)
	if monkey_name in ['L','A']:
		get_RF_variance_across_stimuli(monkey_stats, w_size=25, specific_dataset_types = ['RF_large','RF_thin'], monkey=monkey_name)
	# save macaque stats
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
	end_time = time.time()
	elapsed_time = (end_time - start_time)/60
	print(f'Yay! work for monkey {monkey_name} is completed. Took {elapsed_time:.4f} minutes to complete')


######################################### MACAQUE L RF OVERLAP EVAR COMPARISONS ##############################

num_seeds = 10
random.seed(17)
# Create a list of random seeds
seeds = [random.randint(1, 10000) for _ in range(num_seeds)]

all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200, 'RF_large_spont':200}


monkey_names = ['L']
for monkey_name in monkey_names:
	start_time = time.time()
	monkey_stats_path = os.path.join(project_root, 'results/fig_6',f'monkey_{monkey_name}_stats.pkl')
	with open(monkey_stats_path, 'rb') as handle:
		monkey_stats = pickle.load(handle)
	get_electrode_ids_all_dates(monkey_stats, monkey=monkey_name)
	condition_types = ['SNR','SNR_spont','RF_thin', 'RF_large', 'RS']
	if monkey_name == 'A':
		condition_types = ['SNR','SNR_spont','RF_thin', 'RF_large','RF_thin_spont', 'RF_large_spont']

	w_size=25
	n_splits=10

	percent_over=80
	percent_under=10
	if monkey_name == 'A':
		target_x_n=10
	else:
		target_x_n=14

	ref_area='V4'
	for condition_type in condition_types:
		
		get_predictor_indices_elec_ids(monkey_stats, condition_type, get_get_condition_type(condition_type), target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_xtarget_predictor_indices_elecs(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, ref_area, target_x_n, percent_over, percent_under, monkey=monkey_name)

		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=False, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		print(condition_type, 'done')

	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)

	ref_area='V1'
	for condition_type in condition_types:
		get_predictor_indices_elec_ids(monkey_stats, condition_type, get_get_condition_type(condition_type), target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_xtarget_predictor_indices_elecs(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, ref_area, target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=False, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		print(condition_type, 'done')
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)

	# shuffle control
	ref_area='V4'
	for condition_type in condition_types:
		get_predictor_indices_elec_ids(monkey_stats, condition_type, get_get_condition_type(condition_type), target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_xtarget_predictor_indices_elecs(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, ref_area, target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=True, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		print(condition_type, 'done')
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)
	# shuffle control
	ref_area='V1'
	for condition_type in condition_types:
		get_predictor_indices_elec_ids(monkey_stats, condition_type, get_get_condition_type(condition_type), target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_xtarget_predictor_indices_elecs(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, ref_area, target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=True, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		print(condition_type, 'done')
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)

	end_time = time.time()
	elapsed_time = (end_time - start_time)/60
	print(f'yay! RF overlap evar comparisons for monkey {monkey_name} is now completed')
	print(f'Took {elapsed_time:.4f} minutes to complete')
	# save macaque stats
print('All done with both monkeys!')



monkey_names = ['A']

for monkey_name in monkey_names:
	start_time = time.time()
	print(f'Processing monkey {monkey_name}')
	if not os.path.exists(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')):
		print('Creating monkey stats')
		monkey_stats= create_empty_monkey_stats_dict(monkey=monkey_name)
		get_split_half_r_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=dataset_type_dict[monkey_name])
		get_SNR_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=dataset_type_dict[monkey_name])
		store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name])
		get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name], control_shuffle=False)
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name], control_shuffle=True)
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=spont_dataset_type_dict[monkey_name], control_shuffle=False)
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=spont_dataset_type_dict[monkey_name], control_shuffle=True)
		get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=spont_dataset_type_dict[monkey_name])
	else:
		print(f'Using previous monkey stats')
		with open(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
			monkey_stats = pickle.load(handle)
	get_one_vs_rest_r_monkey_all_dates(monkey_stats, w_size=25, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])

	# save macaque stats
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
	end_time = time.time()
	elapsed_time = (end_time - start_time)/60
	print(f'Yay! work for monkey {monkey_name} is completed. Took {elapsed_time:.4f} minutes to complete')


######################################### MACAQUE A RF OVERLAP EVAR COMPARISONS ##############################


num_seeds = 10
random.seed(17)
# Create a list of random seeds
seeds = [random.randint(1, 10000) for _ in range(num_seeds)]

all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200, 'RF_large_spont':200}


monkey_names = ['A']
for monkey_name in monkey_names:
	start_time = time.time()
	with open(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
		monkey_stats = pickle.load(handle)
	monkey_stats_path = os.path.join(project_root, 'results/fig_6',f'monkey_{monkey_name}_stats.pkl')
	with open(monkey_stats_path, 'rb') as handle:
		monkey_stats = pickle.load(handle)
	get_electrode_ids_all_dates(monkey_stats, monkey=monkey_name)
	condition_types = ['SNR','SNR_spont','RF_thin', 'RF_large', 'RS']
	if monkey_name == 'A':
		condition_types = ['SNR','SNR_spont','RF_thin', 'RF_large','RF_thin_spont', 'RF_large_spont']

	w_size=25
	n_splits=10

	percent_over=80
	percent_under=10

	ref_area='V4'
	if monkey_name == 'A' and ref_area=='V1':
		target_x_n=10
	else:
		target_x_n=14
	for condition_type in condition_types:
		get_predictor_indices_elec_ids(monkey_stats, condition_type, get_get_condition_type(condition_type), target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_xtarget_predictor_indices_elecs(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, ref_area, target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=False, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=True, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		print(condition_type, 'done')

	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)

	ref_area='V1'
	if monkey_name == 'A' and ref_area=='V1':
		target_x_n=10
		print('target x n is 10 for V1')
	else:
		target_x_n=14

	for condition_type in condition_types:
		get_predictor_indices_elec_ids(monkey_stats, condition_type, get_get_condition_type(condition_type), target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_xtarget_predictor_indices_elecs(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, ref_area, target_x_n, percent_over, percent_under, monkey=monkey_name)
		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=False, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		get_x_target_overlap_evars(monkey_stats, condition_type, get_get_condition_type(condition_type), seeds, 
								ref_area, target_x_n=target_x_n, percent_over=percent_over, 
								percent_under=percent_under, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
								frames_to_reduce=all_frames_reduced[condition_type], n_splits=n_splits, 
								control_shuffle=True, monkey=monkey_name, date_used=date_used_dict[monkey_name],
								condition_type_used=condition_type_used_dict[monkey_name])
		print(condition_type, 'done')
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)

print('All done with both monkeys!')
