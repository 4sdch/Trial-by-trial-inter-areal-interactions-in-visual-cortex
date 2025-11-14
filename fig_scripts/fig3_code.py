
from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)

import pickle
import os
import json
import time

results_dir = os.path.join(project_root,'results/fig_2/')
previous_results_dir = os.path.join(project_root,'results/fig_3/')

# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)

import sys
sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)
from utils.neuron_properties_functions import store_mouse_alphas, get_evars_all_mice, get_split_half_r_all_mice, get_SNR_all_mice
from utils.neuron_properties_functions import  get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates, create_empty_monkey_stats_dict, get_evar_monkey_all_dates, store_macaque_alphas


from utils.neuron_properties_functions import create_empty_mouse_stats_dict, get_split_half_r_all_mice, get_SNR_all_mice, get_max_corr_vals_all_mice, get_evars_all_mice, get_evar_monkey_all_dates, store_mouse_alphas
from utils.neuron_properties_functions import create_empty_monkey_stats_dict, get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates, get_max_corr_vals_monkey_all_dates, get_variance_within_trial_across_timepoints, get_variance_within_timepoints_across_trials, get_RF_variance_across_stimuli
from utils.fig_3_functions import store_L23_indices, store_mouse_directionality_alphas, get_directionality_evars_mice, get_directionality_max_corr_vals_mice
from utils.fig_3_functions import store_V1_indices, store_macaque_directionality_alphas, get_directionality_evars_monkey, get_directionality_maxcorrvals_monkey
import utils.mouse_data_functions as cs

################################### MOUSE PREDICTIONS ####################################
# if there i no mouse stats in the previous figures, create empty mouse stat directory and store SNR, split-half r val, and inter-area prediction performance

previous_mouse_stats_path = os.path.join(previous_results_dir, 'mouse_stats.pkl')
if not os.path.exists(previous_mouse_stats_path):
	mouse_stats= create_empty_mouse_stats_dict(main_dir)
	get_SNR_all_mice(main_dir, mouse_stats)
	get_split_half_r_all_mice(main_dir, mouse_stats)
	store_mouse_alphas(main_dir, mouse_stats, activity_type='resp', verbose=True)

else:
	with open(previous_mouse_stats_path, 'rb') as handle:
		mouse_stats = pickle.load(handle)

store_L23_indices(mouse_stats)
store_mouse_directionality_alphas(main_dir, mouse_stats, activity_type='resp', verbose=False)
get_directionality_max_corr_vals_mice(main_dir, mouse_stats,  activity_type='resp')
get_directionality_evars_mice(main_dir, mouse_stats, activity_type='resp',control_shuffle=False)
get_directionality_evars_mice(main_dir, mouse_stats, activity_type='resp',control_shuffle=True)


with open(os.path.join(results_dir, 'mouse_stats.pkl'), 'wb') as file:
    pickle.dump(mouse_stats, file)

# supplemental
get_max_corr_vals_all_mice(main_dir, mouse_stats)
get_directionality_max_corr_vals_mice(main_dir, mouse_stats,  activity_type='spont')
get_directionality_evars_mice(main_dir, mouse_stats, activity_type='spont',control_shuffle=False)
get_directionality_evars_mice(main_dir, mouse_stats, activity_type='spont',control_shuffle=True)
store_mouse_alphas(main_dir, mouse_stats, activity_type='resp', verbose=True)
store_mouse_alphas(main_dir, mouse_stats, activity_type='spont', verbose=True)
if not os.path.exists(previous_mouse_stats_path):
	get_evars_all_mice(main_dir, mouse_stats, activity_type='resp')
	get_evars_all_mice(main_dir, mouse_stats, activity_type='spont')
	get_evars_all_mice(main_dir, mouse_stats, activity_type='resp', control_shuffle=True)
	get_evars_all_mice(main_dir, mouse_stats, activity_type='spont', control_shuffle=True)

# isolate nonvisual neurons only
nonvisual_neurons = True
# store_L23_indices(mouse_stats, nonvisual_neurons=nonvisual_neurons) #this took about 1 hour
store_mouse_directionality_alphas(main_dir, mouse_stats, activity_type='resp', verbose=False, nonvisual_neurons=nonvisual_neurons) #took 2 minutes to complete

# supplemental
get_directionality_max_corr_vals_mice(main_dir, mouse_stats,  activity_type='spont', nonvisual_neurons=nonvisual_neurons)
get_directionality_evars_mice(main_dir, mouse_stats, activity_type='spont',control_shuffle=False, nonvisual_neurons=nonvisual_neurons)
get_directionality_evars_mice(main_dir, mouse_stats, activity_type='spont',control_shuffle=True, nonvisual_neurons=nonvisual_neurons)



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



resp_dataset_type_dict = {'A': ['SNR','RF_thin', 'RF_large'],
                    'L': ['SNR','RF_thin', 'RF_large'],
                    'D': ['SNR']}

spont_dataset_type_dict = {'A': ['SNR_spont','RF_thin_spont', 'RF_large_spont', 'RS','RS_open', 'RS_closed'],
					'L': ['SNR_spont','RF_thin_spont', 'RF_large_spont', 'RS','RS_open', 'RS_closed'],
					'D': ['SNR_spont','RS','RS_open']
					}

date_used_dict = {'L': '090817', 'A': '041018', 'D': '250225'}
condition_type_used_dict = {'L':'RS','A':'SNR','D':'RS'}

monkey_names = ['L', 'A', 'D']
reli_theshold=0.8
for monkey_name in monkey_names:
	reli_theshold=0.8
	previous_monkey_stats_path = os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')
	if not os.path.exists(previous_monkey_stats_path):
		monkey_stats=create_empty_monkey_stats_dict(monkey=monkey_name)
		get_split_half_r_monkey_all_dates(monkey_stats, monkey=monkey_name)
		get_SNR_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=resp_dataset_type_dict[monkey_name])
		store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name])
	else:
		print(f'retrieved info from {previous_monkey_stats_path}')
		with open(previous_monkey_stats_path, 'rb') as handle:
			monkey_stats = pickle.load(handle)
	if monkey_name in ['D']:
		reli_theshold=0.6
	store_V1_indices(monkey_stats, condition_types=resp_dataset_type_dict[monkey_name], verbose=False, reli_threshold=reli_theshold, predictor_min=6)

	del monkey_stats['monkey_directionality_alphas']
	store_macaque_directionality_alphas(monkey_stats, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name], monkey=monkey_name)
	print('directionality alphas stored')
	get_directionality_maxcorrvals_monkey(monkey_stats, dataset_types=resp_dataset_type_dict[monkey_name], monkey=monkey_name)
	print('directional max corr vals stored')
	get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=resp_dataset_type_dict[monkey_name])
	get_directionality_maxcorrvals_monkey(monkey_stats, dataset_types=spont_dataset_type_dict[monkey_name], monkey=monkey_name)
	get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types=spont_dataset_type_dict[monkey_name])
	get_variance_within_trial_across_timepoints(monkey_stats, specific_dataset_types = resp_dataset_type_dict[monkey_name], 
                                             monkey=monkey_name)
	get_variance_within_timepoints_across_trials(monkey_stats, specific_dataset_types = resp_dataset_type_dict[monkey_name], 
                                              monkey=monkey_name)
	if monkey_name in ['L','A']:
		get_RF_variance_across_stimuli(monkey_stats, specific_dataset_types = ['RF_large','RF_thin'], monkey=monkey_name)
	get_directionality_evars_monkey(monkey_stats, control_shuffle=False, dataset_types=resp_dataset_type_dict[monkey_name], monkey=monkey_name)
	print('directional evars stored ')
	get_directionality_evars_monkey(monkey_stats, control_shuffle=True, dataset_types=resp_dataset_type_dict[monkey_name], monkey=monkey_name)
	print('directionality evars control stored')

	with open(os.path.join(results_dir + f'monkey_{monkey_name}_stats.pkl'), 'wb') as file:
		pickle.dump(monkey_stats, file)

	###### suplemental ##########
	print('\n calculating supplemental shenanigans')
	# spont
	get_directionality_evars_monkey(monkey_stats, control_shuffle=False, dataset_types=spont_dataset_type_dict[monkey_name], monkey=monkey_name)
	get_directionality_evars_monkey(monkey_stats, control_shuffle=True, dataset_types=spont_dataset_type_dict[monkey_name], monkey=monkey_name)
	# regular properties

	# in case im supposed to calculate also spont regular stuff
	if not os.path.exists(previous_monkey_stats_path):
		get_evar_monkey_all_dates(monkey_stats, control_shuffle=False, dataset_types=resp_dataset_type_dict[monkey_name]+spont_dataset_type_dict[monkey_name], monkey=monkey_name)
		get_evar_monkey_all_dates(monkey_stats, control_shuffle=True, dataset_types=resp_dataset_type_dict[monkey_name]+spont_dataset_type_dict[monkey_name], monkey=monkey_name)

	get_evar_monkey_all_dates(monkey_stats, control_shuffle=False, dataset_types=spont_dataset_type_dict[monkey_name], monkey=monkey_name)
	get_evar_monkey_all_dates(monkey_stats, control_shuffle=True, dataset_types=spont_dataset_type_dict[monkey_name], monkey=monkey_name)
	
	with open(os.path.join(results_dir + f'monkey_{monkey_name}_stats.pkl'), 'wb') as file:
		pickle.dump(monkey_stats, file)
	print('done!')


############################### subsampled shenanigans ####################################



with open(os.path.join(results_dir, 'subsample_seeds.json'), 'r') as f:
    subsample_seeds = json.load(f)

main_monkey_name = 'L'
subsample_monkey_names = ['A']

start_time = time.time()

condition_type_used = 'SNR'
for subsample_monkey_name in subsample_monkey_names:
	if subsample_monkey_name in ['D']:
		reli_theshold=0.6
	print('Subsampling monkey L to match monkey ', subsample_monkey_name)
	for seed in subsample_seeds:
		subsampled_monkey_stats_path = os.path.join(project_root,f'results/fig_3/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
		with open (subsampled_monkey_stats_path, 'rb') as handle:
			subsampled_monkey_stats = pickle.load(handle)
		subsampled_indices = True
		date_used  = list(subsampled_monkey_stats['SNR'].keys())[0]
		stim_datataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and 'spont' not in k and 'RS' not in k]
		spont_dataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and ('spont' in k or 'RS' in k)]

		get_max_corr_vals_monkey_all_dates(subsampled_monkey_stats, monkey=main_monkey_name, dataset_types=stim_datataset_types_, subsampled_indices=subsampled_indices)
		get_max_corr_vals_monkey_all_dates(subsampled_monkey_stats, monkey=main_monkey_name, dataset_types=spont_dataset_types_, subsampled_indices=subsampled_indices)
		conditions_dates_not_used = store_V1_indices(subsampled_monkey_stats, condition_types=stim_datataset_types_, verbose=False, reli_threshold=reli_theshold, predictor_min=6, return_conditions_not_used=True)
		print('V1 indices stored')
		if (condition_type_used, date_used) in conditions_dates_not_used:
			all_dates = subsampled_monkey_stats[condition_type_used].keys()
			dates_to_use = [d for d in all_dates if (condition_type_used, d) not in conditions_dates_not_used]
			if len(dates_to_use)==0:
				print(f'No dates available for condition {condition_type_used} after applying reliability and predictor thresholds. Skipping this seed.')
				######### save itttt 
				with open(subsampled_monkey_stats_path, 'wb') as file:
					pickle.dump(subsampled_monkey_stats, file)
				print(f'subsampled stats saved to {subsampled_monkey_stats_path}')
				continue
			else:
				date_used = dates_to_use[0]
				print(f'Using date {date_used} for condition {condition_type_used} instead.')
		store_macaque_directionality_alphas(subsampled_monkey_stats, date_used=date_used, condition_type_used=condition_type_used, monkey=main_monkey_name)
		print('directionality alphas stored')
		get_directionality_evars_monkey(subsampled_monkey_stats, control_shuffle=False, dataset_types=stim_datataset_types_, monkey=main_monkey_name, subsampled_indices=subsampled_indices)
		print('directional evars stored ')
		get_directionality_evars_monkey(subsampled_monkey_stats, control_shuffle=True, dataset_types=stim_datataset_types_, monkey=main_monkey_name, subsampled_indices=subsampled_indices)
		print('directionality evars control stored')
		get_directionality_maxcorrvals_monkey(subsampled_monkey_stats, dataset_types=stim_datataset_types_, monkey=main_monkey_name, subsampled_indices=subsampled_indices)
		get_directionality_maxcorrvals_monkey(subsampled_monkey_stats, dataset_types=spont_dataset_types_, monkey=main_monkey_name, subsampled_indices=subsampled_indices)
		print('directional max corr vals stored')
		

		############ just do supplemental here too
		get_directionality_evars_monkey(subsampled_monkey_stats, control_shuffle=False, dataset_types=spont_dataset_types_, monkey=main_monkey_name, subsampled_indices=subsampled_indices)
		get_directionality_evars_monkey(subsampled_monkey_stats, control_shuffle=True, dataset_types=spont_dataset_types_, monkey=main_monkey_name, subsampled_indices=subsampled_indices)
		######### save itttt 
		with open(subsampled_monkey_stats_path, 'wb') as file:
			pickle.dump(subsampled_monkey_stats, file)
		print(f'subsampled stats saved to {subsampled_monkey_stats_path}')
end_time = time.time()
# print the elapsed time in minutes
elapsed_time = (end_time - start_time) / 60
print(f"Elapsed time: {elapsed_time:.2f} minutes")
