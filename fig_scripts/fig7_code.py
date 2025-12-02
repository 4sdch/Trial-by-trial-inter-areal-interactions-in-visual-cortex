from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)

import sys
import pickle
import time
import os


results_dir = os.path.join(project_root,'results/fig_7/')
previous_results_dir = os.path.join(project_root,'results/fig_6/')

# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)

sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)

import utils.mouse_data_functions as cs
from utils.neuron_properties_functions import create_empty_mouse_stats_dict, get_split_half_r_all_mice, get_SNR_all_mice, get_max_corr_vals_all_mice, get_evars_all_mice, store_mouse_alphas
from utils.macaque_data_functions import get_get_condition_type, get_resps, get_img_resp_avg_sem
from utils.fig_7_functions import trial_randomize, process_timelag_shenanigans
from utils.neuron_properties_functions import extract_mouse_name
from utils.ridge_regression_functions import get_predictions_evars_parallel
from utils.neuron_properties_functions import create_empty_monkey_stats_dict, get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates,get_max_corr_vals_monkey_all_dates,get_evar_monkey_all_dates, store_macaque_alphas




########################################## MOUSE SHUFFLE TRIALS #########################################

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


print('starting mouse shuffle trial repeats')
start_time = time.time()

seed = 17
dataset_types = ['ori32','natimg32']
area='L23'
area2='L4'
activity_type = 'resp'
n_splits=10
sample_size=500


for d, dataset_type in enumerate(dataset_types):
    mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
    mousenames= sorted(mt.filenames)
    for mouse in mousenames:
        
        resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse)
        if len(resp_L1)<1000:
            continue
        istim = mt.istim
        
        alpha = mouse_stats['mouse_alphas'][sample_size][(extract_mouse_name(mouse))][area]
        alpha2 = mouse_stats['mouse_alphas'][sample_size][(extract_mouse_name(mouse))][area2]
        
        if mouse_stats['mouse_alphas'][sample_size][(extract_mouse_name(mouse))]['dataset_type_used']==dataset_type:
                resp_L23=resp_L23[sample_size:]
                resp_L4=resp_L4[sample_size:]
                istim = istim[sample_size:]
                
        
        shuffled_istim_indices = trial_randomize(istim=istim, seed=seed)
        
        
        _, evars = get_predictions_evars_parallel(resp_L4[shuffled_istim_indices], 
                                                    resp_L23, n_splits=n_splits, alpha=alpha, 
                                                    frames_reduced=5)
        _, evars2 = get_predictions_evars_parallel(resp_L23[shuffled_istim_indices], 
                                                    resp_L4, n_splits=n_splits, alpha=alpha2, 
                                                    frames_reduced=5)
        mouse_stats[dataset_type][mouse][area]['shuffled_istim_indices']=shuffled_istim_indices
        mouse_stats[dataset_type][mouse][area2]['shuffled_istim_indices']=shuffled_istim_indices

        mouse_stats[dataset_type][mouse][area]['evar_shuffled_istims']=evars
        mouse_stats[dataset_type][mouse][area2]['evar_shuffled_istims']=evars2
        print(mouse, 'done')

# SAVE MOUSE STATS
with open(os.path.join(results_dir, 'mouse_stats.pkl'), 'wb') as handle:
	pickle.dump(mouse_stats, handle)

end_time = time.time()
# Calculate the elapsed time
elapsed_time = (end_time - start_time)/60
print(f'yay! mouse shuffle trial is completed')
print(f'Took {elapsed_time:.4f} minutes to complete')



###################################### MONKEY SHUFFLE TRIALS ######################################

monkey_names = ['A','L','D']
date_used_dict = {'L': '090817', 'A': '290818', 'D': '260225'} 
condition_type_used_dict = {'L':'RS','A':'RF_thin','D':'RS'}


all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                      'RS_open':20, 'RS_closed': 20, 
                      'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                      'RS_open':None, 'RS_closed': None, 
                      'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200, 'RF_large_spont':200}

frames_to_reduce=5
n_splits=10
area='V4'
area2='V1'
w_size=25

sample_size = 500
shuttle_trial_repeat_start_time = time.time()

for monkey_name in monkey_names:
	start_time = time.time()
	print(f'Processing monkey {monkey_name}')
	if not os.path.exists(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')):
		print('Creating monkey stats')
		monkey_stats= create_empty_monkey_stats_dict(monkey=monkey_name)
		get_SNR_monkey_all_dates(monkey_stats, monkey=monkey_name)
		get_split_half_r_monkey_all_dates(monkey_stats, monkey=monkey_name)
		get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name)
		store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name])
		get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name)
	else:
		print(f'Using previous monkey stats')
		with open(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
			monkey_stats = pickle.load(handle)
	print('printing monkey stats keys 1st time:')
	print(list(monkey_stats.keys()))
	dates = monkey_stats['SNR'].keys()
	monkey_alphas=monkey_stats['monkey_alphas'][sample_size]
	alpha = monkey_alphas[area]
	alpha2=monkey_alphas[area2]

	print('calculating shuffle trials for SNR and SNR_spont')
	for condition_type in ['SNR','SNR_spont']:
		for date in monkey_stats[condition_type]:
			# skip the dates with no V4 electrodes
			if date in ['140819', '150819', '160819']:
				continue
			get_condition_type = get_get_condition_type(condition_type)
			resp_V4_shuffled, resp_V1=get_resps(condition_type=get_condition_type, date=date, shuffle=True,
										w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], monkey=monkey_name)
			_, evars = get_predictions_evars_parallel(resp_V1, resp_V4_shuffled, n_splits=n_splits,
											frames_reduced=frames_to_reduce, alpha=alpha)
			_, evars2 = get_predictions_evars_parallel(resp_V4_shuffled, resp_V1, n_splits=n_splits,
											frames_reduced=frames_to_reduce, alpha=alpha2)
			
			monkey_stats[condition_type][date][area]['evar_shuffled_istims']=evars
			monkey_stats[condition_type][date][area2]['evar_shuffled_istims']=evars2
		
		print(condition_type, 'done')
	if monkey_name in ['L','A']:
		print('calculating shuffle trials for RF_thin and RF_large')
		## for RF data 
		for condition_type in ['RF_thin','RF_large','RF_thin_spont', 'RF_large_spont']: #also 'RF_large','RF_thin'
			for date in monkey_stats[condition_type]:
				resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
												get_RF_labels=True, monkey=monkey_name)
				if 'spont' in condition_type:
					chunk_size = int(300/25)
				else:
					chunk_size = None
				binned_epochs = get_img_resp_avg_sem(resp_V4, condition_type=get_get_condition_type(condition_type), chunk_size=chunk_size, get_chunks=True)
				binned_labels = cond_labels[:,0,0]
				shuffled_labels = trial_randomize(binned_labels, seed=None)
				shuffled_resp_V4 = binned_epochs[shuffled_labels].reshape(-1, binned_epochs.shape[2])
				_, shuffled_evars = get_predictions_evars_parallel(resp_V1, shuffled_resp_V4, n_splits=10,frames_reduced=all_frames_reduced[condition_type],alpha=alpha)
				_, shuffled_evars2 = get_predictions_evars_parallel(shuffled_resp_V4, resp_V1, n_splits=10,frames_reduced=all_frames_reduced[condition_type], alpha=alpha2)

				monkey_stats[condition_type][date][area]['evar_shuffled_istims']=shuffled_evars
				monkey_stats[condition_type][date][area2]['evar_shuffled_istims']=shuffled_evars2

	# SAVE MONKEY STATS
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)
	end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = (end_time - start_time)/60
	print(f'yay! shuffle trials for monkey {monkey_name} is completed')
	print(f'Took {elapsed_time:.4f} minutes to complete')
trial_repeat_end_time = time.time()
# Calculate the elapsed time
elapsed_time = (trial_repeat_end_time - shuttle_trial_repeat_start_time)/60
print(f'yay! shuffle trials for all monkeys is completed')
print(f'Took {elapsed_time:.4f} minutes to complete')


####################################### MONKEY TIME WINDOW OFFSETS ########################################
ref_duration = 200
w_size=25
monkey_names = ['A','L','D']
time_offset_start_time = time.time()
for monkey_name in monkey_names:
	start_time = time.time()
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
		monkey_stats = pickle.load(handle)
	print(list(monkey_stats.keys()))
	ref_area = 'V4'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in monkey_stats[condition_type]:
			# skip the dates with no V4 electrodes
			if date in ['140819', '150819', '160819']:
				continue
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									monkey_stats, w_size, control_neurons=True, monkey=monkey_name)
	ref_area = 'V1'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in monkey_stats[condition_type]:
			# skip the dates with no V4 electrodes
			if date in ['140819', '150819', '160819']:
				continue
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									monkey_stats, w_size, control_neurons=True, monkey=monkey_name)
	end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = (end_time - start_time)/60
	print(f'yay! time window offsets for monkey {monkey_name} is completed')
	print(f'Took {elapsed_time:.4f} minutes to complete')
	# SAVE MONKEY STATS	
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)
time_offset_end_time = time.time()
# Calculate the elapsed time
elapsed_time = (time_offset_end_time - time_offset_start_time)/60
print(f'yay! time window offsets for all monkeys is completed')
print(f'Took {elapsed_time:.4f} minutes to complete')

ref_duration = 200
w_size=10
monkey_names = ['L','A','D']
time_offset_start_time = time.time()
for monkey_name in monkey_names:
	start_time = time.time()
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
		monkey_stats = pickle.load(handle)
	print(list(monkey_stats.keys()))
	ref_area = 'V4'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in monkey_stats[condition_type]:
			# skip the dates with no V4 electrodes
			if date in ['140819', '150819', '160819']:
				continue
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									monkey_stats, w_size, control_neurons=True, monkey=monkey_name)
	ref_area = 'V1'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in monkey_stats[condition_type]:
			# skip the dates with no V4 electrodes
			if date in ['140819', '150819', '160819']:
				continue
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									monkey_stats, w_size, control_neurons=True, monkey=monkey_name)
	end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = (end_time - start_time)/60
	print(f'yay! time window offsets for monkey {monkey_name} is completed')
	print(f'Took {elapsed_time:.4f} minutes to complete')
	# SAVE MONKEY STATS	
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)
time_offset_end_time = time.time()
# Calculate the elapsed time
elapsed_time = (time_offset_end_time - time_offset_start_time)/60
print(f'yay! time window offsets for all monkeys is completed')
print(f'Took {elapsed_time:.4f} minutes to complete')



################# run then again with control_neurons=False
ref_duration = 200
w_size=10
monkey_names = ['L','A','D']
time_offset_start_time = time.time()
control_neurons = False
for monkey_name in monkey_names:
	start_time = time.time()
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
		monkey_stats = pickle.load(handle)
	print(list(monkey_stats.keys()))
	ref_area = 'V4'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in monkey_stats[condition_type]:
			# skip the dates with no V4 electrodes
			if date in ['140819', '150819', '160819']:
				continue
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									monkey_stats, w_size, control_neurons=control_neurons, monkey=monkey_name)
	ref_area = 'V1'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in monkey_stats[condition_type]:
			# skip the dates with no V4 electrodes
			if date in ['140819', '150819', '160819']:
				continue
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									monkey_stats, w_size, control_neurons=control_neurons, monkey=monkey_name)
	end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = (end_time - start_time)/60
	print(f'yay! time window offsets for monkey {monkey_name} is completed')
	print(f'Took {elapsed_time:.4f} minutes to complete')
	# SAVE MONKEY STATS	
	with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
		pickle.dump(monkey_stats, handle)
time_offset_end_time = time.time()
# Calculate the elapsed time
elapsed_time = (time_offset_end_time - time_offset_start_time)/60
print(f'yay! time window offsets for all monkeys is completed')
print(f'Took {elapsed_time:.4f} minutes to complete')



# ################################ subsample trial repeats for monkey L to match monkey A and D ################################
import json
import time

overall_start = time.time()
with open(os.path.join(results_dir, 'subsample_seeds.json'), 'r') as f:
    subsample_seeds = json.load(f)
subsample_monkey_names = ['A','D']
main_monkey_name = 'L'

main_trial_repeat_start_time = time.time()	
for subsample_monkey_name in subsample_monkey_names:
	shuttle_trial_repeat_start_time = time.time()
	print('Subsampling monkey L to match monkey ', subsample_monkey_name)
	for seed in subsample_seeds:
		subsampled_monkey_stats_path = os.path.join(project_root,f'results/fig_7/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
		with open (subsampled_monkey_stats_path, 'rb') as handle:
			subsampled_monkey_stats = pickle.load(handle)
		subsampled_indices = True
		date_used  = list(subsampled_monkey_stats['SNR'].keys())[0]
		stim_datataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and 'spont' not in k and 'RS' not in k]
		spont_dataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and ('spont' in k or 'RS' in k)]
		start_time = time.time()
		print(f'Processing monkey {subsample_monkey_name} seed {seed}')
		dates = subsampled_monkey_stats['SNR'].keys()
		monkey_alphas=subsampled_monkey_stats['monkey_alphas'][sample_size]
		alpha = monkey_alphas[area]
		alpha2=monkey_alphas[area2]

		print('calculating shuffle trials for SNR and SNR_spont')
		for condition_type in ['SNR','SNR_spont']:
			for date in subsampled_monkey_stats[condition_type]:
				get_condition_type = get_get_condition_type(condition_type)
				resp_V4_shuffled, resp_V1=get_resps(condition_type=get_condition_type, date=date, shuffle=True,
											w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], monkey=main_monkey_name)
				subsampled_indices_dict = {'V1':subsampled_monkey_stats[condition_type][date]['V1'][f'monkey_{main_monkey_name}_subsample_indices'],
                                    'V4':subsampled_monkey_stats[condition_type][date]['V4'][f'monkey_{main_monkey_name}_subsample_indices']}
				resp_V4_shuffled = resp_V4_shuffled[:, subsampled_indices_dict['V4']]
				resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]

				_, evars = get_predictions_evars_parallel(resp_V1, resp_V4_shuffled, n_splits=n_splits,
												frames_reduced=frames_to_reduce, alpha=alpha)
				_, evars2 = get_predictions_evars_parallel(resp_V4_shuffled, resp_V1, n_splits=n_splits,
												frames_reduced=frames_to_reduce, alpha=alpha2)
				
				subsampled_monkey_stats[condition_type][date][area]['evar_shuffled_istims']=evars
				subsampled_monkey_stats[condition_type][date][area2]['evar_shuffled_istims']=evars2
			
			print(condition_type, 'done')
		if subsample_monkey_name in ['L','A']:
			print('calculating shuffle trials for RF_thin and RF_large')
			## for RF data 
			for condition_type in ['RF_thin','RF_large','RF_thin_spont', 'RF_large_spont']: #also 'RF_large','RF_thin'
				for date in subsampled_monkey_stats[condition_type]:
					resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size, stim_on=0, stim_off=all_ini_stim_offs[condition_type], 
													get_RF_labels=True, monkey=main_monkey_name)
					subsampled_indices_dict = {'V1':subsampled_monkey_stats[condition_type][date]['V1'][f'monkey_{main_monkey_name}_subsample_indices'],
									'V4':subsampled_monkey_stats[condition_type][date]['V4'][f'monkey_{main_monkey_name}_subsample_indices']}
					resp_V4 = resp_V4[:, subsampled_indices_dict['V4']]
					resp_V1 = resp_V1[:, subsampled_indices_dict['V1']]
					if 'spont' in condition_type:
						chunk_size = int(300/25)
					else:
						chunk_size = None
					binned_epochs = get_img_resp_avg_sem(resp_V4, condition_type=get_get_condition_type(condition_type), chunk_size=chunk_size, get_chunks=True)
					binned_labels = cond_labels[:,0,0]
					shuffled_labels = trial_randomize(binned_labels, seed=None)
					shuffled_resp_V4 = binned_epochs[shuffled_labels].reshape(-1, binned_epochs.shape[2])
					_, shuffled_evars = get_predictions_evars_parallel(resp_V1, shuffled_resp_V4, n_splits=10,frames_reduced=all_frames_reduced[condition_type],alpha=alpha)
					_, shuffled_evars2 = get_predictions_evars_parallel(shuffled_resp_V4, resp_V1, n_splits=10,frames_reduced=all_frames_reduced[condition_type], alpha=alpha2)

					subsampled_monkey_stats[condition_type][date][area]['evar_shuffled_istims']=shuffled_evars
					subsampled_monkey_stats[condition_type][date][area2]['evar_shuffled_istims']=shuffled_evars2

		# SAVE
		with open(subsampled_monkey_stats_path, 'wb') as handle:
			pickle.dump(subsampled_monkey_stats, handle)
		print(f'Saved subsampled monkey stats to {subsampled_monkey_stats_path}')
		end_time = time.time()
		# Calculate the elapsed time
		elapsed_time = (end_time - start_time)/60
		print(f'yay! shuffle trials for monkey {subsample_monkey_name} seed {seed} is completed')
		print(f'Took {elapsed_time:.4f} minutes to complete')
	trial_repeat_end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = (trial_repeat_end_time - shuttle_trial_repeat_start_time)/60
	print(f'yay! shuffle trials for all seeds in {subsample_monkey_name} is completed')
	print(f'Took {elapsed_time:.4f} minutes to complete')
main_trial_repeat_end_time = time.time()
print(f'All done with subsampling monkey L to match {subsample_monkey_names}. took {(main_trial_repeat_end_time - main_trial_repeat_start_time)/60:.4f} hours to complete')



# ############################ subsample timelag shenanigans for monkey L to match monkey A only ################################

w_size=10
ref_duration = 200
timelag = time.time()
subsample_monkey_name = 'A'
print('Subsampling monkey L to match monkey ', subsample_monkey_name)

for seed in subsample_seeds:
	start_time = time.time()
	subsampled_monkey_stats_path = os.path.join(project_root,f'results/fig_7/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
	with open (subsampled_monkey_stats_path, 'rb') as handle:
		subsampled_monkey_stats = pickle.load(handle)
	subsampled_indices = True
	date_used  = list(subsampled_monkey_stats['SNR'].keys())[0]
	stim_datataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and 'spont' not in k and 'RS' not in k]
	spont_dataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and ('spont' in k or 'RS' in k)]
	time_offset_start_time = time.time()
	start_time = time.time()
	ref_area = 'V4'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in subsampled_monkey_stats[condition_type]:
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									subsampled_monkey_stats, w_size, control_neurons=True, monkey=main_monkey_name,
         							subsampled_indices=subsampled_indices)
	ref_area = 'V1'
	condition_types = ['SNR','SNR_spont']
	for condition_type in condition_types:
		for date in subsampled_monkey_stats[condition_type]:
			process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
									subsampled_monkey_stats, w_size, control_neurons=True, 
         							monkey=main_monkey_name, subsampled_indices=subsampled_indices)
	end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = (end_time - start_time)/60
	print(f'yay! time window offsets for monkey {main_monkey_name} subsampled to {subsample_monkey_name} seed {seed} is completed')	
	# SAVE 
	with open(subsampled_monkey_stats_path, 'wb') as handle:
		pickle.dump(subsampled_monkey_stats, handle)

time_offset_end_time = time.time()
# Calculate the elapsed time
elapsed_time = (time_offset_end_time - time_offset_start_time)/60
print(f'yay! time window offsets for all seeds in {subsample_monkey_name} is completed')

over_all_end = time.time()
print(f'All done with subsamplings monkey for both functions. took {(over_all_end - overall_start)/60:.4f} hours to complete')