
# Description: This file contains functions for analyzing neural data, specifically for mouse and monkey datasets
import time
import pickle
import json

from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)


import os
results_dir = os.path.join(project_root,'results/fig_3/')


# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)

import sys
sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)


from utils.fig_7_functions import process_timelag_shenanigans

# Morales-Gregorio/Chen bands
BANDS = {
    "low_2_12":   (2, 12),
    "beta_12_30": (12, 30),
    "gamma_30_45":(30, 45),
    "hgamma_55_95":(55, 95),
}

# timelags w_size = 10 ms
ref_duration = 200
w_size=10
monkey_names = ['L','A']

for monkey_name in monkey_names:
	monkey_stats_path = os.path.join(main_dir, 'results/fig_7',f'monkey_{monkey_name}_stats.pkl')
	with open(monkey_stats_path, 'rb') as f:
		monkey_stats = pickle.load(f)

	
	## all bands but 1 took 41 minutes plus 13 min fro the last band
	time_offset_start_time = time.time()
	for band in list(BANDS.keys()):
	# for band in ['hgamma_55_95']: # redo this band because i got rid of some artifacts at around 60Hz
		print(f'Processing band: {band}')
		start_time = time.time()
		ref_area = 'V4'
		condition_types = ['SNR','SNR_spont']
		for condition_type in condition_types:
			for date in monkey_stats[condition_type]:
				if date in ['140819', '150819', '160819','250225']:
					continue
				process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
										monkey_stats, w_size, control_neurons=True, monkey=monkey_name,
										recording_type='LFP', band=band)
		ref_area = 'V1'
		for condition_type in condition_types:
			for date in monkey_stats[condition_type]:
				# skip the dates with no V4 electrodes
				if date in ['140819', '150819', '160819','250225']:
					continue
				process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
										monkey_stats, w_size, control_neurons=True, monkey=monkey_name,
										recording_type='LFP', band=band)
		end_time = time.time()
		# Calculate the elapsed time
		elapsed_time = (end_time - start_time)/60
		print(f'yay! time window offsets for monkey {monkey_name} is completed')
		print(f'Took {elapsed_time:.4f} minutes to complete')
		# SAVE MONKEY STATS	
		with open(monkey_stats_path, 'wb') as handle:
			pickle.dump(monkey_stats, handle)
		print(f'Saved monkey stats to {monkey_stats_path} for band {band}')
	time_offset_end_time = time.time()
	# Calculate the elapsed time
	elapsed_time = (time_offset_end_time - time_offset_start_time)/60
	print(f'yay! time window offsets for all monkeys is completed')
	print(f'Took {elapsed_time:.4f} minutes to complete')


###########################subsample monkey L to match monkey A number of units


with open(os.path.join(main_dir, 'results/fig_7', 'subsample_seeds.json'), 'r') as f:
    subsample_seeds = json.load(f)
subsample_monkey_name='A'
main_monkey_name = 'L'

control_neurons = True


# timelags w_size = 10 ms
ref_duration = 200
w_size=10
## all bands but 1 took 41 minutes plus 13 min fro the last band
time_offset_start_time = time.time()
for band in list(BANDS.keys()):
	band_start_time = time.time()
	for seed in subsample_seeds:
		start_time = time.time()
		subsampled_monkey_stats_path = os.path.join(main_dir,f'results/fig_7/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
		with open (subsampled_monkey_stats_path, 'rb') as handle:
			subsampled_monkey_stats = pickle.load(handle)
		subsampled_indices = True
		ref_area = 'V4'
		condition_types = ['SNR','SNR_spont']
		for condition_type in condition_types:
			for date in subsampled_monkey_stats[condition_type]:
				process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
										subsampled_monkey_stats, w_size, control_neurons=control_neurons, monkey=main_monkey_name,
										recording_type='LFP', band=band, subsampled_indices=subsampled_indices)
		ref_area = 'V1'
		for condition_type in condition_types:
			for date in subsampled_monkey_stats[condition_type]:
				process_timelag_shenanigans(condition_type, date, ref_area, ref_duration, 
										subsampled_monkey_stats, w_size, control_neurons=control_neurons, monkey=main_monkey_name,
										recording_type='LFP', band=band, subsampled_indices=subsampled_indices)
		end_time = time.time()
		# Calculate the elapsed time
		elapsed_time = (end_time - start_time)/60
		print(f'yay! time window offsets for subsampled monkey L is completed')
		print(f'Took {elapsed_time:.4f} minutes to complete')
		# SAVE MONKEY STATS	
		with open(subsampled_monkey_stats_path, 'wb') as handle:
			pickle.dump(subsampled_monkey_stats, handle)
		print(f'Saved monkey stats to {subsampled_monkey_stats_path} for band {band} and seed {seed}')
	band_end_time = time.time()
	print(f'Took {(band_end_time - band_start_time)/60:.4f} minutes to complete band {band}')
time_offset_end_time = time.time()
# Calculate the elapsed time
elapsed_time = (time_offset_end_time - time_offset_start_time)/60
print(f'yay! time window offsets for all monkeys is completed')
print(f'Took {elapsed_time:.4f} minutes to complete')