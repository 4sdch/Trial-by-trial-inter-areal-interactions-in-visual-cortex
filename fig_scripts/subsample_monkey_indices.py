
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


# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)

import sys
sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)
from utils.neuron_properties_functions import store_mouse_alphas, get_evars_all_mice,create_empty_mouse_stats_dict, get_split_half_r_all_mice, get_SNR_all_mice
from utils.neuron_properties_functions import  get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates, create_empty_monkey_stats_dict, get_evar_monkey_all_dates, store_macaque_alphas

num_seeds = 20
random.seed(17)
seeds = [random.randint(1, 10000) for _ in range(num_seeds)]
#create 20 subsamples of monkey L to match monkey A and D site counts

import json
with open(os.path.join(results_dir, 'subsample_seeds.json'), 'w') as f:
    json.dump(seeds, f)

# monkey L_{A} should replicate the number of sites in monkey A along with the number of sessions per dataset
# monkey L 3 sessions of checkerboard (along with its gray screen, RS_open, and RS_closed) (SNR), 1 session of RF_thin (along with its gray screen), 1 session of RF_large (along with its gray screen)
# monkey A 1 session of checkerboard (along with its gray screen) (SNR), 1 session of RF_thin (along with its gray screen), and 1 session of RF_large (along with its gray screen)
# monkey L_{D} should replicate the number of sites in monkey D along with the number of sessions per dataset
# monkey D has 2 sessions of checkerboard (along with its gray screen and RS_open) (SNR)



def get_dates(condition_type, monkey='L'):
	if monkey=='L':
		if 'SNR' in condition_type or 'RS' in condition_type:
			return ['090817', '100817', '250717']
		elif 'large' in condition_type:
			return ['260617']
		elif 'thin' in condition_type:
			return ['280617']
		else:
			return None
	elif monkey=='A':
		if 'SNR' in condition_type:
			return ['041018']
		elif 'large' in condition_type:
			return ['280818']
		elif 'thin' in condition_type:
			return ['290818']
		else:
			return None
	elif monkey=='D':
		if 'SNR' in condition_type:
			return ['250225','260225']
		elif condition_type == 'RS_open':
			return ['250225','260225']
		else:
			return None
	else:
		raise ValueError('monkey not found')

def get_reli_condition(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string
    
def create_empty_monkey_stats_dict_subsampled(monkey = 'L', monkey_to_match = 'A', rng=None):
	"""
	Create an empty dictionary for monkey statistics.

	Returns:
	- monkey_stats (dict): Empty dictionary for monkey statistics.
	"""
	if rng is None:
		rng = random.Random(0)
	monkey_stats={}
	for dataset_type in ['SNR', 'RF_thin','RF_large']:
		big_dates = get_dates(dataset_type,monkey)
		small_dates = get_dates(dataset_type,monkey_to_match)
		if small_dates is None or big_dates is None:
			continue
		assert len(small_dates) <= len(big_dates), "The monkey to match has more sessions than the monkey being subsampled."
		monkey_stats[dataset_type]={}
		if len(small_dates) == len(big_dates):
			dates = big_dates
		else:
			# set random seed for reproducibility
			dates = rng.sample(big_dates, len(small_dates))
		for date in dates:
			monkey_stats[dataset_type][date]={}
			monkey_stats[dataset_type][date]['V4']={}
			monkey_stats[dataset_type][date]['V1']={}
	for dataset_type in ['SNR_spont','RS','RS_open','RS_closed','RF_thin_spont','RF_large_spont']:
		big_dates = get_dates(dataset_type,monkey)
		small_dates = get_dates(dataset_type,monkey_to_match)
		if small_dates is None or big_dates is None:
			continue
		## assign the same dates as the non-spontaneous version
		assert len(small_dates) <= len(big_dates), "The monkey to match has more sessions than the monkey being subsampled."
		monkey_stats[dataset_type]={}
		if len(small_dates) == len(big_dates):
			dates = big_dates
		else:
			# set random seed for reproducibility
			dates = monkey_stats[get_reli_condition(dataset_type)].keys()
		for date in dates:
			monkey_stats[dataset_type][date]={}
			monkey_stats[dataset_type][date]['V4']={}
			monkey_stats[dataset_type][date]['V1']={}
	return monkey_stats


monkey_site_counts_per_session = {
	'A':{
		'SNR':{
			'041018':{
				'V1': 571,
				'V4': 76,
			}},
		'RF_thin':{
			'290818':{
				'V1': 571,
				'V4': 76,
			}},
		'RF_large':{
			'280818':{
				'V1': 571,
				'V4': 76,
			}},
		},
	'D':{
		'SNR':{
			'250225':{
				'V1': 16,
				'V4': 17,
			},
			'260225':{
				'V1': 16,
				'V4': 18,
			}},
		},
	'L':{
		'SNR':{
			'090817':{
				'V1': 627,
				'V4': 96,
			},
			'100817':{
				'V1': 688,
				'V4': 115,
			},
			'250717':{
				'V1': 645,
				'V4': 86,
			}},
		'RF_thin':{
			'280617':{
				'V1': 645,
				'V4': 86,
			}},
		'RF_large':{
			'260617':{
				'V1': 645,
				'V4': 86,
			}},
		},
	}




# sanity check that site counts are correct
previous_results_dir = os.path.join(project_root,'results/fig_6/')
verbose = True
for monkey_name in ['L','A','D']:
	monkey_stats_path = os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')
	with open(monkey_stats_path, 'rb') as handle:
		monkey_stats = pickle.load(handle)
	for dataset_type in monkey_stats.keys():
		if monkey_name =='D' and dataset_type not in ['SNR','SNR_spont','RS_open']:
			continue
		if 'alphas' in dataset_type:
			continue
		for date in monkey_stats[dataset_type].keys():
			if date in ['140819','150819','160819']: # these are extra sessions for monkey A that we are not using
				continue
			if verbose:
				print(monkey_name, dataset_type, date)
			for area in ['V1','V4']:
				# print(monkey_stats[dataset_type][date][area].keys())
				num_sites = len(monkey_stats[dataset_type][date][area]['evars'])
				num_sites_dict = monkey_site_counts_per_session[monkey_name][get_reli_condition(dataset_type)][date][area]
				assert num_sites == num_sites_dict, f"Site count mismatch for {monkey_name} {date} {area} {dataset_type}: expected {num_sites_dict}, got {num_sites}"
				if verbose:
					print(f'num sites for {monkey_name} {date} {area} {dataset_type}: {num_sites}, expected: {num_sites_dict}')


# subsample monkey L to match monkey A
monkey_name = 'L'
monkeys_to_match = ['A','D']
neuron_properties = ['split_half_r','SNR_meanspont','1_vs_rest_r','var_within_trial_across_timepoints','var_across_trials_within_timepoints','max_corr_val','var_across_stimuli']

#create new dir called monkey_L_subsampled_to_A and monkey_L_subsampled_to_D

for monkey_to_match in monkeys_to_match:
	results_dir = os.path.join(project_root,f'results/fig_2/monkey_L_subsampled_to_{monkey_to_match}')
	os.makedirs(results_dir, exist_ok=True)
	for seed in seeds:
		log_rows = []
		# save seeds for reference
		rng = random.Random(seed)
		print(f'Creating subsampled monkey stats for {monkey_name} to match {monkey_to_match} with seed {seed}')
		subsampled_monkey_stats = create_empty_monkey_stats_dict_subsampled(monkey=monkey_name, monkey_to_match=monkey_to_match, rng=rng) # this monkey stats already subsampled for the same number of sessions per dataset type
		previous_monkey_stats_path = os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')
		with open(previous_monkey_stats_path, 'rb') as handle:
			full_monkey_stats = pickle.load(handle)
		for dataset_type in subsampled_monkey_stats.keys():
			full_dates = list(subsampled_monkey_stats[dataset_type].keys())  # preserves insertion order
			small_dates = list(get_dates(dataset_type, monkey_to_match))
			# thesse should be the same length
			assert len(full_dates) == len(small_dates), f"Date length mismatch for {dataset_type}: expected {len(small_dates)}, got {len(subsampled_monkey_stats[dataset_type].keys())}"
			for full_date, small_date in zip(full_dates, small_dates):
				for area in ['V1','V4']:
					num_sites_to_match = monkey_site_counts_per_session[monkey_to_match][get_reli_condition(dataset_type)][small_date][area]
					full_num_sites = len(full_monkey_stats[dataset_type][full_date][area]['evars'])
					assert num_sites_to_match <= full_num_sites, f"Cannot subsample {num_sites_to_match} from {full_num_sites} for {monkey_name} {full_date} {area} {dataset_type}"
					subsample_indices = sorted(rng.sample(range(full_num_sites), num_sites_to_match))
					log_rows.append({
					"target": monkey_to_match,
					"seed": seed,
					"dataset": dataset_type,
					"L_date": full_date,
					"AorD_date": small_date,
					"area": area,
					"n_pool": full_num_sites,
					"n_target": num_sites_to_match,
					"indices": ",".join(map(str, subsample_indices)),   # or "site_ids": ",".join(chosen_ids)
				})

					if verbose:
						print(f'subsampling {num_sites_to_match} from {full_num_sites} for {monkey_name} {full_date} {area} {dataset_type}')
					subsampled_monkey_stats[dataset_type][full_date][area]['monkey_L_subsample_indices'] = subsample_indices # store the subsample indices for reference when using function get_resp
					for key in full_monkey_stats[dataset_type][full_date][area].keys():
						if key in neuron_properties:
							# neuron properties could be a numpy array or a list of length num_sites OR a list of length num_sites with each element being a numpy array (e.g. split_half_r)
							if len(full_monkey_stats[dataset_type][full_date][area][key]) == full_num_sites:
								if isinstance(full_monkey_stats[dataset_type][full_date][area][key], np.ndarray):
									subsampled_monkey_stats[dataset_type][full_date][area][key] = full_monkey_stats[dataset_type][full_date][area][key][subsample_indices]
								elif isinstance(full_monkey_stats[dataset_type][full_date][area][key], list):
									subsampled_monkey_stats[dataset_type][full_date][area][key] = [full_monkey_stats[dataset_type][full_date][area][key][i] for i in subsample_indices]
								else:
									raise ValueError(f"Unexpected data type for {key} in {monkey_name} {full_date} {area} {dataset_type}: {type(full_monkey_stats[dataset_type][full_date][area][key])}")
							else:
								if full_monkey_stats[dataset_type][full_date][area][key][0].shape[0] == full_num_sites:
									subsampled_monkey_stats[dataset_type][full_date][area][key] = np.array([full_monkey_stats[dataset_type][full_date][area][key][i][subsample_indices] for i in range(len(full_monkey_stats[dataset_type][full_date][area][key]))])
		subsampled_monkey_stats["meta"] = {
											"target": monkey_to_match,
											"source": monkey_name,
											"seed": seed,
											"strata": ["dataset_type","date","area"],
											"generator": "stratified_subsample_v1",
										}
		# now save the subsampled monkey stats to results dir
		subsampled_monkey_stats_path = os.path.join(results_dir, f'monkey_{monkey_name}_subsampled_to_{monkey_to_match}_seed{seed}_stats.pkl')
		with open(subsampled_monkey_stats_path, 'wb') as handle:
			pickle.dump(subsampled_monkey_stats, handle)
		log_df = pd.DataFrame(log_rows)
		log_path = os.path.join(results_dir, f'subsample_log_seed{seed}.csv')
		log_df.to_csv(log_path, index=False)
		log_rows.clear()
				
	print(f'finished creating subsampled monkey stats for {monkey_name} to match {monkey_to_match} with all seeds')

######## match indices in SNR to be the same for SNR_spont and RS ########

with open(os.path.join(project_root, 'results/fig_5', 'subsample_seeds.json'), 'r') as f:
    subsample_seeds = json.load(f)
    
main_monkey_name = 'L'
subsample_monkey_name = 'D'


for seed in subsample_seeds:
	subsampled_monkey_stats_path = os.path.join(project_root,f'results/fig_5/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
	with open(subsampled_monkey_stats_path, 'rb') as f:
		subsampled_monkey_stats = pickle.load(f)
	spont_dataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and ('spont' in k or 'RS' in k)]
	for spont_dataset_type in spont_dataset_types_:
		print(f'Processing {main_monkey_name} {spont_dataset_type} seed {seed} to match indices with {get_reli_condition(spont_dataset_type)}')
		non_spont_dataset_type = get_reli_condition(spont_dataset_type)
		for date in subsampled_monkey_stats[spont_dataset_type].keys():
			for area in ['V1','V4']:
				assert 'monkey_L_subsample_indices' in subsampled_monkey_stats[non_spont_dataset_type][date][area], f"Subsample indices not found for {main_monkey_name} {date} {area} {non_spont_dataset_type}"
				previous_subsample_indices = subsampled_monkey_stats[spont_dataset_type][date][area]['monkey_L_subsample_indices']
				subsample_indices = subsampled_monkey_stats[non_spont_dataset_type][date][area]['monkey_L_subsample_indices']
				subsampled_monkey_stats[spont_dataset_type][date][area]['monkey_L_subsample_indices'] = subsample_indices
				print('previous spont indices:', previous_subsample_indices[:10], 'new spont indices:', subsample_indices[:10])
				print('previous length of spont indices:', len(previous_subsample_indices), 'new length of spont indices:', len(subsample_indices))
	with open(subsampled_monkey_stats_path, 'wb') as f:
		pickle.dump(subsampled_monkey_stats, f)
	print('finished matching spont indices to non-spont indices for all seeds')