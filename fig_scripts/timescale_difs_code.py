
from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)


import pickle
import os
results_dir = os.path.join(project_root,'results/fig_3/')


# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)

import sys
sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)
from utils.neuron_properties_functions import  get_evar_monkey_all_dates
from utils.fig_3_functions import get_directionality_evars_monkey


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

spont_dataset_type_dict = {'A': ['SNR_spont','RF_thin_spont', 'RF_large_spont', 'RS_open', 'RS_closed'],
					'L': ['SNR_spont','RF_thin_spont', 'RF_large_spont', 'RS_open', 'RS_closed'],
					'D': ['SNR_spont','RS_open']
					}

dataset_type_dict = {'A': resp_dataset_type_dict['A'] + spont_dataset_type_dict['A'],
					'L': resp_dataset_type_dict['L'] + spont_dataset_type_dict['L'],
					'D': resp_dataset_type_dict['D'] + spont_dataset_type_dict['D']
					}
date_used_dict = {'L': '090817', 'A': '041018', 'D': '250225'}
condition_type_used_dict = {'L':'RS','A':'SNR','D':'RS'}

bins=[10,25,50,100,200]
monkey_names = ['L','A','D' ]
for monkey_name in monkey_names:
	monkey_stats_path = os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl')
	with open(monkey_stats_path, 'rb') as handle:
			monkey_stats = pickle.load(handle)
	for w_size in bins:
		get_evar_monkey_all_dates(monkey_stats, w_size=w_size, control_shuffle=False, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])
		get_evar_monkey_all_dates(monkey_stats, w_size=w_size, control_shuffle=True, monkey=monkey_name, dataset_types=dataset_type_dict[monkey_name])
	with open(monkey_stats_path, 'wb') as file:
		pickle.dump(monkey_stats, file)


bins=[5,10,25,50,100,200]
for monkey_name in ['L','A']:
	monkey_stats_path = os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl')
	with open(monkey_stats_path, 'rb') as handle:
			monkey_stats = pickle.load(handle)
	for w_size in bins:
		get_directionality_evars_monkey(monkey_stats, w_size = w_size, control_shuffle=False, dataset_types=resp_dataset_type_dict[monkey_name], monkey=monkey_name)
		get_directionality_evars_monkey(monkey_stats, w_size = w_size, control_shuffle=True, dataset_types=resp_dataset_type_dict[monkey_name], monkey=monkey_name)
	with open(monkey_stats_path, 'wb') as file:
		pickle.dump(monkey_stats, file)