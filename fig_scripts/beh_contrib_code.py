import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import scipy.stats as stats
import os
import scipy.io as sio
import time
from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)

os.chdir(project_root)


import sys
sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)
from utils.neuron_properties_functions import extract_mouse_name
from utils.ridge_regression_functions import get_predictions_evars_parallel
import utils.mouse_data_functions as cs
from utils.macaque_data_functions import get_resps, get_get_condition_type
from utils.beh_contrib_functions import retrive_mouse_beh_data, open_closed_mask_25ms_from_csv
from utils.neuron_properties_functions import get_SNR_all_mice, get_split_half_r_all_mice

save_figs = True
temp_fig_dir = os.path.join(project_root, 'results/paper_figures/revisions/')

mouse_stats_path = os.path.join(project_root, 'results/fig_2',f'mouse_stats.pkl')
with open(mouse_stats_path, 'rb') as f:
	mouse_stats = pickle.load(f)




############################### mouse #########################################

dataset_type = 'stimspont'
mouse_stats[dataset_type]={}
mouse_stats[f'{dataset_type}_spont']={}
mt = cs.mt_retriever(project_root, dataset_type=dataset_type)
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

# get SNR and responsiveness for each neuron in each mouse
# plot evar distributions for datasets in stimspont

dataset_type = 'stimspont'


get_SNR_all_mice(project_root, mouse_stats, dataset_types=[dataset_type])
get_split_half_r_all_mice(project_root, mouse_stats, dataset_types=[dataset_type])

with open(mouse_stats_path, 'wb') as f:
	pickle.dump(mouse_stats, f)


dataset_type = 'stimspont'
mt = cs.mt_retriever(project_root, dataset_type=dataset_type)
dataset_names = mt.mts.keys()
sample_size=500
n_splits=10
frames_reduced=5
# resp only

area='L23'
area2='L4'

activity_types = ['resp','spont']
control_shuffles = [True, False]

for activity_type in activity_types:
	for control_shuffle in control_shuffles:
		control_con = '_null' if control_shuffle is True else ''
		spont_con = '_spont' if activity_type =='spont' else ''
		for d_name in dataset_names:
			resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, d_name)
			_, motion_face_svd, beh_run_speed = retrive_mouse_beh_data(d_name, project_root, resp_or_spont=activity_type)
			
			if activity_type=='resp':
				#move when istim is max 
				istim = mt.mt['stim'][0]['istim'][0]
				istim -= 1 # get out of MATLAB convention
				istim = istim[:,0]
				nimg = istim.max() # these are blank stims (exclude them) 
				motion_face_svd = motion_face_svd[istim < nimg]
				beh_run_speed = beh_run_speed[istim < nimg]

			alpha = mouse_stats['mouse_alphas'][sample_size][(extract_mouse_name(d_name))][area]
			alpha2 = mouse_stats['mouse_alphas'][sample_size][(extract_mouse_name(d_name))][area2]
			# prediction using L4 to predict L23
			_, evars = get_predictions_evars_parallel(resp_L4, resp_L23, alpha=alpha, n_splits=n_splits, frames_reduced=frames_reduced, control_shuffle=control_shuffle)
			_, evars2 = get_predictions_evars_parallel(resp_L23, resp_L4, alpha=alpha2, n_splits=n_splits, frames_reduced=frames_reduced, control_shuffle=control_shuffle)
			# prediction using behavior to predict L23
			X_beh = np.hstack([motion_face_svd, beh_run_speed])
			_, evars_beh = get_predictions_evars_parallel(X_beh, resp_L23, alpha=alpha, n_splits=n_splits, frames_reduced=frames_reduced, control_shuffle=control_shuffle)
			_, evars_beh2 = get_predictions_evars_parallel(X_beh, resp_L4, alpha=alpha2, n_splits=n_splits, frames_reduced=frames_reduced, control_shuffle=control_shuffle)
			# prediction using both L4 and behavior to predict L23
			X_combined = np.hstack([resp_L4, X_beh])
			_, evars_combined = get_predictions_evars_parallel(X_combined, resp_L23, alpha=alpha, n_splits=n_splits, frames_reduced=frames_reduced, control_shuffle=control_shuffle)
			_, evars_combined2 = get_predictions_evars_parallel(np.hstack([resp_L23, X_beh]), resp_L4, alpha=alpha2, n_splits=n_splits, frames_reduced=frames_reduced, control_shuffle=control_shuffle)
			print(f"\nDataset: {d_name}")
			# print(f"Mean EVar for L4 to L23 prediction: {np.nanmean(evars):.4f}. Mean EVar for L23 to L4 prediction: {np.nanmean(evars2):.4f}")
			print(f"Mean EVar for Behavior to L23 prediction: {np.nanmean(evars_beh):.4f}. Mean EVar for Behavior to L4 prediction: {np.nanmean(evars_beh2):.4f}")
			print(f"Mean EVar for L4 + Behavior to L23 prediction: {np.nanmean(evars_combined):.4f}. Mean EVar for L23 + Behavior to L4 prediction: {np.nanmean(evars_combined2):.4f}")
			
			# save evars to mouse_stats
			mouse_stats[dataset_type + spont_con][d_name][area]['evars' + control_con ]=evars
			mouse_stats[dataset_type + spont_con][d_name][area2]['evars' + control_con ]=evars2
			mouse_stats[dataset_type + spont_con][d_name][area]['evars_beh' + control_con ]=evars_beh
			mouse_stats[dataset_type + spont_con][d_name][area2]['evars_beh' + control_con ]=evars_beh2
			mouse_stats[dataset_type + spont_con][d_name][area]['evars_combined' + control_con ]=evars_combined
			mouse_stats[dataset_type + spont_con][d_name][area2]['evars_combined' + control_con ]=evars_combined2

			# save mouse_stats
			mouse_stats_path = os.path.join(project_root, 'results/fig_2',f'mouse_stats.pkl')
			with open(mouse_stats_path, 'wb') as f:
				pickle.dump(mouse_stats, f)
			print(f"Mouse stats saved to {mouse_stats_path}")


############################### monkey ###############################################

all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 300, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':300, 
                    'RF_large_spont':300}
#depending on the dataset type, there are different times of autocorrelation to mitigate
all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}



monkey_name='L'
monkey=monkey_name
monkey_stats_path = os.path.join(project_root, 'results/fig_2',f'monkey_{monkey_name}_stats.pkl')
with open(monkey_stats_path, 'rb') as f:
	monkey_stats = pickle.load(f)

start_time = time.time()
area='V4'
area2='V1'
frame_size=500
n_splits=10
w_size=25
dataset_types =['RS_open']

monkey_alphas=monkey_stats['monkey_alphas'][frame_size]
alpha = monkey_alphas[area]
alpha2=monkey_alphas[area2]    

for control_shuffle in [True, False]:
	control_con = '_null' if control_shuffle is True else ''
	for condition_type in dataset_types:
		if 'monkey' in condition_type:
			continue
		for date in monkey_stats[condition_type]:
			eyes_open_closed_path = f'data/chen/metadata/monkey_{monkey}/epochs_{monkey}_RS_{date}.csv'
			npz = np.load(f'data/chen/monkey_{monkey}/{date}/{monkey}_RS_{date}_pupil_25ms.npz', allow_pickle=True)
			time_ms_25 = npz["time_ms_25"]
			pupil_40hz = npz["pupil_40hz"]
			# add new axis to pupil so that its shape is (time, 1)
			pupil_40hz = pupil_40hz[:, np.newaxis]
			open_mask_25 = open_closed_mask_25ms_from_csv(
				time_ms_25,
				epochs_csv=eyes_open_closed_path,  # your CSV path
				t0_ms_csv=0                                # change if your CSV time origin isn’t 0
			)
			# now subset:
			pupil_open_25  = pupil_40hz[open_mask_25]
			# skip certain dates for now
			if date == '041018' and 'RS' in condition_type or date =='250717':
				continue
			get_condition_type = get_get_condition_type(condition_type)
			resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date, 
										w_size=w_size,stim_off=all_ini_stim_offs[condition_type], monkey=monkey)
			if condition_type ==monkey_alphas['condition_type_used'] and date==monkey_alphas['date_used']:
				resp_V4=resp_V4[frame_size:]
				resp_V1=resp_V1[frame_size:]
			_, evars = get_predictions_evars_parallel(resp_V1, resp_V4, n_splits=n_splits, alpha=alpha,
												frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
			_, evars2 = get_predictions_evars_parallel(resp_V4, resp_V1, n_splits=n_splits, alpha=alpha2,
												frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
			monkey_stats[condition_type][date][area]['evars' + control_con]=evars
			monkey_stats[condition_type][date][area2]['evars' + control_con]=evars2
			_, evars_beh = get_predictions_evars_parallel(pupil_open_25[:len(resp_V1)], resp_V4, n_splits=n_splits, alpha=alpha,
												frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
			_, evars2_beh = get_predictions_evars_parallel(pupil_open_25[:len(resp_V1)], resp_V1, n_splits=n_splits, alpha=alpha2,
												frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
			monkey_stats[condition_type][date][area]['evars_beh' + control_con]=evars_beh
			monkey_stats[condition_type][date][area2]['evars_beh' + control_con]=evars2_beh

			# predict using combined  and pupil
			pred_input = np.concatenate((pupil_open_25[:len(resp_V1)], resp_V1), axis=1)
			_, evars_comb = get_predictions_evars_parallel(pred_input, resp_V4, n_splits=n_splits, alpha=alpha,
												frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
			monkey_stats[condition_type][date][area]['evars_combined' + control_con]=evars_comb
			pred_input2 = np.concatenate((pupil_open_25[:len(resp_V1)], resp_V4), axis=1)
			_, evars2_comb = get_predictions_evars_parallel(pred_input2, resp_V1, n_splits=n_splits, alpha=alpha2,
												frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
			monkey_stats[condition_type][date][area2]['evars_combined' + control_con]=evars2_comb

			print(date,'evar done')
		print(condition_type, 'done')
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f'yay! it took {elapsed_time:.2f} minutes to finish all dataset types!')

# save monkey stats
with open(monkey_stats_path, 'wb') as f:
	pickle.dump(monkey_stats, f)
print('monkey stats saved!')