
from fig_5_functions import get_property_dataset_type_monkey
from fig_3_functions import get_reli_condition
import numpy as np
import pandas as pd


# Morales-Gregorio/Chen bands
BANDS = {
    "low_2_12":   (2, 12),
    "beta_12_30": (12, 30),
    "gamma_30_45":(30, 45),
    "hgamma_55_95":(55, 95),
}
def make_df_timelags(monkey_stats_timelags, condition_type,ref_area, 
                    ref_ons,ref_offs, ref_duration, control_neurons=False, w_size=25,
                    spont_stim_off=300, specific_dates=None):
	w_size_ = f'_{w_size}' if w_size!=25 else ''
	if 'spont' in condition_type:
		real_dur = int(spont_stim_off/w_size)
		act_type = 'gray screen'
	elif 'SNR' in condition_type:
		real_dur = int(400/w_size)
		act_type = 'stimulus'
	ref_dur = int(ref_duration/w_size)
	data=[]
	
	for band in BANDS.keys():
		band_ = f'_{band}'
		for date, areas_data in monkey_stats_timelags[condition_type].items():
			if date in ['140819', '150819', '160819','250225']:
				continue
			if specific_dates is not None and date not in specific_dates:
				continue
			print(f'processing {condition_type} {date} {ref_area} {band}')
			relis = monkey_stats_timelags[get_property_dataset_type_monkey(condition_type)][date][ref_area]['split_half_r']
			snrs = monkey_stats_timelags[get_property_dataset_type_monkey(condition_type)][date][ref_area]['SNR_meanspont']
			if ref_area=='V1':
				pred_label='V4→V1'
				# seed_indices = np.concatenate(monkey_stats_timelags[extract_condition(condition_type)][date]['V1']['V1_chosen_indices'])
				if 'big_chosen_indices' in list(monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area].keys()):
					seed_indices = np.concatenate(monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area]['big_chosen_indices'])
				elif 'small_chosen_indices' in list(monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area].keys()):
					v1_indices = monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area]['small_chosen_indices']
					seed_indices = np.concatenate(np.tile(v1_indices, (10,1)))
					print('v1 small chosen indices',len(v1_indices), len(np.unique(v1_indices)))
				else:
					print(f'no V1 indices found for {condition_type} {date} {ref_area}. skipping this condition type and date')
					break
			elif ref_area=='V4':
				pred_label='V1→V4'
				# seed_indices = np.concatenate(monkey_stats_timelags[extract_condition(condition_type)][date]['V1']['V1_chosen_indices'])
				if 'big_chosen_indices' in list(monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area].keys()):
					seed_indices = np.concatenate(monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area]['big_chosen_indices'])
					print('v4 big chosen indices',len(seed_indices), len(np.unique(seed_indices)))
				elif 'small_chosen_indices' in list(monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area].keys()):
					v4_indices = monkey_stats_timelags[get_reli_condition(condition_type)][date][ref_area]['small_chosen_indices']
					seed_indices = np.concatenate(np.tile(v4_indices, (10,1)))
					print('v4 small chosen indices',len(v4_indices), len(np.unique(v4_indices)))
			for ref_on, ref_off in zip(ref_ons, ref_offs):
				values = areas_data[ref_area]
				if control_neurons is True:
					timelag_evars = values[f'timelag_evars_{ref_on}_{ref_off}_all_seeds{band_}{w_size_}']
				else:
					timelag_evars = values[f'timelag_evars_{ref_on}_{ref_off}_all_neurons{band_}{w_size_}']
				timelags = np.arange(-ref_on, -ref_on+(real_dur - ref_dur)+1)
				for t, timelag in enumerate(timelags): 
					if control_neurons is True:
						evars = np.concatenate(timelag_evars[t])
						lag0evars = np.concatenate(timelag_evars[np.argwhere(timelags==0)[0,0]])
						permutations = np.concatenate([np.ones([int(len(seed_indices)/10)],dtype=int)*count for count in range(10)])
						# print(permutations.shape)
						for n, evar in enumerate(evars):
							data.append({
								'Date': date,
								'Area': ref_area,
								'EV': evar,
								'Offset(ms)': timelag*w_size,
								'Direction':pred_label,
								'Ref_Times': f'{int(ref_on*w_size)}:{int(ref_off*w_size)}',
								'Mean_Norm_EV':evar/np.nanmean(lag0evars),
								'SNR': snrs[seed_indices[n]],
								'split-half r': relis[seed_indices[n]],
								'Permutation': permutations[n],
								'Neuron':seed_indices[n],
								'Band': band
							})
					else:
						lag0evars = timelag_evars[np.argwhere(timelags==0)[0,0]]
						for n, (evar, reli, snr) in enumerate(zip(timelag_evars[t], 
											relis, 
											snrs)):
							if ref_area=='V4':
								pred_label='V1→V4'
							elif ref_area=='V1':
								pred_label='V4→V1'
							data.append({
								'Date': date,
								'Area': ref_area,
								'EV': evar,
								'Offset(ms)': timelag*w_size,
								'SNR': snr,
								'split-half r': reli,
								'Direction':pred_label,
								'Ref_Times': f'{int(ref_on*w_size)}:{int(ref_off*w_size)}',
								'Mean_Norm_EV':evar/np.nanmean(lag0evars),
								'Neuron':n,
								'Band': band
							})
	# Create a DataFrame from the flattened data
	df = pd.DataFrame(data)
	return df