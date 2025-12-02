import numpy as np
import pandas as pd
import seaborn as sns
import os
import scipy.io as sio


from neuron_properties_functions import extract_mouse_name
from fig_2_functions import get_property_dataset_type


def retrive_mouse_beh_data(dataset_name, project_root, resp_or_spont='resp'):
	mt_stim_spont = sio.loadmat(os.path.join(project_root, 'data/stringer', dataset_name))
	motion_face_svd = mt_stim_spont['beh'][0]['face'][0]['motionSVD'][0][0]
	beh_run_speed = mt_stim_spont['beh'][0]['runSpeed'][0][:,0]

	n_planes = len(mt_stim_spont['stim'][0]['stimtimes'][0][0])
	stimtimes_arrays= mt_stim_spont['stim'][0]['stimtimes'][0][0][n_planes-1]
	stimtimes = [ stimtimes_arrays[i][0][0] for i in range(len(stimtimes_arrays))]  # Python list of planes * 3 + 3 arrays
	plane_idx = n_planes # chosen plane idx. just need to pick one. 11 is the best plane since it results in the least number of mismatches
	n_bins = 3 # number of frames to bin together according to authors 
	n_comp = motion_face_svd.shape[1]
	if resp_or_spont == 'resp':
		stim_onsets = []
		for b in range(3):
			# print(f"Block {b}\n")
			block_indices = {0:plane_idx, 1:plane_idx+n_planes,2:plane_idx+(n_planes*2)}
			#conver to zero-based index
			block_indices = {k:v-1 for k,v in block_indices.items()}
			# print(f'block indices: {block_indices}')

			block_times = stimtimes[block_indices[b]] - 1 # convert to zero-based index for actual block times
			# print(block_times[:200])
			# print(f'len of block_times before any filtering: {len(block_times)}')
			# print(f'len of resp: {mt_stim_spont["stim"]["resp"][0][0].shape[0]}')
			if len(block_times) ==mt_stim_spont['stim']['resp'][0][0].shape[0]: # if the block times are already the same length as resp, then just use them directly
				stim_onsets.extend(block_times)
				break
			else: # the data acquisition is divided into 3 blocks, each block should be exactly 1/3 of the total number of frames in resp
				stimonset_after_idx = np.concatenate([[0],np.argwhere(np.diff(block_times) > 4)[:,0] + 1])
				# print(len(stimonset_after_idx))
				# print(f'len of åstimonset after_idx: {len(stimonset_after_idx)}')
				# print(block_times[stimonset_after_idx])
				stim_onsets.extend(block_times)
				# print(f'len of stim_onsets after block {b}: {len(stim_onsets)}')
		if len(stim_onsets)!=mt_stim_spont["stim"]["resp"][0][0].shape[0]:
			raise ValueError(f"Length of stim_onsets {len(stim_onsets)} does not match length of resp {mt_stim_spont['stim']['resp'][0][0].shape[0]} after processing blocks")
		
		binned3_resp = np.zeros((len(stim_onsets),len(mt_stim_spont['Fsp'])))
		binned3_motion_face_svd = np.zeros((len(stim_onsets), n_comp))
		binned3_beh_run_speed = np.zeros((len(stim_onsets), 1))

		for i, onset in enumerate(stim_onsets):
			if dataset_name=='stimspont_M150824_MP019_20160323':
				onset = onset -1
			# grab up to 3 frames starting at `onset`
			# if you’re at the very end and get fewer than n_bins frames,
			# this will still average whatever is there
			binned3_motion_face_svd[i] = motion_face_svd[onset : onset + n_bins].mean(axis=0)
			binned3_beh_run_speed[i] = beh_run_speed[onset : onset + n_bins].mean(axis=0)
			binned3_resp[i] = np.nanmean(mt_stim_spont['Fsp'][:,onset : onset + n_bins],axis=1)
		return binned3_resp, binned3_motion_face_svd, binned3_beh_run_speed
	
	elif resp_or_spont == 'spont':	
		stimpt = mt_stim_spont['stimtpt'][:,0].astype(np.float64)  # Convert to float64 to allow negative value
		spontaneous_activity_unbinned_indices = np.where(stimpt == 0)[0]
		binned3_spont = np.zeros((len(spontaneous_activity_unbinned_indices[::3]),len(mt_stim_spont['Fsp'])))
		binned3_motion_face_svd_spont = np.zeros((len(spontaneous_activity_unbinned_indices[::3]), n_comp))
		binned3_beh_run_speed_spont = np.zeros((len(spontaneous_activity_unbinned_indices[::3]), 1))

		for i, onset in enumerate(spontaneous_activity_unbinned_indices[::3]):
			segment = mt_stim_spont['Fsp'][:,onset : onset + 3]
			binned3_spont[i] = segment.mean(axis=1)
			binned3_motion_face_svd_spont[i] = motion_face_svd[onset : onset + 3].mean(axis=0)
			binned3_beh_run_speed_spont[i] = beh_run_speed[onset : onset + 3].mean(axis=0)

		return binned3_spont, binned3_motion_face_svd_spont, binned3_beh_run_speed_spont
	else:
		raise ValueError("resp_or_spont must be either 'resp' or 'spont'")



def make_mouse_df(mouse_stats_, dataset_types=['ori32','natimg32']):
    """This function `make_mouse_df` is creating a DataFrame from the provided 
    `mouse_stats_` data for different dataset types. It iterates over the dataset types, 
    extracts relevant information for each mouse and area, and then appends this information 
    to a list called `data`. Finally, it creates a DataFrame `df_mouse_all` from the 
    collected data and returns it.

    Args:
        mouse_stats_ (_type_): _description_
        dataset_types (list, optional): _description_. Defaults to ['ori32','natimg32'].

    Returns:
        _type_: _description_
    """
    data = []
    for dataset_type in dataset_types:
        if '_spont' in dataset_type:
            act_type = 'gray screen'
        else:
            act_type = 'stimulus'
        for mouse, areas_data in mouse_stats_[dataset_type].items():
            mouse_name = extract_mouse_name(mouse)
            
            for area, values in areas_data.items():
                if area=='L23':
                    direction = 'L4→L2/3'
                    area_ = 'L2/3'
                else:
                    direction = 'L2/3→L4'
                    area_=area
                # Get the split-half correlation values for the current area
                split_half_rs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['split_half_r']
                # Get the SNR values for the current area
                SNRs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['SNR_meanspont']
                # Iterate over each pair of split-half correlation, SNR, EV, and null EV values
                for cell_n, (split_half_r, snr, evar, null_evar, evar_beh, evar_combined, evar_beh_timelag, evar_combined_timelag, evar_beh_null, evar_combined_null, evar_beh_timelag_null, evar_combined_timelag_null) in enumerate(zip(split_half_rs, SNRs,
                            values['evars'],values['evars_null'], values['evars_beh'], values['evars_combined'],values['evars_beh_timelag'],values['evars_combined_timelag'], values['evars_beh_null'], values['evars_combined_null'],values['evars_beh_timelag_null'],values['evars_combined_timelag_null'])):
                    # Append data for the actual experiment (control_shuffle = False)
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area_,
                        'EV': evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':False,
                        'Cell Number': cell_n,
                        'EV (behavior)': evar_beh,
						'EV (combined)': evar_combined,
						'Direction': direction,
						'EV (behavior w/ timelag)':evar_beh_timelag,
						'EV (combined w/ timelag)': evar_combined_timelag
                    })
                    # Append data for the shuffled experiment (control_shuffle = True)
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area_,
                        'EV': null_evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':True, 
                        'Cell Number': cell_n,
                        'EV (behavior)': evar_beh_null,
						'EV (combined)': evar_combined_null,
      					'Direction': direction,
						'EV (behavior w/ timelag)':evar_beh_timelag_null,
						'EV (combined w/ timelag)': evar_combined_timelag_null
                    })
    # Create a DataFrame from the flattened data
    df_mouse_all = pd.DataFrame(data)
    return df_mouse_all


import numpy as np
import pandas as pd

def open_closed_mask_25ms_from_csv(time_ms_25, epochs_csv, t0_ms_csv=0):
    """
    time_ms_25 : int64 array of 25 ms bin centers (absolute ms)
    epochs_csv : path to the authors' CSV with columns t_start, t_stop, state (seconds)
    t0_ms_csv  : offset (ms) if the CSV times aren't referenced to 0. Usually 0.

    Returns:
      open_mask_25 : boolean array, True=Open_eyes aligned to time_ms_25
    """
    df = pd.read_csv(epochs_csv)

    # Keep only Open_eyes intervals and convert to integer ms (stop is exclusive)
    df = df[df["state"] == "Open_eyes"].copy()
    starts_ms = np.round(df["t_start"].to_numpy(dtype=float)*1000.0).astype(np.int64) + t0_ms_csv
    stops_ms  = np.round(df["t_stop"].to_numpy(dtype=float )*1000.0).astype(np.int64)  + t0_ms_csv

    # Ensure sorted (CSV usually is, but be safe)
    order = np.argsort(starts_ms)
    starts_ms = starts_ms[order]
    stops_ms  = stops_ms[order]

    # For each bin center t, we are open if #(starts <= t) - #(stops <= t) > 0 (intervals are [start, stop))
    # Vectorized via searchsorted:
    t = np.asarray(time_ms_25, dtype=np.int64)
    n_started = np.searchsorted(starts_ms, t, side="right")
    n_stopped = np.searchsorted(stops_ms,  t, side="right")
    open_mask_25 = (n_started - n_stopped) > 0
    return open_mask_25

def three_plot(df, x, y, hue, ax,label_order, hue_order, **args):
    sns.violinplot(x=x, y=y, hue=hue, 
                data=df,ax=ax,order=label_order, hue_order=hue_order,
                inner='box',
                inner_kws={'box_width':2, 'whis_width':0.5,
                            'marker':'_', 'markersize':3,
                            'markeredgewidth':0.8,
                            },
                            **args
                            )
    ax.set(xlabel=None
        )

def get_property_dataset_type_monkey(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string 

def make_monkey_df(monkey_stats_, dataset_types=['SNR', 'RF_thin', 'RF_large']):
    """
    Create a DataFrame from the provided monkey statistics data for different dataset types.

    Args:
    - monkey_stats_ (_type_): Monkey statistics data.
    - dataset_types (list, optional): List of dataset types. Defaults to ['SNR', 'RF_thin', 'RF_large'].

    Returns:
    - pandas.DataFrame: DataFrame containing the collected monkey data.
    """
    data = []
    for dataset_type in dataset_types:
        # print(dataset_type)
        if 'spont' in dataset_type:
            act_type = 'gray screen'
        elif 'RS' in dataset_type:
            act_type = 'lights off'
        else:
            act_type = 'stimulus'
        for date, areas_data in monkey_stats_[dataset_type].items():
            # skip the dates with no V4 electrodes  
            if date in ['140819', '150819', '160819','250717']:
                        continue
            print(date)
            for area, values in areas_data.items():
                # print(area)
                if area =='V4':
                    direction= 'V1→V4'
                else:
                    direction = 'V4→V1'
                # Get the split-half orrelation values for the current area
                split_half_rs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['split_half_r']
                # Get the SNR values for the current area
                SNRs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['SNR_meanspont']
                for cell_n, (split_half_r, snr, evar, null_evar, evar_beh, evar_comb, evar_beh_null, evar_comb_null) in enumerate(zip(split_half_rs, SNRs,values['evars'],values['evars_null'],values['evars_beh'], values['evars_combined'],values['evars_beh_null'], values['evars_combined_null'],)):
                    # Append data for the actual experiment (control_shuffle = False)
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date':date,
                        'Area': area,
                        'EV': evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':False,
                        'Cell Number': cell_n,
                        'EV (behavior)': evar_beh,
						'EV (combined)': evar_comb,

						'Direction': direction
                        
                    })
                    # Append data for the shuffled experiment (control_shuffle = True
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date': date,
                        'Area': area,
                        'EV': null_evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':True, 
                        'Cell Number': cell_n,
                        'EV (behavior)': evar_beh_null,
						'EV (combined)': evar_comb_null,
						'Direction': direction
                    })
    # Create a DataFrame from the flattened data
    df_monkey_all = pd.DataFrame(data)
    return df_monkey_all
