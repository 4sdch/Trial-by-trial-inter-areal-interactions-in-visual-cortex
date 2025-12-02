from set_home_directory import get_project_root_homedir_in_sys_path
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")
if project_root is None:
    raise RuntimeError(f"Project root not found: ensure a folder named '{project_root}' exists in one of the sys.path entries.")
print("Project root found:", project_root)
from joblib import Parallel, delayed
import pickle
import numpy as np
import time
import sys
import os 
import json

results_dir = os.path.join(project_root,'results/fig_5/')

# ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)
os.chdir(project_root)


sys.path.insert(0,os.path.join(main_dir,'utils/'))
sys.path.insert(0,main_dir)

from utils.neuron_properties_functions import create_empty_mouse_stats_dict, get_split_half_r_all_mice, get_SNR_all_mice, get_max_corr_vals_all_mice, get_evars_all_mice, get_evar_monkey_all_dates, store_mouse_alphas
from utils.neuron_properties_functions import create_empty_monkey_stats_dict, get_SNR_monkey_all_dates, get_split_half_r_monkey_all_dates, get_max_corr_vals_monkey_all_dates, get_evar_monkey_all_dates, store_macaque_alphas
from utils.neuron_properties_functions import get_dates, store_macaque_alphas, get_variance_within_trial_across_timepoints, get_variance_within_timepoints_across_trials,get_RF_variance_across_stimuli
import utils.mouse_data_functions as cs
from utils.fig_5_functions import process_evar_subsample_seeds
from utils.macaque_data_functions import get_resps, get_get_condition_type
from joblib import Parallel, delayed

########################### MOUSE PREDICTIONS ####################################
previous_results_dir = os.path.join(project_root,'results/fig_4/')
dataset_types = ['ori32', 'natimg32']
min_stimulus_frame_lengths = []
for dataset_type in dataset_types:
    mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
    mousenames= sorted(mt.filenames)
    for mouse in mousenames:
        _, resp_L23,_,  _, resp_L4 = mt.retrieve_layer_activity('resp', mouse)
        min_stimulus_frame_lengths.append(resp_L23.shape[0])
        _, resp_L23_spont, _, _, resp_L4_spont = mt.retrieve_layer_activity('spont', mouse)
        if len(resp_L23_spont)<1000:
            continue
        print(resp_L23.shape[0], resp_L23_spont.shape[0])

print('no need to control for frame size as the frames are similar in size')
if not os.path.exists(os.path.join(previous_results_dir, 'mouse_stats.pkl')):
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
    with open(os.path.join(previous_results_dir, 'mouse_stats.pkl'), 'rb') as handle:
        mouse_stats = pickle.load(handle)
    # save the mouse stats
    with open(os.path.join(results_dir, 'mouse_stats.pkl'), 'wb') as handle:
        pickle.dump(mouse_stats, handle) 
        
        

# ############################# MONKEY PREDICTIONS ####################################

start_time = time.time()
# monkey_names = ['L','A']
monkey_names = ['D'] 
date_used_dict = {'L': '090817', 'A': '041018', 'D': '260225'} 
condition_type_used_dict = {'L':'RS','A':'SNR','D':'RS'}

areas = ['V1','V4']
w_size=25
condition_type1 = 'SNR'
condition_type2 = 'SNR_spont'
initial_seed = 17
sample_size=500
rng = np.random.default_rng(initial_seed)
num_seeds = 10
area='V4'
area2='V1'
seed=17

for monkey_name in monkey_names:
    specific_dataset_types = ['SNR','RF_large','RF_thin']
    spont_types = ['RS', 'RS_open', 'RS_closed', 'SNR_spont']
    if monkey_name in ['D']:
        specific_dataset_types = ['SNR']
        previous_results_dir = os.path.join(project_root,'results/fig_3/') # using figure 3 here because monkey D does not have fig 4 results
    else:
        previous_results_dir = os.path.join(project_root,'results/fig_4/')
    if monkey_name =='A':
        spont_types = ['SNR_spont']
    elif monkey_name == 'D':
        spont_types = ['SNR_spont', 'RS_open'] # only had a 3 second period in 1 day and the next day had no eyes closed moments 
    print(f'Processing monkey {monkey_name}')
    
    if not os.path.exists(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl')):
        monkey_stats= create_empty_monkey_stats_dict(monkey=monkey_name)
        get_SNR_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=specific_dataset_types)
        get_split_half_r_monkey_all_dates(monkey_stats, monkey=monkey_name, specific_dataset_types=specific_dataset_types)
        store_macaque_alphas(main_dir, monkey_stats, verbose=True, monkey=monkey_name, date_used=date_used_dict[monkey_name], condition_type_used=condition_type_used_dict[monkey_name])
        get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types = specific_dataset_types + spont_types) 
        get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types = specific_dataset_types+ spont_types)
        get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types = specific_dataset_types+ spont_types, control_shuffle=True)

    else:
        with open(os.path.join(previous_results_dir, f'monkey_{monkey_name}_stats.pkl'), 'rb') as handle:
            monkey_stats = pickle.load(handle)
        get_max_corr_vals_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types = specific_dataset_types + spont_types) 
        get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types = spont_types)
        get_evar_monkey_all_dates(monkey_stats, monkey=monkey_name, dataset_types = spont_types, control_shuffle=True)
    
    dates = get_dates('SNR', monkey_name)
    SNR_lengths = []
    SNR_spont_lengths = []
    RS_closed_lengths = [] if monkey_name not in ['A','D'] else None
    min_lengths = {}
    for d, date in enumerate(dates):
        # print(area)
        resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(condition_type1), date=date, w_size=w_size, monkey=monkey_name)
        SNR_lengths.append(resp_V4.shape[0])
        resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(condition_type2), date=date, w_size=w_size, monkey=monkey_name)
        SNR_spont_lengths.append(resp_V4.shape[0])
        min_lengths[date]=min(SNR_lengths[-1],SNR_spont_lengths[-1])
        if RS_closed_lengths is not None:
            resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type('RS_closed'), date=date, w_size=w_size, monkey=monkey_name)
            RS_closed_lengths.append(resp_V4.shape[0])
            min_lengths[date]=min(min_lengths[date], RS_closed_lengths[-1])
    
    print('SNR', SNR_lengths)
    print('SNR_spont', SNR_spont_lengths)
    if RS_closed_lengths is not None:
        print('RS_closed', RS_closed_lengths)
    print(f'min lengths for monkey {monkey_name}',min_lengths)
    
    # Generate 10 random seed values
    
    seeds = rng.integers(low=0, high=np.iinfo(np.int32).max, size=num_seeds)

    

    alpha=monkey_stats['monkey_alphas'][sample_size]['V4']
    alpha2=monkey_stats['monkey_alphas'][sample_size]['V1']
    
    condition_types = ['SNR'] + spont_types
    for condition_type in condition_types:
        for date in monkey_stats[condition_type]:
            resp_V4, resp_V1, =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size,monkey=monkey_name)
            if resp_V4.shape[0]==min_lengths[date]:
                monkey_stats[condition_type][date][area]['spont_comparison_evars']=monkey_stats[condition_type][date][area]['evars']
                monkey_stats[condition_type][date][area2]['spont_comparison_evars']=monkey_stats[condition_type][date][area2]['evars']
                continue
            print(f'condition type {condition_type} date {date} resp_V4 shape {resp_V4.shape} resp_V1 shape {resp_V1.shape}')
            if condition_type ==monkey_stats['monkey_alphas'][sample_size]['condition_type_used'] and date==monkey_stats['monkey_alphas'][sample_size]['date_used']:
                resp_V4=resp_V4[sample_size:]
                resp_V1=resp_V1[sample_size:]
                
            results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths, seed, alpha, 
                                                                                alpha2, condition_type) for seed in seeds)
            all_v4_evars = np.array([a for a,_ in results])
            all_v1_evars = np.array([a for _,a in results])
            monkey_stats[condition_type][date][area]['spont_comparison_evars']=all_v4_evars
            monkey_stats[condition_type][date][area2]['spont_comparison_evars']=all_v1_evars
        print(f'{condition_type} done')
    print('performing shuffle control')
    ### shuffle control
    for condition_type in condition_types:
        for date in monkey_stats[condition_type]:
            resp_V4, resp_V1, =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size, monkey=monkey_name)
            if resp_V4.shape[0]==min_lengths[date]:
                monkey_stats[condition_type][date][area]['spont_comparison_evars_null']=monkey_stats[condition_type][date][area]['evars_null']
                monkey_stats[condition_type][date][area2]['spont_comparison_evars_null']=monkey_stats[condition_type][date][area2]['evars_null']
                continue
            if condition_type ==monkey_stats['monkey_alphas'][sample_size]['condition_type_used'] and date==monkey_stats['monkey_alphas'][sample_size]['date_used']:
                resp_V4=resp_V4[sample_size:]
                resp_V1=resp_V1[sample_size:]
            
            results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths, seed, alpha,
                                                                                alpha2, condition_type, control_shuffle=True) for seed in seeds)
            all_v4_evars = np.array([a for a,_ in results])
            all_v1_evars = np.array([a for _,a in results])
            monkey_stats[condition_type][date][area]['spont_comparison_evars_null']=all_v4_evars
            monkey_stats[condition_type][date][area2]['spont_comparison_evars_null']=all_v1_evars
        print(f'{condition_type} done')
    with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
            pickle.dump(monkey_stats, handle)
            
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'Yay! work for monkey {monkey_name} is completed. Took {elapsed_time:.4f} minutes to complete') 
    
    
    
    ############################ MONKEY SUPPLEMENTAL MOVING BARS #############################
    if monkey_name in ['L','A']:
        print('Monkey supplemental')
        start_time = time.time()
        dates = get_dates('RF_thin', monkey_name) + get_dates('RF_large', monkey_name)
        get_condition_type = 'RF_spont'
        min_lengths = {}
        for d, date in enumerate(dates):
            # print(area)
            resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date, w_size=w_size,monkey=monkey_name)
            min_lengths[date]=resp_V4.shape[0]
        print('min lengths',min_lengths)
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
        print(f'Took {elapsed_time:.4f} minutes to complete')
        condition_types = ['RF_large','RF_thin']
        for condition_type in condition_types:
            for date in monkey_stats[condition_type]:
                resp_V4, resp_V1, =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size, monkey=monkey_name)
                results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths, seed, alpha, alpha2, condition_type) for seed in seeds)
                all_v4_evars = np.array([a for a,_ in results])
                all_v1_evars = np.array([a for _,a in results])
                monkey_stats[condition_type][date][area]['spont_comparison_evars']=all_v4_evars
                monkey_stats[condition_type][date][area2]['spont_comparison_evars']=all_v1_evars
            print(f'{condition_type} done')
        print('Monkey supplemental shuffle control')
        ### shuffle control
        for condition_type in condition_types:
            for date in monkey_stats[condition_type]:
                resp_V4, resp_V1, =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size, monkey=monkey_name)
                results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths, seed, 
                                                                                    alpha, alpha2, condition_type, control_shuffle=True) for seed in seeds)
                all_v4_evars = np.array([a for a,_ in results])
                monkey_stats[condition_type][date][area]['spont_comparison_evars_null']=all_v4_evars
                monkey_stats[condition_type][date][area2]['spont_comparison_evars_null']=all_v1_evars
            print(f'{condition_type} done')
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
        print(f'Yay! work is completed. Took {elapsed_time:.4f} minutes to complete') 
    # save the monkey stats
    with open(os.path.join(results_dir, f'monkey_{monkey_name}_stats.pkl'), 'wb') as handle:
            pickle.dump(monkey_stats, handle)
    
    
    
############################## subsampled monkey L to match site count of A and D ####################################
    
def get_property_dataset_type_monkey(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string 
    
results_dir = os.path.join(project_root,'results/fig_7/')

with open(os.path.join(results_dir, 'subsample_seeds.json'), 'r') as f:
    subsample_seeds = json.load(f)

main_monkey_name = 'L'
subsample_monkey_names = ['A','D']

start_time = time.time()
initial_seed = 17
rng = np.random.default_rng(initial_seed)
num_seeds = 10
subsampled_indices = True
w_size=25
sample_size=500
area='V4'
area2='V1'
for subsample_monkey_name in subsample_monkey_names:
    print('Subsampling monkey L to match monkey ', subsample_monkey_name)
    for seed in subsample_seeds:
        seed_start_time = time.time()
        subsampled_monkey_stats_path = os.path.join(project_root,f'results/fig_5/monkey_L_subsampled_to_{subsample_monkey_name}',f'monkey_{main_monkey_name}_subsampled_to_{subsample_monkey_name}_seed{seed}_stats.pkl')
        with open (subsampled_monkey_stats_path, 'rb') as handle:
            subsampled_monkey_stats = pickle.load(handle)

        stim_datataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and 'spont' not in k and 'RS' not in k]
        spont_dataset_types_ = [k for k in subsampled_monkey_stats.keys() if k not in ['meta','monkey_alphas','monkey_alphas_glm','monkey_directionality_alphas'] and ('spont' in k or 'RS' in k)]
        get_max_corr_vals_monkey_all_dates(subsampled_monkey_stats, monkey=main_monkey_name, dataset_types=stim_datataset_types_, subsampled_indices=subsampled_indices)
        get_max_corr_vals_monkey_all_dates(subsampled_monkey_stats, monkey=main_monkey_name, dataset_types=spont_dataset_types_, subsampled_indices=subsampled_indices)
        get_variance_within_trial_across_timepoints(subsampled_monkey_stats, specific_dataset_types = stim_datataset_types_, 
                                                monkey=main_monkey_name, subsampled_indices = subsampled_indices)
        get_variance_within_timepoints_across_trials(subsampled_monkey_stats,specific_dataset_types = stim_datataset_types_, 
                                                 monkey=main_monkey_name, subsampled_indices=subsampled_indices)
        if any('RF' in dt for dt in stim_datataset_types_):
            get_RF_variance_across_stimuli(subsampled_monkey_stats, specific_dataset_types = [dt for dt in stim_datataset_types_ if 'RF' in dt],
                                        monkey=main_monkey_name, subsampled_indices = subsampled_indices)
        dates = list(subsampled_monkey_stats['SNR'].keys()) # use checkerboard dates since that is the same date as gray screen and resting state presentation (if resting state done)
        
        SNR_spont_dataset_types = ['SNR'] + [dt for dt in spont_dataset_types_ if 'RF' not in dt]
        min_lengths = {}
        for d, date in enumerate(dates):
            # print(area)
            SNR_and_spont_lengths = {}
            for dtset in ['SNR']+ SNR_spont_dataset_types:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dtset), date=date, w_size=w_size, monkey=main_monkey_name)
                SNR_and_spont_lengths[dtset] = resp_V4.shape[0]
            # get min length and dataset type. store min length for each date and print dtset that is min
            min_dtset = min(SNR_and_spont_lengths, key=SNR_and_spont_lengths.get)
            min_lengths[date]=SNR_and_spont_lengths[min_dtset]
            print(f'date {date} min length {min_lengths[date]} from dataset type {min_dtset}')
        
        # Generate 10 random seed values
    
        seeds = rng.integers(low=0, high=np.iinfo(np.int32).max, size=num_seeds)

        alpha=subsampled_monkey_stats['monkey_alphas'][sample_size]['V4']
        alpha2=subsampled_monkey_stats['monkey_alphas'][sample_size]['V1']
        
        ## checkerbboard first 
        for condition_type in SNR_spont_dataset_types: # need to redo this because subsampled_indices are different for spont and stim. will redo other figs with the correct indices
            for date in subsampled_monkey_stats[condition_type]:
                resp_V4, resp_V1, =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size,monkey=main_monkey_name)
                if resp_V4.shape[0]==min_lengths[date]:
                    subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars']=subsampled_monkey_stats[condition_type][date][area]['evars']
                    subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars']=subsampled_monkey_stats[condition_type][date][area2]['evars']
                    subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars_null']=subsampled_monkey_stats[condition_type][date][area]['evars_null']
                    subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars_null']=subsampled_monkey_stats[condition_type][date][area2]['evars_null']
                    continue
                if subsampled_indices is True:
                    assert subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V1'][f'monkey_{main_monkey_name}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                    assert subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V4'][f'monkey_{main_monkey_name}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                    subsampled_indices_dict = {'V1':subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V1'][f'monkey_{main_monkey_name}_subsample_indices'],
                                                'V4':subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V4'][f'monkey_{main_monkey_name}_subsample_indices']}
                    resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                    resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
                print(f'condition type {condition_type} date {date} resp_V4 shape {resp_V4.shape} resp_V1 shape {resp_V1.shape}')
                results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths, seed, alpha, 
                                                                                    alpha2, condition_type) for seed in seeds)
                all_v4_evars = np.array([a for a,_ in results])
                all_v1_evars = np.array([a for _,a in results])
                subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars']=all_v4_evars
                subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars']=all_v1_evars
                
                results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths, seed, alpha,
                                                                                    alpha2, condition_type, control_shuffle=True) for seed in seeds)
                all_v4_evars = np.array([a for a,_ in results])
                all_v1_evars = np.array([a for _,a in results])
                subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars_null']=all_v4_evars
                subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars_null']=all_v1_evars
                
            print(f'{condition_type} done')
        
        # if RF in any of the spont_dataset_types_, do RF as well
        spont_RF_dataset_types = [dt for dt in spont_dataset_types_ if 'RF' in dt]
        stim_RF_dataset_types = [dt for dt in stim_datataset_types_ if 'RF' in dt]
        if len(spont_RF_dataset_types)==0:
            print('No RF dataset types found, skipping RF subsampling')
        else:
            # we know that RF_spont has min lengths so will use those lengths to subsample
            min_lengths_RF = {}
            for dtset in spont_RF_dataset_types:
                for d, date in enumerate(subsampled_monkey_stats[dtset].keys()):
                    resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dtset), date=date, w_size=w_size,monkey=main_monkey_name)
                    min_lengths_RF[date]=resp_V4.shape[0]
                    print(f'date {date} min length {min_lengths_RF[date]} from dataset type {dtset}')
            # moving bars
            for condition_type in stim_RF_dataset_types + spont_RF_dataset_types:
                for date in subsampled_monkey_stats[condition_type]:
                    resp_V4, resp_V1, =get_resps(condition_type=get_get_condition_type(condition_type), date=date, w_size=w_size,monkey=main_monkey_name)
                    if resp_V4.shape[0]==min_lengths_RF[date]:
                        subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars']=subsampled_monkey_stats[condition_type][date][area]['evars']
                        subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars']=subsampled_monkey_stats[condition_type][date][area2]['evars']
                        subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars_null']=subsampled_monkey_stats[condition_type][date][area]['evars_null']
                        subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars_null']=subsampled_monkey_stats[condition_type][date][area2]['evars_null']
                        continue
                    if subsampled_indices is True:
                        assert subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V1'][f'monkey_{main_monkey_name}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V1"
                        assert subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V4'][f'monkey_{main_monkey_name}_subsample_indices'] is not None, "subsampled_indices is set to True but no subsampled indices found in monkey_stats V4"
                        subsampled_indices_dict = {'V1':subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V1'][f'monkey_{main_monkey_name}_subsample_indices'],
                                                    'V4':subsampled_monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V4'][f'monkey_{main_monkey_name}_subsample_indices']}
                        resp_V4 = resp_V4[:, subsampled_indices_dict[area]]
                        resp_V1 = resp_V1[:, subsampled_indices_dict[area2]]
                    print(f'condition type {condition_type} date {date} resp_V4 shape {resp_V4.shape} resp_V1 shape {resp_V1.shape}')   
                    results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths_RF, seed, alpha, 
                                                                                        alpha2, condition_type) for seed in seeds)
                    all_v4_evars = np.array([a for a,_ in results])
                    all_v1_evars = np.array([a for _,a in results])
                    subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars']=all_v4_evars
                    subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars']=all_v1_evars
                    
                    results = Parallel(n_jobs=-1)(delayed(process_evar_subsample_seeds)(resp_V1, resp_V4, date, min_lengths_RF, seed, alpha,
                                                                                        alpha2, condition_type, control_shuffle=True) for seed in seeds)
                    all_v4_evars = np.array([a for a,_ in results])
                    all_v1_evars = np.array([a for _,a in results])
                    subsampled_monkey_stats[condition_type][date][area]['spont_comparison_evars_null']=all_v4_evars
                    subsampled_monkey_stats[condition_type][date][area2]['spont_comparison_evars_null']=all_v1_evars
                    
                print(f'{condition_type} done')
        
        # save the monkey stats
        with open(subsampled_monkey_stats_path, 'wb') as handle:
            pickle.dump(subsampled_monkey_stats, handle)
        seed_end_time = time.time()
        print(f'Saved subsampled stats to {subsampled_monkey_stats_path}. Took {(seed_end_time - seed_start_time)/60:.4f} minutes to complete for seed {seed}')

final_end_time = time.time()
elapsed_time = (final_end_time - start_time)/60
print(f'Yay! All subsampling work is completed. Took {elapsed_time:.4f} minutes to complete')