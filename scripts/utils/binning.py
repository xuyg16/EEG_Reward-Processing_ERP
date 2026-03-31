import numpy as np
import mne
import pandas as pd
import config
from pipeline.s10_rewp_calculation import rewp_calculation

def binning(epochs, conditions_dict, bin_num=4):
    '''
    This function takes in the epochs and the conditions dictionary, and returns a dictionary of binned epochs and a dataframe of trial counts per condition per bin.
    '''
    
    binned_epochs = []
    
    # 2. Define the broad groups for chronological slicing
    groups = {'low': [], 'mid': [], 'high': []}
    for condition, marker_list in conditions_dict.items():
        if condition in ['Low-Low Win', 'Low-Low Loss']:
            groups['low'].extend(marker_list)
        elif 'Mid-' in condition:
            groups['mid'].extend(marker_list)
        elif 'High-High' in condition:
            groups['high'].extend(marker_list)

    # Loop through all the task groups (three in total)
    for group_name, markers in groups.items():
        #print(f'DEBUG: Processing group {group_name} with markers {markers}')
        #available_markers = [m for m in markers if m in epochs.event_id]    #conditions in each group
        
        # Select epochs for this task group
        current_epochs = epochs[markers]
        #print(f'DEBUG: the number of trials before dropping is {len(current_epochs)}(not sure if this is before dropping or after)')
        
        n_group_trials = len(current_epochs)
        
        # Calculate chronological cut points
        group_cut_points = np.linspace(0, n_group_trials, bin_num + 1, dtype=int)
        #print(f'DEBUG: Cut points for group {group_name}: {group_cut_points}')

        for i in range(bin_num):
            start, end = group_cut_points[i], group_cut_points[i+1]
            epochs_bin = current_epochs[start:end]
            
            binned_epochs.append({
                'bin': i + 1, 
                'epochs': epochs_bin,
            })
            #print(f'DEBUG: Created bin {i + 1} for group {group_name} with {len(epochs_bin)} trials.')

    # Combine bins across groups
    binned_epochs_combined = {}
    for entry in binned_epochs:
        bin_index = entry['bin']
        if bin_index not in binned_epochs_combined:
            # If it's the first time seeing this bin, initialize it
            binned_epochs_combined[bin_index] = [entry['epochs']]
        else:
            # If the bin exists, extend the existing list with new epochs
            binned_epochs_combined[bin_index].append(entry['epochs'])

    for bin_index in binned_epochs_combined:
        binned_epochs_combined[bin_index] = mne.concatenate_epochs(binned_epochs_combined[bin_index])


    # Count trials per condition per bin
    bin_trial_counts = {}
    for bin_idx, bin_epochs in binned_epochs_combined.items():
        bin_trial_counts[bin_idx] = {}
        for cond, markers in conditions_dict.items():
            # select using the raw stimulus string(s)
            available_markers = [m for m in markers if m in bin_epochs.event_id]
            if available_markers:
                bin_trial_counts[bin_idx][cond] = len(bin_epochs[available_markers])
            else:
                bin_trial_counts[bin_idx][cond] = 0

        df_counts = pd.DataFrame(bin_trial_counts).T
        df_counts.index.name = 'bin'  

    return binned_epochs_combined, df_counts



def get_group_binned_rewp(n_bins, subjects, epoch_dict, binned_group_evokeds, learners_only=False):
    '''
    This function takes in the binned group evokeds and calculates the RewP for each bin and subject, returning a dictionary of RewP values per condition per bin.
    '''
    conditions = ['Low-Low', 'Mid-Low', 'Mid-High', 'High-High']
    rewp_per_subject_binned= {cond: np.zeros((len(subjects), n_bins)) for cond in conditions}

    for s_idx, subj in enumerate(subjects):
        if learners_only and not config.SUBJECT_INFO[subj]['learner']:
            continue
        for i in range(n_bins):
            subj_evokeds = binned_group_evokeds[subj][i]
            subj_results = rewp_calculation(subj_evokeds, epoch_dict, verbose=False)
            for cond in conditions:
                rewp_per_subject_binned[cond][s_idx, i] = subj_results[cond]['mean']
   
    return rewp_per_subject_binned
