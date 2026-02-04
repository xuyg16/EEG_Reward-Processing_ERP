import numpy as np
import mne

def binning(epochs, conditions_dict, bin_num=4):
    '''
    Docstring for binning
    
    :param epochs: Description
    :param conditions_dict: Description
    :param bin_num: Description
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
                    
    return binned_epochs_combined