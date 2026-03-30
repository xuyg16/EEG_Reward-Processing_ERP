from scipy.stats import trim_mean
import numpy as np
import mne

def get_trimmed_mean(epochs, proportiontocut):
    '''
    Calculate the trimmed mean ERP from epochs.

    :param epochs: MNE Epochs object
    :param proportiontocut: Proportion of trials to cut from each end of the distribution
    :return: MNE EvokedArray object containing the trimmed mean ERP
    :rtype: mne.EvokedArray

    :returns: trimmed_evoked -- the trimmed mean ERP as an Evoked object
    '''
    data = epochs.get_data()
    n_trials = len(epochs)
    trimmed_erp_data = np.apply_along_axis(
        trim_mean,
        axis=0, # along the trial axis
        arr=data,
        proportiontocut=proportiontocut
    ) # (n_channels, n_times)
    # Create the final Evoked object
    trimmed_evoked = mne.EvokedArray(
        trimmed_erp_data, 
        epochs.info, 
        tmin=epochs.times[0],
        nave=n_trials
    )
    return trimmed_evoked



def get_evoked(conditions_dict, epochs, proportiontocut=0.05, verbose=True):
    '''
    Generate evoked ERPs for different conditions using trimmed mean.
    
    :param conditions_dict: dictionary mapping condition names to event markers
    :param epochs: MNE Epochs object
    :param proportiontocut: Proportion of trials to cut from each end of the distribution

    :return: Dictionary of Evoked objects for each condition
    '''
    all_evokeds = {}
    for name, marker in conditions_dict.items():
        # 1. Select the epochs for this condition

        try:
            epoch_cond = epochs[marker]
            # 2. Generate the ERP using your trimmed mean function
            erp_cond = get_trimmed_mean(epoch_cond, proportiontocut=proportiontocut)
            # Set the comment so it shows up in the object summary
            erp_cond.comment = name
            all_evokeds[name] = erp_cond
        
        except KeyError:
            if verbose:
                print(f"Warning: No trials found for condition {name}")
            continue

    return all_evokeds


def get_evoked_difference(all_evokeds):
    '''
    Calculate difference waves (Win - Loss) for each condition pair.
    
    :param all_evokeds: Dictionary of Evoked objects for each condition
    
    :return: Dictionary of Evoked difference waves
    '''
    diff_evokeds = {}
    cases = [
        ('Low-Low', 'Low-Low Win', 'Low-Low Loss'),
        ('Mid-Low', 'Mid-Low Win', 'Mid-Low Loss'),
        ('Mid-High', 'Mid-High Win', 'Mid-High Loss'),
        ('High-High', 'High-High Win', 'High-High Loss')
    ]

    for case_name, win_cond, loss_cond in cases:
        # Calculate Difference: Win - Loss
        diff = mne.combine_evoked(
            [all_evokeds[win_cond], all_evokeds[loss_cond]],
            weights=[1, -1]
        )
        diff.comment = case_name # Set name for plotting
        diff_evokeds[case_name] = diff
    return diff_evokeds
    

def compute_grand_average(epoch_dict, group_evokeds):
    '''
    Compute grand average ERPs across all subjects for each condition.
    
    :param epoch_dict: Dictionary of conditions
    :param group_evokeds: Dictionary mapping subject IDs to their Evoked objects
    
    :return: Dictionary of grand average Evoked objects for each condition
    '''
    grand_averages = {}
    for condition in epoch_dict.keys():
        evokeds_list = [group_evokeds[subject_id][condition] for subject_id in group_evokeds.keys() if condition in group_evokeds[subject_id]]
        if evokeds_list:
            grand_averages[condition] = mne.grand_average(evokeds_list)

    return grand_averages