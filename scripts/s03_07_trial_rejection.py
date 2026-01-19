import numpy as np
import mne

### ------------- Customized Trial Rejection ------------------
def trial_rejection_cust(eeg, evts, evts_dict_stim, maxMin=500e-6, level=500e-6, step=40e-6, lowest=0.1e-6, tmin=0, tmax=3, baseline=None):
    '''
    Customized trial rejection based on four artifact checks:
        1. MaxMin: whether the peak-to-peak amplitude range is over the threshold
        2. Level: whether the max abs amplitude level is over the threhold
        3. Step: whether the difference between adjacent timestep is over teh threshold
        4. Lowest: whether ALL the data points are below the threshold -> if yes, dead channel
    
    :param eeg: eeg data in mne.Raw format
    :param evts: event array
    :param evts_dict_stim: event dictionary
    :param maxMin: max-min amplitude threshold
    :param level: amplitude level threshold
    :param step: step threshold
    :param lowest: lowest amplitude threshold

    :return trials: trials after rejecting bad trials
    :return rejected_info: dictionary containing the info of rejected trials: reason(channel name) for each dropped trial
    '''
    # dividing the data into trials
    trials = mne.Epochs(eeg, evts, evts_dict_stim, tmin=tmin, tmax=tmax, baseline=baseline, verbose=False) 

    # get all artifacts
    artifact_mask = find_artifacts(trials, maxMin, level, step, lowest)  # shape: (n_epochs, n_channels)

    # get the artifact info
    is_artifact = np.any(artifact_mask, axis=1).squeeze()  # shape: (n_epochs, )
    is_artifacts_idx = np.where(is_artifact)[0]

    # store the info before dropping (otherwise the indices will change after dropping)
    rejected_info = {}
    for idx in is_artifacts_idx:
        ch_indiced = np.where(artifact_mask[idx])[0]
        rejected_info[idx] = [trials.ch_names[ch_idx] for ch_idx in ch_indiced]

    
    # drop bad trials
    trials.drop(is_artifacts_idx)   # in place operation

    return trials, rejected_info


def find_artifacts(trials, maxMin, level, step, lowest):
    """
    Find artifacts in the given trials based on four criteria (see docstring of `trial_rejection_cust` for details).

    :param trials: mne.Epochs object containing the trials to be checked
    :param maxMin: max-min amplitude threshold
    :param level: amplitude level threshold
    :param step: step threshold
    :param lowest: lowest amplitude threshold

    :return: A boolean array of shape (n_epochs, n_channels) indicating whether each channel in each epoch is marked as an artifact
    """

    data = trials.get_data(verbose=False)    # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape

    is_artifact = np.zeros((n_epochs, n_channels), dtype=bool)

    # CHECKPOINT 1: MaxMin
    max_amp = np.max(data, axis=2)
    min_amp = np.min(data, axis=2)
    is_minMax_artifact = np.abs(max_amp - min_amp) > maxMin
    is_artifact |= is_minMax_artifact      # return True if either of left or right variable is True

    # CHECKPOINT 2: Level
    is_level_artifact =  np.max(np.abs(data), axis=2)  > level
    is_artifact |= is_level_artifact

    # CHEKPOINT 3: Step
    diff = np.diff(data, axis=2)
    is_step_artifact = np.any(diff > step, axis=2)
    is_artifact |= is_step_artifact

    # CHECKPOINT 4: lowest
    is_lowest_artifact = np.all(np.abs(data) < lowest, axis=2)
    is_artifact |= is_lowest_artifact

    return is_artifact



### ------------- Trial Rejection by MNE Methods------------------
### NOTE: need some adjustments to fit our data structure
def trial_rejection_mne(eeg, evts, evts_dict_stim, max=500e-6, min=0.1e-6, tmin=0, tmax=3, baseline=None):
    '''
    Trial rejection using MNE built-in methods based on peak-to-peak amplitude and flat signal checks.
    
    :param eeg: eeg data in mne.Raw format
    :param evts: event array
    :param evts_dict_stim: event dictionary
    :param max: max amplitude threshold
    :param min: min amplitude threshold
    :param tmin: start time of the epoch
    :param tmax: end time of the epoch
    :param baseline: baseline period (None for no baseline correction)

    :return: trials after rejecting bad trials
    '''
    ### peak-to-peak amp check
    reject_criteria = dict(
        eeg=max
    )

    ### peak-to-peak min check
    flat_criteria = dict(
        eeg=min
    )

    trials = mne.Epochs(
        eeg, 
        evts,
        evts_dict_stim,
        tmin=tmin, 
        tmax=tmax, 
        reject=reject_criteria,        
        flat=flat_criteria,
        baseline=baseline,            
        # reject_by_annotation=True if there are segments marked as 'BAD'  
        preload=True
    )

    return trials