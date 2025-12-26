import numpy as np
import mne

### ------------- Customized Trial Rejection ------------------
def trial_rejection_cust(eeg_ica, evts, evts_dict_stim, maxMin=500e-6, level=500e-6, step=40e-6, lowest=0.1e-6):
    # dividing the data into trials
    trials = mne.Epochs(eeg_ica, evts, evts_dict_stim, tmin=0, tmax=3, baseline=None)    #NOTE: from 0s to 3s. maybe extend the window?
    print(trials.get_data().shape)

    #NOTE: parameters used are the same as the author, may consider changing them
    is_artifacts = np.any(find_artifacts(trials, maxMin, level, step, lowest), axis=1).squeeze()
    # get the indices of trials of artifacts
    is_artifacts_idx = np.where(is_artifacts)[0]
    #print(len(is_artifacts_idx))

    # drop bad trials
    trials.copy().drop(is_artifacts_idx)

    return trials 


def find_artifacts(trials, maxMin, level, step, lowest):
    """
    Four Artifacts checks:
        1. MaxMin: whether the peak-to-peak amplitude range is over the threshold.
        2. Level: whether the max abs amplitude level is over the threhold
        3. Step: whether the difference between adjacent timestep is over teh threshold
        4. Lowest: whether ALL the data points are below the threshold -> if yes, dead channel

    INPUT:
        EEG: mne.Epochs
    OUPUTS:
        is_artifact (n_epochs, n_channels)
    """
    
    data = trials.get_data()    # shape: (n_epochs, n_channels, n_times)
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
def trial_rejection_mne(eeg, evts, evts_dict_stim, max=500e-6, min=0.1e-6, tmin=0, tmax=3, baseline=None):
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