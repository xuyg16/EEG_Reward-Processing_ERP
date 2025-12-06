import numpy as np
import mne

def find_artifacts(trials, maxMin=500e-6, level=500e-6, step=40e-6, lowest=0.1e-6):
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