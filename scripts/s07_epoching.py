import mne
from s03_07_trial_rejection import trial_rejection_cust, trial_rejection_mne
from tools import get_event_dict

def epoching(conditions_dict, eeg, max=150e-6, min=0.1e-6, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0)):
    '''
    Epoching the continuous EEG data based on the provided conditions dictionary,
    and applying trial rejection.

    :param conditions_dict: dictionary mapping condition names to event markers
    :param eeg: MNE Raw object containing the continuous EEG data
    :param max: maximum threshold for trial rejection
    :param min: minimum threshold for trial rejection
    :param tmin: start time for epoching
    :param tmax: end time for epoching
    :param baseline: tuple defining the baseline correction period

    :return: MNE Epochs object containing the epoched and cleaned data
    '''

    # get all the feedback-locked stimulus
    all_markers = []
    for markers in conditions_dict.values():
        all_markers.extend(markers)

    # Create the filtered event dictionary
    evts, evts_dict_stim = get_event_dict(eeg, all_markers)

    epochs_all = trial_rejection_mne(eeg, evts, evts_dict_stim, max=max, min=min,  tmin=tmin, tmax=tmax, baseline=baseline)

    return epochs_all


def epoching_cust(conditions_dict, eeg, maxMin=150e-6, level=150e-6, step=40e-6, lowest=0.1e-6, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0)):
    '''
    Epoching the continuous EEG data based on the provided conditions dictionary,
    and applying trial rejection.

    :param conditions_dict: dictionary mapping condition names to event markers
    :param eeg: MNE Raw object containing the continuous EEG data
    :param max: maximum threshold for trial rejection
    :param min: minimum threshold for trial rejection
    :param tmin: start time for epoching
    :param tmax: end time for epoching
    :param baseline: tuple defining the baseline correction period

    :return: MNE Epochs object containing the epoched and cleaned data
    '''
    
    all_markers = []
    for markers in conditions_dict.values():
        all_markers.extend(markers)

    # Create the filtered event dictionary
    evts, evts_dict_stim = get_event_dict(eeg, all_markers)

    epochs_all = trial_rejection_cust(eeg, evts, evts_dict_stim, maxMin=maxMin, level=level, step=step, lowest=lowest
                                                       , tmin=tmin, tmax=tmax, baseline=baseline)

    return epochs_all