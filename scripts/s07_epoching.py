import mne
from s03_07_trial_rejection import trial_rejection_cust, trial_rejection_mne

def epoching(conditions_dict, eeg, max=150e-6, min=0.1e-6, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0)):
    evts, event_id = mne.events_from_annotations(eeg)

    all_markers = []
    for markers in conditions_dict.values():
        all_markers.extend(markers)

    # Create the filtered event dictionary
    evts_dict_stim = {k: event_id[k] for k in event_id.keys() if k in all_markers}

    epochs_all = trial_rejection_mne(eeg, evts, evts_dict_stim, max=max, min=min,  tmin=tmin, tmax=tmax, baseline=baseline)

    return epochs_all