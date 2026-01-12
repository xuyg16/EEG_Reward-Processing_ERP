import mne
from mne_icalabel import label_components


def get_ica(trials_mne, method='picard'):
    '''
    Fit ICA on the given MNE Epochs object.
    
    :param trials_mne: MNE Epochs object containing EEG data.
    :param method: ICA method to use.

    :return: Fitted ICA object.
    '''

    ica = mne.preprocessing.ICA(method=method, random_state=97)     # random_state for reproducibility
    ica.fit(trials_mne,verbose=True)

    return ica


def get_iclabel(trials, ica, method='iclabel'):
    '''
    Get IC labels using the specified method.

    :param trials: MNE Epochs object containing EEG data.
    :param ica: Fitted ICA object.
    :param method: Method to use for labeling components (the only one available is 'iclabel'?).

    :return: IC labels.
    '''
    # IC_label expects filtered between 1 and 100 Hz, reference to be common average and ica method to be infomax
    # so it's better to combine the result of autolabel and munual check in our case
    trials.load_data()
    ic_labels = label_components(trials, ica, method=method)

    return ic_labels


def iccomponent_removal(eeg, ica, ic_labels, criteria):
    '''
    Remove bad IC components based on the given criteria.

    :param eeg: MNE Raw object containing EEG data.
    :param ica: Fitted ICA object.
    :param ic_labels: IC labels.
    :param criteria: Confidence threshold for component removal.

    :return: Cleaned MNE Raw object.
    '''
    # authors only remove eye components and leaves other intact
    exclude_idx = []
    for i, label in enumerate(ic_labels['labels'] ):
        #print(label)
        if label == 'eye blink' and ic_labels['y_pred_proba'][i] > criteria:
            exclude_idx.append(i)

    ica.exclude = exclude_idx
    ica.apply(eeg)

    return eeg