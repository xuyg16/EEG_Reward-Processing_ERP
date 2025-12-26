import mne
from mne_icalabel import label_components


def get_ica(trials_mne, method='picard'):
    ica = mne.preprocessing.ICA(method=method)
    ica.fit(trials_mne,verbose=True)

    return ica


def get_iclabel(trials, ica, method='iclabel'):
    # IC_label expects filtered between 1 and 100 Hz, reference to be common average and ica method to be infomax
    # so it's better to combine the result of autolabel and munual check in our case
    trials.load_data()
    ic_labels = label_components(trials, ica, method=method)

    return ic_labels


def iccomponent_removal(eeg, ica, ic_labels, criteria):
    ### exclude the bad IC components and reconstruct the eeg data
    # authors only remove eye components and leaves other intact
    # criteria: confidence > 50%
    exclude_idx = []
    for i, label in enumerate(ic_labels['labels'] ):
        #print(label)
        if label == 'eye blink' and ic_labels['y_pred_proba'][i] > criteria:
            exclude_idx.append(i)

    ica.exclude = exclude_idx
    ica.apply(eeg)

    return ica