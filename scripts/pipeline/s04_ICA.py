import mne
from mne_icalabel import label_components
from mne_icalabel.iclabel import iclabel_label_components
from stats.logging_utils import log_ica_exclusion
from visualization import iclabel_visualize
import config


### TO DO LIST: add the code to show which components are removed ###

def get_ica(trials, method='picard', save_path=None):
    '''
    Fit ICA on the given MNE Epochs object.
    
    :param trials_mne: MNE Epochs object containing EEG data.
    :param method: ICA method to use.

    :return: Fitted ICA object.
    '''
    if method == 'picard':
        ica = mne.preprocessing.ICA(method=method, fit_params=dict(ortho=False, extended=True), random_state=2016)     # random_state for reproducibility (the authors used 2016)
    else:
        ica = mne.preprocessing.ICA(method=method, fit_params=dict(extended=True), random_state=2016) 
    ica.fit(trials,verbose=True)

    if save_path:
        ica.save(save_path, overwrite=True)
        print(f"ICA object saved to {save_path}")

    return ica

### NOTE: not used by label_components does not give the probability of each possible label but only the one with the highes probability
# def get_iclabel(trials, ica, method='iclabel'):
#     '''
#     Get IC labels using the specified method.

#     :param trials: MNE Epochs object containing EEG data.
#     :param ica: Fitted ICA object.
#     :param method: Method to use for labeling components (the only one available is 'iclabel'?).

#     :return: IC labels.
#     '''
#     # IC_label expects filtered between 1 and 100 Hz, reference to be common average and ica method to be infomax
#     # so it's better to combine the result of autolabel and munual check in our case
#     trials.load_data()
#     ic_labels = label_components(trials, ica, method=method)

#     return ic_labels


# def iccomponent_removal(eeg, ica, ic_labels, criteria):
#     '''
#     Remove bad IC components based on the given criteria.

#     :param eeg: MNE Raw object containing EEG data.
#     :param ica: Fitted ICA object.
#     :param ic_labels: IC labels.
#     :param criteria: Confidence threshold for component removal.

#     :return: Cleaned MNE Raw object.
#     '''
#     # authors only remove eye components and leaves other intact
#     exclude_idx = []
#     for i, label in enumerate(ic_labels['labels'] ):
#         #print(label)
#         if label == 'eye blink' and ic_labels['y_pred_proba'][i] > criteria:
#             exclude_idx.append(i)

#     ica.exclude = exclude_idx
#     ica.apply(eeg)

#     return eeg


#NOTE: TBD - combine these two functions into one
def iccomponent_removal_author(eeg, trials, ica, subject_id, logger=None, save_path=None):
    '''
    Remove bad IC components based on the given criteria.

    :param eeg: MNE Raw object containing EEG data.
    :param trials: MNE Epochs object for ICA fitting.
    :param ica: Fitted ICA object.
    :param exclude_idx: List of component indices to exclude. If None, find the components to exclude

    :return: Cleaned MNE Raw object.
    '''
    exclude_idx = config.SUBJECT_INFO[subject_id]['ic_excluded']['original']
    if exclude_idx is None:
        exclude_idx = []
        trials.load_data()
        # authors only remove eye components and leaves other intact
        label_dict = {
            'brain': 0,
            'muscle': 1,
            'eye blink': 2,
            'eye movement': 3,
            'heart': 4,
            'line noise': 5,
            'channel noise': 6,
            'other': 7
        }
        all_labels = iclabel_label_components(trials, ica)
        for i, probabilities in enumerate(all_labels):
            #print(label)
            if probabilities[label_dict['eye blink']] > probabilities[label_dict['brain']]:
                exclude_idx.append(i)

    # logging 
    if logger:
        log_ica_exclusion(logger, subject_id, exclude_idx, ica.n_components_)

    if save_path:
        # Save a plot of the excluded components to a folder
        iclabel_visualize(ica, all_labels[exclude_idx], exclude_idx=exclude_idx, show=False, save_path=save_path)


    ica.exclude = exclude_idx
    ica.apply(eeg)

    return eeg

def iccomponent_removal_new(eeg, trials, ica, subject_id, logger=None, save_path=None):
    '''
    Remove bad IC components based on the given criteria.

    :param eeg: MNE Raw object containing EEG data.
    :param trials: MNE Epochs object for ICA fitting.
    :param ica: Fitted ICA object.
    :param exclude_idx: List of component indices to exclude. If None, find the components to exclude

    :return: Cleaned MNE Raw object.
    '''
    exclude_idx = config.SUBJECT_INFO[subject_id]['ic_excluded']['proposed']

    if exclude_idx is None:
        exclude_idx = []
        trials.load_data()
        # authors only remove eye components and leaves other intact
        label_dict = ['brain', 'muscle', 'eye blink', 'eye movement', 'heart', 'line noise', 'channel noise', 'other']
        all_labels = iclabel_label_components(trials, ica)
        for i, probabilities in enumerate(all_labels):
            #print(label)
            if probabilities[label_dict.index('brain')] < 0.4:
                exclude_idx.append(i)

    # logging 
    if logger:
        log_ica_exclusion(logger, subject_id, exclude_idx, ica.n_components_)

    if save_path:
        # Save a plot of the excluded components to a folder
        iclabel_visualize(ica, all_labels[exclude_idx], exclude_idx=exclude_idx, show=False, save_path=save_path)

    ica.exclude = exclude_idx
    ica.apply(eeg)

    return eeg