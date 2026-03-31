import mne
# from mne_icalabel import label_components NOTE: only shows the component with the highest probability
from mne_icalabel.iclabel import iclabel_label_components
from utils.logger import log_ica_exclusion
from utils.visualization import iclabel_visualize
import config


def get_ica(trials, method='picard', save_path=None):
    '''
    Fit ICA on the given MNE Epochs object.
    
    :param trials_mne: MNE Epochs object containing EEG data.
    :param method: ICA method to use.
    :param save_path: path to save the fitted ICA object. If None, the ICA object will not be saved.

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


def iccomponent_removal(eeg, trials, ica, subject_id, active_pipeline, logger=None, save_path=None):
    '''
    Remove bad IC components based on the given criteria. 

    :param eeg: MNE Raw object containing EEG data.
    :param trials: MNE Epochs object for ICA fitting.
    :param ica: Fitted ICA object.
    :param subject_id: ID of the subject being processed.
    :param active_pipeline: the name of the active pipeline, used to determine the criteria for excluding components.
    :param logger: Logger object for logging the exclusion process. If None, logging will be skipped.
    :param save_path: path to save the visualization of excluded components. If None, the visualization will not be saved.

    :return: Cleaned MNE Raw object.
    '''
    exclude_idx = config.SUBJECT_INFO[subject_id]['ic_excluded'][active_pipeline]   # check is exclude idx is already saved

    if exclude_idx is None:
        exclude_idx = []
        trials.load_data()
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
            if active_pipeline == 'original':
                if probabilities[label_dict['eye blink']] > probabilities[label_dict['brain']] or \
                    probabilities[label_dict['eye movement']] > probabilities[label_dict['brain']]:
                    exclude_idx.append(i)
            elif active_pipeline == 'proposed':
                 if probabilities[label_dict.index('brain')] < 0.4:
                    exclude_idx.append(i)
            else:
                raise ValueError(f'Unknown pipeline: {active_pipeline}')
        if save_path:
        # Save a plot of the excluded components to a folder
            iclabel_visualize(ica, all_labels[exclude_idx], exclude_idx=exclude_idx, show=False, save_path=save_path)
    # logging 
    if logger:
        log_ica_exclusion(logger, subject_id, exclude_idx, ica.n_components_)


    ica.exclude = exclude_idx
    ica.apply(eeg)

    return eeg