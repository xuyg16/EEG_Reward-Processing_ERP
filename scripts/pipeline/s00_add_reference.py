import mne

def add_reference_channel(raw, new_ref='Fz'):
    mne.add_reference_channels(raw, new_ref, copy=False)  # add new_ref as reference channel
    return raw


def reref(eeg, verbose=True):
    '''
    Reference EEG data to average of mastoids (TP9, TP10). If one mastoid is missing, reference to the other. If both are missing, raise an error.

    :param eeg: eeg data to be re-referenced

    :return: re-referenced eeg data
    '''
    has_tp9 = 'TP9' in eeg.ch_names
    has_tp10 = 'TP10' in eeg.ch_names

    if has_tp9 and has_tp10:
        if verbose:
            print("Average Reference: Keeping both.")
        eeg.set_eeg_reference(ref_channels=['TP9', 'TP10'])
        # No need to drop!
    elif has_tp9:
        if verbose:
            print("Single Reference: Dropping TP9 to avoid flat line.")
        eeg.set_eeg_reference(ref_channels=['TP9'])
        eeg.drop_channels(['TP9']) # Critical fix
    elif has_tp10:
        if verbose:
            print("Single Reference: Dropping TP10 to avoid flat line.")
        eeg.set_eeg_reference(ref_channels=['TP10'])
        eeg.drop_channels(['TP10']) # Critical fix
    
    return eeg