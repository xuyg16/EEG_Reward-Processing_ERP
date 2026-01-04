
def drop_bad_channels(subject_id, eeg):
    '''
    Drop bad channels based on subject ID. Bad channels are found after the first trial processing step.
    
    :param subject_id: subject id of the eeg data
    :param eeg: eeg data to drop bad channels from

    :return: eeg data with bad channels dropped
    '''
    if subject_id == '27' or subject_id == '28' or subject_id == '34':
        bad_channels = []   
    elif subject_id == '28':
        bad_channels =['Fp1']
    # elif subject_id == '35':
    #     bad_channels = ['Fp1', 'TP9', 'TP10']
    else:
        bad_channels = []
        print(f'bad channels unknown for {subject_id}')
    eeg.info['bads'] = bad_channels
    eeg_ica = eeg.copy().drop_channels(bad_channels)

    return eeg_ica


def reref(eeg):
    '''
    Reference EEG data to average of mastoids (TP9, TP10). If one mastoid is missing, reference to the other. If both are missing, raise an error.

    :param eeg: eeg data to be re-referenced

    :return: re-referenced eeg data
    '''
    has_tp9 = 'TP9' in eeg.ch_names
    has_tp10 = 'TP10' in eeg.ch_names

    if has_tp9 and has_tp10:
        print("Re-referencing to average of mastoids (TP9, TP10).")
        eeg.set_eeg_reference(ref_channels=['TP9', 'TP10'])
    elif has_tp9:
        print("TP10 is missing. Re-referencing to TP9 only.")
        eeg.set_eeg_reference(ref_channels=['TP9'])
    elif has_tp10:
        print("TP9 is missing. Re-referencing to TP10 only.")
        eeg.set_eeg_reference(ref_channels=['TP10'])
    else:
        raise RuntimeError("No mastoid channels (TP9 or TP10) found for re-referencing")
    
    return eeg