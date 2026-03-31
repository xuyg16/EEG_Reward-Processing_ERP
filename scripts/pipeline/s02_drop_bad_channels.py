

def drop_bad_channels(bad_channels, eeg):
    '''
    Drop bad channels based on subject ID. Bad channels are found after the first trial processing step.
    
    :param bad_channels: list of bad channels to drop
    :param eeg: eeg data to drop bad channels from

    :return: eeg data with bad channels dropped
    '''
    eeg.info['bads'] = bad_channels
    eeg_ica = eeg.copy().drop_channels(bad_channels)

    return eeg_ica