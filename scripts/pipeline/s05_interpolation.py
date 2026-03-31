
def interpolation(eeg, verbose=True):
    '''
    Interpolates bad channels in the EEG data.
    
    :param eeg: eegh object with bad channels marked
    :param verbose: whether to print information about the interpolation process

    :return: eeg object with bad channels interpolated
    '''
    
    if len(eeg.info['bads']) > 0:
        if verbose:
            print(f"Interpolating bad channels: {eeg.info['bads']}")
        eeg.interpolate_bads(reset_bads=True)
    else:
        if verbose:
            print("No bad channels marked for interpolation")

    return eeg