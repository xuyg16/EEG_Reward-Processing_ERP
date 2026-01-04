
def interpolation(eeg):
    '''
    Interpolates bad channels in the EEG data.
    
    :param eeg: eegh object with bad channels marked

    :return: eeg object with bad channels interpolated
    '''
    # common_montage_path = "C:\\Users\\Zheng\\Desktop\\fourth semester\\EEG\\common.locs"
    # montage_common = mne.channels.read_custom_montage(common_montage_path)
    # eeg_band_notch.set_montage(montage_common, on_missing='ignore') 

    if len(eeg.info['bads']) > 0:
        print(f"Interpolating bad channels: {eeg.info['bads']}")
        eeg.interpolate_bads(reset_bads=True)
    else:
        print("No bad channels marked for interpolation")

    return eeg