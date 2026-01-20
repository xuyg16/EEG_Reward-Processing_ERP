from meegkit.dss import dss_line
import numpy as np

def down_sampling(eeg, new_sfreq=250, verbose=True):
    '''
    Downsample the eeg signal.
    
    :param eeg: eeg signal to be processed
    :param new_sfreq: the frequency after downsampling

    :return: downsampled eeg signal
    '''
    eeg_down = eeg.copy().resample(new_sfreq, npad='auto')

    if verbose:
        print(f"Original Sampling Rate: {eeg.info['sfreq']} Hz")
        print(f"New Sampling Rate: {eeg_down.info['sfreq']} Hz")

    return eeg_down


# NOTE: add and manipulate the cutoff frequencies if needed
def band_filter(eeg, f_low=0.1, f_high=30):
    '''
    Perform bandpass filtering on the downsampled eeg signal.
    
    :param eeg: eeg signal to be processed
    :param f_low: low cutoff frequency
    :param f_high: high cutoff frequency

    :return: bandpass filtered eeg signal
    '''
    eeg_band = eeg.copy().filter(l_freq=f_low, h_freq=f_high)

    return eeg_band


def notch_filter(eeg, line_freq=50):
    '''
    Perform notch filtering on the bandpass filtered eeg signal.

    :param eeg: eeg signal to be processed
    :param line_freq: line frequency to be removed

    :return: notch filtered eeg signal
    '''
    eeg_band_notch = eeg.copy().notch_filter(line_freq)

    return eeg_band_notch


def zapline_filter(eeg, line_freq=50):
    '''
    Perform zapline filtering on the bandpass filtered eeg signal.
    
    :param eeg: eeg signal to be processed
    :param line_freq: line frequency to be removed

    :return: zapline filtered eeg signal
    '''
    # input & output of dss_line are of shape: (n_samples, n_channels, n_trial)
    eeg_band_zap = eeg.copy()
    band_sfreq = eeg.info['sfreq']
    eeg_band_zap_array, _ = dss_line(np.expand_dims(eeg.get_data().T, axis=2), fline=line_freq, sfreq=band_sfreq)
    # convert back to shape: (n_channels, n_samples)
    eeg_band_zap_array = eeg_band_zap_array.squeeze().T
    eeg_band_zap._data = eeg_band_zap_array

    return eeg_band_zap
