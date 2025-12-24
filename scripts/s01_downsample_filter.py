from meegkit.dss import dss_line
import numpy as np
import mne

def down_sampling(raw, new_sfreq=250):
    eeg_down = raw.copy().resample(new_sfreq, npad='auto')
    print(f"Original Sampling Rate: {raw.info['sfreq']} Hz")
    print(f"New Sampling Rate: {eeg_down.info['sfreq']} Hz")
    
    return eeg_down


def band_filter(eeg_down, f_low=0.1, f_high=30):
    # manipulate the cutoff frequencies if needed
    eeg_band = eeg_down.copy().filter(l_freq=f_low, h_freq=f_high)

    return eeg_band


def notch_filter(eeg_band, line_freq=50):
    eeg_band_notch = eeg_band.copy().notch_filter(line_freq)

    return eeg_band_notch

def zapline_filter(eeg_band, line_freq=50):
    # input & output of dss_line are of shape: (n_samples, n_channels, n_trial)
    eeg_band_zap = eeg_band.copy()
    band_sfreq = eeg_band.info['sfreq']
    eeg_band_zap_array, _ = dss_line(np.expand_dims(eeg_band.get_data().T, axis=2), fline=50, sfreq=band_sfreq)
    # convert back to shape: (n_channels, n_samples)
    eeg_band_zap_array = eeg_band_zap_array.squeeze().T
    eeg_band_zap._data = eeg_band_zap_array

    return eeg_band_zap
