from scipy.stats import trim_mean
import numpy as np
import mne

def get_trimmed_mean(epochs, proportiontocut):
    data = epochs.get_data()
    trimmed_erp_data = np.apply_along_axis(
        trim_mean,
        axis=0, # along the trial axis
        arr=data,
        proportiontocut=proportiontocut
    ) # (n_channels, n_times)
    # Create the final Evoked object
    trimmed_evoked = mne.EvokedArray(
        trimmed_erp_data, 
        epochs.info, 
        tmin=epochs.times[0]
    )
    return trimmed_evoked

'''return the dictionary for evoked erps for different contitions'''
def get_evoked(conditions_dict, epochs, proportiontocut=0.05):
    all_evokeds = {}
    for name, marker in conditions_dict.items():
        epoch_cond = epochs[marker]
        erp_cond = get_trimmed_mean(epoch_cond, proportiontocut=proportiontocut)
        all_evokeds[name] = erp_cond

    return all_evokeds



# using mean amplitude to measure erp amplitude
def calculate_mean_amplitude(evoked, channel_name, tmin, tmax):
    """Calculates the mean amplitude for a channel within a time window."""
    
    # Select the specific channel
    data = evoked.get_data(picks=channel_name)[0] 
    
    # Get the time indices corresponding to the window (tmin, tmax)
    i_start, i_end = evoked.time_as_index([tmin, tmax])
    
    # calculate the mean across time (axis 1)
    mean_val = np.mean(data[i_start:i_end + 1])
    
    return mean_val * 1e6 # Convert Volts to microvolts (µV)

# using peak-to-peak
def calculate_peak_to_peak(evoked, channel_name, tmin, tmax):
    """
    Calculates peak-to-peak within a single time window.
    Finds the minimum (negative peak) and maximum (positive peak) in the same window.
    """
    data = evoked.get_data(picks=channel_name)[0]
    times = evoked.times
    
    # Single search window for both peaks
    i_start, i_end = evoked.time_as_index([tmin, tmax])
    
    # Find minimum (most negative) and maximum (most positive) in this window
    n_idx = i_start + np.argmin(data[i_start:i_end + 1])
    p_idx = i_start + np.argmax(data[i_start:i_end + 1])
    
    n_amplitude = data[n_idx]
    p_amplitude = data[p_idx]
    n_time = times[n_idx]
    p_time = times[p_idx]
    
    # Peak-to-peak difference
    peak_to_peak = (p_amplitude - n_amplitude) * 1e6
    
    return peak_to_peak, n_time * 1000, p_time* 1000, n_amplitude, p_amplitude
