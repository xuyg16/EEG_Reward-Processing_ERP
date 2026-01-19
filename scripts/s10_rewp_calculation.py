import mne
import numpy as np


def calculate_mean_amplitude(evoked, channel_name, tmin, tmax):
    """Calculates the mean amplitude for a channel within a time window.
    
    :param evoked: MNE Evoked object
    :param channel_name: Name of the channel to analyze
    :param tmin: start time of the window (in seconds)
    :param tmax: end time of the window (in seconds)
    
    :return: Mean amplitude in microvolts (µV)"""
    
    # Select the specific channel
    data = evoked.get_data(picks=channel_name)[0] 
    
    # Get the time indices corresponding to the window (tmin, tmax)
    i_start, i_end = evoked.time_as_index([tmin, tmax])
    
    # calculate the mean across time (axis 1)
    mean_val = np.mean(data[i_start:i_end + 1])
    
    return mean_val * 1e6 # Convert Volts to microvolts (µV)



def calculate_peak_to_peak(evoked, channel_name, tmin, tmax):
    """
    Calculates peak-to-peak within a single time window.
    Finds the minimum (negative peak) and maximum (positive peak) in the same window.

    :param evoked: MNE Evoked object
    :param channel_name: Name of the channel to analyze
    :param tmin: start time of the window (in seconds)
    :param tmax: end time of the window (in seconds)
    
    :return: peak-to-peak amplitude in microvolts (µV), time of negative peak (ms), time of positive peak (ms), amplitude of negative peak (V), amplitude of positive peak (V)
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



def rewp_calculation(all_evokeds, epoch_dict):
    """
    Calculate RewP metrics (Mean Amplitude and Peak-to-Peak) based on difference waves (Win - Loss).

    :param all_evokeds: Dictionary of MNE Evoked objects for all conditions
    :param epoch_dict: Dictionary defining the epoch conditions
    
    :return: None (prints the RewP metrics)"""

    win = [k for k in epoch_dict.keys() if 'Win' in k]
    loss = [k for k in epoch_dict.keys() if 'Loss' in k]
    valid_win_names = [name for name in win if name in all_evokeds]
    valid_loss_names = [name for name in loss if name in all_evokeds]


    # calculate average amplitude for win and lose cases
    win_evokeds_list = [all_evokeds[name] for name in valid_win_names]
    loss_evokeds_list = [all_evokeds[name] for name in valid_loss_names]

    erp_grand_win = mne.combine_evoked(win_evokeds_list, weights='equal')
    erp_grand_loss = mne.combine_evoked(loss_evokeds_list, weights='equal')
    rewp_diff = mne.combine_evoked([erp_grand_win, erp_grand_loss], weights=[1, -1])

    # Parameters
    channel = 'FCz'
    mean_window = [0.240, 0.340] # as set by authors

    # Metrics
    rewp_mean_amplitude = calculate_mean_amplitude(rewp_diff, channel, *mean_window)
    p2p_amp, n_time, p_time, n_val, p_val = calculate_peak_to_peak(rewp_diff, channel, *mean_window)


    print(f"RewP Mean Amplitude: {rewp_mean_amplitude:.2f} µV")
    print(f"RewP Peak-to-Peak: {p2p_amp:.2f} µV")



