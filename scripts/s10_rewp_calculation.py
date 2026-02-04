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



def rewp_calculation(all_evokeds, epoch_dict, verbose=True):
    """
    Calculate RewP metrics (Mean Amplitude and Peak-to-Peak) based on difference waves (Win - Loss).

    :param all_evokeds: Dictionary of MNE Evoked objects for all conditions
    :param epoch_dict: Dictionary defining the epoch conditions
    
    :return: None (prints the RewP metrics)"""
    condition_pairs = [
        ('Low-Low',   'Low-Low Win',   'Low-Low Loss'),
        ('Mid-Low',   'Mid-Low Win',   'Mid-Low Loss'),
        ('Mid-High',  'Mid-High Win',  'Mid-High Loss'),
        ('High-High', 'High-High Win', 'High-High Loss')
    ]

    channel = 'FCz'
    mean_window = [0.240, 0.340]
    results = {}

    for label, win_key, loss_key in condition_pairs:
        if win_key in all_evokeds and loss_key in all_evokeds:
            # Create the Difference Wave: RewP = Win - Loss
            # weights=[1, -1] performs the subtraction
            rewp_diff = mne.combine_evoked(
                [all_evokeds[win_key], all_evokeds[loss_key]], 
                weights=[1, -1]
            )

            mean_amp = calculate_mean_amplitude(rewp_diff, channel, *mean_window)
            p2p_amp, n_t, p_t, n_amp, p_amp = calculate_peak_to_peak(rewp_diff, channel, *mean_window)

            results[label] = {'mean': mean_amp, 'p2p': p2p_amp}
            
            if verbose:
                print(f"[{label}] Mean: {mean_amp:5.2f} µV | P2P: {p2p_amp:5.2f} µV")
        else:
            print(f"[{label}] Missing data for win/loss pair.")
            results[label] = {'mean': np.nan, 'p2p': np.nan}

    return results



