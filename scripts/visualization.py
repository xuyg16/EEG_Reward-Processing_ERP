from matplotlib import pyplot as plt
import numpy as np

'''
help to plot psd graph of selected eeg data
'''
def show_single_psd(eeg_data, y_min=-50, y_max=50, picks=None, title=None):
    eeg_psd = eeg_data.compute_psd().plot(picks=picks, show=False)
    ax = eeg_psd.axes[0]
    y_min = y_min
    y_max = y_max
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    plt.show()

'''
help to compare the psd after different preprocessing strategies
'''
def psd_compare(eegs, labels, title, figsize=(8, 6), picks=['FCz'], y_min=-46.4, y_max=-46, x_min=49.5, x_max=50.5):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    freqs = eegs[0].compute_psd(picks=picks).freqs  # assuming the same sfreq
    
    # Iterate through EEG objects and plot the PSD data
    for i, eeg in enumerate(eegs):
        # Compute the PSD for the selected picks
        psd = eeg.compute_psd(picks=picks)
        
        # Get the power data (it returns channels x frequencies, so we average across channels)
        # Note: If picks contains multiple channels, this averages them for a cleaner comparison.
        # 1e12 for scaling, sticking to the default psd calculation in psd
        power_data = np.mean(10 * np.log10(psd.get_data() * 1e12), axis=0)
        
        ax.plot(
            freqs, 
            power_data, 
            label=labels[i], 
            linewidth=2
        )

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB/Hz)')

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.show()