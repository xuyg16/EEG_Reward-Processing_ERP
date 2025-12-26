from matplotlib import pyplot as plt
import numpy as np
import mne

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


'''help to visualize IClables'''
def iclabel_visualize(ica, ic_labels, trials):
    labels = ic_labels['labels'] 
    probs = ic_labels['y_pred_proba'] 

    # plot out scalp picture and the auto ic label
    titles = []
    for i, label in enumerate(labels):
        probability = np.max(probs[i])
        
        title = f"{label.capitalize()} ({probability:.0%})"
        titles.append(title)


    figs = ica.plot_components(inst=trials, show=False)

    if not isinstance(figs, list):
        figs = [figs]

    comp_idx = 0
    for fig in figs:
        fig.subplots_adjust(hspace=0.6, wspace=0.1, bottom=0.15)
        for ax in fig.axes:
            if comp_idx >= len(titles):
                break
                
            ax.text(0.5, -0.3, titles[comp_idx], 
                    transform=ax.transAxes, 
                    ha='center', va='top', fontsize=9, color='black', fontweight='bold')
            
            comp_idx += 1

    plt.show()


def plot_erp(evokeds, channel, mean_window, colors=None, linestyles=None, title=None):

    # Create figure
    fig, axes = plt.subplots(1, 1, figsize=(12, 5), sharex=True, sharey=True)
    mne.viz.plot_compare_evokeds(
        evokeds,
        picks=[channel],
        colors=colors,
        linestyles=linestyles,
        axes=axes,
        title=title,   #NOTE: to be changed
        legend='upper right',
        ci=True,
        show=False
    )
    # Add shading for Mean Amplitude window
    axes.axvspan(mean_window[0], mean_window[1], color='gray', alpha=0.2, label=f'Mean Window ({mean_window[0]*1000:.0f}-{mean_window[1]*1000:.0f}ms)')
    plt.show();