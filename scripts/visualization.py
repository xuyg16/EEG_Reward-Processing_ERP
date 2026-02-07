from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import mne



def show_single_psd(eeg_data, y_min=-50, y_max=50, picks=None, title=None):
    '''
    Plot the power spectral density (PSD) of EEG data.

    :param eeg_data: MNE Raw or Epochs object
    :param y_min: Minimum y-axis value for the plot
    :param y_max: Maximum y-axis value for the plot
    :param picks: Channels to include in the plot
    :param title: Title for the plot
    '''
    eeg_psd = eeg_data.compute_psd().plot(picks=picks, show=False)
    ax = eeg_psd.axes[0]
    y_min = y_min
    y_max = y_max
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    plt.show()



def psd_compare(eegs, labels, title, figsize=(8, 6), picks=['FCz'], y_min=-46.4, y_max=-46, x_min=49.5, x_max=50.5):
    '''
    Compare power spectral density (PSD) of EEG data across different preprocessing strategies.

    :param eegs: List of MNE Raw or Epochs objects
    :param labels: List of labels for each EEG object
    :param title: Title for the plot
    :param figsize: Figure size tuple (width, height)
    :param picks: Channels to include in the plot
    :param y_min: Minimum y-axis value for the plot
    :param y_max: Maximum y-axis value for the plot
    :param x_min: Minimum x-axis value for the plot
    :param x_max: Maximum x-axis value for the plot
    '''
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



def iclabel_visualize(ica, ic_labels, exclude_idx=None, show=False, trials=None, save_path=None):
    '''
    Visualize Independent Component Analysis (ICA) components with their corresponding labels and probabilities.
    
    :param ica: ica object
    :param ic_labels: Dictionary containing 'labels' and 'y_pred_proba'
    :param trials: MNE Epochs object for plotting components. Only needed if want interactive plot with topographies.
    '''
    label_dict = ['brain', 'muscle', 'eye blink', 'eye movement', 'heart', 'line noise', 'channel noise', 'other']

    # plot out scalp picture and the auto ic label
    titles = []
    for probabilities in ic_labels:
        max_prob = np.max(probabilities)
        label_idx = np.argmax(probabilities)
        label = label_dict[label_idx]
        print(f'The label with highest probabiliy ({max_prob}) is {label}')
        
        title = f"{label.capitalize()} ({max_prob:.0%})"
        titles.append(title)


    figs = ica.plot_components(picks=exclude_idx, inst=trials, show=False)

    if not isinstance(figs, list):
        figs = [figs]

    comp_idx = 0
    for fig in figs:
        fig.set_layout_engine(None)
        fig.subplots_adjust(hspace=0.6, wspace=0.1, bottom=0.15)
        for ax in fig.axes:
            if comp_idx >= len(titles):
                break
                
            ax.text(0.5, -0.3, titles[comp_idx], 
                    transform=ax.transAxes, 
                    ha='center', va='top', fontsize=9, color='black', fontweight='bold')
            
            comp_idx += 1

    if show:
        plt.show();
    if save_path:
        with PdfPages(save_path) as pdf:
            for fig in figs:
                pdf.savefig(fig)
        print(f'Save ')


def plot_erp(evokeds, channel='FCz', mean_window=[0.240, 0.340], ylim=[-5, 10], diff=False, title=None):
    '''
    Plot ERP waveforms with mean amplitude window shading.
    
    :param evokeds: Dictionary of MNE Evoked objects
    :param channel: Channel name to plot
    :param mean_window: Tuple indicating the start and end of the mean amplitude window (in seconds)
    :param colors: colors for each condition
    :param linestyles: linestyles for each condition
    :param title: Title for the plot
    '''
    if diff:
        colors = {
        'Low-Low': '#4C72B0',   # Muted Blue
        'Mid-Low': '#64B5CD',   # Soft Cyan
        'Mid-High': '#E1BC66',  # Sand/Gold
        'High-High': '#C44E52'  # Muted Crimson 
        }
        linestyles = None
    else:
        colors = {
        'Low-Low Win': 'red', 'Low-Low Loss': 'blue',
        'Mid-Low Win': 'red', 'Mid-Low Loss': 'blue',
        'Mid-High Win': 'red', 'Mid-High Loss': 'blue',
        'High-High Win': 'red', 'High-High Loss': 'blue'
        }
        linestyles = {
            'Low-Low Win': '-', 'Low-Low Loss': '-',
            'Mid-Low Win': '--', 'Mid-Low Loss': '--',
            'Mid-High Win': '-.', 'Mid-High Loss': '-.',
            'High-High Win': ':', 'High-High Loss': ':'
        }

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
        show=False,
        show_sensors=False,
        ylim=dict(eeg=ylim)
    )
    # Add shading for Mean Amplitude window
    axes.axvspan(mean_window[0], mean_window[1], color='gray', alpha=0.2, label=f'Mean Window ({mean_window[0]*1000:.0f}-{mean_window[1]*1000:.0f}ms)')
    plt.show();

# NOTE: need to change to average calculation instead of timepoints
def plot_topo_serires(evokeds, times = [0.18, 0.22, 0.26, 0.30, 0.34, 0.38], vlimit = (-5, 5)):
    '''
    Plot topo sereies for the grand average erps
    
    :param evokeds: input grand averages
    :param times: time points for each single topography
    :param vlimit: amplitude limites for the topographies
    '''
    for condition, evoked in evokeds.items():
        print(f"Plotting Topomap for: {condition}")
        evoked.plot_topomap(times=times, ch_type='eeg', colorbar=True, vlim=vlimit)



def plot_binning_results(rewp_differences, title='RewP Mean Amplitude Across Chronological Bins', figsize=(12, 6)):
    plt.figure(figsize=figsize)

    target_conditions = ['Low-Low', 'Mid-Low', 'Mid-High', 'High-High']
    colors = {
        'Low-Low': '#4C72B0',   # Muted Blue
        'Mid-Low': '#64B5CD',   # Soft Cyan
        'Mid-High': '#E1BC66',  # Sand/Gold
        'High-High': '#C44E52'  # Muted Crimson 
        }
    
    for i, cond in enumerate(target_conditions):
        # 1. Extract the bin-by-bin means
        y_values = [bin_res[cond]['mean'] for bin_res in rewp_differences if cond in bin_res]
        x_values = list(range(1, len(y_values) + 1))
        
        if y_values:
            color = colors[cond]
            
            # 2. Calculate the global average for this condition across all bins
            avg_value = np.mean(y_values)
            
            # 3. Plot the "Average" line (Lower Saturation / Faint)
            # alpha=0.3 makes it transparent/desaturated
            plt.axhline(y=avg_value, color=color, linestyle='--', alpha=0.3, 
                        label=f'{cond} Avg') # Labeling first one as example

            # 4. Plot the main "Trend" line (Full Saturation)
            plt.plot(x_values, y_values, marker='o', label=cond, 
                    linewidth=2.5, markersize=8, color=color)

    # Formatting
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Chronological Bin', fontsize=12)
    plt.ylabel('Mean Amplitude (µV)', fontsize=12)
    plt.xticks(range(1, len(y_values) + 1))
    plt.axhline(0, color='black', alpha=0.2) # Zero baseline

    # Refined Legend
    plt.legend(title='Reward Level', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()

    plt.show()