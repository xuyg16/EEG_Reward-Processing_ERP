from stats.logging_utils import log_bad_channels

def find_bad_channels(epochs, reject_criteria, subject_id, custom=False, rejection_info=None, logger=None, verbose=True):
    '''
    Find and print channels that exceed the rejection criteria based on epoch drops.

    :param epochs: MNE Epochs object
    :param reject_criteria: rejection criteria (e.g., 0.2 for 20%)
    :param custom: boolean indicating whether custom trial rejection was used (default: False)

    :return: List of bad channels exceeding the rejection criteria
    '''
    n_total_epochs = len([log for log in epochs.drop_log if 'IGNORED' not in log])   # number of epochs before dropping
    print(n_total_epochs)
    if not custom:
        # Get the drop log which lists the channels responsible for each drop
        drop_log = epochs.drop_log
    elif custom and rejection_info is not None:
        # Construct drop log from custom rejection info
        drop_log = []
        for epoch_idx in range(n_total_epochs):
            if epoch_idx in rejection_info:
                drop_log.append(tuple(rejection_info[epoch_idx]))
            else:
                drop_log.append(())
    else:
        raise ValueError("Custom rejection info must be provided when 'custom' is True.")

    # Initialize a dictionary to count how many times each channel caused a drop
    channel_drop_counts = {}
    ch_names = epochs.ch_names

    # Iterate through the drop log and count
    for reasons in drop_log:
        # 'reasons' is a tuple of channel names (e.g., ('Fz', 'T7') or ())
        if reasons:  # Only process epochs that were actually dropped
            for ch_name in reasons:
                channel_drop_counts[ch_name] = channel_drop_counts.get(ch_name, 0) + 1

    # Identify channels whose rejection rate exceeds the 20% threshold
    bad_channels_to_mark = []
    if verbose:
        print("--- Channel Rejection Summary ---")

    for ch_name in ch_names:
        # Use .get() to handle channels that never caused a drop (count = 0)
        drop_count = channel_drop_counts.get(ch_name, 0)
        rejection_rate = drop_count / n_total_epochs

        if verbose:
            print(f"{ch_name}: {drop_count}/{n_total_epochs} drops ({rejection_rate:.1%})")

        if rejection_rate > reject_criteria:
            bad_channels_to_mark.append(ch_name)
    
    if verbose:
        print("---------------------------------")
        print(f"Channels exceeding {reject_criteria:.0%} threshold: {bad_channels_to_mark}")

    if logger:
        log_bad_channels(logger, subject_id, bad_channels_to_mark)