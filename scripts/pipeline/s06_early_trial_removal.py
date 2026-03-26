import mne
import numpy as np

def exclude_early_trials(data, num_to_exclude=10, verbose=True):
    '''
    Exclude first few trials (default: 10) for each task type from the Epochs data.
        'S  1' = start of low-value task fixation (all low cue)
        'S 11' = start of mixed task fixation(low cue)
        'S 21' = start of mixed task fixation(high cue)
        'S 31' = start of high-value task fixation(all high cue)

    :param data: Epochs object
    :param num_to_exclude: Number of early trials to exclude per task type

    :return: Epochs object with early trials excluded
    '''
    events, event_dict = mne.events_from_annotations(data)
    
    # trial counts per task type
    trial_count = {'S 1': 0, 'S 11_S 21': 0, 'S 31': 0}

    task_start_ids = {}
    
    # Find the event ID for task starters so that we could know when one trial stops
    for event_name, event_id in event_dict.items():
        if 'Stimulus:S  1' == event_name:
            task_start_ids['S 1'] = [event_id]
        elif 'Stimulus:S 11' == event_name or 'Stimulus:S 21'== event_name:
            if 'S 11_S 21' not in task_start_ids: 
                task_start_ids['S 11_S 21'] = []
            task_start_ids['S 11_S 21'].append(event_id)
        elif 'Stimulus:S 31' == event_name:
            task_start_ids['S 31'] = [event_id]

    # flatten all start ids
    all_task_starts = []
    for ids in task_start_ids.values():
        all_task_starts.extend(ids)

    events_to_exclude = []
    i = 0
    while i < len(events):
        event_id = events[i, 2] # ith event's eventid (2 means event id)
        
        # Determine if this event marks the start of a trial
        current_task_type = None
        if 'S 1' in task_start_ids and event_id in task_start_ids['S 1']:
            current_task_type = 'S 1'
        elif 'S 11_S 21' in task_start_ids and event_id in task_start_ids['S 11_S 21']:
            current_task_type = 'S 11_S 21'
        elif 'S 31' in task_start_ids and event_id in task_start_ids['S 31']:
            current_task_type = 'S 31'

        # If it is a trial start
        if current_task_type is not None:
            # Check if the trial should be excluded
            if trial_count[current_task_type] < num_to_exclude:
                events_to_exclude.append(i)
                
                # Mark subsequent events until the next trial starts
                j = 1
                while (i + j) < len(events):
                    next_id = events[i + j, 2]
                    if next_id in all_task_starts:
                        break 
                    events_to_exclude.append(i + j)
                    j += 1
            
            trial_count[current_task_type] += 1
        
        i += 1

    # Remove excluded events
    # Create mask: True = keep, False = exclude
    keep_mask = np.ones(len(events), dtype=bool)
    keep_mask[events_to_exclude] = False

    # Keep only non-excluded events
    events_filtered = events[keep_mask]

    data_clean = data.copy()
    
    # Add back the filtered events
    new_annot = mne.annotations_from_events(
        events_filtered,
        data_clean.info['sfreq'],
        event_desc={v: k for k, v in event_dict.items()}
    )
    data_clean.set_annotations(new_annot)
    
    if verbose:
        print(f"Excluded {len(events_to_exclude)} events (first {num_to_exclude} trials of each block).")
    return data_clean