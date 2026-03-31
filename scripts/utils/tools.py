import mne
import re


def _normalize_event_key(key):
    '''
    Normalize an event key by stripping whitespace, replacing "Stimulus/" with "Stimulus:", and collapsing multiple spaces.
    '''
    key = str(key).strip()
    key = key.replace("Stimulus/", "Stimulus:").replace("Stimulus :", "Stimulus:")
    if key.startswith("Stimulus:"):
        key = key[len("Stimulus:"):]
    # collapse multiple spaces
    key = " ".join(key.split())
    return key



def _flatten_conditions(condition_dict):
    '''
    Flatten a condition dictionary which may contain nested dicts or lists of conditions.
    '''
    if isinstance(condition_dict, dict):
        items = []
        for v in condition_dict.values():
            if isinstance(v, (list, tuple, set)):
                items.extend(v)
            else:
                items.append(v)
        return items
    if isinstance(condition_dict, (list, tuple, set)):
        return list(condition_dict)
    return [condition_dict]


def get_event_dict(eeg, condition_dict):
    '''
    get the dictionary for the events under specified conditions
    '''
    evts, evts_dict = mne.events_from_annotations(eeg)

    # Build normalized lookup for actual event keys
    norm_to_orig = {}
    for k in evts_dict.keys():
        nk = _normalize_event_key(k)
        if nk not in norm_to_orig:
            norm_to_orig[nk] = k

    cond_list = _flatten_conditions(condition_dict)
    evts_dict_stim = {}
    missing = []
    for cond in cond_list:
        nk = _normalize_event_key(cond)
        if nk in norm_to_orig:
            orig_key = norm_to_orig[nk]
            evts_dict_stim[cond] = evts_dict[orig_key]
        else:
            missing.append(cond)

    if missing:
        print(f"[get_event_dict] Warning: missing event keys: {missing}")

    return evts, evts_dict_stim