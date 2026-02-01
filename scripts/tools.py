import mne
import re

def get_valid_input(prompt, valid_options):
    """
    Prompt the user for input until a valid option is provided.

    :param prompt: The prompt message to display to the user
    :param valid_options: A list of valid options
    :return: The valid input provided by the user
    """
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() in [opt.lower() for opt in valid_options]:
            print(f'Input accepted:{user_input}')
            return user_input
        else:
            print(f"Invalid input. Please choose from {valid_options}.")

def _normalize_event_key(key):
    key = str(key).strip()
    key = key.replace("Stimulus/", "Stimulus:").replace("Stimulus :", "Stimulus:")
    if key.startswith("Stimulus:"):
        key = key[len("Stimulus:"):]
    # collapse multiple spaces
    key = " ".join(key.split())
    return key



def _flatten_conditions(condition_dict):
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
    
    :param eeg: input eeg data in mne.Raw format
    :param condition_dict: conditions for epoching

    :return evts: identity and timing of the epochs. 
    :return conidtion_dict: events dictionary filtered by conditions
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