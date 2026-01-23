import mne

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



def get_event_dict(eeg, condition_dict):
    '''
    get the dictionary for the events under specified conditions
    
    :param eeg: input eeg data in mne.Raw format
    :param condition_dict: conditions for epoching

    :return evts: identity and timing of the epochs. 
    :return conidtion_dict: events dictionary filtered by conditions
    '''
    evts, evts_dict = mne.events_from_annotations(eeg)
    evts_dict_stim = {k: evts_dict[k] for k in evts_dict.keys() if k in condition_dict}
    
    return evts, evts_dict_stim