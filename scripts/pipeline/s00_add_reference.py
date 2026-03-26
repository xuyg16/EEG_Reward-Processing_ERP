import mne

def add_reference_channel(raw, new_ref='Fz'):
    mne.add_reference_channels(raw, new_ref, copy=False)  # add new_ref as reference channel
    return raw