# -------- General parameters --------
BIDS_ROOT = {
    'qian': "/Users/qianyueli/Documents/UniStuttgart/26WI/EEG/EEG_project/reward_dataset",
    'zheng': "C:\\Users\\Zheng\\Desktop\\fourth semester\\EEG\\reward_dataset\\reward_dataset\\reward_dataset",
    'xu': "/Users/xuyg/GitHub/EEG_Reward-Processing_ERP/ds004147"
}
LOCS_FILENAME = {   #NOTE: download .locs file from author's Github, put it under root_of_dataset/code/..
    'site2': 'site2channellocations.locs',
    'common': 'common.locs',
}

# -------- Data parameters --------
SUBJECT_INFO = {
    '27': {'learner': True,
           'bad_channels': []},
    '28': {'learner': True,
           'bad_channels': []},
    '29': {'learner': False,
           'bad_channels': []},
    '30': {'learner': False,
           'bad_channels': []},
    '31': {'learner': True,
           'bad_channels': []},
    '32': {'learner': False,
           'bad_channels': []},
    '33': {'learner': False,
           'bad_channels': []},
    '34': {'learner': True,
           'bad_channels': []},
    '35': {'learner': True,
           'bad_channels': []},
    '36': {'learner': True,
           'bad_channels': []},
    '37': {'learner': True,
           'bad_channels': []},
    '38': {'learner': True,
           'bad_channels': []},
}

CONDITIONS_DICT = {
    'onset_locked':[
        'Stimulus:S  1', 'Stimulus:S 11', 'Stimulus:S 21', 'Stimulus:S 31'
    ],
    'feedback_locked':{
    'Low-Low Win':   ['Stimulus:S  6'], 
    'Low-Low Loss':  ['Stimulus:S  7'], 
    'Mid-Low Win':   ['Stimulus:S 16'], 
    'Mid-Low Loss':  ['Stimulus:S 17'], 
    'Mid-High Win':  ['Stimulus:S 26'], 
    'Mid-High Loss': ['Stimulus:S 27'],
    'High-High Win': ['Stimulus:S 36'], 
    'High-High Loss':['Stimulus:S 37'],
    }
}

# -------- Preprocessing parameters --------
SAMPLING_RATE = 250  # in Hz
BANDPASS_FREQS = (0.1, 30)  # in Hz
NOTCH_FREQS = 50  # in Hz

# -------- Pipeline parameters --------
## tbd: double check these parameters and add more if needed
PIPELINES = {
    'original':{
        'ica_method': 'infomax',
        'ica_criteria': None, #tbd to be edited if needed
        'trial_rejection_method': 'custom',
        'rejection_params': { #trial rejection
            'ica':{
                'maxMin': 500e-6,
                'level': 500e-6,
                'step': 40e-6,
                'lowest': 0.1e-6,
                'tmin': 0,
                'tmax': 3,
                'baseline': None
            },
            'erp':{
                'maxMin': 150e-6,
                'level': 150e-6,
                'step': 40e-6,
                'lowest': 0.1e-6,
                'tmin': -0.2,
                'tmax': 0.6,
                'baseline': (-0.2, 0)
            }
        },
        'bad_channels_rejection_criteria': 0.2, # 20%
        'epoch_tmin': -0.2,
        'epoch_tmax': 0.6,
        'early_trial_deletion': 10, # delete first 10 trials to avoid learning effects
        'evoked_proportiontocut': 0.00 # no trimming
    },
    'proposed':{
        'ica_method': 'picard',
        'ica_criteria': None, #tbd to be edited if needed
        'trial_rejection_method': 'mne',
        'rejection_params': { #trial rejection
            'ica':{
                'max': 500e-6,
                'min': 0.1e-6,
                'tmin': 0,
                'tmax': 3,
                'baseline': None
            },
            'erp':{
                'max': 150e-6,
                'min': 0.1e-6,
                'tmin': -0.2,
                'tmax': 0.6,
                'baseline': (-0.2, 0)
            }
        },
        'bad_channels_rejection_criteria': 0.2, # 20%
        'epoch_tmin': -0.2,
        'epoch_tmax': 0.6,
        'early_trial_deletion': 10, # delete first 10 trials to avoid learning effects
        'evoked_proportiontocut': 0.05 # 5% trimming
    }
}


# -------- Other parameters --------
N_BINS = 5