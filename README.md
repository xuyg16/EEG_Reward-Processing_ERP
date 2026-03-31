# EEG Reward Processing ERP

**Authors:** *Qianyue Li*, *Zheng Lin*, *Yanhong Xu*  
**Course:** *Signal Processing and Analysis of Human Brain Potentials (EEG)*  
**Semester:** *Winter semester 2025/2026*

This repository reproduces the ERP analysis from *Task-level value affects trial-level reward processing* and includes additional decoding analyses based on saved feedback-locked epochs.

## Quick Start

Download dataset

This project utilizes the `ds004147` dataset from Nemar, which contains EEG recordings from 12 participants performing the "Casino Task". The original paper uses dataset from 38 participants, recorded from 2 locations (UVic and Oxford). However, participants from the UVic testing site did not consent for their data to be publicly shared, so only data from location 2 could be accessed. 

You can download the required BIDS-compliant dataset from: [Download dataset here](https://nemar.org/dataexplorer/detail?dataset_id=ds004147)

Then:

1. Edit `scripts/config.py` so `BIDS_ROOT` points to your local `ds004147` path.
2. Run `scripts/single_subject_processing.ipynb` or `scripts/multi_subject_processing.ipynb`.
3. Run `scripts/decoding/make_feedback_epochs.ipynb` to save epochs.
4. Run `scripts/decoding/time_resolved_decoding.ipynb` and/or `scripts/decoding/window_decoding.ipynb`.

## What This Repository Contains

- EEG preprocessing and ERP/RewP analysis notebooks
- Shared pipeline, utility, and statistics modules under `scripts/`
- Decoding notebooks that reuse saved feedback epochs
- A Quarto report with methods, results, and discussion in `report/`

## Repository Structure

```text
EEG_Reward-Processing_ERP/               
├── ds004147/                 # BIDS dataset
├── scripts/
│   ├── config.py             # local paths and analysis parameters
│   ├── pipeline/             # preprocessing steps
│   ├── utils/                # helper functions and logging
│   ├── stats/                # RewP/statistical analysis code
│   ├── decoding/             # epoch I/O and decoding notebooks
│   ├── single_subject_processing.ipynb
│   └── multi_subject_processing.ipynb
├── output_mne/               # generated outputs
│   ├── logs/                 # runtime logs, cleared on a fresh rerun
│   ├── stats/                # saved result tables / summaries
│   ├── epochs/               # reusable saved epochs
│   ├── ICA_objects/          # saved ICA decompositions
│   └── plots/                # exported figures
├── report/                   # Quarto report and rendered output
├── presentation/
└── research/
└── README.md 
└── requirements.txt          # List of modules and packages that are used for this 
```



## Recommended Reproduction Order

### 1. ERP / RewP preprocessing

Use one of the main notebooks in `scripts/`:

- `single_subject_processing.ipynb`: inspect and process one participant
- `multi_subject_processing.ipynb`: run group-level ERP/RewP analysis and summary statistics

In the notebooks, check:

- `USER`: selects the path key used in `config.BIDS_ROOT`
- `ACTIVE_PIPELINE`: choose `original` or `proposed`
- subject selection if you only want a subset

### 2. Save feedback-locked epochs for decoding

Run:

- `scripts/decoding/make_feedback_epochs.ipynb`

This creates reusable feedback epochs with metadata under `output_mne/epochs/`.

### 3. Run decoding analyses

Then run one or both:

- `scripts/decoding/time_resolved_decoding.ipynb`
- `scripts/decoding/window_decoding.ipynb`


## Suggested Rule of Thumb
- If a reader asks "How do I run this project?", the answer belongs in `README.md`.  
- If a reader asks "What did you do, why, and what did you find?", the answer belongs in the report.


## Useful Resource

- [COBIDAS MEEG: how to report EEG/MEG processing clearly](https://cobidasmeeg.wordpress.com/)
- [mne.preprocessing.ICA](https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA)
- [Overview of artifact detection](https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#tut-artifact-overview)