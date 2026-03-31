import mne
import numpy as np
import pandas as pd

from functools import lru_cache
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import utils.ccs_eeg_utils as ccs_eeg_utils
from pipeline.s00_add_reference import add_reference_channel
from pipeline.s01_downsample_filter import down_sampling, band_filter, notch_filter
from pipeline.s02_drop_bad_channels import drop_bad_channels, reref
from pipeline.s03_07_trial_rejection import trial_rejection_cust, trial_rejection_mne
from pipeline.s04_ICA import get_ica, iccomponent_removal_author, iccomponent_removal_new
from pipeline.s05_interpolation import interpolation
from pipeline.s07_epoching import epoching, epoching_cust
import config


REPO_ROOT = Path(__file__).resolve().parents[2] # Adjust as needed to point to the root of the repository
EPOCHS_DIR = REPO_ROOT / "output_mne" / "epochs" 
ICA_DIR_CANDIDATES = (
    REPO_ROOT / "output_mne" / "ICA_objects",
    REPO_ROOT / "ICA_objects",
)


TASK_LABELS = {
    1: "low_task",
    2: "mid_task",
    3: "high_task",
}


CONTEXT_LABELS = {
    (1, 50): "low_low",
    (2, 50): "mid_low",
    (2, 80): "mid_high",
    (3, 80): "high_high",
}


FEEDBACK_CODE_MAP = {
    6: {"context": "low_low", "outcome": 1, "outcome_label": "win"},
    7: {"context": "low_low", "outcome": 0, "outcome_label": "loss"},
    16: {"context": "mid_low", "outcome": 1, "outcome_label": "win"},
    17: {"context": "mid_low", "outcome": 0, "outcome_label": "loss"},
    26: {"context": "mid_high", "outcome": 1, "outcome_label": "win"},
    27: {"context": "mid_high", "outcome": 0, "outcome_label": "loss"},
    36: {"context": "high_high", "outcome": 1, "outcome_label": "win"},
    37: {"context": "high_high", "outcome": 0, "outcome_label": "loss"},
}


LOCK_TO_CONDITIONS = {
    "feedback": "feedback_locked",
    "onset": "onset_locked",
}


@lru_cache(maxsize=None)
def _load_site2_montage(montage_path: str):
    '''
    Load the custom montage for site 2. Caches the result to avoid redundant file reads.
    '''
    return mne.channels.read_custom_montage(montage_path)


def _get_ica_path(subject_id: str, pipeline_name: str) -> Path:
    '''
    Search for the ICA file corresponding to the given subject and pipeline in the candidate directories.
    '''
    for base_dir in ICA_DIR_CANDIDATES:
        candidate = base_dir / f"{pipeline_name}-sub{subject_id}_ica.fif"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"ICA file not found for sub-{subject_id} pipeline '{pipeline_name}'. "
        f"Looked in: {[str(path) for path in ICA_DIR_CANDIDATES]}"
    )


def _fit_or_load_ica(
    trials: mne.Epochs,
    subject_id: str,
    pipeline_name: str,
):
    '''
    Attempt to load a pre-fitted ICA object for the given subject and pipeline. If not found, fit a new ICA on the provided trials and save it for future use.
    '''
    try:
        return mne.preprocessing.read_ica(_get_ica_path(subject_id, pipeline_name))
    except FileNotFoundError:
        save_dir = ICA_DIR_CANDIDATES[0]
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{pipeline_name}-sub{subject_id}_ica.fif"
        return get_ica(
            trials,
            method=config.PIPELINES[pipeline_name]["ica_method"],
            save_path=save_path,
        )


def _load_bids_raw(subject_id: str, bids_root: Path, pipeline_name: str = "proposed") -> mne.io.BaseRaw:
    '''
    Load the raw EEG data for a given subject from the BIDS directory, apply the custom montage, and perform initial preprocessing steps (downsampling, filtering, bad channel handling, and re-referencing) according to the specified pipeline.
    '''
    bids_root = Path(bids_root)
    bids_path = BIDSPath(
        subject=subject_id,
        task="casinos",
        datatype="eeg",
        suffix="eeg",
        root=bids_root,
    )
    raw = read_raw_bids(bids_path, verbose="ERROR")
    ccs_eeg_utils.read_annotations_core(bids_path, raw)
    raw.load_data()

    montage_path = bids_root / "code" / config.LOCS_FILENAME["site2"]
    montage = _load_site2_montage(str(montage_path))
    cfg = config.PIPELINES[pipeline_name]
    bad_channels = config.SUBJECT_INFO[subject_id]["bad_channels"]

    raw = add_reference_channel(raw, "Fz")
    raw.set_montage(montage, match_case=False)

    eeg_down = down_sampling(raw, verbose=False)
    eeg_band = band_filter(eeg_down)
    eeg_band_notch = notch_filter(eeg_band)
    eeg_band_notch = drop_bad_channels(bad_channels, eeg_band_notch)
    eeg_band_notch = reref(eeg_band_notch, verbose=False)

    if pipeline_name == "original":
        ica_trials, _ = trial_rejection_cust(
            eeg_band_notch,
            config.CONDITIONS_DICT["onset_locked"],
            **cfg["rejection_params"]["ica"],
        )
    else:
        eeg_ica = band_filter(eeg_down.copy(), f_low=1, f_high=100)
        eeg_ica = drop_bad_channels(bad_channels, eeg_ica)
        eeg_ica = reref(eeg_ica, verbose=False)
        ica_trials = trial_rejection_mne(
            eeg_ica,
            config.CONDITIONS_DICT["onset_locked"],
            **cfg["rejection_params"]["ica"],
        )

    ica = _fit_or_load_ica(ica_trials, subject_id, pipeline_name)
    if pipeline_name == "original":
        eeg_clean = iccomponent_removal_author(
            eeg_band_notch,
            ica_trials,
            ica,
            subject_id,
        )
    else:
        eeg_clean = iccomponent_removal_new(
            eeg_band_notch,
            ica_trials,
            ica,
            subject_id,
        )

    return interpolation(eeg_clean, verbose=False)


def build_feedback_epochs_from_raw(
    raw: mne.io.BaseRaw,
    pipeline_name: str = "proposed",
):
    '''
    Build feedback-locked epochs from the preprocessed raw data according to the specified pipeline. Applies epoching and trial rejection steps as defined in the pipeline configuration.
    '''
    cfg = config.PIPELINES[pipeline_name]
    rejection_params = cfg["rejection_params"]["erp"]

    if pipeline_name == "original":
        epochs, rejection_info = epoching_cust(
            conditions_dict=config.CONDITIONS_DICT["feedback_locked"],
            eeg=raw,
            **rejection_params,
        )
    else:
        epochs = epoching(
            conditions_dict=config.CONDITIONS_DICT["feedback_locked"],
            eeg=raw,
            **rejection_params,
        )
        rejection_info = None

    return epochs, rejection_info


def build_and_save_feedback_epochs(
    raw: mne.io.BaseRaw,
    subject_id: str,
    pipeline_name: str,
    bids_root: Path,
    root_dir: Path | None = None,
    logger=None,
    overwrite: bool = True,
):
    '''
    Build feedback-locked epochs from the raw data, attach metadata from the behavior table, and save the epochs to disk according to the specified pipeline and directory structure.
    '''
    epochs, rejection_info = build_feedback_epochs_from_raw(
        raw=raw,
        pipeline_name=pipeline_name,
    )

    epochs = attach_feedback_metadata(
        epochs=epochs,
        raw=raw,
        subject_id=subject_id,
        pipeline_name=pipeline_name,
        bids_root=bids_root,
        logger=logger,
    )

    save_path = save_epochs(
        epochs=epochs,
        subject_id=subject_id,
        pipeline_name=pipeline_name,
        lock="feedback",
        overwrite=overwrite,
        root_dir=root_dir,
        logger=logger,
    )

    return epochs, save_path, rejection_info



def get_epochs_path(subject_id: str, pipeline_name: str, lock: str, root_dir: Path | None = None) -> Path:
    '''
    Construct the file path for saving/loading epochs based on the subject ID, pipeline name, lock type, and optional root directory.
    '''
    if lock not in LOCK_TO_CONDITIONS:
        raise ValueError(f"Unknown lock '{lock}'. Expected one of {sorted(LOCK_TO_CONDITIONS)}")
    base_dir = EPOCHS_DIR if root_dir is None else Path(root_dir)
    return base_dir / pipeline_name / f"sub-{subject_id}_{lock}-epo.fif"


def load_behavior_table(subject_id: str, early_trials_to_exclude: int, bids_root: Path | None = None) -> pd.DataFrame:
    '''
    Load the behavior table for a given subject from the BIDS directory, perform necessary preprocessing steps (e.g., converting columns to numeric, creating new columns for trial indexing and validity), and return a cleaned DataFrame ready for merging with EEG metadata.
    '''
    if bids_root is None:
        raise ValueError("bids_root must be provided explicitly.")
    bids_root = Path(bids_root)
    beh_path = bids_root / f"sub-{subject_id}" / "beh" / f"sub-{subject_id}_task-casinos_beh.tsv"
    beh = pd.read_csv(beh_path, sep="\t")

    numeric_cols = ["block", "trial", "task", "cue", "prob", "response", "early", "invalid", "outcome", "optimal"]
    for col in numeric_cols:
        beh[col] = pd.to_numeric(beh[col], errors="coerce")
    beh["rt"] = pd.to_numeric(beh["rt"], errors="coerce")

    beh["trial_index_within_task"] = beh.groupby("task").cumcount() # zero-indexed trial number within each task
    beh["is_early_familiarization"] = beh["trial_index_within_task"] < early_trials_to_exclude

    beh["has_feedback_outcome"] = beh["outcome"].isin([0, 1])
    beh["is_behavior_valid"] = (beh["early"] == 0) & (beh["invalid"] == 0) & beh["has_feedback_outcome"]

    beh["task_context"] = beh["task"].map(TASK_LABELS)
    beh["context"] = [CONTEXT_LABELS[(int(task), int(prob))] for task, prob in zip(beh["task"], beh["prob"])]
    beh["cue_value"] = np.where(beh["prob"] == 80, "high", "low")
    beh["outcome_label"] = np.where(beh["outcome"] == 1, "win", "loss")
    beh["subject_id"] = subject_id
    beh["behavior_row_index"] = np.arange(len(beh))
    return beh.reset_index(drop=True)


def build_feedback_metadata(raw: mne.io.BaseRaw, behavior: pd.DataFrame) -> pd.DataFrame:
    '''
    Build a metadata DataFrame for feedback-locked epochs by aligning the feedback events extracted from the raw EEG data with the corresponding rows in the behavior DataFrame. Validates that the contexts and outcomes match between the two sources and constructs a comprehensive metadata table for downstream analysis.
    '''
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    event_code_map = {}
    for event_name, actual_id in event_id.items():
        digits = "".join(ch for ch in str(event_name) if ch.isdigit())
        if not digits:
            continue
        stim_code = int(digits)
        if stim_code in FEEDBACK_CODE_MAP:
            event_code_map[actual_id] = stim_code

    feedback_mask = np.isin(events[:, 2], list(event_code_map.keys()))
    feedback_events = events[feedback_mask]
    feedback_codes = [event_code_map[int(actual_id)] for actual_id in feedback_events[:, 2]]

    # only keep behavior rows that have valid feedback outcomes, do not delete early trials
    behavior_feedback = behavior.loc[behavior["has_feedback_outcome"]].copy().reset_index(drop=True)

    if len(feedback_events) != len(behavior_feedback):
        raise RuntimeError(
            f"Behavior/feedback mismatch: {len(behavior_feedback)} behavior rows vs {len(feedback_events)} feedback events."
        )

    event_context = [FEEDBACK_CODE_MAP[int(code)]["context"] for code in feedback_codes]
    event_outcome = [FEEDBACK_CODE_MAP[int(code)]["outcome"] for code in feedback_codes]
    event_outcome_label = [FEEDBACK_CODE_MAP[int(code)]["outcome_label"] for code in feedback_codes]

    if behavior_feedback["context"].tolist() != event_context:
        raise RuntimeError("Behavior rows and feedback event contexts do not align.")
    if behavior_feedback["outcome"].astype(int).tolist() != event_outcome:
        raise RuntimeError("Behavior rows and feedback event outcomes do not align.")

    metadata = behavior_feedback.copy()
    metadata["feedback_event_index"] = np.where(feedback_mask)[0].astype(int)
    metadata["event_code"] = np.asarray(feedback_codes, dtype=int)
    metadata["event_sample"] = feedback_events[:, 0].astype(int)
    metadata["event_onset_sec"] = feedback_events[:, 0] / raw.info["sfreq"]
    metadata["event_outcome_label"] = event_outcome_label
    return metadata


def exclude_early_trials_epochs(epochs:mne.Epochs) -> mne.Epochs:
    '''
    Exclude epochs corresponding to early familiarization trials based on the 'is_early_familiarization' column in the epochs metadata. This function assumes that the metadata has already been attached to the epochs and contains the necessary column for identifying early trials.
    '''
    if epochs.metadata is None or "is_early_familiarization" not in epochs.metadata.columns:
        raise ValueError("Epochs metadata must contain 'is_early_familiarization' column to exclude early trials.")

    keep_mask = ~epochs.metadata["is_early_familiarization"].to_numpy()
    out = epochs[keep_mask].copy()
    out.metadata = out.metadata.reset_index(drop=True)
    return out


def attach_feedback_metadata(
    epochs: mne.Epochs,
    raw: mne.io.BaseRaw,
    subject_id: str,
    pipeline_name: str,
    bids_root: Path | None = None,
    logger=None,
) -> mne.Epochs:
    '''
    Attach feedback-related metadata to the epochs by merging the behavior table with the event information extracted from the raw EEG data. Validates that the number of epochs matches the number of metadata entries and that the event samples align correctly before attaching the metadata to the epochs object.
    '''
    cfg = config.PIPELINES[pipeline_name]
    behavior = load_behavior_table(subject_id, cfg["early_trial_deletion"], bids_root=bids_root)
    metadata_full = build_feedback_metadata(raw, behavior)
    metadata_kept = (
        metadata_full
        .set_index("feedback_event_index")
        .loc[epochs.selection]
        .reset_index(drop=True)
    )

    if len(metadata_kept) != len(epochs):
        raise RuntimeError(f"Metadata/epochs mismatch: {len(metadata_kept)} vs {len(epochs)}")

    if not np.array_equal(metadata_kept["event_sample"].to_numpy(), epochs.events[:, 0]):
        raise RuntimeError("Metadata event_sample does not match epochs.events[:, 0].")

    epochs.metadata = metadata_kept
    epochs = exclude_early_trials_epochs(epochs)

    if logger is not None:
        logger.info(
            "Attached feedback metadata for sub-%s: kept %s epochs after early-trial exclusion",
            subject_id,
            len(epochs),
        )
    return epochs


def save_epochs(epochs: mne.Epochs, subject_id: str, pipeline_name: str, lock: str = "feedback",
                overwrite: bool = False, root_dir: Path | None = None, logger=None) -> Path:
    '''
    Save the given epochs to disk at a path determined by the subject ID, pipeline name, and lock type. Creates the necessary directories if they do not exist and optionally logs the saving action.
'''
    path = get_epochs_path(subject_id, pipeline_name, lock, root_dir=root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(path, overwrite=overwrite)
    if logger is not None:
        logger.info("Saved %s epochs for sub-%s to %s", lock, subject_id, path)
    return path


def load_epochs(subject_id: str, pipeline_name: str, lock: str = "feedback", preload: bool = True,
                root_dir: Path | None = None, logger=None) -> mne.Epochs:
    '''
    Load epochs from disk for a given subject, pipeline, and lock type. Validates that the epochs file exists and optionally logs the loading action.
'''
    path = get_epochs_path(subject_id, pipeline_name, lock, root_dir=root_dir)
    if not path.exists():
        raise FileNotFoundError(f"Epochs file not found: {path}")
    if logger is not None:
        logger.info("Loading saved %s epochs for sub-%s from %s", lock, subject_id, path)
    return mne.read_epochs(path, preload=preload, verbose="ERROR")
