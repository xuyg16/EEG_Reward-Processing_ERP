import mne
import numpy as np
import pandas as pd

from pathlib import Path
from scripts.utils.tools import get_event_dict
from scripts import config

REPO_ROOT = Path(__file__).resolve().parents[2] # Adjust as needed to point to the root of the repository
EPOCHS_DIR = REPO_ROOT / "output_mne" / "epochs" 

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


def get_epochs_path(subject_id: str, pipeline_name: str, lock: str, root_dir: Path | None = None) -> Path:
    if lock not in LOCK_TO_CONDITIONS:
        raise ValueError(f"Unknown lock '{lock}'. Expected one of {sorted(LOCK_TO_CONDITIONS)}")
    base_dir = EPOCHS_DIR if root_dir is None else Path(root_dir)
    return base_dir / pipeline_name / f"sub-{subject_id}_{lock}-epo.fif"


def load_behavior_table(subject_id: str, early_trials_to_exclude: int, bids_root: Path | None = None) -> pd.DataFrame:
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
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    event_code_map = {}
    for event_name, actual_id in event_id.items():
        stim_code = get_event_dict(event_name)
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
    metadata["feedback_event_index"] = np.arange(len(feedback_events), dtype=int)
    metadata["event_code"] = np.asarray(feedback_codes, dtype=int)
    metadata["event_sample"] = feedback_events[:, 0].astype(int)
    metadata["event_onset_sec"] = feedback_events[:, 0] / raw.info["sfreq"]
    metadata["event_outcome_label"] = event_outcome_label
    return metadata


def exclude_early_trials_epochs(epochs:mne.Epochs) -> mne.Epochs:
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
    path = get_epochs_path(subject_id, pipeline_name, lock, root_dir=root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(path, overwrite=overwrite)
    if logger is not None:
        logger.info("Saved %s epochs for sub-%s to %s", lock, subject_id, path)
    return path


def load_epochs(subject_id: str, pipeline_name: str, lock: str = "feedback", preload: bool = True,
                root_dir: Path | None = None, logger=None) -> mne.Epochs:
    path = get_epochs_path(subject_id, pipeline_name, lock, root_dir=root_dir)
    if logger is not None:
        logger.info("Loading saved %s epochs for sub-%s from %s", lock, subject_id, path)
    return mne.read_epochs(path, preload=preload, verbose="ERROR")
