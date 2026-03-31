import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer

try:
    from .epoch_io import load_epochs
except ImportError:
    from decoding.decoding_utils.epoch_io import load_epochs


def decode_context_window(
    epochs: mne.Epochs,
    context: str,
    window_start: float = 0.24,
    window_end: float = 0.34,
    n_splits: int = 5,
):
    '''
    Perform window-decoding for a single subject and context.'''
    mask = epochs.metadata["context"] == context
    if int(mask.sum()) == 0:
        raise RuntimeError(f"No epochs available for context '{context}'.")

    context_epochs = (
        epochs[mask.to_numpy()]
        .copy()
        .pick("eeg")
        .crop(tmin=window_start, tmax=window_end)
    )

    y = context_epochs.metadata["outcome"].to_numpy(dtype=int)
    unique_y = np.unique(y)
    if set(unique_y) != {0, 1}:
        raise ValueError(f"Outcome must be coded as 0/1, got {unique_y}")

    class_counts = np.bincount(y, minlength=2)
    min_class_n = int(class_counts.min())
    if min_class_n < 2:
        raise RuntimeError(f"Not enough trials to decode {context}: class counts {class_counts.tolist()}")

    cv_splits = min(n_splits, min_class_n)
    X = context_epochs.get_data(copy=True)

    estimator = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000),
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    return {
        "fold_auc": scores,
        "mean_auc": float(scores.mean()),
        "std_auc": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "n_trials": int(len(context_epochs)),
        "n_win": int((y == 1).sum()),
        "n_loss": int((y == 0).sum()),
        "cv_splits": int(cv_splits),
        "window_start_sec": float(window_start),
        "window_end_sec": float(window_end),
    }


def summarize_subject_window(subject_id: str, context: str, result: dict) -> dict:
    '''
    Create a summary dictionary for a single subject-context decoding result.
    '''
    return {
        "subject_id": subject_id,
        "context": context,
        "n_trials": result["n_trials"],
        "n_win": result["n_win"],
        "n_loss": result["n_loss"],
        "cv_splits": result["cv_splits"],
        "window_start_sec": result["window_start_sec"],
        "window_end_sec": result["window_end_sec"],
        "mean_auc": result["mean_auc"],
        "std_auc": result["std_auc"],
    }


def compute_group_stats_window(summary_df: pd.DataFrame, contexts: list[str]) -> dict:
    '''
    Compute group-level statistics for window-decoding results.
    '''
    group_stats = {
        "contexts": contexts,
        "subjects": sorted(summary_df["subject_id"].unique().tolist()),
    }

    for context in contexts:
        subset = summary_df.loc[summary_df["context"] == context, "mean_auc"].dropna()
        group_stats[context] = {
            "mean_auc": float(subset.mean()),
            "std_auc": float(subset.std(ddof=1)) if len(subset) > 1 else 0.0,
        }

    if len(contexts) == 2:
        pivot = summary_df.pivot(index="subject_id", columns="context", values="mean_auc").dropna()
        if len(pivot) >= 2:
            diff = pivot[contexts[1]] - pivot[contexts[0]]
            t_stat, p_value = stats.ttest_rel(pivot[contexts[1]], pivot[contexts[0]])
            group_stats["paired_comparison"] = {
                "contrast": f"{contexts[1]} - {contexts[0]}",
                "n_subjects": int(len(pivot)),
                "mean_difference": float(diff.mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
    return group_stats


def save_outputs_window(
    output_dir: Path,
    pipeline_name: str,
    contexts: list[str],
    summary_df: pd.DataFrame,
    group_stats: dict,
    auc_store: dict,
):
    '''
    Save window-decoding outputs to disk, including summary CSV, group stats JSON, and fold AUCs in NPZ format.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"decoding_feedback_outcome_window_{pipeline_name}_{'_vs_'.join(contexts)}"
    summary_path = output_dir / f"{base_name}_summary.csv"
    stats_path = output_dir / f"{base_name}_group_stats.json"
    npz_path = output_dir / f"{base_name}_folds.npz"

    summary_df.to_csv(summary_path, index=False)
    stats_path.write_text(json.dumps(group_stats, indent=2))

    arrays = {}
    for subject_id, context_map in auc_store.items():
        for context, payload in context_map.items():
            arrays[f"{subject_id}__{context}__fold_auc"] = payload["fold_auc"]
    np.savez(npz_path, **arrays)

    return summary_path, stats_path, npz_path


def run_group_decoding_window(
    subjects: list[str],
    pipeline_name: str,
    contexts: list[str],
    window_start: float = 0.24,
    window_end: float = 0.34,
    root_dir: Path | None = None,
    logger=None,
):
    '''
    Run window-decoding for a group of subjects and contexts, returning summary DataFrame, group stats, and AUC store.
    '''
    summary_rows = []
    auc_store = {}

    for subject_id in subjects:
        if logger is not None:
            logger.info("Processing sub-%s...", subject_id)

        epochs = load_epochs(
            subject_id,
            pipeline_name,
            lock="feedback",
            preload=True,
            root_dir=root_dir,
        )
        auc_store[subject_id] = {}

        for context in contexts:
            try:
                result = decode_context_window(
                    epochs,
                    context=context,
                    window_start=window_start,
                    window_end=window_end,
                )
            except (RuntimeError, ValueError) as exc:
                message = f"Skipping sub-{subject_id} {context}: {exc}"
                if logger is not None:
                    logger.warning(message)
                else:
                    warnings.warn(message)
                continue

            auc_store[subject_id][context] = result
            summary_rows.append(summarize_subject_window(subject_id, context, result))

    if not summary_rows:
        raise RuntimeError("No valid subject-context decoding results were produced.")

    summary_df = pd.DataFrame(summary_rows)
    group_stats = compute_group_stats_window(summary_df, contexts)
    return summary_df, group_stats, auc_store
