import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .epoch_io import load_epochs
except ImportError:
    from decoding.decoding_utils.epoch_io import load_epochs


def decode_context(epochs: mne.Epochs, context: str, n_splits: int = 5):
    '''
    Perform time-resolved decoding of feedback outcome (win vs. loss) for a single context.
    '''
    mask = epochs.metadata["context"] == context
    if int(mask.sum()) == 0:
        raise RuntimeError(f"No epochs available for context '{context}'.")

    context_epochs = epochs[mask.to_numpy()].copy().pick("eeg")
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
        StandardScaler(),
        LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000),
    )
    time_decoder = SlidingEstimator(estimator, scoring="roc_auc", n_jobs=1, verbose=False)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_multiscore(time_decoder, X, y, cv=cv, n_jobs=1)

    return {
        "times": context_epochs.times.copy(),
        "scores": scores,
        "mean_scores": scores.mean(axis=0),
        "n_trials": int(len(context_epochs)),
        "n_win": int((y == 1).sum()),
        "n_loss": int((y == 0).sum()),
        "cv_splits": int(cv_splits),
    }


def summarize_subject(subject_id: str, context: str, result: dict, window_start: float, window_end: float) -> dict:
    '''
    Extract summary metrics for a single subject-context decoding result, focusing on a specified time window.
    '''
    times = result["times"]
    window_mask = (times >= window_start) & (times <= window_end)
    if not np.any(window_mask):
        raise RuntimeError(f"No samples found in summary window {window_start:.3f}-{window_end:.3f}s")

    mean_scores = result["mean_scores"]
    peak_idx = int(np.argmax(mean_scores))
    return {
        "subject_id": subject_id,
        "context": context,
        "n_trials": result["n_trials"],
        "n_win": result["n_win"],
        "n_loss": result["n_loss"],
        "cv_splits": result["cv_splits"],
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "window_auc_mean": float(mean_scores[window_mask].mean()),
        "peak_auc": float(mean_scores[peak_idx]),
        "peak_time_sec": float(times[peak_idx]),
    }


def compute_group_stats(summary_df: pd.DataFrame, contexts: list[str]) -> dict:
    '''
    Compute group-level statistics from the summary DataFrame, including means, stds, and paired comparisons if applicable.
    '''
    group_stats = {
        "contexts": contexts,
        "subjects": sorted(summary_df["subject_id"].unique().tolist()),
    }

    for context in contexts:
        subset = summary_df.loc[summary_df["context"] == context, "window_auc_mean"]
        group_stats[context] = {
            "mean_window_auc": float(subset.mean()),
            "std_window_auc": float(subset.std(ddof=1)) if len(subset) > 1 else 0.0,
        }

    if len(contexts) == 2:
        pivot = summary_df.pivot(index="subject_id", columns="context", values="window_auc_mean").dropna()
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


def save_outputs(
    output_dir: Path,
    pipeline_name: str,
    contexts: list[str],
    summary_df: pd.DataFrame,
    group_stats: dict,
    timecourse_store: dict,
):
    '''
    Save the summary DataFrame, group statistics, and time-resolved decoding results to disk in a structured format.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"decoding_feedback_outcome_timeresolved_{pipeline_name}_{'_vs_'.join(contexts)}"
    summary_path = output_dir / f"{base_name}_summary.csv"
    stats_path = output_dir / f"{base_name}_group_stats.json"
    npz_path = output_dir / f"{base_name}_timecourses.npz"

    summary_df.to_csv(summary_path, index=False)
    stats_path.write_text(json.dumps(group_stats, indent=2))

    arrays = {}
    for subject_id, context_map in timecourse_store.items():
        for context, payload in context_map.items():
            arrays[f"{subject_id}__{context}__times"] = payload["times"]
            arrays[f"{subject_id}__{context}__mean_auc"] = payload["mean_scores"]
            arrays[f"{subject_id}__{context}__fold_auc"] = payload["scores"]
    np.savez(npz_path, **arrays)

    return summary_path, stats_path, npz_path


def run_group_decoding(
    subjects: list[str],
    pipeline_name: str,
    contexts: list[str],
    window_start: float = 0.24,
    window_end: float = 0.34,
    root_dir: Path | None = None,
    logger=None,
):
    '''
    Run the full group-level time-resolved decoding analysis, including loading epochs, performing decoding, summarizing results, and computing group statistics.
    '''
    summary_rows = []
    timecourse_store = {}

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
        timecourse_store[subject_id] = {}

        for context in contexts:
            try:
                result = decode_context(epochs, context)
            except (RuntimeError, ValueError) as exc:
                message = f"Skipping sub-{subject_id} {context}: {exc}"
                if logger is not None:
                    logger.warning(message)
                else:
                    warnings.warn(message)
                continue

            timecourse_store[subject_id][context] = result
            summary_rows.append(
                summarize_subject(subject_id, context, result, window_start, window_end)
            )

    if not summary_rows:
        raise RuntimeError("No valid subject-context decoding results were produced.")

    summary_df = pd.DataFrame(summary_rows)
    group_stats = compute_group_stats(summary_df, contexts)
    return summary_df, group_stats, timecourse_store
