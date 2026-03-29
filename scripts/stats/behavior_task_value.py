from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.stats.inference_parametric import rm_anova_oneway, rm_ttest
from scripts.utils.logger import log


def _normalize_subject_id(subject) -> str:
    subject = str(subject).strip()
    if subject.startswith("sub-"):
        subject = subject[4:]
    if subject.isdigit():
        subject = f"{int(subject):02d}"
    return subject


def outcome_to_win01(series: pd.Series) -> np.ndarray:
    """
    Convert outcome to:
    1 -> win
    0 -> loss
    -1 / anything else -> NaN
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    out[x == 1] = 1.0
    out[x == 0] = 0.0
    return out


def masked_mean(values: np.ndarray, keep_rows) -> float:
    keep_rows = np.asarray(keep_rows, dtype=bool)
    selected = values[keep_rows]
    return float(np.nanmean(selected)) if selected.size else np.nan


def mean_ci_t(x, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, (np.nan, np.nan), 0
    m = float(np.mean(x))
    if n < 2:
        return m, (np.nan, np.nan), n
    sd = float(np.std(x, ddof=1))
    tval = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci = tval * sd / np.sqrt(n)
    return m, (m - ci, m + ci), n


def compute_subject_behavior_summary(beh_path: str | Path):
    beh_path = Path(beh_path)
    df = pd.read_csv(beh_path, sep="\t")

    required = ["task", "prob", "optimal", "early", "invalid", "outcome"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{beh_path} missing columns: {missing}")

    valid_trials = (df["early"] == 0) & (df["invalid"] == 0)
    high_value_cues = df["prob"] == 80

    outcome = outcome_to_win01(df["outcome"])
    optimal = df["optimal"].to_numpy(dtype=float)

    low_task_trials = (df["task"] == 1) & valid_trials
    mid_task_trials = (df["task"] == 2) & valid_trials
    high_task_trials = (df["task"] == 3) & valid_trials

    mid_task_high_value_trials = mid_task_trials & high_value_cues
    high_task_high_value_trials = high_task_trials & high_value_cues

    low = masked_mean(outcome, low_task_trials)
    mid = masked_mean(outcome, mid_task_trials)
    high = masked_mean(outcome, high_task_trials)

    mid_high_acc = masked_mean(optimal, mid_task_high_value_trials)
    high_high_acc = masked_mean(optimal, high_task_high_value_trials)

    return {
        "low": low,
        "mid": mid,
        "high": high,
        "mid_high_acc": mid_high_acc,
        "high_high_acc": high_high_acc,
        "n_mid_high": int(np.sum(mid_task_high_value_trials)),
        "n_high_high": int(np.sum(high_task_high_value_trials)),
        "beh_path": str(beh_path),
    }


def collect_subject_behavior_summary(bids_root: str | Path, subjects, logger=None):
    bids_root = Path(bids_root)

    rows = []
    missing = []

    for s in subjects:
        s_str = _normalize_subject_id(s)
        beh_path = bids_root / f"sub-{s_str}" / "beh" / f"sub-{s_str}_task-casinos_beh.tsv"
        if not beh_path.exists():
            missing.append(s_str)
            continue

        row = compute_subject_behavior_summary(beh_path)
        row["subject"] = s_str
        rows.append(row)

    if missing:
        raise FileNotFoundError(f"Missing behavior files for subjects: {missing}")

    if not rows:
        raise FileNotFoundError(f"No behavior files found under {bids_root}")

    df = pd.DataFrame(rows)
    df["_subject_order"] = pd.to_numeric(df["subject"], errors="coerce")
    df = df.sort_values(["_subject_order", "subject"]).drop(columns="_subject_order").reset_index(drop=True)
    log(logger, "Collected behavior summary for %s subjects", len(df))
    return df


def run_behavior_stats(df, logger=None):
    x = df[["low", "mid", "high"]].to_numpy(float)
    mask = np.all(np.isfinite(x), axis=1)
    x_use = x[mask]

    if x_use.size == 0:
        raise ValueError("No complete subjects for task win-rate ANOVA.")

    anova_res = rm_anova_oneway(x_use, logger=logger)

    y = df[["mid_high_acc", "high_high_acc"]].to_numpy(float)
    perf_mask = np.all(np.isfinite(y), axis=1)
    df_perf = df.loc[perf_mask].reset_index(drop=True)

    if df_perf.empty:
        raise ValueError("No complete subjects for high-value cue accuracy t-test.")

    perf_res = rm_ttest(
        df_perf["mid_high_acc"].to_numpy(float),
        df_perf["high_high_acc"].to_numpy(float),
        normality_on="each",
        logger=logger,
    )

    return {
        "anova": anova_res,
        "performance_ttest": perf_res,
    }


def plot_task_winrates(df, title="Behavioral manipulation check", figsize=(6, 4)):
    cols = ["low", "mid", "high"]
    labels = ["Low", "Mid", "High"]
    x = np.arange(1, 4)

    values = df[cols].to_numpy(float)
    values = values[np.all(np.isfinite(values), axis=1)]

    fig, ax = plt.subplots(figsize=figsize)

    for row in values:
        ax.plot(x, row * 100, color="0.8", linewidth=1, alpha=0.7)

    means, ci_low, ci_high = [], [], []
    for i in range(3):
        m, (lo, hi), _ = mean_ci_t(values[:, i])
        means.append(m * 100)
        ci_low.append(lo * 100)
        ci_high.append(hi * 100)

    means = np.array(means)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    ax.plot(x, means, color="black", marker="o", linewidth=2)
    ax.errorbar(x, means, yerr=[means - ci_low, ci_high - means], fmt="none", color="black", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reward (% wins)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_high_value_accuracy(df, title="Performance on high-value cues", figsize=(5, 4)):
    cols = ["mid_high_acc", "high_high_acc"]
    labels = ["Mid", "High"]
    x = np.arange(1, 3)

    values = df[cols].to_numpy(float)
    values = values[np.all(np.isfinite(values), axis=1)]

    fig, ax = plt.subplots(figsize=figsize)

    for row in values:
        ax.plot(x, row * 100, color="0.8", linewidth=1, alpha=0.7)

    means, ci_low, ci_high = [], [], []
    for i in range(2):
        m, (lo, hi), _ = mean_ci_t(values[:, i])
        means.append(m * 100)
        ci_low.append(lo * 100)
        ci_high.append(hi * 100)

    means = np.array(means)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    ax.plot(x, means, color="black", marker="o", linewidth=2)
    ax.errorbar(x, means, yerr=[means - ci_low, ci_high - means], fmt="none", color="black", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Performance (% correct)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()