from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from stats.inference_parametric import rm_anova_oneway, paired_ttest
from utils.logger import log


def _normalize_subject_id(subject) -> str:
    '''
    Normalize subject ID to a consistent format (e.g., "01", "02", ..., "10", "11", etc.)
    '''
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
    '''
    Compute mean of values where keep_rows is True, ignoring NaNs.
    '''
    keep_rows = np.asarray(keep_rows, dtype=bool)
    selected = values[keep_rows]
    return float(np.nanmean(selected)) if selected.size else np.nan


def mean_ci_t(x, alpha=0.05):
    '''
    Compute mean and confidence interval using t-distribution.
    '''
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
    '''
    Compute summary of subject behavior from the casino task.
    '''
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
    '''
    Collect behavior summary for each subject from the casino task.
    '''
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
    '''
    Run stats on behavior summary:
    '''
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

    perf_res = paired_ttest(
        df_perf["mid_high_acc"].to_numpy(float),
        df_perf["high_high_acc"].to_numpy(float),
        logger=logger,
    )

    return {
        "anova": anova_res,
        "performance_ttest": perf_res,
    }


def summarize_task_winrates(df, decimals=2, include_anova=False, logger=None):
    """
    Summarize task win rates (low/mid/high) with 95% CIs.
    Optionally include one-way repeated-measures ANOVA.

    Returns
    -------
    dict
        {
            "low": {"mean": ..., "ci": (..., ...), "n": ...},
            "mid": {"mean": ..., "ci": (..., ...), "n": ...},
            "high": {"mean": ..., "ci": (..., ...), "n": ...},
            "anova": {...} or None,
            "text": str,
        }
    """
    cols = ["low", "mid", "high"]
    labels = ["low-value", "mid-value", "high-value"]

    x = df[cols].to_numpy(float)

    # use complete subjects across all three task conditions
    complete_subject_mask = np.all(np.isfinite(x), axis=1)
    x_use = x[complete_subject_mask]

    if x_use.size == 0:
        raise ValueError("No complete subjects for task win-rate summary.")

    n = x_use.shape[0]
    summary = {}

    lines = [f"The proportion of winning trials among different tasks (n={n}):"]

    for idx, (label, col) in enumerate(zip(labels, cols)):
        m, (lo, hi), n_col = mean_ci_t(x_use[:, idx])
        summary[col] = {
            "mean": float(m),
            "ci": (float(lo), float(hi)),
            "n": int(n_col),
        }
        lines.append(
            f"• {label}: {m*100:.{decimals}f}%, "
            f"95% CI [{lo*100:.{decimals}f}, {hi*100:.{decimals}f}]"
        )

    anova_res = None
    if include_anova:
        anova_res = rm_anova_oneway(x_use, logger=logger)
        p = anova_res["p"]
        p_str = "< .001" if np.isfinite(p) and p < 0.001 else f"= {p:.3f}"

        lines.append(
            f"F({anova_res['df1']},{anova_res['df2']}) = {anova_res['F']:.2f}, "
            f"p {p_str}, "
            f"ηp² = {anova_res['partial_eta2']:.2f}, "
            f"ηg² = {anova_res['generalized_eta2']:.2f}"
        )

    return {
        "low": summary["low"],
        "mid": summary["mid"],
        "high": summary["high"],
        "anova": anova_res,
        "text": "\n".join(lines),
    }

def summarize_mean_performance(df, decimals=2, logger=None):
    """
    Summarize mean performance in mid/high blocks with 95% CIs and paired t-test.
    """
    x_mid = df["mid_high_acc"].to_numpy(float)
    x_high = df["high_high_acc"].to_numpy(float)

    # use only complete paired subjects
    valid_pair_mask = np.isfinite(x_mid) & np.isfinite(x_high)
    x_mid = x_mid[valid_pair_mask]
    x_high = x_high[valid_pair_mask]

    if x_mid.size == 0:
        raise ValueError("No complete subjects for mean performance summary.")

    mid_m, (mid_lo, mid_hi), n_mid = mean_ci_t(x_mid)
    high_m, (high_lo, high_hi), n_high = mean_ci_t(x_high)

    t_res = paired_ttest(
        x_mid,
        x_high,
        check_normality=True,
        logger=logger,
    )

    p = t_res["p"]
    p_str = "< .001" if np.isfinite(p) and p < 0.001 else f"= {p:.3f}"

    lines = [
        f"mid-value block: {mid_m*100:.{decimals}f}%, 95% CI [{mid_lo*100:.{decimals}f}, {mid_hi*100:.{decimals}f}]",
        f"high-value block: {high_m*100:.{decimals}f}%, 95% CI [{high_lo*100:.{decimals}f}, {high_hi*100:.{decimals}f}]",
        f"t({t_res['df']}) = {t_res['t']:.2f}, p {p_str}, Cohen's d = {t_res['cohen_dz']:.2f}",
    ]

    return {
        "mid": {"mean": float(mid_m), "ci": (float(mid_lo), float(mid_hi)), "n": int(n_mid)},
        "high": {"mean": float(high_m), "ci": (float(high_lo), float(high_hi)), "n": int(n_high)},
        "ttest": t_res,
        "text": "\n".join(lines),
    }


def plot_task_winrates(df, title="Behavioral manipulation check", figsize=(6, 4)):
    '''
    Plot mean win rates for low/mid/high task conditions with 95% CIs.
    '''
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
    ax.set_xlabel("Task value")
    ax.set_ylabel("Reward (% wins)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_mean_performance(df, title="Performance on high-value cues", figsize=(5, 4)):
    '''
    Plot mean performance on high-value cues in mid/high blocks with CIs.
    '''
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
    ax.set_xlabel("Task value")
    ax.set_ylabel("Performance (% correct)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()