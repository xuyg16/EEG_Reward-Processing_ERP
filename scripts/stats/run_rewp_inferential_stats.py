import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from stats.inference_parametric import paired_ttest
from stats.inference_permutation_test import paired_permutation_test


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


def _format_p_value(p):
    '''
    Format p-value for display.
    '''
    if not np.isfinite(p):
        return "NaN"
    if p < 0.001:
        return "< .001"
    s = f"{p:.3f}"
    if s.startswith("0"):
        s = s[1:]
    return f"= {s}"


def summarize_rewp_comparison(
    scores,
    idx_a,
    idx_b,
    label_a,
    label_b,
    comparison_name,
    logger=None,
):
    '''
    Summarize comparison between two conditions (e.g. MH vs HH) with mean, CI, t-test, and permutation test.
    '''
    scores = np.asarray(scores, float)
    x1 = scores[:, idx_a]
    x2 = scores[:, idx_b]

    valid_pair_mask = np.isfinite(x1) & np.isfinite(x2)
    x1 = x1[valid_pair_mask]
    x2 = x2[valid_pair_mask]

    m1, (lo1, hi1), n1 = mean_ci_t(x1)
    m2, (lo2, hi2), n2 = mean_ci_t(x2)

    # suppress low-level logging here; only keep final summary text
    t_res = paired_ttest(x1, x2, check_normality=True, logger=None)
    perm_res = paired_permutation_test(x1, x2, logger=None)

    lines = [
        f"[{comparison_name}]",
        f"{label_a}: {m1:.2f} μV, 95% CI [{lo1:.2f}, {hi1:.2f}]",
        f"{label_b}: {m2:.2f} μV, 95% CI [{lo2:.2f}, {hi2:.2f}]",
        f"t({t_res['df']}) = {t_res['t']:.2f}, p {_format_p_value(t_res['p'])}, Cohen's d = {t_res['cohen_dz']:.2f}",
        f"Exact paired permutation: p {_format_p_value(perm_res['p'])}",
    ]

    return {
        "a": {"mean": float(m1), "ci": (float(lo1), float(hi1)), "n": int(n1)},
        "b": {"mean": float(m2), "ci": (float(lo2), float(hi2)), "n": int(n2)},
        "ttest": t_res,
        "permutation": perm_res,
        "text": "\n".join(lines),
    }

def plot_rewp_performance_correlation_old(
    scores,
    subjects,
    behavior_df,
    learner_cutoff=0.60,
    figsize=(6, 5),
    title="ΔRewP vs performance",
):
    """

    x-axis:
        mean performance across learnable conditions
        = mean(mid_high_acc, high_high_acc)

    y-axis:
        delta RewP = MH - HH

    markers:
        learner = open circle
        non-learner = star

    regression:
        fit across all subjects
    """
    scores = np.asarray(scores, float)

    # RewP part
    rewp_df = pd.DataFrame({
        "subject": [str(s) for s in subjects],
        "MH": scores[:, 2],
        "HH": scores[:, 3],
    })
    rewp_df["delta_rewp"] = rewp_df["MH"] - rewp_df["HH"]

    # Behavior part
    beh = behavior_df.copy()
    beh["subject"] = beh["subject"].astype(str)

    # If behavior_df stores subjects like '01' and scores subjects are ints/strings,
    # normalize lightly here
    def _norm_sub(x):
        x = str(x).strip()
        if x.startswith("sub-"):
            x = x[4:]
        return x

    rewp_df["subject"] = rewp_df["subject"].map(_norm_sub)
    beh["subject"] = beh["subject"].map(_norm_sub)

    beh["performance"] = beh[["mid_high_acc", "high_high_acc"]].mean(axis=1)

    beh["learner"] = (
        (beh["mid_high_acc"] >= learner_cutoff) &
        (beh["high_high_acc"] >= learner_cutoff)
    )

    # Merge
    df_plot = rewp_df.merge(
        beh[["subject", "performance", "learner"]],
        on="subject",
        how="inner",
    )

    df_plot = df_plot[
        np.isfinite(df_plot["performance"]) &
        np.isfinite(df_plot["delta_rewp"])
    ].copy()

    if len(df_plot) < 3:
        raise ValueError("Not enough valid subjects to plot the correlation.")

    # Correlation and regression across ALL participants
    r, p = stats.pearsonr(df_plot["performance"], df_plot["delta_rewp"])
    slope, intercept, *_ = stats.linregress(df_plot["performance"], df_plot["delta_rewp"])

    learners = df_plot["learner"]
    nonlearners = ~learners

    fig, ax = plt.subplots(figsize=figsize)

    # non-learners: star
    ax.scatter(
        df_plot.loc[nonlearners, "performance"],
        df_plot.loc[nonlearners, "delta_rewp"],
        marker="*",
        s=180,
        color="black",
        label="Non-Learners",
    )

    # learners: open circle
    ax.scatter(
        df_plot.loc[learners, "performance"],
        df_plot.loc[learners, "delta_rewp"],
        marker="o",
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Learners",
    )

    # regression line across all subjects
    xline = np.linspace(df_plot["performance"].min(), df_plot["performance"].max(), 200)
    yline = intercept + slope * xline
    ax.plot(xline, yline, color="black", linewidth=1.5)

    ax.set_xlabel("Performance (% correct)")
    ax.set_ylabel(r"$\Delta$ RewP ($\mu V$)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")

    # optional: make axes look closer to the paper
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-10, 10)

    # stats text
    p_text = "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")
    ax.text(
        0.98, 0.05,
        f"r = {r:.2f}, p {p_text}, n = {len(df_plot)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    plt.tight_layout()
    plt.show()

    return df_plot, {"r": float(r), "p": float(p), "n": int(len(df_plot))}


def plot_rewp_performance_correlation(
    scores,
    subjects,
    behavior_df,
    learner_cutoff=0.60,
    figsize=(6.5, 5.5),
    title="ΔRewP vs performance",
):
    '''
    Updated version with nicer aesthetics and more robust handling of edge cases.
    '''
    scores = np.asarray(scores, float)

    rewp_df = pd.DataFrame({
        "subject": [str(s) for s in subjects],
        "MH": scores[:, 2],
        "HH": scores[:, 3],
    })
    rewp_df["delta_rewp"] = rewp_df["MH"] - rewp_df["HH"]

    beh = behavior_df.copy()
    beh["subject"] = beh["subject"].astype(str)

    def _norm_sub(x):
        x = str(x).strip()
        if x.startswith("sub-"):
            x = x[4:]
        return x

    rewp_df["subject"] = rewp_df["subject"].map(_norm_sub)
    beh["subject"] = beh["subject"].map(_norm_sub)

    beh["performance"] = beh[["mid_high_acc", "high_high_acc"]].mean(axis=1)
    beh["learner"] = (
        (beh["mid_high_acc"] >= learner_cutoff) &
        (beh["high_high_acc"] >= learner_cutoff)
    )

    df_plot = rewp_df.merge(
        beh[["subject", "performance", "learner"]],
        on="subject",
        how="inner",
    )

    df_plot = df_plot[
        np.isfinite(df_plot["performance"]) &
        np.isfinite(df_plot["delta_rewp"])
    ].copy()

    if len(df_plot) < 3:
        raise ValueError("Not enough valid subjects to plot the correlation.")

    r, p = stats.pearsonr(df_plot["performance"], df_plot["delta_rewp"])
    slope, intercept, *_ = stats.linregress(df_plot["performance"], df_plot["delta_rewp"])

    learners = df_plot["learner"]
    nonlearners = ~learners

    fig, ax = plt.subplots(figsize=figsize)

    # learners
    ax.scatter(
        df_plot.loc[learners, "performance"] * 100,
        df_plot.loc[learners, "delta_rewp"],
        marker="o",
        s=90,
        facecolors="white",
        edgecolors="#4C78A8",
        linewidths=1.8,
        label="Learners",
        zorder=3,
    )

    # non-learners
    ax.scatter(
        df_plot.loc[nonlearners, "performance"] * 100,
        df_plot.loc[nonlearners, "delta_rewp"],
        marker="*",
        s=180,
        color="#F58518",
        label="Non-learners",
        zorder=4,
    )

    # regression line
    xline = np.linspace(df_plot["performance"].min(), df_plot["performance"].max(), 200)
    yline = intercept + slope * xline
    ax.plot(xline * 100, yline, color="#333333", linewidth=2)

    ax.set_xlabel("Performance (% correct)")
    ax.set_ylabel("Δ RewP (μV)")
    ax.set_title(title)

    ax.grid(True, linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    p_text = "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")
    ax.text(
        0.98, 0.05,
        f"r = {r:.2f}, p {p_text}, n = {len(df_plot)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

    return df_plot, {"r": float(r), "p": float(p), "n": int(len(df_plot))}