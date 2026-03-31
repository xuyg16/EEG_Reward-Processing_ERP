import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from utils.logger import log


def paired_permutation_test(x1, x2, logger=None):
    """
    Exact paired sign-flip permutation test (two-sided),
    using the mean paired difference as the test statistic.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # keep only subjects with valid paired observations in both conditions
    valid_pair_mask = np.isfinite(x1) & np.isfinite(x2)
    x1 = x1[valid_pair_mask]
    x2 = x2[valid_pair_mask]

    diff = x1 - x2
    n = diff.size

    if n < 2:
        return {
            "n": int(n),
            "stat": np.nan,
            "p": np.nan,
            "mean_diff": np.nan,
            "cohen_dz": np.nan,
            "method": "exact_signflip",
        }

    stat_obs = float(np.mean(diff))

    sign_matrix = np.array(list(product([-1.0, 1.0], repeat=n)), dtype=float)
    perm_stats = (sign_matrix * diff[None, :]).mean(axis=1)

    p_perm = float(np.mean(np.abs(perm_stats) >= np.abs(stat_obs)))

    sd_diff = np.std(diff, ddof=1)
    dz = np.mean(diff) / sd_diff if sd_diff > 0 else np.nan

    log(logger, f"Permutation (exact sign-flip): stat = {stat_obs:.4g}, p = {p_perm:.4g}, n = {n}")
    log(logger, f"Mean difference = {np.mean(diff):.4g}")
    log(logger, f"Cohen's dz = {dz:.4g}")

    return {
        "n": int(n),
        "stat": float(stat_obs),
        "p": float(p_perm),
        "mean_diff": float(np.mean(diff)),
        "cohen_dz": float(dz),
        "method": "exact_signflip",
    }


def plot_exact_permutation_null(
    x1,
    x2,
    comparison_name="MH vs HH",
    xlabel="Mean difference in RewP (μV)",
    title="Exact paired permutation null distribution",
    logger=None,
):
    """
    Plot the exact sign-flip null distribution for paired data.
    Test statistic = mean paired difference.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # keep only complete paired observations
    valid_pair_mask = np.isfinite(x1) & np.isfinite(x2)
    x1 = x1[valid_pair_mask]
    x2 = x2[valid_pair_mask]

    diff = x1 - x2
    n = diff.size

    if n < 2:
        raise ValueError("Need at least 2 paired observations.")

    stat_obs = float(np.mean(diff))

    sign_matrix = np.array(list(product([-1.0, 1.0], repeat=n)), dtype=float)
    perm_stats = (sign_matrix * diff[None, :]).mean(axis=1)

    p_perm = float(np.mean(np.abs(perm_stats) >= np.abs(stat_obs)))

    p_text = "< .001" if p_perm < 0.001 else f"= {p_perm:.3f}".replace("0.", ".")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(perm_stats, bins=30, edgecolor="black", alpha=0.8)

    ax.axvline(
        stat_obs,
        linewidth=2,
        linestyle="--",
        label=f"Observed = {stat_obs:.2f}",
    )
    ax.axvline(
        -stat_obs,
        linewidth=1.5,
        linestyle=":",
        alpha=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    ax.text(
        0.98, 0.95,
        f"{comparison_name}\n"
        f"n = {n}\n"
        f"p {p_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.show()

    log(
        logger,
        f"[Permutation plot | {comparison_name}] "
        f"observed mean diff = {stat_obs:.4g}, p = {p_perm:.4g}, n = {n}"
    )

    return {
        "n": int(n),
        "stat_obs": float(stat_obs),
        "p": float(p_perm),
        "perm_stats": perm_stats,
    }


def plot_rewp_performance_correlation(
    scores,
    subjects,
    behavior_df,
    learner_cutoff=0.60,
    figsize=(6.5, 5.5),
    title="ΔRewP vs performance",
):
    '''
    Plot the correlation between ΔRewP (MH - HH) and behavioral performance.
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats

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

    # learners: blue hollow circles
    ax.scatter(
        df_plot.loc[learners, "performance"] * 100,
        df_plot.loc[learners, "delta_rewp"],
        marker="o",
        s=95,
        facecolors="white",
        edgecolors="#4C78A8",
        linewidths=2,
        label="Learners",
        zorder=3,
    )

    # non-learners: orange stars
    ax.scatter(
        df_plot.loc[nonlearners, "performance"] * 100,
        df_plot.loc[nonlearners, "delta_rewp"],
        marker="*",
        s=220,
        color="#F58518",
        label="Non-learners",
        zorder=4,
    )

    # regression line
    xline = np.linspace(df_plot["performance"].min(), df_plot["performance"].max(), 200)
    yline = intercept + slope * xline
    ax.plot(xline * 100, yline, color="#444444", linewidth=2.2)

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