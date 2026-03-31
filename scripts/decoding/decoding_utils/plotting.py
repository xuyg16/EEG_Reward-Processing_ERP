
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_window_decoding_summary(
    summary_df: pd.DataFrame,
    contexts: list[str],
    window_start: float,
    window_end: float,
    *,
    figsize: tuple[float, float] = (10, 4.8),
    random_seed: int = 42,
):
    pivot = summary_df.pivot(index="subject_id", columns="context", values="mean_auc").dropna()
    if pivot.empty:
        raise RuntimeError("No paired window-decoding results are available to plot.")

    xpos = np.arange(len(contexts), dtype=float)
    rng = np.random.default_rng(random_seed)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    ax_subjects, ax_group = axes
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    for _, row in pivot.iterrows():
        y = row[contexts].to_numpy(dtype=float)
        ax_subjects.plot(xpos, y, color="0.80", linewidth=1.0, alpha=0.9, zorder=1)

    for idx, context in enumerate(contexts):
        y = pivot[context].to_numpy(dtype=float)
        jitter = rng.normal(0.0, 0.03, size=len(y))
        ax_subjects.scatter(
            np.full(len(y), xpos[idx]) + jitter,
            y,
            s=38,
            color=colors[idx % len(colors)],
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )

    means = pivot[contexts].mean(axis=0)
    sems = pivot[contexts].sem(axis=0).fillna(0.0)
    ax_subjects.errorbar(
        xpos,
        means.to_numpy(),
        yerr=sems.to_numpy(),
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=2.0,
        capsize=4,
        markersize=8,
        zorder=3,
    )

    ax_subjects.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax_subjects.set_xticks(xpos)
    ax_subjects.set_xticklabels([ctx.replace("_", "\n") for ctx in contexts])
    ax_subjects.set_ylabel("AUC")
    ax_subjects.set_title(f"Per-subject decoding ({len(pivot)} paired subjects)")
    ax_subjects.grid(axis="y", alpha=0.2)

    bar_heights = means.to_numpy()
    bar_errors = sems.to_numpy()
    bars = ax_group.bar(
        xpos,
        bar_heights,
        yerr=bar_errors,
        capsize=4,
        width=0.62,
        color=[colors[idx % len(colors)] for idx in range(len(contexts))],
        edgecolor="none",
        alpha=0.9,
    )
    ax_group.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax_group.set_xticks(xpos)
    ax_group.set_xticklabels([ctx.replace("_", "\n") for ctx in contexts])
    ax_group.set_ylabel("Mean AUC")
    ax_group.set_title(f"Group mean ({window_start:.2f}-{window_end:.2f} s)")
    ax_group.grid(axis="y", alpha=0.2)

    for idx, bar in enumerate(bars):
        ax_group.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + bar_errors[idx] + 0.01,
            f"{bar_heights[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    y_min = min(0.45, float(pivot.min().min()) - 0.05)
    y_max = max(0.8, float(pivot.max().max()) + 0.05)
    ax_subjects.set_ylim(y_min, y_max)
    ax_group.set_ylim(y_min, y_max)
    fig.suptitle("Feedback outcome decoding", y=1.02, fontsize=13)
    plt.tight_layout()
    return fig, axes, pivot


def plot_time_resolved_decoding_summary(
    timecourse_store: dict,
    subjects_to_run: list[str],
    contexts: list[str],
    window_start: float,
    window_end: float,
    *,
    figsize: tuple[float, float] = (8, 5),
):
    fig, ax = plt.subplots(figsize=figsize)
    plotted_contexts: list[str] = []

    for context in contexts:
        context_curves = []
        times = None
        for subject_id in subjects_to_run:
            payload = timecourse_store.get(subject_id, {}).get(context)
            if payload is None:
                continue
            times = payload["times"]
            context_curves.append(payload["mean_scores"])

        if not context_curves:
            continue

        context_curves = np.vstack(context_curves)
        mean_curve = context_curves.mean(axis=0)
        if context_curves.shape[0] > 1:
            sem_curve = context_curves.std(axis=0, ddof=1) / np.sqrt(context_curves.shape[0])
        else:
            sem_curve = np.zeros_like(mean_curve)

        ax.plot(times, mean_curve, label=context, linewidth=2)
        ax.fill_between(times, mean_curve - sem_curve, mean_curve + sem_curve, alpha=0.2)
        plotted_contexts.append(context)

    if not plotted_contexts:
        raise RuntimeError("No time-resolved decoding results are available to plot.")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.axvspan(window_start, window_end, color="grey", alpha=0.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("AUC")
    ax.set_title("Time-resolved outcome decoding")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    return fig, ax, plotted_contexts
