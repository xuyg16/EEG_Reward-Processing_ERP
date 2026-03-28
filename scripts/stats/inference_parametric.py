import numpy as np
import csv
import json
from pathlib import Path
from scipy import stats
from utils.logger import log


def swtest(x, alpha=0.05):
    """
    Shapiro-Wilk normality test.

    :param x: 1D array
    :param alpha: significance level
    :return: (violated, p)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return False, np.nan
    _, p = stats.shapiro(x)
    return p < alpha, p


def rm_ttest(x1, x2, alpha=0.05, normality_on='diff', logger=None):
    """
    Paired t-test with normality checks.

    :param x1: 1D array
    :param x2: 1D array
    :param alpha: significance level
    :param normality_on: 'diff' or 'each'
    :return: dict with t/p/df/d and normality p
    """
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    diff = x1 - x2

    normality_p = np.nan
    if normality_on == 'diff':
        h, pN = swtest(diff, alpha=alpha)
        normality_p = pN
        log(logger, f"Normality diff: {'violated' if h else 'met'} (p={pN:.4g})")
    else:
        h1, p1 = swtest(x1, alpha=alpha)
        h2, p2 = swtest(x2, alpha=alpha)
        log(logger, f"Normality var1: {'violated' if h1 else 'met'} (p={p1:.4g})")
        log(logger, f"Normality var2: {'violated' if h2 else 'met'} (p={p2:.4g})")

    t, p = stats.ttest_rel(x1, x2, nan_policy='omit')
    n = np.sum(np.isfinite(diff))
    df = n - 1
    sd = np.nanstd(diff, ddof=1)
    d = np.nanmean(diff) / sd if sd > 0 else np.nan
    log(logger, f"t({df}) = {t:.4g}, p = {p:.4g}")
    log(logger, f"Cohen's d = {d:.4g}")
    return {
        't': float(t),
        'p': float(p),
        'df': int(df),
        'n': int(n),
        'cohen_d': float(d),
        'normality_p': float(normality_p),
    }


def rm_anova_oneway(x, logger=None):
    """
    One-way repeated-measures ANOVA + Friedman.

    :param x: (n_subjects, n_conditions)
    :return: dict with ANOVA and Friedman results
    """
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    for i in range(k):
        h, p = swtest(x[:, i])
        log(logger, f"Normality var {i+1}: {'violated' if h else 'met'} (p={p:.4g})")

    grand_mean = np.nanmean(x)
    cond_means = np.nanmean(x, axis=0)
    subj_means = np.nanmean(x, axis=1)

    ss_total = np.nansum((x - grand_mean) ** 2)
    ss_conditions = n * np.nansum((cond_means - grand_mean) ** 2)
    ss_subjects = k * np.nansum((subj_means - grand_mean) ** 2)
    ss_error = ss_total - ss_conditions - ss_subjects

    df_conditions = k - 1
    df_subjects = n - 1
    df_error = df_conditions * df_subjects

    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error

    f_val = ms_conditions / ms_error
    p_val = 1.0 - stats.f.cdf(f_val, df_conditions, df_error)

    etap = ss_conditions / (ss_conditions + ss_error) if (ss_conditions + ss_error) > 0 else np.nan
    etag = ss_conditions / (ss_subjects + ss_conditions + ss_error) if (ss_subjects + ss_conditions + ss_error) > 0 else np.nan

    log(logger, f"RM ANOVA: F({df_conditions},{df_error}) = {f_val:.4g}, p = {p_val:.4g}")
    log(logger, f"partial eta^2 = {etap:.4g}")
    log(logger, f"generalized eta^2 = {etag:.4g}")

    valid_mask = np.all(np.isfinite(x), axis=1)
    x_clean = x[valid_mask]
    if x_clean.shape[0] >= 3 and k >= 2:
        chi2, p_friedman = stats.friedmanchisquare(*[x_clean[:, j] for j in range(k)])
    else:
        chi2, p_friedman = np.nan, np.nan
    log(logger, f"Friedman: chi2 = {chi2:.4g}, p = {p_friedman:.4g}")

    return {
        'F': float(f_val),
        'p': float(p_val),
        'df1': int(df_conditions),
        'df2': int(df_error),
        'partial_eta2': float(etap),
        'generalized_eta2': float(etag),
        'friedman_chi2': float(chi2),
        'friedman_p': float(p_friedman),
        'friedman_n': int(x_clean.shape[0]),
    }

def run_score_parametric_tests(scores, comparisons, condition_labels=None, subjects=None, logger=None):
    scores = np.asarray(scores, float)
    log(logger, 'Score matrix shape: %s', scores.shape)
    if condition_labels is not None:
        log(logger, 'Condition labels: %s', condition_labels)

    log(logger, '\n=== rmANOVA ===')
    anova_res = rm_anova_oneway(scores, logger=logger)

    pairwise = {}
    log(logger, '\n=== Paired t-tests ===')
    for label, idx_a, idx_b in comparisons:
        log(logger, label)
        pairwise[label] = rm_ttest(scores[:, idx_a], scores[:, idx_b], logger=logger)

    return {
        'anova': anova_res,
        'pairwise': pairwise,
        'subjects': subjects,
        'condition_labels': condition_labels,
    }


def save_parametric_results(param_res, out_path, logger=None):
    out_path = Path(out_path)
    if out_path.suffix.lower() != '.csv':
        out_path = out_path.with_suffix('.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.json")

    header = [
        "test", "comparison", "stat", "df1", "df2", "p",
        "cohen_d", "n", "normality_p",
        "partial_eta2", "generalized_eta2",
        "friedman_chi2", "friedman_p", "friedman_n",
    ]

    rows = []
    anova = param_res.get("anova", {})
    labels = param_res.get("condition_labels")
    rows.append({
        "test": "rmANOVA",
        "comparison": "/".join(labels) if labels else "",
        "stat": anova.get("F"),
        "df1": anova.get("df1"),
        "df2": anova.get("df2"),
        "p": anova.get("p"),
        "partial_eta2": anova.get("partial_eta2"),
        "generalized_eta2": anova.get("generalized_eta2"),
        "friedman_chi2": anova.get("friedman_chi2"),
        "friedman_p": anova.get("friedman_p"),
        "friedman_n": anova.get("friedman_n"),
    })

    for label, res in param_res.get("pairwise", {}).items():
        rows.append({
            "test": "rmTTest",
            "comparison": label,
            "stat": res.get("t"),
            "df1": res.get("df"),
            "p": res.get("p"),
            "cohen_d": res.get("cohen_d"),
            "n": res.get("n"),
            "normality_p": res.get("normality_p"),
        })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(param_res, f, indent=2, ensure_ascii=False)

    log(logger, f"Saved parametric results -> {out_path}")
    log(logger, f"Saved raw parametric JSON -> {raw_path}")
    return out_path