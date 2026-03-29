import numpy as np
from scipy import stats
from utils.logger import log


def swtest(x):
    """
    Shapiro-Wilk normality test.

    Parameters
    ----------
    x : array-like
        1D data vector.

    Returns
    -------
    float
        Shapiro-Wilk p value.
        Returns np.nan when the test is not applicable (n < 3).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 3:
        return np.nan

    _, p = stats.shapiro(x)
    return float(p)


def paired_ttest(x1, x2, check_normality=True, logger=None):
    """
    Paired-samples t-test.

    Parameters
    ----------
    x1, x2 : array-like
        Paired observations from two conditions.
    check_normality : bool
        Whether to run Shapiro-Wilk on paired differences.
    logger : logging.Logger | None

    Returns
    -------
    dict
        {
            "n": int,
            "t": float,
            "p": float,
            "df": int,
            "mean_diff": float,
            "cohen_dz": float,
            "normality_p": float
        }
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
            "t": np.nan,
            "p": np.nan,
            "df": np.nan,
            "mean_diff": np.nan,
            "cohen_dz": np.nan,
            "normality_p": np.nan,
        }

    normality_p = np.nan
    if check_normality:
        normality_p = swtest(diff)
        if np.isfinite(normality_p):
            log(logger, f"Shapiro-Wilk on paired differences: p = {normality_p:.4g}")
        else:
            log(logger, "Shapiro-Wilk on paired differences: not testable (n < 3)")

    t, p = stats.ttest_rel(x1, x2)
    sd_diff = np.std(diff, ddof=1)
    dz = np.mean(diff) / sd_diff if sd_diff > 0 else np.nan

    log(logger, f"Paired t-test: t({n - 1}) = {t:.4g}, p = {p:.4g}")
    log(logger, f"Mean difference = {np.mean(diff):.4g}")
    log(logger, f"Cohen's dz = {dz:.4g}")

    return {
        "n": int(n),
        "t": float(t),
        "p": float(p),
        "df": int(n - 1),
        "mean_diff": float(np.mean(diff)),
        "cohen_dz": float(dz),
        "normality_p": float(normality_p),
    }


def rm_anova_oneway(x, logger=None):
    """
    One-way repeated-measures ANOVA.

    Parameters
    ----------
    x : array-like, shape (n_subjects, n_conditions)
        Repeated-measures data matrix.

    Returns
    -------
    dict
        {
            "n": int,
            "k": int,
            "F": float,
            "p": float,
            "df1": int,
            "df2": int,
            "partial_eta2": float,
            "generalized_eta2": float,
        }
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_subjects, n_conditions)")

    # keep only complete subjects across all repeated-measures conditions
    complete_subject_mask = np.all(np.isfinite(x), axis=1)
    x = x[complete_subject_mask]

    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 subjects and 2 conditions for rmANOVA")

    grand_mean = np.mean(x)
    cond_means = np.mean(x, axis=0)
    subj_means = np.mean(x, axis=1)

    ss_total = np.sum((x - grand_mean) ** 2)
    ss_conditions = n * np.sum((cond_means - grand_mean) ** 2)
    ss_subjects = k * np.sum((subj_means - grand_mean) ** 2)
    ss_error = ss_total - ss_conditions - ss_subjects

    df1 = k - 1
    df_subjects = n - 1
    df2 = df1 * df_subjects

    ms_conditions = ss_conditions / df1
    ms_error = ss_error / df2
    F = ms_conditions / ms_error
    p = 1.0 - stats.f.cdf(F, df1, df2)

    partial_eta2 = (
        ss_conditions / (ss_conditions + ss_error)
        if (ss_conditions + ss_error) > 0 else np.nan
    )
    generalized_eta2 = (
        ss_conditions / (ss_conditions + ss_subjects + ss_error)
        if (ss_conditions + ss_subjects + ss_error) > 0 else np.nan
    )

    log(logger, f"RM ANOVA: F({df1},{df2}) = {F:.4g}, p = {p:.4g}")
    log(logger, f"partial eta^2 = {partial_eta2:.4g}")
    log(logger, f"generalized eta^2 = {generalized_eta2:.4g}")

    return {
        "n": int(n),
        "k": int(k),
        "F": float(F),
        "p": float(p),
        "df1": int(df1),
        "df2": int(df2),
        "partial_eta2": float(partial_eta2),
        "generalized_eta2": float(generalized_eta2),
    }