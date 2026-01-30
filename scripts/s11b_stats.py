import numpy as np
from scipy import stats


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


def rm_ttest(x1, x2, alpha=0.05, normality_on='diff'):
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
        print(f"Normality diff: {'violated' if h else 'met'} (p={pN:.4g})")
    else:
        h1, p1 = swtest(x1, alpha=alpha)
        h2, p2 = swtest(x2, alpha=alpha)
        print(f"Normality var1: {'violated' if h1 else 'met'} (p={p1:.4g})")
        print(f"Normality var2: {'violated' if h2 else 'met'} (p={p2:.4g})")

    t, p = stats.ttest_rel(x1, x2, nan_policy='omit')
    n = np.sum(np.isfinite(diff))
    df = n - 1
    sd = np.nanstd(diff, ddof=1)
    d = np.nanmean(diff) / sd if sd > 0 else np.nan
    print(f"t({df}) = {t:.4g}, p = {p:.4g}")
    print(f"Cohen's d = {d:.4g}")
    return {
        't': float(t),
        'p': float(p),
        'df': int(df),
        'n': int(n),
        'cohen_d': float(d),
        'normality_p': float(normality_p),
    }


def rm_anova_oneway(x):
    """
    One-way repeated-measures ANOVA + Friedman.

    :param x: (n_subjects, n_conditions)
    :return: dict with ANOVA and Friedman results
    """
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    for i in range(k):
        h, p = swtest(x[:, i])
        print(f"Normality var {i+1}: {'violated' if h else 'met'} (p={p:.4g})")

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

    print(f"RM ANOVA: F({df_conditions},{df_error}) = {f_val:.4g}, p = {p_val:.4g}")
    print(f"partial eta^2 = {etap:.4g}")
    print(f"generalized eta^2 = {etag:.4g}")

    valid_mask = np.all(np.isfinite(x), axis=1)
    x_clean = x[valid_mask]
    if x_clean.shape[0] >= 3 and k >= 2:
        chi2, p_friedman = stats.friedmanchisquare(*[x_clean[:, j] for j in range(k)])
    else:
        chi2, p_friedman = np.nan, np.nan
    print(f"Friedman: chi2 = {chi2:.4g}, p = {p_friedman:.4g}")

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

