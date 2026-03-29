import numpy as np
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



