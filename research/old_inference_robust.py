import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from utils.logger import log


def paired_permutation_test(x1, x2, n_perm=10000, stat="t", seed=0, alternative="two-sided", logger=None):
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    diff = x1 - x2
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n < 3:
        return {"n": int(n), "stat_obs": np.nan, "p_perm": np.nan, "perm_stats": None}

    rng = np.random.default_rng(seed)

    if stat == "t":
        stat_obs = float(stats.ttest_1samp(diff, 0.0).statistic)
    elif stat == "mean":
        stat_obs = float(np.mean(diff))
    else:
        raise ValueError("stat must be 't' or 'mean'")

    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    diff_perm = signs * diff

    if stat == "t":
        mu = diff_perm.mean(axis=1)
        sd = diff_perm.std(axis=1, ddof=1)
        den = sd / np.sqrt(n)
        perm_stats = np.where(den > 0, mu / den, 0.0)
    else:
        perm_stats = diff_perm.mean(axis=1)

    if alternative == "two-sided":
        p_perm = (np.sum(np.abs(perm_stats) >= np.abs(stat_obs)) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p_perm = (np.sum(perm_stats >= stat_obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p_perm = (np.sum(perm_stats <= stat_obs) + 1) / (n_perm + 1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return {
        "n": int(n),
        "stat_obs": float(stat_obs),
        "p_perm": float(p_perm),
        "perm_stats": perm_stats,
        "stat": stat,
        "alternative": alternative,
        "n_perm": int(n_perm),
    }


def paired_bootstrap_ci(x1, x2, n_boot=10000, seed=0, ci=0.95, logger=None):
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    diff = x1 - x2
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n < 3:
        return {"n": int(n), "mean_diff": np.nan, "d": np.nan,
                "ci_mean": (np.nan, np.nan), "ci_d": (np.nan, np.nan)}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx]
    boot_mean = boot.mean(axis=1)

    boot_sd = boot.std(axis=1, ddof=1)
    boot_d = np.where(boot_sd > 0, boot_mean / boot_sd, np.nan)

    alpha = (1 - ci) / 2
    lo, hi = alpha, 1 - alpha

    ci_mean = (float(np.nanquantile(boot_mean, lo)), float(np.nanquantile(boot_mean, hi)))
    ci_d = (float(np.nanquantile(boot_d, lo)), float(np.nanquantile(boot_d, hi)))

    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    d = float(mean_diff / sd) if sd > 0 else np.nan

    return {
        "n": int(n),
        "mean_diff": mean_diff,
        "d": d,
        "ci_mean": ci_mean,
        "ci_d": ci_d,
        "n_boot": int(n_boot),
        "ci": float(ci),
    }


def run_score_robustness(scores, comparisons, n_perm=10000, n_boot=10000, seed=0, logger=None):
    scores = np.asarray(scores, float)
    results = {}

    log(logger, "\n=== Permutation tests ===")
    for offset, (label, idx_a, idx_b) in enumerate(comparisons):
        log(logger, label)
        res = paired_permutation_test(scores[:, idx_a], scores[:, idx_b], n_perm=n_perm, seed=seed + offset)
        log(logger, f"perm p = {res['p_perm']:.4g}, stat = {res['stat_obs']:.4g}, n = {res['n']}")
        results[f"perm_{label}"] = res

    log(logger, "\n=== Bootstrap CIs ===")
    for offset, (label, idx_a, idx_b) in enumerate(comparisons):
        log(logger, label)
        res = paired_bootstrap_ci(scores[:, idx_a], scores[:, idx_b], n_boot=n_boot, seed=seed + offset)
        log(logger, f"mean diff = {res['mean_diff']:.4g}, CI = {res['ci_mean']}")
        log(logger, f"Cohen's d = {res['d']:.4g}, CI = {res['ci_d']}")
        results[f"boot_{label}"] = res

    return results


def save_robustness_results(robust_res, out_path, logger=None):
    out_path = Path(out_path)
    if out_path.suffix.lower() != '.csv':
        out_path = out_path.with_suffix('.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.json")

    header = [
        "method", "comparison", "stat_type", "stat_obs", "p_perm", "n_perm", "n",
        "mean_diff", "d", "ci_mean_low", "ci_mean_high", "ci_d_low", "ci_d_high", "ci", "n_boot",
    ]
    rows = []

    for key, res in robust_res.items():
        if key.startswith("perm_"):
            rows.append({
                "method": "permutation",
                "comparison": key.removeprefix("perm_"),
                "stat_type": res.get("stat"),
                "stat_obs": res.get("stat_obs"),
                "p_perm": res.get("p_perm"),
                "n_perm": res.get("n_perm"),
                "n": res.get("n"),
            })
        elif key.startswith("boot_"):
            ci_mean = res.get("ci_mean", (None, None))
            ci_d = res.get("ci_d", (None, None))
            rows.append({
                "method": "bootstrap",
                "comparison": key.removeprefix("boot_"),
                "mean_diff": res.get("mean_diff"),
                "d": res.get("d"),
                "ci_mean_low": ci_mean[0],
                "ci_mean_high": ci_mean[1],
                "ci_d_low": ci_d[0],
                "ci_d_high": ci_d[1],
                "ci": res.get("ci"),
                "n_boot": res.get("n_boot"),
                "n": res.get("n"),
            })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})

    def _to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items() if k != "perm_stats"}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        return obj

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(robust_res), f, indent=2, ensure_ascii=False)

    log(logger, f"Saved robustness results -> {out_path}")
    log(logger, f"Saved raw robustness JSON -> {raw_path}")
    return out_path
