import csv
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter
from stats.logging_utils import log, log_scores
from stats.rewp_parametric import rm_anova_oneway, rm_ttest



def _tokenize(k:str):
    # lowercase alphanumeric tokens, split "Mid-High Win" -> ["mid","high","win"]
    return re.findall(r"[a-z0-9]+", k.lower())

def _find_key(keys, includes, excludes=()):
    inc = Counter(s.lower() for s in includes) # required counts
    exc = set(s.lower() for s in excludes) # forbidden tokens

    for k in keys:
        toks = _tokenize(k)
        tok_count = Counter(toks)

        # check excludes
        if any(e in toks for e in exc):
            continue

        if all(tok_count[tok] >= count for tok, count in inc.items()):
            return k

    return None


def _get_key_map(keys):
    patterns = {
        'LL_win': (['low', 'low', 'win'], []),
        'LL_loss': (['low', 'low', 'loss'], []),
        'ML_win': (['mid', 'low', 'win'], []),
        'ML_loss': (['mid', 'low', 'loss'], []),
        'MH_win': (['mid', 'high', 'win'], []),
        'MH_loss': (['mid', 'high', 'loss'], []),
        'HH_win': (['high', 'high', 'win'], []),
        'HH_loss': (['high', 'high', 'loss'], []),
    }
    key_map = {name: _find_key(keys, inc, exc) for name, (inc, exc) in patterns.items()}
    missing = [name for name, key in key_map.items() if key is None]
    if missing:
        raise RuntimeError(f"Missing condition keys: {missing}. Check condition names.")

    # sanity checks
    if key_map["MH_win"] == key_map["HH_win"]:
        raise RuntimeError(f"MH_win and HH_win mapped to same key: {key_map['MH_win']}")
    if key_map["MH_loss"] == key_map["HH_loss"]:
        raise RuntimeError(f"MH_loss and HH_loss mapped to same key: {key_map['MH_loss']}")

    return key_map

def compute_rewp_scores(group_evokeds, ch_name='FCz', tmin=0.240, tmax=0.340, logger=None):
    """
    Build RewP mean-amplitude scores (Win-Loss) for LL/ML/MH/HH.

    :param group_evokeds: {subject_id: {condition_name: Evoked}}
    :return: scores (n_subjects, 4), subjects list, key_map
    """
    subjects = list(group_evokeds.keys())
    example_keys = list(next(iter(group_evokeds.values())).keys())
    key_map = _get_key_map(example_keys)

    def _mean_fc(evoked):
        e = evoked.copy().pick(ch_name).crop(tmin, tmax)
        return float(e.data.mean())

    scores = []
    for sid in subjects:
        ev = group_evokeds[sid]
        ll = _mean_fc(ev[key_map['LL_win']]) - _mean_fc(ev[key_map['LL_loss']])
        ml = _mean_fc(ev[key_map['ML_win']]) - _mean_fc(ev[key_map['ML_loss']])
        mh = _mean_fc(ev[key_map['MH_win']]) - _mean_fc(ev[key_map['MH_loss']])
        hh = _mean_fc(ev[key_map['HH_win']]) - _mean_fc(ev[key_map['HH_loss']])
        scores.append([ll, ml, mh, hh])

    scores = np.asarray(scores)
    log_scores(scores, subjects, logger=logger)
    return scores, subjects, key_map


def save_rewp_scores(scores, subjects, key_map, out_path,
                     ch_name='FCz', tmin=0.240, tmax=0.340, logger=None):
    """
    Save RewP scores to CSV with JSON metadata.

    :param scores: (n_subjects, 4) array
    :param subjects: list of subject ids
    :param key_map: dict of condition name mapping
    :param out_path: file path (.csv)
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != '.csv':
        out_path = out_path.with_suffix('.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'score_labels': ['LL', 'ML', 'MH', 'HH'],
        'ch_name': ch_name,
        'tmin': float(tmin),
        'tmax': float(tmax),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")

    scores = np.asarray(scores, float)
    subjects = np.asarray(subjects, int)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "LL", "ML", "MH", "HH"])
        for sid, row in zip(subjects, scores):
            writer.writerow([int(sid)] + [float(x) for x in row])

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"key_map": key_map, "meta": meta}, f, indent=2, ensure_ascii=False)

    log(logger, f"Saved RewP scores -> {out_path}")
    log(logger, f"Saved metadata -> {meta_path}")
    return out_path


def load_rewp_scores(path, logger=None):
    """
    Load RewP scores from csv saved by save_rewp_scores.
    """
    path = Path(path)
    if path.suffix.lower() != '.csv':
        path = path.with_suffix('.csv')
    meta_path = path.with_name(path.stem + "_meta.json")

    subjects = []
    scores = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.append(int(row["subject"]))
            scores.append([float(row["LL"]), float(row["ML"]), float(row["MH"]), float(row["HH"])])

    key_map = None
    meta = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        key_map = payload.get("key_map")
        meta = payload.get("meta")

    return np.asarray(scores, float), subjects, key_map, meta


def run_rewp_parametric_from_scores(scores, subjects=None, key_map=None, logger=None):
    """
    Run rmANOVA + rmTTest from precomputed scores.

    :param scores: (n_subjects, 4) array
    """
    scores = np.asarray(scores, float)
    log(logger, 'RewP diff score shape:', scores.shape)
    if key_map is not None:
        log(logger, 'Condition mapping:', key_map)

    log(logger, '\n=== rmANOVA on RewP diffs (LL, ML, MH, HH) ===')
    anova_res = rm_anova_oneway(scores, logger=logger)

    log(logger, '\n=== rmTTest ===')
    log(logger, 'Mid-Low vs Low-Low')
    t1 = rm_ttest(scores[:, 1], scores[:, 0], logger=logger)
    log(logger, 'Mid-High vs High-High')
    t2 = rm_ttest(scores[:, 2], scores[:, 3], logger=logger)

    return {
        'anova': anova_res,
        'ttest_ml_ll': t1,
        'ttest_mh_hh': t2,
        'subjects': subjects,
    }


def save_parametric_results(param_res, out_path, logger=None):
    """
    Save parametric results to CSV + raw JSON.

    :param param_res: dict returned by run_rewp_parametric_from_scores
    :param out_path: csv path
    """
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
    rows.append({
        "test": "rmANOVA",
        "comparison": "LL/ML/MH/HH",
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

    t_ml_ll = param_res.get("ttest_ml_ll", {})
    rows.append({
        "test": "rmTTest",
        "comparison": "ML-LL",
        "stat": t_ml_ll.get("t"),
        "df1": t_ml_ll.get("df"),
        "p": t_ml_ll.get("p"),
        "cohen_d": t_ml_ll.get("cohen_d"),
        "n": t_ml_ll.get("n"),
        "normality_p": t_ml_ll.get("normality_p"),
    })

    t_mh_hh = param_res.get("ttest_mh_hh", {})
    rows.append({
        "test": "rmTTest",
        "comparison": "MH-HH",
        "stat": t_mh_hh.get("t"),
        "df1": t_mh_hh.get("df"),
        "p": t_mh_hh.get("p"),
        "cohen_d": t_mh_hh.get("cohen_d"),
        "n": t_mh_hh.get("n"),
        "normality_p": t_mh_hh.get("normality_p"),
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