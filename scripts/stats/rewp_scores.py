import csv
import json
import numpy as np
from pathlib import Path
from pipeline.s10_rewp_calculation import rewp_calculation
from utils.logger import log, log_scores


KEY_MAP = {
    'LL': 'Low-Low',
    'ML': 'Mid-Low',
    'MH': 'Mid-High',
    'HH': 'High-High',
}


def compute_rewp_scores(group_evokeds, ch_name='FCz', tmin=0.240, tmax=0.340, logger=None):
    """
    Build RewP mean-amplitude scores (Win-Loss) for LL/ML/MH/HH.

    :param group_evokeds: {subject_id: {condition_name: Evoked}}
    :return: scores (n_subjects, 4), subjects list, key_map
    """
    if not group_evokeds:
        raise ValueError("group_evokeds is empty.")

    subjects = list(group_evokeds.keys())
    scores = []
    for sid in subjects:
        ev = group_evokeds[sid]
        score_map = rewp_calculation(
            ev,
            epoch_dict=None,
            verbose=False,
            channel=ch_name,
            mean_window=(tmin, tmax),
        )
        scores.append([
            score_map[KEY_MAP['LL']]['mean'],
            score_map[KEY_MAP['ML']]['mean'],
            score_map[KEY_MAP['MH']]['mean'],
            score_map[KEY_MAP['HH']]['mean'],
        ])

    scores = np.asarray(scores)
    log_scores(scores, subjects, logger=logger)
    return scores, subjects, KEY_MAP.copy()


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