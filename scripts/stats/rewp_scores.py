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
    for subject_id in subjects:
        ev = group_evokeds[subject_id]
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
    #log_scores(scores, subjects, logger=logger)
    return scores, subjects, KEY_MAP.copy()


def save_rewp_scores(scores, subjects, out_path, logger=None):
    """
    Save RewP scores to CSV.
    Columns: subject, LL, ML, MH, HH
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != '.csv':
        out_path = out_path.with_suffix('.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    scores = np.asarray(scores, float)
    #subjects = np.asarray(subjects, int)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "LL", "ML", "MH", "HH"])
        for subject_id, row in zip(subjects, scores):
            writer.writerow([(subject_id)] + [float(x) for x in row])

    log(logger, f"Saved RewP scores -> {out_path}")
    return out_path


def load_rewp_scores(path, logger=None):
    """
    Load RewP scores from csv saved by save_rewp_scores.
    """
    path = Path(path)
    if path.suffix.lower() != '.csv':
        path = path.with_suffix('.csv')

    subjects = []
    scores = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.append(int(row["subject"]))
            scores.append([float(row["LL"]), float(row["ML"]), float(row["MH"]), float(row["HH"])])

    scores = np.asarray(scores, dtype=float)
    log(logger, f"Loaded RewP scores <- {path}")
    return scores, subjects, KEY_MAP.copy()