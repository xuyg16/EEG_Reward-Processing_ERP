import numpy as np
import logging
from pathlib import Path

_LOGGER = {}

def _close_logger_handlers(logger: logging.Logger):
    for handler in logger.handlers[:]:
        handler.close()
    logger.handlers.clear()


def log(logger, msg, *args, level="info"):
    """Print if no logger; otherwise send the message to the chosen log level."""
    if logger is None:
        if args:
            msg = msg % args
        print(msg)
        return
    if isinstance(level, str):
        level_value = logging.getLevelName(level.upper())
        if not isinstance(level_value, int):
            raise ValueError(f"Unknown log level: {level}")
    else:
        level_value = int(level)
    logger.log(level_value, msg, *args)


def setup_logger(name: str, out_dir: Path, console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.log"

    if name in _LOGGERS:
        logger = _LOGGERS[name]
        existing_log_path = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                existing_log_path = Path(handler.baseFilename).resolve()
                handler.setLevel(file_level)
            elif isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)
        if existing_log_path == log_path.resolve():
            return logger, log_path
        _close_logger_handlers(logger)
        _LOGGERS.pop(name, None)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console_fmt = logging.Formatter("%(message)s")
    file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(console_fmt)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)

    _close_logger_handlers(logger)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    logger.info("Logger initialized -> %s", log_path)
    return logger, log_path


def log_scores(scores, subjects, logger=None, head=8):
    try:
        import pandas as pd
    except Exception:
        pd = None

    scores = np.asarray(scores, float)
    subjects = list(subjects)

    if pd is not None:
        df = pd.DataFrame(scores, columns=["LL", "ML", "MH", "HH"])
        df.insert(0, "subject", subjects)
        summ = df[["LL", "ML", "MH", "HH"]].agg(["count", "mean", "std"])
        log(logger, "RewP scores shape: %s\nSummary:\n%s", scores.shape, summ.to_string())
        log(logger, "Scores preview (head=%s):\n%s", head, df.head(head).to_string(index=False))
        return df

    means = np.nanmean(scores, axis=0)
    stds = np.nanstd(scores, axis=0)
    log(
        logger,
        "RewP scores shape: %s\nMean: LL=%.4g, ML=%.4g, MH=%.4g, HH=%.4g\nStd:  LL=%.4g, ML=%.4g, MH=%.4g, HH=%.4g",
        scores.shape,
        means[0], means[1], means[2], means[3],
        stds[0], stds[1], stds[2], stds[3],
    )
    return None


def setup_rewp_logger(group_label, out_dir=None, repo_root=None, name_prefix="rewp_stats",
                      console_level=logging.INFO, file_level=logging.DEBUG,
                      print_summary=True):
    """
    Convenience helper to create a logger + output dir, with clean notebook prints.

    Returns: (logger, out_dir, log_path)
    """
    if repo_root is None:
        repo_root = Path.cwd().resolve()
        if repo_root.name == "scripts":
            repo_root = repo_root.parent
    if out_dir is None:
        out_dir = repo_root / "output_mne"
    out_dir = Path(out_dir)
    log_dir = out_dir / "logs"
    name = f"{name_prefix}_{group_label}"
    log_path = log_dir / f"{name}.log"
    logger = get_logger(log_dir, name=name, console_level=console_level, file_level=file_level)
    log(logger, f"Output dir: {out_dir}")
    log(logger, f"Check log file: {log_path}")
    return logger, out_dir, log_path


def log_ica_exclusion(logger, subject_id, exclude_idx, total_components):
    """
    log the results of ICA component exclusion for a subject.
    
    :param logger: preset up logger
    :param subject_id: id of the subject being processed
    :param exclude_idx: indices of IC components rejected
    :param total_components: total number of IC components generated
    """
    msg = (
        f"Subject {subject_id}: Excluded {len(exclude_idx)}/{total_components} ICs -> {exclude_idx}"
    )
    log(logger, msg)


def log_bad_channels(logger, subject_id, bad_channels):
    """
    Log the list of bad channels for a given subject.

    :param logger: preset up logger
    :param subject_id: id of the subject being processed
    :param bad_channels: list of bad channel names
    """
    msg = (
        f"Subject {subject_id}: Bad channels detected -> {bad_channels}"
    )
    log(logger, msg)