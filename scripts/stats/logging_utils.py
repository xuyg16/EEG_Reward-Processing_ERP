import numpy as np
import logging
from pathlib import Path

_LOGGER = {}

def get_logger(out_dir, name="rewp", console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up a logger that logs to both console and a file.

    Args:
        out_dir (str or Path): Directory where the log file will be saved.
        name (str): Name of the logger.
        level (int): Logging level.
    """
    global _LOGGER
    if name in _LOGGER:
        logger = _LOGGER[name]
        # update handler levels if reusing the logger
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(console_level)
            elif isinstance(h, logging.FileHandler):
                h.setLevel(file_level)
        return logger
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter
    logger.propagate = False  # Prevent log messages from being propagated to the root logger

    if not logger.handlers:
        fmt_console = logging.Formatter('%(message)s')
        fmt_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console (notebbook/terminal)
        sh = logging.StreamHandler()
        sh.setLevel(console_level)
        sh.setFormatter(fmt_console)

        # File
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setLevel(file_level)  # keep full details in log file
        fh.setFormatter(fmt_file)

        logger.addHandler(sh)
        logger.addHandler(fh)
    
    logger.info("Logger initialized -> %s", log_path)
    _LOGGER[name] = logger
    return logger


def log(logger, msg, level="info"):
    """ Print if no logger; otherwise log."""
    if logger:
        logger.info(msg)
    else:
        print(msg)


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
        msg = (
            f"RewP scores shape: {scores.shape}\n"
            f"Summary:\n{summ.to_string()}"
        )
        log(logger, msg)
        log(logger, f"Scores preview (head={head}):\n{df.head(head).to_string(index=False)}")
        return df

    means = np.nanmean(scores, axis=0)
    stds = np.nanstd(scores, axis=0)
    msg = (
        f"RewP scores shape: {scores.shape}\n"
        f"Mean: LL={means[0]:.4g}, ML={means[1]:.4g}, MH={means[2]:.4g}, HH={means[3]:.4g}\n"
        f"Std:  LL={stds[0]:.4g}, ML={stds[1]:.4g}, MH={stds[2]:.4g}, HH={stds[3]:.4g}"
    )
    log(logger, msg)
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

