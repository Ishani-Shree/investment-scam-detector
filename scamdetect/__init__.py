"""Multimodal pattern recognition for investment scam detection."""
import os


def _set_loky_cpu_count():
    """joblib/loky counts physical cores via `wmic`, which newer Windows 11 builds lack,
    and warns on every run. It skips that probe when LOKY_MAX_CPU_COUNT < logical cores."""
    if "LOKY_MAX_CPU_COUNT" in os.environ:
        return
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
    except Exception:
        cores = None
    logical = os.cpu_count() or 1
    os.environ["LOKY_MAX_CPU_COUNT"] = str(min(cores or max(logical // 2, 1), max(logical - 1, 1)))


_set_loky_cpu_count()
