import time
from memory_profiler import memory_usage
import os
import sys


def profile_function(func, *args, **kwargs):
    start_time = time.perf_counter()
    mem_usage, result = memory_usage((func, args, kwargs), retval=True)
    end_time = time.perf_counter()
    return result, mem_usage, end_time - start_time


def get_current_env():
    # Conda environment name
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        return conda_env

    # venv or virtualenv path
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        return os.path.basename(venv_path)

    # fallback: sys.prefix check (less reliable for conda)
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_prefix:
        return os.path.basename(sys.prefix)

    return None  # likely system Python
