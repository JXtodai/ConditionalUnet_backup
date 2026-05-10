"""Compatibility helpers for Matplotlib and porespy in this environment.

This environment has two import-time problems:
1. ``matplotlib.cbook`` resolves to a package directory that does not expose
   ``_is_pandas_dataframe``, while the real implementation lives in
   ``matplotlib/cbook.py``.
2. ``porespy`` imports numba-decorated functions with ``cache=True`` and that
   fails in this setup.

This module applies the needed shims before importing ``matplotlib`` or
``porespy``.
"""

from __future__ import annotations

import importlib.util
import pathlib


def _patch_matplotlib_cbook() -> None:
    import matplotlib.cbook as cbook_pkg

    if hasattr(cbook_pkg, "_is_pandas_dataframe"):
        return

    cbook_py = pathlib.Path(cbook_pkg.__file__).resolve().parent.parent / "cbook.py"
    if pathlib.Path(cbook_pkg.__file__).resolve() == cbook_py or not cbook_py.exists():
        raise ImportError(
            "matplotlib.cbook is missing _is_pandas_dataframe, and no fallback "
            f"cbook.py was found at {cbook_py}"
        )

    spec = importlib.util.spec_from_file_location("_matplotlib_cbook_file", cbook_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load matplotlib cbook module from {cbook_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in dir(module):
        if hasattr(cbook_pkg, name):
            continue
        setattr(cbook_pkg, name, getattr(module, name))


def _patch_numba_cache() -> None:
    import numba

    if getattr(numba, "_porespy_compat_patched", False):
        return

    original_jit = numba.jit
    original_njit = numba.njit

    def jit_no_cache(*args, **kwargs):
        kwargs.pop("cache", None)
        return original_jit(*args, **kwargs)

    def njit_no_cache(*args, **kwargs):
        kwargs.pop("cache", None)
        return original_njit(*args, **kwargs)

    numba.jit = jit_no_cache
    numba.njit = njit_no_cache
    numba._porespy_compat_patched = True


def prepare_matplotlib() -> None:
    _patch_matplotlib_cbook()


def prepare() -> None:
    _patch_matplotlib_cbook()
    _patch_numba_cache()


def import_porespy():
    prepare()
    import porespy as ps

    return ps
