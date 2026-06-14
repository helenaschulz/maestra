"""Sandbox worker — executes ONE generated feature candidate in an isolated process.

Invoked as ``python -m maestra._sandbox_worker <tmpdir>``. Reads ``code.py``, ``train.pkl``,
``val.pkl`` and ``meta.json`` from the temp dir; on success writes ``train.npy`` + ``val.npy``
(numeric feature values) and ``result.json`` ``{"status": "ok"}``. On ANY failure it writes
``result.json`` ``{"status": "error", "error": ...}`` and exits 0 — the parent never crashes.

The candidate's ``transform`` is only ever given the data WITHOUT the target column, so a
feature cannot read the label it is being scored against (the leakage guarantee).

Hardening here is *bounded execution*, not a security boundary against adversarial code:
RLIMIT_CPU/RLIMIT_AS, blocked sockets, an import whitelist and restricted builtins. The real
safety against a useless or sneaky feature is the CV gate in the parent, not this worker.
"""
from __future__ import annotations

import builtins as _builtins
import json
import os
import resource
import socket
import sys

import numpy as np
import pandas as pd

# Modules a candidate may import. Everything else (os, sys, subprocess, socket, ...) is blocked.
_ALLOWED_MODULES = {"pandas", "numpy", "math", "statistics", "re", "itertools", "collections"}

# Builtins exposed to candidate code. Notably absent: open, eval, exec, compile, input,
# __import__ (replaced by a guarded one), globals, vars, breakpoint.
_SAFE_BUILTINS = (
    "abs all any bool bytes callable chr complex dict divmod enumerate filter float frozenset "
    "getattr hasattr hash hex int isinstance issubclass iter len list map max min next ord pow "
    "range repr reversed round set slice sorted str sum tuple type zip print format "
    # common exception types — feature code legitimately raises/catches these
    "Exception ValueError TypeError KeyError IndexError AttributeError RuntimeError NameError "
    "ZeroDivisionError ArithmeticError OverflowError FloatingPointError StopIteration".split()
)


def _harden(cpu_seconds: int, mem_mb: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass
    try:  # macOS frequently ignores RLIMIT_AS — the wall-clock timeout is the hard limit
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    except (ValueError, OSError):
        pass

    def _no_network(*_args, **_kwargs):
        raise OSError("network access is disabled in the sandbox")

    socket.socket = _no_network  # type: ignore[assignment]


def _restricted_namespace() -> dict:
    real_import = _builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] not in _ALLOWED_MODULES:
            raise ImportError(f"import of {name!r} is blocked in the sandbox")
        return real_import(name, *args, **kwargs)

    safe = {n: getattr(_builtins, n) for n in _SAFE_BUILTINS if hasattr(_builtins, n)}
    safe["__import__"] = guarded_import
    return {"__builtins__": safe, "pd": pd, "np": np}


def _as_numeric_values(obj, n: int) -> np.ndarray:
    """Validate a transform output: a numeric Series/array of length ``n``, not all-NaN."""
    series = obj if isinstance(obj, pd.Series) else pd.Series(obj)
    if len(series) != n:
        raise ValueError(f"feature length {len(series)} != expected {n}")
    arr = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        raise ValueError("feature is entirely non-finite")
    return arr


def main(tmpdir: str) -> None:
    meta = json.load(open(os.path.join(tmpdir, "meta.json")))
    _harden(meta["cpu_seconds"], meta["mem_mb"])

    def fail(message: str) -> None:
        with open(os.path.join(tmpdir, "result.json"), "w") as fh:
            json.dump({"status": "error", "error": message[:500]}, fh)
        sys.exit(0)

    try:
        train_df = pd.read_pickle(os.path.join(tmpdir, "train.pkl"))
        val_df = pd.read_pickle(os.path.join(tmpdir, "val.pkl"))
        target = meta["target"]
        with open(os.path.join(tmpdir, "code.py")) as fh:
            code = fh.read()

        ns = _restricted_namespace()
        exec(code, ns)  # noqa: S102 - executing the candidate is the whole point; see module docstring
        if "fit" not in ns or "transform" not in ns:
            fail("code must define fit(train_df) and transform(df, params)")

        params = ns["fit"](train_df)  # fit may use the target (train only)
        train_features = train_df.drop(columns=[target], errors="ignore")  # transform never sees the target
        val_features = val_df.drop(columns=[target], errors="ignore")
        train_out = _as_numeric_values(ns["transform"](train_features, params), len(train_df))
        val_out = _as_numeric_values(ns["transform"](val_features, params), len(val_df))
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - any candidate failure becomes a clean error
        fail(f"{type(exc).__name__}: {exc}")

    np.save(os.path.join(tmpdir, "train.npy"), train_out)
    np.save(os.path.join(tmpdir, "val.npy"), val_out)
    with open(os.path.join(tmpdir, "result.json"), "w") as fh:
        json.dump({"status": "ok"}, fh)


if __name__ == "__main__":
    main(sys.argv[1])
