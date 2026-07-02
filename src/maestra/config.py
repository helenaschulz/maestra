"""Shared environment/config helpers used by every entry point (maestra, -bench, -mlebench).

Lives outside ``cli.py`` so library-level modules never import from the CLI layer.
"""
from __future__ import annotations

import os


def load_dotenv(path: str = ".env") -> None:
    """Load ``KEY=VALUE`` lines from a local ``.env`` into ``os.environ``.

    A tiny hand-rolled loader so the project carries no extra dependency just to read
    one API key. Existing environment variables take precedence (``setdefault``).
    """
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
