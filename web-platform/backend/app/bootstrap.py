from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_repo_on_path() -> Path:
    root = os.environ.get("ORBITAL_REPO_ROOT")
    resolved = Path(root).resolve() if root else REPO_ROOT
    as_text = str(resolved)
    if as_text not in sys.path:
        sys.path.append(as_text)
    return resolved


ensure_repo_on_path()
