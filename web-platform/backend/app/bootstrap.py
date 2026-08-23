"""Put the repository root on ``sys.path``.

The API deliberately reuses the existing packages rather than duplicating them:

* ``orbital_core``  the validated physics engine
* ``pilot``         the scenario catalogue, input validation, run path,
                    analytics, exports and educational content

Both live at the repository root, one level above ``web-platform/``. Importing
them rather than copying them is the whole point: there is exactly one
definition of each scenario, each bound and each metric, shared by the legacy
Streamlit application and this API. A copy would drift.

This mirrors the convention the repository already uses in ``experiments/*.py``,
which insert the root on ``sys.path`` the same way.

Import this module before any ``orbital_core`` or ``pilot`` import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: web-platform/backend/app/bootstrap.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_repo_on_path() -> Path:
    """Append the repository root to ``sys.path``. Idempotent.

    **Appended, never inserted at the front.** The repository root contains
    ``app.py`` -- the legacy Streamlit MVP -- and this backend package is also
    called ``app``. If the root came first on ``sys.path``, ``import app`` would
    resolve to that single module instead of this package, and every
    ``from app.services import ...`` would fail with the confusing
    "'app' is not a package".

    Appending means the backend's own directory (which is ``sys.path[0]`` when
    running ``uvicorn app.main:app`` from ``web-platform/backend``) keeps
    priority for ``app``, while ``orbital_core`` and ``pilot`` still resolve
    from the root.
    """
    root = os.environ.get("ORBITAL_REPO_ROOT")
    resolved = Path(root).resolve() if root else REPO_ROOT
    as_text = str(resolved)
    if as_text not in sys.path:
        sys.path.append(as_text)
    return resolved


ensure_repo_on_path()
