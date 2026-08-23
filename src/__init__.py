"""Compatibility shim: ``src`` is now :mod:`orbital_core`.

The physics engine moved from ``src/`` to ``orbital_core/`` so that the package
name says what it is, and so that the FastAPI backend, the legacy Streamlit
application and the experiment scripts all import one obvious module. The move
was done with ``git mv``, so every file keeps its full history.

**No code was changed by the move.** This module re-exports ``orbital_core``
under the old name so that:

* the existing test suite keeps importing ``from src import orbit`` unchanged,
  which is what makes it a genuine regression check on the move rather than a
  test of newly-edited imports; and
* any script or notebook outside this repository that imported ``src`` keeps
  working.

``src.orbit is orbital_core.orbit`` evaluates to ``True`` -- these are aliases
for the same module objects, not copies, so there is exactly one definition of
every constant and function.

New code should import ``orbital_core`` directly.
"""

from __future__ import annotations

import sys as _sys

import orbital_core as _core
from orbital_core import (  # noqa: F401
    collision,
    controller,
    cw,
    external_validation,
    lvlh,
    orbit,
    perturbations,
    plotting,
    scenarios,
    validation,
)

__version__ = _core.__version__
__doc_original__ = _core.__doc__

# Register the submodules under their old dotted names so that
# ``import src.orbit`` and ``from src.cw import propagate_cw`` resolve to the
# very same module objects rather than re-importing anything.
for _name in (
    "orbit",
    "perturbations",
    "lvlh",
    "cw",
    "controller",
    "validation",
    "external_validation",
    "collision",
    "scenarios",
    "plotting",
):
    _sys.modules[f"{__name__}.{_name}"] = getattr(_core, _name)

del _name

__all__ = [
    "orbit",
    "perturbations",
    "lvlh",
    "cw",
    "controller",
    "validation",
    "external_validation",
    "collision",
    "scenarios",
    "plotting",
    "__version__",
]
