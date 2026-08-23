"""Pilot platform layer for the orbital rendezvous research engine.

This package holds everything that is *not* physics: configuration, the
analytics database, input validation, resource safeguards, exports, admin
authentication, and the Streamlit user interface.

The simulation engine in ``src/`` is not modified by anything here. Every
simulation goes through ``src/scenarios.py``, the frontend-agnostic boundary
the engine already provided, so the validated numerical results are unchanged.

Named ``pilot`` rather than ``platform`` on purpose: a top-level package called
``platform`` would shadow the standard-library module of the same name.
"""

__version__ = "1.0.0"
