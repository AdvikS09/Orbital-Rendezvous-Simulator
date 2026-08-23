"""Orbital Rendezvous Simulator - FastAPI backend.

Presentation-layer replacement for the Streamlit application. Contains no
physics: every calculation goes through ``orbital_core``, and every scenario
definition, bound, metric and piece of educational content comes from
``pilot``, shared unchanged with the legacy application.
"""

__version__ = "1.0.0"
