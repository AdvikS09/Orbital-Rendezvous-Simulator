"""Shared helpers for the runnable scripts.

Adds the project root to ``sys.path`` (so ``import src`` works when a script is
run directly) and defines the output directories for figures and tables.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "tables")


def ensure_output_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def figure_path(name):
    ensure_output_dirs()
    return os.path.join(FIGURES_DIR, name)


def result_path(name):
    ensure_output_dirs()
    return os.path.join(RESULTS_DIR, name)
