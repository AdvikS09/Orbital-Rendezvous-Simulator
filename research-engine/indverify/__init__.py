"""Independent closed-loop verifier for the CW rendezvous study.

Implements indverify/SPEC.md using only the Python standard library, NumPy
and SciPy.  It does not import, copy or call any production code.

Public interface
----------------
run_case(scenario, setting, branches=("ref", "cw", "so"), fine_dt=0.01,
         keep_history=False) -> (summary, history)
Command line: python -m indverify --cases CASES.json --settings SETTINGS.json --out OUTDIR
"""

from ._version import __version__
from .runner import run_case

IMPLEMENTS = "indverify/SPEC.md"

__all__ = ["run_case", "__version__", "IMPLEMENTS"]
