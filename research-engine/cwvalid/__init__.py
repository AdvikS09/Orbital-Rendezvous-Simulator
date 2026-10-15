"""cwvalid: closed-loop CW validity extension to the Orbital Stability Platform engine.

Modules
-------
constants   physical constants (from src.orbit) and study defaults
dynamics    CW, second-order, exact nonlinear relative (NERM) and inertial models
control     PD law with direction-preserving saturation; command sources
simulate    adaptive DOP853 / fixed RK4 propagation with dense output
metrics     diagnostics and outcome definitions on a common time basis
scenario    scenario specification, deterministic IDs, initial-state builders
experiment  paired-scenario runner (CW vs nonlinear, independent loops)
runner      configuration-driven batch execution with checkpoints
"""

__version__ = "0.1.0"
