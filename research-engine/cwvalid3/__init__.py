"""v3 extension library (does not modify cwvalid/).

Modules (see v3_extension/methods/V3_CONVENTIONS.md):
  sampled      batch fixed-step propagation of held-command (zero-order hold) branches
  hybrid       held-command first-variation estimators A_H, B_H, A_naive, C_H (batch)
  hybrid_map   independent discrete-map implementation of the held-command estimators
  controllers  general 3x6 matrix-gain law normsat(-K s), PD special case, LQR design
  metrics3     robust extremum, crossing and arrival resolution with explicit statuses
"""

__version__ = "0.1.0"
