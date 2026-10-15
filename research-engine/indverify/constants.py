"""Physical constants and fixed numerical parameters (SPEC sections 1, 5, 6, 8).

Everything in this package is in SI units (m, s, m/s, m/s^2, rad).
"""

import math

# --- SPEC section 1: constants and target orbit -----------------------------
MU = 3.986004418e14            # Earth gravitational parameter [m^3/s^2]
R_E = 6378137.0                # Earth equatorial radius [m]
INCLINATION_DEG = 51.6         # target orbit inclination (fixed for definiteness)
INCLINATION = math.radians(INCLINATION_DEG)
DEFAULT_ALTITUDE = 400000.0    # default target altitude [m]

# --- SPEC section 5: physical-domain check of the inertial branch -----------
ALTITUDE_FLOOR = 100000.0      # chaser altitude below which the reference stops [m]

# --- SPEC section 6: saturation-switch restarts ------------------------------
SWITCH_RESTART_CAP = 2000      # give up switch-event restarts after this many
NO_PROGRESS_MIN_ADVANCE = 1e-9 # a restart that advanced less than this [s] ...
NO_PROGRESS_LIMIT = 10         # ... this many times in a row disables switch events

# --- SPEC section 7: solver defaults -----------------------------------------
DEFAULT_MAX_STEP = 20.0        # [s]
POS_ATOL_SCALE_REL = 1000.0    # relative-state position atol = rtol * 1 km
VEL_ATOL_SCALE_REL = 1.0       # relative-state velocity atol = rtol * 1 m/s
POS_ATOL_SCALE_INERTIAL = 1.0e6  # inertial position atol = rtol * 1000 km
VEL_ATOL_SCALE_INERTIAL = 1.0e3  # inertial velocity atol = rtol * 1 km/s

# Divergence guard for the CW / quadratic branches (no physical-domain check is
# applied to them; this only turns a run-away state into a reported numerical
# failure before it overflows).  1e15 m is ~1e8 Earth radii.  It goes beyond
# the failure modes listed in SPEC section 5 (step-size underflow, non-finite
# state); the failure is reported at the start of the step that crossed the
# limit, so its time depends on the step size (e.g. 4105.72 s with Radau
# 1e-10 vs 4105.5 s with RK4 h = 0.5 s): compare failure times across solvers
# only to within one step.
DIVERGENCE_LIMIT = 1.0e15

# --- SPEC section 8: assessed quantities ---------------------------------------
TERMINAL_POS_M = 1.0           # terminal set: |rho| < 1 m
TERMINAL_VEL_MPS = 0.01        # ... and |rhodot| < 0.01 m/s
DWELL_REQUIRED_S = 300.0       # arrival needs dwell >= 300 s
PROD_DT = 0.5                  # production monitoring grid [s]
FINE_DT = 0.01                 # fine monitoring grid [s]
GRID_SNAP = 1e-9               # last grid point within this of T is replaced by T
PROD_ENTRY_XTOL = 1e-6         # Brent xtol for the production entry time [s]
FINE_ROOT_XTOL = 1e-9          # Brent xtol for fine-convention / saturation roots [s]
PEAK_XATOL = 1e-6              # bounded maximisation tolerance for peaks [s]

# Branch status strings
COMPLETED = "completed"
DOMAIN_EXIT = "domain_exit"
NUMERICAL_FAILURE = "numerical_failure"
