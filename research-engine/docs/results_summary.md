### Reference orbit and orbital parameters

The target (the "chief") sits on a circular orbit at 400 km altitude, so its radius is
r0 = 6,778,137 m, and I've set the inclination to 51.6 degrees (ISS-like). An orbit is fixed
by the six classical elements (a, e, i, RAAN, argp, nu). For a circular orbit e = 0 and only
the true anomaly nu changes with time, advancing at the mean motion n = sqrt(mu/a^3) =
1.131e-3 rad/s. That gives a period T = 2*pi*sqrt(a^3/mu) = 5553.6 s, about 92.6 minutes.
(mu = 3.986e14 m^3/s^2.)

### The LVLH frame

I describe the chaser's motion in the target's rotating LVLH (local-vertical / local-
horizontal) frame, built from the target's inertial position and velocity (r, v):

- x, radial (R-bar) = r_hat, pointing away from Earth;
- z, cross-track (H-bar) = (r x v) normalised, the orbit normal;
- y, along-track (V-bar) = z_hat cross x_hat, which completes a right-handed set and points
  along the velocity for a circular orbit.

The key point is that this frame rotates with the orbit, at angular velocity
omega = (r x v)/r^2 (magnitude n), so it is not inertial. The velocity transform into the
frame therefore picks up the transport term omega x rho. That rotation is exactly what
produces the Coriolis and centrifugal terms in the next part.

### Deriving the CW equations

Start from Newton's two-body law for each spacecraft in the inertial frame:

```
target:  r_t'' = -mu * r_t / |r_t|^3
chaser:  r_c'' = -mu * r_c / |r_c|^3,    with   r_c = r_t + rho,   rho = (x, y, z)
```

The derivation is four steps:

1. Subtract the two equations. The relative motion is then driven by the difference between
   the two gravity vectors, i.e. the tidal (differential) gravity.
2. Write rho in the rotating LVLH frame, whose angular velocity is omega = n*z_hat. In a
   rotating frame the inertial acceleration of rho expands (transport theorem) to
   rho'' + 2*(omega x rho') + omega x (omega x rho), so the relative equation of motion is

   ```
   rho'' = g_tidal - 2*(omega x rho') - omega x (omega x rho)
   ```

   Note the Coriolis term and the centrifugal term both come in with minus signs.
3. Linearise the differential gravity for a small separation (|rho| << r0). Gravity goes as
   mu/r^2, so expanding to first order about r0 (and using n^2 = mu/r0^3), the radial part
   gets weaker with height, leaving a net outward +2 n^2 x, while a sideways displacement
   tilts the gravity vector back towards Earth, giving -n^2 y and -n^2 z. So
   g_tidal = (+2 n^2 x, -n^2 y, -n^2 z).
4. Work out the transport terms with omega = n*z_hat: -omega x (omega x rho) = +n^2 (x, y, 0)
   (the centrifugal stiffness) and -2*(omega x rho') = (+2n y', -2n x', 0) (the Coriolis
   coupling).

Adding the tidal and centrifugal parts axis by axis:

| Axis | Tidal gravity | Centrifugal | Net |
|------|---------------|-------------|-----|
| radial x | +2 n^2 x | +n^2 x | +3 n^2 x |
| along-track y | -n^2 y | +n^2 y | 0 (they cancel) |
| cross-track z | -n^2 z | none | -n^2 z |

With control accelerations (a_x, a_y, a_z) this gives the CW equations, which is what I've
coded in src/cw.py:

```
x'' = 3 n^2 x + 2n y' + a_x      (radial)
y'' =         - 2n x' + a_y      (along-track)
z'' = - n^2 z         + a_z      (cross-track)
```

### What each term means

- Coriolis terms (+2n y', -2n x'): because the frame is rotating, radial and along-track
  motion deflect each other, like the sideways push you feel walking outward on a spinning
  roundabout. This couples the in-plane motion, so the chaser can't just move in a straight
  line towards the target.
- Radial term (+3 n^2 x): at a larger radius gravity is weaker (the +2 n^2 x gravity-gradient
  part) and the rotating-frame centrifugal term omega^2 x adds another +n^2 x, so the net
  push is outward and grows with x. Radial position is unstable. The 3 is just 2 (gravity
  gradient) plus 1 (centrifugal).
- Cross-track term (-n^2 z): gravity always points at Earth's centre, so an out-of-plane
  displacement feels a small restoring pull, giving z'' = -n^2 z. That is simple harmonic
  motion at the orbital rate, and it is completely decoupled from the in-plane motion, just
  oscillating once per orbit.

### Things I checked in the simulator

The closed-form state-transition matrix Phi(t) (a 4x4 in-plane block plus a 2x2 cross-track
block) reproduces the standard textbook behaviour:

- a pure along-track offset doesn't drift (it's a natural place to park);
- a radial offset x0 produces a steady along-track drift of -12*pi*x0 per orbit. The
  along-track row of Phi(t) has the term 6(sin nt - nt), whose growing part is -6 n t x0,
  which is -12*pi*x0 after one orbit (nt = 2*pi). Physically a slightly different radius
  means a slightly different period, so the chaser slowly slides ahead or behind;
- the natural bounded relative orbit is a 2:1 along-track-to-radial ellipse.

Validity: CW assumes a circular reference orbit, two-body gravity, and a small separation
(that's the linearisation). The CW vs higher-fidelity comparison below shows where that starts to fail.

## PD controller implementation

The controller commands an acceleration (force per unit mass) to drive the relative state to
a target, which by default is the LVLH origin with zero relative velocity:

```
u = -Kp (r - r_target) - Kd (v - v_target)
```

(src/controller.py). The gains can be scalars, per-axis vectors, or 3x3 matrices. For a
unit-mass second-order system x'' + Kd x' + Kp x = 0, critical damping is Kd = 2*sqrt(Kp);
writing Kp = wn^2 makes Kd = 2*wn. By default I use a natural frequency wn = 6n (about six
times the orbital rate), so

```
Kp = (6n)^2 = 4.608e-5  (1/s^2),    Kd = 2*(6n) = 1.358e-2  (1/s)
```

There is also a thrust-saturation limit (default |u| <= 0.05 m/s^2) to stand in for a real,
finite thruster. I reuse the same controller on the high-fidelity model: its LVLH command is
rotated into the inertial frame (rotation from src/lvlh.py) and applied to the deputy inside
the high-fidelity propagator (src/validation.py).

## Simulator setup

The code lives in src/, all SI units, with the inertial (ECI) state stored as
[x, y, z, vx, vy, vz]:

| Module | What it does |
|--------|--------------|
| orbit.py | constants, two-body dynamics, fixed-step RK4, orbital-element conversions |
| perturbations.py | J2 (zonal) and atmospheric drag, combined in a togglable ForceModel |
| lvlh.py | inertial to LVLH transforms (including the omega x rho transport term) |
| cw.py | CW dynamics, the closed-form Phi(t), RK4 propagation |
| controller.py | the PD controller, critical-damping helper, saturation |
| validation.py | conservation checks, rendezvous metrics, gain sweep, CW vs high-fidelity |
| scenarios.py | a thin "run a scenario, return data" API the Streamlit app calls |
| external_validation.py | (scaffold, future work) CSV/JSON export plus external-ephemeris import and compare |
| collision.py | (scaffold, future work) keep-out-zone and debris minimum-distance detection |

Force models: two-body, plus J2 (J2 = 1.083e-3), plus drag (exponential atmosphere,
rho0 = 3.9e-12 kg/m^3 at 400 km, scale height 60 km, Cd = 2.2, A = 1 m^2, m = 100 kg), or J2
and drag together.

Propagator check: over one orbit the fixed-step RK4 holds specific energy to a relative
drift of 1.4e-14 and angular momentum to 6.6e-15, and recovers the analytic period to
1.3e-13 (outputs/data/orbit_validation.csv). So the integrator itself is not a meaningful
error source on these timescales.

## Initial results and trends

### Controlled rendezvous (CW + PD)

Starting from a relative state of [200, -300, 80] m with zero relative velocity (a 369.3 m
offset), critically damped at wn = 6n, over one orbit. Figures:
cw_rendezvous_lvlh_trajectory.png, cw_rendezvous_position.png, cw_rendezvous_velocity.png,
cw_rendezvous_control.png.

| Quantity | Value |
|----------|-------|
| Final position error | 1.5e-7 m (0.15 micrometres) |
| Final velocity error | 6.6e-10 m/s |
| Convergence time (error under 1 m, and stays there) | 1591 s, about 0.29 orbit |
| Peak control acceleration | 0.017 m/s^2 (below the 0.05 cap, so not saturated) |
| Total delta-v | 2.33 m/s |

So the controller pulls a 369 m offset down to sub-micrometre in under a third of an orbit,
using modest, unsaturated thrust.

### PD gain tuning

A 4x4 sweep over Kp in {1e-5, 4.6e-5, 1e-4, 2e-4} and Kd in {5e-3, 1e-2, 1.36e-2, 3e-2}
(outputs/data/gain_sweep.csv, figure gain_sweep_comparison.png). A few representative rows:

| Kp | Kd | Convergence (s) | Peak accel (m/s^2) | delta-v (m/s) | Regime |
|----|----|-----------------|--------------------|---------------|--------|
| 1.0e-5 | 0.0136 | did not converge within 1 orbit | 0.004 | 2.7 | too weak / over-damped |
| 4.6e-5 | 0.0136 | 1598 | 0.017 | 2.32 | critically damped (wn = 6n) |
| 1.0e-4 | 0.0136 | 994 | 0.037 | 4.05 | faster, more fuel |
| 2.0e-4 | 0.005 | 2331 (about 0.42 orbit) | 0.074 | 19.9 | aggressive / under-damped |

Trends: for this fixed starting offset, peak thrust is basically proportional to Kp (0.004 up
to 0.074 m/s^2 across the four Kp values). Stronger gains settle faster and to a tighter
final error, but delta-v climbs steeply (up to about 20 m/s for the most aggressive case).
Gains that are too weak, or have too much Kd relative to Kp (over-damped), don't reach the
1 m band inside an orbit. The critically damped wn = 6n point is a reasonable compromise on
the speed-versus-delta-v trade within the grid I searched (it is the best of this grid, not
a proven optimum): fairly fast settling (about 0.29 orbit), low delta-v (about 2.3 m/s), and
modest thrust. A faster point does exist (Kp = 1e-4, Kd = 0.0136 settles in 994 s at 4.05 m/s
of delta-v), so it really is a trade-off, which is part of Question 2.

### CW vs higher-fidelity model

I run the same scenario in CW and in the full non-linear model (two-body with the
perturbations), both expressed in LVLH, and look at how far apart they drift. That gap is the
linearisation (and perturbation) error. Figures: cw_vs_high_fidelity_error.png,
cw_vs_high_fidelity_toggles.png.

(a) Uncontrolled free drift from a 1 km along-track offset, over one orbit, with each force
model switched on (outputs/data/cw_vs_high_fidelity_toggles.csv, max position error):

| Force model | CW vs model error |
|-------------|-------------------|
| two-body | 2.8 m |
| plus J2 | 5.8 m |
| plus drag | 2.8 m |
| plus J2 and drag | 5.8 m |

Adding J2 roughly doubles the gap over one orbit; drag barely matters at this altitude and
timescale. The gap also depends a lot on the geometry of the offset: if I redo the same
one-orbit free drift for a radial 1 km offset instead of along-track, the gap is about 110 m
for two-body, roughly 40 times larger, because a radial offset sets off a large steady
along-track drift and brings in more non-linearity. (That radial number I recomputed from the
engine for this note; only the along-track table above is a saved output.) So CW is
trustworthy for small, gently-excited separations and not for large radial offsets.

(b) Closed-loop, with the same PD driving both models, from [200, -300, 80] under J2 and
drag: CW and the high-fidelity model stay within 1.7 cm of each other across the whole orbit
(final 3e-9 m). Because the feedback keeps correcting, the linearisation error never builds
up. That is an encouraging first sign, for this one scenario and gain set, that a CW-tuned
controller still works on the higher-fidelity model. I'd want to confirm it across more
starting conditions, and the high-fidelity model itself is still not validated against an
external tool (still future work, listed below).

## Limitations and possible extensions

Current limitations (deliberate, for this stage):

- CW is linear, so it is only valid for small separations near a circular orbit (see the CW vs higher-fidelity comparison above).
- Continuous-thrust PD only, with just saturation standing in for the thruster; no impulsive
  or optimal targeting yet.
- Perturbations are limited to J2 and drag (no solar radiation pressure, higher-order
  gravity, or third-body).
- No navigation or measurement model; I assume perfect knowledge of the state.
- Fixed-step RK4 (fine here, but not adaptive).

Possible extensions (scaffolded, future work, not core results):

- External validation against GMAT or STK. Export a scenario (initial state, force-model
  settings, step, duration, ephemeris) as CSV/JSON, import the external tool's ephemeris back
  in, and compute position and velocity max/final/mean/RMS errors. This would validate the
  higher-fidelity propagator, which is the reference for the CW comparison
  (src/external_validation.py).
- Collision and keep-out detection: a spherical keep-out zone around the target plus an
  optional debris point on its own CW drift, reporting minimum distances and risk flags.
  Detection only, no avoidance (src/collision.py).
- A simple debris-avoidance scenario would be the natural next step after detection.
