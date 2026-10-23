from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from orbital_core import orbit
from orbital_core import scenarios as sc


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str
    default: object
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str = ""
    help: str = ""
    options: tuple = ()
    advanced: bool = False


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    category: str
    feature_category: str
    model: str
    summary: str
    explanation: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    params: tuple[Param, ...]
    runner: Callable
    result_kind: str
    levels: tuple[str, ...] = ("guided", "professional", "free")
    reading_guide: str = ""
    references: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)


ALTITUDE = Param("altitude_km", "Target altitude", "slider", 400.0,
                 200.0, 1200.0, 25.0, "km",
                 "Height above Earth's surface of the target's circular orbit. "
                 "400 km is roughly the International Space Station's altitude.")
INCLINATION = Param("inclination_deg", "Inclination", "slider", 51.6,
                    0.0, 90.0, 0.5, "deg",
                    "Tilt of the orbit plane relative to the equator. 51.6 deg is "
                    "the ISS inclination. J2 effects depend on this; pure "
                    "Clohessy-Wiltshire results do not.")
N_ORBITS = Param("n_orbits", "Number of orbits", "slider", 2.0, 1.0, 20.0, 1.0, "",
                 "How long to propagate, measured in orbital periods.")

REL_POS = (
    Param("rel_x", "Radial offset (R-bar, x)", "number", 0.0, -50_000.0, 50_000.0,
          10.0, "m", "Positive is away from Earth. A radial offset is the "
                     "least forgiving case: it drives a large along-track drift."),
    Param("rel_y", "Along-track offset (V-bar, y)", "number", 1000.0,
          -50_000.0, 50_000.0, 10.0, "m",
          "Positive is ahead of the target along its velocity direction. A pure "
          "along-track offset is a natural parking position."),
    Param("rel_z", "Cross-track offset (H-bar, z)", "number", 0.0,
          -50_000.0, 50_000.0, 10.0, "m",
          "Out of the orbit plane. This axis is decoupled and simply oscillates "
          "once per orbit."),
)
REL_VEL = (
    Param("rel_vx", "Radial rate", "number", 0.0, -100.0, 100.0, 0.01, "m/s",
          "Initial relative velocity, radial component.", advanced=True),
    Param("rel_vy", "Along-track rate", "number", 0.0, -100.0, 100.0, 0.01, "m/s",
          "Initial relative velocity, along-track component.", advanced=True),
    Param("rel_vz", "Cross-track rate", "number", 0.0, -100.0, 100.0, 0.01, "m/s",
          "Initial relative velocity, cross-track component.", advanced=True),
)

DT = Param("dt_s", "Timestep", "number", 5.0, 0.01, 600.0, 1.0, "s",
           "Fixed RK4 integration step. Smaller is more accurate and slower. "
           "The published results use 500-2000 steps per orbit.", advanced=True)
DURATION = Param("duration_s", "Duration", "number", 5553.62, 10.0, 300_000.0, 100.0, "s",
                 "How long to simulate. One orbit at 400 km is about 5554 s.")

CONTROLLED = Param("controlled", "Apply PD control", "checkbox", True, help=
                   "With control on, a Proportional-Derivative controller thrusts "
                   "to drive the chaser to the target. With it off you see natural "
                   "(free-drift) relative motion.")
OMEGA_N = Param("omega_n_factor", "Controller bandwidth", "slider", 6.0, 2.0, 15.0, 0.5,
                "x orbital rate",
                "How aggressive the controller is, as a multiple of the orbital "
                "rate n. Higher converges faster but costs more fuel.")
MAX_ACCEL = Param("max_accel", "Max thrust acceleration", "number", 0.05,
                  1e-6, 1.0, 0.005, "m/s^2",
                  "Thrust limit, standing in for a real finite thruster. The "
                  "commanded acceleration is clipped to this magnitude.")
KP = Param("kp", "Kp (proportional gain)", "number", 4.6e-5, 0.0, 1.0, 1e-5, "1/s^2",
           "Position feedback gain.", advanced=True)
KD = Param("kd", "Kd (derivative gain)", "number", 1.36e-2, 0.0, 10.0, 1e-3, "1/s",
           "Velocity feedback gain. Critical damping is Kd = 2*sqrt(Kp).",
           advanced=True)

USE_J2 = Param("use_j2", "Include J2 (Earth oblateness)", "checkbox", True,
               help="Earth's equatorial bulge. The dominant secular gravitational "
                    "perturbation in low Earth orbit.")
USE_DRAG = Param("use_drag", "Include atmospheric drag", "checkbox", False,
                 help="Exponential-atmosphere drag. With identical spacecraft it "
                      "largely cancels in the relative frame.")
CD = Param("cd", "Drag coefficient Cd", "number", 2.2, 0.0, 10.0, 0.1, "",
           "Dimensionless drag coefficient; 2.2 is typical for a satellite.",
           advanced=True)
AREA = Param("area", "Cross-section area", "number", 1.0, 0.001, 1000.0, 0.5, "m^2",
             "Frontal area presented to the flow.", advanced=True)
MASS = Param("mass", "Mass", "number", 100.0, 0.1, 100_000.0, 10.0, "kg",
             "Spacecraft mass. Drag scales with Cd*A/m.", advanced=True)


ENGINE_ASSUMPTIONS = (
    "Fixed-step 4th-order Runge-Kutta integration (no adaptive stepping).",
    "SI units throughout; states are [x, y, z, vx, vy, vz].",
    "Earth gravity model: point mass, optionally plus the zonal J2 term. No "
    "higher-order gravity, no third-body, no solar radiation pressure.",
    "Perfect, noise-free knowledge of the state -- there is no sensor model, "
    "navigation filter or measurement noise.",
)
CW_ASSUMPTIONS = (
    "The Clohessy-Wiltshire equations linearise relative motion about a "
    "*circular* reference orbit; they are a first-order approximation.",
    "The target is uncontrolled; only the chaser thrusts.",
)
DRAG_ASSUMPTIONS = (
    "Single-scale-height exponential atmosphere (rho0 = 3.9e-12 kg/m^3 at "
    "400 km, scale height 60 km, moderate solar activity) -- not an "
    "NRLMSISE-class model.",
    "Target and chaser share the same ballistic coefficient Cd*A/m, so drag is "
    "common-mode and largely cancels in the relative frame. Differential drag "
    "is not modelled.",
)
CW_LIMITATIONS = (
    "Accuracy degrades with separation. This repository characterises CW "
    "against its non-linear reference out to 50 km, and the published "
    "agreement figures are for separations up to about 1 km over one orbit.",
    "A radial offset is much worse than an along-track one: the same 1 km "
    "offset gives roughly 40x more CW error radially than along-track.",
)
CONTROLLER_LIMITATIONS = (
    "Continuous-thrust PD control with magnitude saturation only. No impulsive, "
    "LQR or MPC guidance, no thrust-direction constraints, no duty cycle.",
    "The controller does not avoid anything: it will fly straight through a "
    "keep-out zone if the target is inside one.",
)


def _rel_vectors(p: dict) -> tuple[list, list]:
    return ([p.get("rel_x", 0.0), p.get("rel_y", 0.0), p.get("rel_z", 0.0)],
            [p.get("rel_vx", 0.0), p.get("rel_vy", 0.0), p.get("rel_vz", 0.0)])


def _run_two_body(p: dict) -> dict:
    return sc.run_orbit(p["altitude_km"], p["inclination_deg"], p["n_orbits"],
                        use_j2=False, use_drag=False, steps_per_orbit=1500)


def _run_j2_effect(p: dict) -> dict:
    return sc.compare_orbit_perturbation(p["altitude_km"], p["inclination_deg"],
                                         p["n_orbits"], use_j2=True, use_drag=False,
                                         steps_per_orbit=800)


def _run_drag_effect(p: dict) -> dict:
    return sc.compare_orbit_perturbation(p["altitude_km"], p["inclination_deg"],
                                         p["n_orbits"], use_j2=False, use_drag=True,
                                         steps_per_orbit=600, cd=p.get("cd", 2.2),
                                         area=p.get("area", 2.0), mass=p.get("mass", 50.0))


def _run_cw_natural(p: dict) -> dict:
    pos, vel = _rel_vectors(p)
    return sc.run_cw_relative(pos, vel, p["altitude_km"],
                              duration_s=p["duration_s"], dt_s=p["dt_s"])


def _run_pd_rendezvous(p: dict) -> dict:
    pos, vel = _rel_vectors(p)
    return sc.run_cw_rendezvous(pos, vel, p["altitude_km"],
                                duration_s=p["duration_s"], dt_s=p["dt_s"],
                                max_accel=p["max_accel"],
                                omega_n_factor=p["omega_n_factor"])


def _run_gain_sweep(p: dict) -> dict:
    pos, vel = _rel_vectors(p)
    return sc.run_gain_sweep(pos, vel, p["altitude_km"],
                             duration_s=p["duration_s"], dt_s=p["dt_s"])


def _run_cw_vs_reference(p: dict) -> dict:
    pos, vel = _rel_vectors(p)
    return sc.run_relative_scenario(
        pos, vel, altitude_km=p["altitude_km"], inclination_deg=p["inclination_deg"],
        duration_s=p["duration_s"], dt_s=p["dt_s"], model="comparison",
        controlled=p.get("controlled", False), max_accel=p.get("max_accel", 0.05),
        omega_n_factor=p.get("omega_n_factor", 6.0),
        use_j2=p.get("use_j2", True), use_drag=p.get("use_drag", False))


def _run_research(p: dict) -> dict:
    pos, vel = _rel_vectors(p)
    kwargs = dict(
        altitude_km=p["altitude_km"], inclination_deg=p["inclination_deg"],
        duration_s=p["duration_s"], dt_s=p["dt_s"], model=p.get("model", "comparison"),
        controlled=p.get("controlled", True), max_accel=p.get("max_accel", 0.05),
        use_j2=p.get("use_j2", True), use_drag=p.get("use_drag", True),
        cd=p.get("cd", 2.2), area=p.get("area", 1.0), mass=p.get("mass", 100.0))
    if p.get("use_explicit_gains"):
        kwargs["kp"] = p.get("kp")
        kwargs["kd"] = p.get("kd")
    else:
        kwargs["omega_n_factor"] = p.get("omega_n_factor", 6.0)
    return sc.run_relative_scenario(pos, vel, **kwargs)


def _run_collision(p: dict) -> dict:
    pos, vel = _rel_vectors(p)
    debris = None
    if p.get("include_debris", True):
        debris = [p.get("debris_x", 10.0), p.get("debris_y", 150.0), p.get("debris_z", 20.0)]
    return sc.run_collision_scenario(
        pos, vel, altitude_km=p["altitude_km"], duration_s=p["duration_s"],
        dt_s=p["dt_s"], controlled=p.get("controlled", True),
        max_accel=p.get("max_accel", 0.05), omega_n_factor=p.get("omega_n_factor", 6.0),
        keep_out_radius=p.get("keep_out_radius", 25.0),
        collision_radius=p.get("collision_radius", 5.0),
        debris_rel_pos=debris)


def _run_convergence(p: dict) -> dict:
    import numpy as np

    radius = sc.radius_from_altitude(p["altitude_km"])
    from orbital_core import cw as cw_module
    n = cw_module.cw_mean_motion(radius, orbit.MU_EARTH)
    period = orbit.orbital_period(radius)
    state0 = np.array([p.get("rel_x", 1000.0), p.get("rel_y", 0.0),
                       p.get("rel_z", 500.0), 0.0, 0.0, 0.0], dtype=float)

    rows = []
    for samples in (126, 251, 501, 1001, 2001, 4001):
        times = np.linspace(0.0, period, samples)
        numeric, _ = cw_module.propagate_cw(state0, times, n, None)
        exact = cw_module.propagate_cw_analytic(state0, times, n)
        error = float(np.max(np.linalg.norm(numeric[:, :3] - exact[:, :3], axis=1)))
        rows.append({"samples_per_orbit": samples,
                     "dt_s": period / (samples - 1),
                     "max_position_error_m": error})
    for previous, current in zip(rows, rows[1:]):
        ratio = previous["max_position_error_m"] / current["max_position_error_m"]
        current["error_ratio_vs_previous"] = ratio
        current["observed_order"] = float(np.log2(ratio))
    return {"rows": rows, "period_s": period, "mean_motion": n, "state0": state0}


def _run_external_export(p: dict) -> dict:
    return sc.build_validation_scenario(
        altitude_km=p["altitude_km"], inclination_deg=p["inclination_deg"],
        duration_s=p["duration_s"], dt_s=p["dt_s"],
        use_j2=p.get("use_j2", True), use_drag=p.get("use_drag", False))


CATEGORY_LABELS = {
    "orbit_propagation": "Orbital propagation",
    "relative_motion": "Relative motion and rendezvous",
    "model_validation": "Model validation",
    "safety": "Proximity safety",
    "external_validation": "External-tool validation",
}

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="two_body_orbit",
        title="Two-body circular orbit",
        category="orbit_propagation",
        feature_category="two_body_propagation",
        model="orbit",
        summary="Propagate a circular orbit under point-mass gravity and check that "
                "energy, angular momentum and the orbital period are conserved.",
        explanation=(
            "A satellite in a circular orbit is in free fall: gravity supplies exactly "
            "the centripetal acceleration its speed requires, so the altitude stays "
            "constant. With only point-mass gravity acting, three quantities must be "
            "conserved -- specific orbital energy, specific angular momentum, and the "
            "orbital period. Checking that the simulator conserves them is the "
            "standard way to show a propagator is trustworthy before you rely on it "
            "for anything harder."),
        assumptions=ENGINE_ASSUMPTIONS,
        limitations=(
            "No perturbations at all in this scenario -- this is the idealised case.",
            "Conservation is checked numerically, so it is only as good as the step size.",
        ),
        params=(ALTITUDE, INCLINATION, N_ORBITS),
        runner=_run_two_body,
        result_kind="orbit",
        reading_guide=(
            "Energy drift and period error should both be extremely small numbers "
            "(around 1e-13 relative). The altitude plot should be a flat line. If "
            "either moves, the integration step is too coarse."),
    ),

    Scenario(
        id="j2_effect",
        title="J2 effect: Earth is not perfectly spherical",
        category="orbit_propagation",
        feature_category="perturbation_j2",
        model="orbit",
        summary="Compare an orbit including Earth's equatorial bulge against the ideal "
                "two-body orbit, and measure how far apart they drift.",
        explanation=(
            "Earth bulges at the equator, so its gravity field is not that of a point "
            "mass. The largest correction is the J2 term, and in low Earth orbit it is "
            "the dominant *secular* gravitational perturbation -- meaning its effect "
            "accumulates orbit after orbit rather than averaging out. It makes the "
            "orbit plane precess. This scenario runs the same initial state twice, "
            "once with J2 and once without, and plots the separation between them."),
        assumptions=ENGINE_ASSUMPTIONS + (
            "Only the zonal J2 term is included -- no J3, J4 or tesseral harmonics.",),
        limitations=(
            "The separation grows with the number of orbits, so the headline number "
            "depends strongly on how long you run for.",
            "J2 effects depend on inclination; at 0 deg the plane cannot precess.",
        ),
        params=(ALTITUDE, INCLINATION, N_ORBITS),
        runner=_run_j2_effect,
        result_kind="orbit_comparison",
        reading_guide=(
            "The divergence curve should grow steadily rather than oscillate about "
            "zero -- that growth is what 'secular' means. Try inclination 0 deg and "
            "then 51.6 deg to see the inclination dependence."),
    ),

    Scenario(
        id="drag_effect",
        title="Atmospheric drag and orbital decay",
        category="orbit_propagation",
        feature_category="perturbation_drag",
        model="orbit",
        summary="Watch a low orbit lose energy to the residual atmosphere and decay.",
        explanation=(
            "Even at hundreds of kilometres there is enough atmosphere to matter. Drag "
            "always opposes motion, so it removes energy from the orbit. The "
            "counter-intuitive result is that the spacecraft ends up going *faster*: "
            "losing energy lowers the orbit, and lower orbits are faster. The decay "
            "rate scales with the ballistic coefficient Cd*A/m, so a light spacecraft "
            "with a large area decays fastest."),
        assumptions=ENGINE_ASSUMPTIONS + DRAG_ASSUMPTIONS,
        limitations=(
            "The exponential atmosphere is calibrated at 400 km. Results well below "
            "that are extrapolated and indicative only.",
            "Real atmospheric density varies by an order of magnitude with solar "
            "activity, time of day and geomagnetic conditions. None of that is modelled.",
        ),
        params=(Param("altitude_km", "Target altitude", "slider", 300.0, 200.0, 600.0,
                      25.0, "km", ALTITUDE.help),
                INCLINATION,
                Param("n_orbits", "Number of orbits", "slider", 10.0, 1.0, 30.0, 1.0, "",
                      N_ORBITS.help),
                Param("area", "Cross-section area", "slider", 2.0, 0.5, 10.0, 0.5, "m^2",
                      AREA.help),
                Param("mass", "Mass", "slider", 50.0, 10.0, 500.0, 10.0, "kg", MASS.help),
                CD),
        runner=_run_drag_effect,
        result_kind="orbit_comparison",
        reading_guide=(
            "Altitude should trend downwards. Halving the mass or doubling the area "
            "should roughly double the decay, because both double Cd*A/m."),
    ),

    Scenario(
        id="cw_natural_motion",
        title="Natural relative motion (Clohessy-Wiltshire)",
        category="relative_motion",
        feature_category="cw_relative_motion",
        model="cw",
        summary="Free drift of a chaser in the target's rotating LVLH frame, with no "
                "thrust at all.",
        explanation=(
            "Seen from a target spacecraft, a nearby chaser does not move in straight "
            "lines. The target's frame rotates once per orbit, which introduces "
            "Coriolis and centrifugal terms, and gravity differs slightly across the "
            "separation. The Clohessy-Wiltshire equations capture all of this to first "
            "order. Two classic results fall straight out: a pure along-track offset "
            "just sits there (it is an equilibrium), while a radial offset drifts "
            "along-track by 12*pi times the offset every orbit, because a different "
            "radius means a different orbital period."),
        assumptions=ENGINE_ASSUMPTIONS + CW_ASSUMPTIONS,
        limitations=CW_LIMITATIONS,
        params=(*REL_POS, *REL_VEL, ALTITUDE, DURATION, DT),
        runner=_run_cw_natural,
        result_kind="relative",
        reading_guide=(
            "Try x = 0, y = 1000, z = 0: the chaser should barely move. Then try "
            "x = 100, y = 0, z = 0: it should drift about -3770 m along-track in one "
            "orbit (-12*pi*100). Cross-track motion is a clean sine wave at the orbital "
            "period, completely decoupled from the in-plane motion."),
        references=("Clohessy & Wiltshire (1960), J. Aerospace Sciences",),
    ),

    Scenario(
        id="pd_rendezvous",
        title="PD-controlled rendezvous",
        category="relative_motion",
        feature_category="pd_rendezvous",
        model="cw",
        summary="Drive the chaser to the target using a Proportional-Derivative "
                "controller, and measure the fuel it costs.",
        explanation=(
            "A Proportional-Derivative controller commands thrust proportional to how "
            "far the chaser is from where it should be (the P term) and to how fast "
            "that error is changing (the D term). It is about the simplest closed-loop "
            "guidance law there is, which is exactly why it is worth characterising: "
            "it is cheap enough to run on board. The bandwidth control sets how "
            "aggressive it is, as a multiple of the orbital rate. The key output is "
            "delta-v, the integral of thrust acceleration over time, which is the "
            "currency of spaceflight -- it maps directly onto propellant mass."),
        assumptions=ENGINE_ASSUMPTIONS + CW_ASSUMPTIONS + (
            "Gains default to critically damped: Kp = omega_n^2, Kd = 2*omega_n.",),
        limitations=CW_LIMITATIONS + CONTROLLER_LIMITATIONS,
        params=(Param("rel_x", REL_POS[0].label, "number", 200.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[0].help),
                Param("rel_y", REL_POS[1].label, "number", -300.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[1].help),
                Param("rel_z", REL_POS[2].label, "number", 80.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[2].help),
                *REL_VEL, ALTITUDE, DURATION, DT, OMEGA_N, MAX_ACCEL),
        runner=_run_pd_rendezvous,
        result_kind="relative",
        reading_guide=(
            "Watch the trade-off. Raising the bandwidth shortens the convergence time "
            "but raises both peak thrust and delta-v. If peak thrust hits the limit "
            "you set, the controller is saturating and the response is no longer the "
            "clean second-order one the gains were designed for. The reference case "
            "in this repository -- start [200, -300, 80] m, bandwidth 6n -- converges "
            "in about 1595 s for 2.33 m/s of delta-v."),
    ),

    Scenario(
        id="gain_tuning",
        title="Controller gain tuning: speed versus fuel",
        category="relative_motion",
        feature_category="gain_sweep",
        model="cw",
        summary="Sweep a grid of Kp and Kd values and map out the trade-off between "
                "converging fast and using little propellant.",
        explanation=(
            "There is no single best controller. Aggressive gains reach the target "
            "quickly but spend propellant doing it; gentle gains are efficient but may "
            "not arrive at all within the time you have. This scenario runs the same "
            "rendezvous for every combination in a 4x4 grid of gains and tabulates "
            "convergence time, peak thrust and delta-v for each, so the trade-off is "
            "visible rather than argued about."),
        assumptions=ENGINE_ASSUMPTIONS + CW_ASSUMPTIONS + (
            "Thrust saturation is disabled for the sweep, so each combination's "
            "control effort reflects its gains directly.",),
        limitations=CW_LIMITATIONS + CONTROLLER_LIMITATIONS + (
            "The grid is fixed and coarse. The best cell is the best *of this grid*, "
            "not a proven optimum.",
            "Cells that never reach the 1 m tolerance within the horizon are reported "
            "as blank rather than hidden.",
        ),
        params=(Param("rel_x", REL_POS[0].label, "number", 200.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[0].help),
                Param("rel_y", REL_POS[1].label, "number", -300.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[1].help),
                Param("rel_z", REL_POS[2].label, "number", 80.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[2].help),
                ALTITUDE, DURATION,
                Param("dt_s", DT.label, "number", 10.0, 0.01, 600.0, 1.0, "s", DT.help,
                      advanced=True)),
        runner=_run_gain_sweep,
        result_kind="gain_sweep",
        reading_guide=(
            "Read the two heatmaps together. The fastest cell and the cheapest cell "
            "are rarely the same one -- in the published sweep, the fastest gains cost "
            "roughly twice the delta-v of the most efficient converging gains."),
    ),

    Scenario(
        id="cw_vs_reference",
        title="How good is the linear model? CW versus a non-linear reference",
        category="model_validation",
        feature_category="model_comparison",
        model="comparison",
        summary="Run the same scenario in the linear CW model and in a full non-linear "
                "two-body (+J2, +drag) propagation, and measure the gap.",
        explanation=(
            "The Clohessy-Wiltshire equations are a linearisation, so they must be "
            "wrong by *something*. This measures by how much. The reference model "
            "propagates the target and the chaser separately under full non-linear "
            "gravity, with J2 and drag optional, then expresses the result in the same "
            "LVLH frame so the two can be differenced directly.\n\n"
            "One design decision makes this measurement meaningful: both models use "
            "the *same* RK4 integrator at the *same* step size. Integration error is "
            "therefore common to both and cancels, so what is left is genuinely "
            "modelling error rather than numerical error."),
        assumptions=ENGINE_ASSUMPTIONS + CW_ASSUMPTIONS + DRAG_ASSUMPTIONS + (
            "The 'reference' is a higher-fidelity model within this engine. It has "
            "NOT been benchmarked against an external tool such as GMAT or STK.",),
        limitations=CW_LIMITATIONS + (
            "The reference model is itself unvalidated against an independent tool. "
            "It is more faithful than CW, not ground truth.",
        ),
        params=(*REL_POS, *REL_VEL, ALTITUDE, INCLINATION, DURATION, DT,
                Param("controlled", CONTROLLED.label, "checkbox", False,
                      help=CONTROLLED.help),
                USE_J2, USE_DRAG, OMEGA_N, MAX_ACCEL),
        runner=_run_cw_vs_reference,
        result_kind="comparison",
        reading_guide=(
            "The reference case in this repository -- 1 km along-track, free drift, one "
            "orbit -- gives about 2.78 m of divergence for two-body and 5.81 m with J2. "
            "Two things to notice. First, J2 roughly doubles the error. Second, turning "
            "the controller on suppresses the divergence almost entirely, because "
            "feedback keeps correcting it: that is worth knowing, but it also means a "
            "closed-loop test is a weak way to judge how good CW is."),
    ),

    Scenario(
        id="integrator_convergence",
        title="Integrator convergence: is the step size small enough?",
        category="model_validation",
        feature_category="integrator_convergence",
        model="cw",
        summary="Halve the timestep repeatedly and watch the integration error fall, "
                "confirming the RK4 propagator is fourth-order accurate.",
        explanation=(
            "Every result from a numerical integrator depends on the step size, so the "
            "honest question is not 'is it accurate?' but 'has it converged?'. The CW "
            "equations have an exact closed-form solution (the state-transition "
            "matrix), which gives a rare opportunity: the true integration error can be "
            "measured exactly rather than estimated. A fourth-order method should cut "
            "its error by a factor of about 2^4 = 16 each time the step is halved.\n\n"
            "Note what this is and is not. The engine has exactly one integrator, RK4. "
            "This is a convergence study of that integrator, not a comparison between "
            "different integration schemes -- the engine has no others to compare."),
        assumptions=ENGINE_ASSUMPTIONS + CW_ASSUMPTIONS,
        limitations=(
            "The exact solution is only available for the unforced linear CW system, "
            "so this measures the integrator on that problem specifically.",
            "The engine offers only RK4; there is no Euler, Verlet or adaptive scheme "
            "to compare against in the current codebase.",
        ),
        params=(Param("rel_x", "Radial offset", "number", 1000.0, -50_000.0, 50_000.0,
                      100.0, "m", REL_POS[0].help),
                Param("rel_y", "Along-track offset", "number", 0.0, -50_000.0, 50_000.0,
                      100.0, "m", REL_POS[1].help),
                Param("rel_z", "Cross-track offset", "number", 500.0, -50_000.0, 50_000.0,
                      100.0, "m", REL_POS[2].help),
                ALTITUDE),
        runner=_run_convergence,
        result_kind="convergence",
        levels=("professional", "free"),
        reading_guide=(
            "The 'observed order' column should sit close to 4. If it does, the "
            "propagator behaves as fourth-order RK4 should and the published results "
            "at 500-2000 steps per orbit are safely converged."),
    ),

    Scenario(
        id="proximity_safety",
        title="Keep-out zone and debris proximity (detection only)",
        category="safety",
        feature_category="collision_detection",
        model="cw",
        summary="Check whether a chaser trajectory enters a keep-out zone around the "
                "target or passes too close to a piece of debris.",
        explanation=(
            "Real proximity operations define a keep-out zone around the target that a "
            "visiting vehicle may not enter without authority. This scenario measures "
            "the chaser's distance to the target and to an optional debris object over "
            "the whole trajectory, and flags any violation.\n\n"
            "Be clear about what this does not do: it detects, it does not avoid. The "
            "PD controller has no avoidance term, so if you aim it at a target sitting "
            "inside a keep-out zone it will fly straight in and the violation flag will "
            "fire. That is the honest demonstration of why avoidance is needed, and it "
            "is deliberately left as future work."),
        assumptions=ENGINE_ASSUMPTIONS + CW_ASSUMPTIONS + (
            "The debris object drifts on its own uncontrolled CW trajectory.",
            "Zones are spheres; no attitude, shape or approach-corridor modelling.",),
        limitations=CONTROLLER_LIMITATIONS + (
            "Detection only -- there is no avoidance, path planning or manoeuvre "
            "generation anywhere in this engine.",
            "No position uncertainty or covariance is modelled, so this is not a "
            "conjunction-assessment tool and must not be used as one.",
        ),
        params=(Param("rel_x", REL_POS[0].label, "number", 0.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[0].help),
                Param("rel_y", REL_POS[1].label, "number", 300.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[1].help),
                Param("rel_z", REL_POS[2].label, "number", 40.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[2].help),
                ALTITUDE, DURATION, DT, CONTROLLED, OMEGA_N, MAX_ACCEL,
                Param("keep_out_radius", "Keep-out zone radius", "slider", 25.0,
                      5.0, 500.0, 5.0, "m",
                      "Radius of the protected sphere around the target."),
                Param("collision_radius", "Debris collision threshold", "slider", 5.0,
                      1.0, 100.0, 1.0, "m",
                      "Chaser-debris separation below which a collision is flagged."),
                Param("include_debris", "Include a debris object", "checkbox", True),
                Param("debris_x", "Debris radial position", "number", 10.0,
                      -50_000.0, 50_000.0, 5.0, "m", advanced=True),
                Param("debris_y", "Debris along-track position", "number", 150.0,
                      -50_000.0, 50_000.0, 5.0, "m", advanced=True),
                Param("debris_z", "Debris cross-track position", "number", 20.0,
                      -50_000.0, 50_000.0, 5.0, "m", advanced=True)),
        runner=_run_collision,
        result_kind="collision",
        reading_guide=(
            "With control on and the target at the origin, the keep-out flag will "
            "always fire -- the controller is flying to the centre of the zone. Turn "
            "control off to see a free-drift pass instead."),
    ),

    Scenario(
        id="external_validation_export",
        title="External-tool validation workflow (GMAT / STK)",
        category="external_validation",
        feature_category="external_validation",
        model="orbit",
        summary="Export a scenario in a form another tool can reproduce, so its "
                "ephemeris can be compared against this engine's.",
        explanation=(
            "Comparing against an independent, widely used tool is how a propagator "
            "earns trust. This scenario produces the exportable half of that workflow: "
            "an initial state, force-model settings, timestep and duration, plus this "
            "engine's own propagated ephemeris, as CSV and JSON.\n\n"
            "**No such comparison has been run for this repository.** What you have "
            "here is the workflow, not a result. To complete it you would reproduce "
            "the exported scenario in GMAT or STK yourself, export that tool's "
            "ephemeris, and upload it on the comparison step. This platform does not "
            "drive GMAT or STK, is not affiliated with NASA or Ansys/AGI, and makes no "
            "claim to have been validated by either."),
        assumptions=ENGINE_ASSUMPTIONS + (
            "Exchange format is SI in the ECI frame with columns "
            "t_s, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps.",),
        limitations=(
            "No benchmark result is shipped with this repository. Nothing here has "
            "been validated against GMAT, STK or any other external tool.",
            "Two-body and J2-only scenarios match external tools most closely; "
            "atmosphere models differ substantially between tools, so a drag-enabled "
            "comparison will show a real and expected mismatch.",
        ),
        params=(ALTITUDE, INCLINATION, DURATION,
                Param("dt_s", DT.label, "number", 10.0, 0.01, 600.0, 1.0, "s", DT.help),
                USE_J2, USE_DRAG),
        runner=_run_external_export,
        result_kind="external_export",
        levels=("professional", "free"),
        reading_guide=(
            "Download both files. The JSON carries the exact initial state to as many "
            "digits as the engine holds -- reproduce that, not a rounded version, or "
            "the comparison measures your rounding rather than the propagators."),
    ),

    Scenario(
        id="research_workbench",
        title="Research workbench (full parameter control)",
        category="relative_motion",
        feature_category="research_workbench",
        model="comparison",
        summary="Every parameter exposed at once: initial state, orbit, controller "
                "gains, force model and which model to run.",
        explanation=(
            "The unguided entry point. Set the initial relative state, the reference "
            "orbit, the controller, and the force model, and choose whether to run the "
            "linear CW model, the non-linear reference, or both side by side. This is "
            "the same code path as every other scenario -- there is nothing extra "
            "under the hood, only fewer guard rails on the inputs."),
        assumptions=ENGINE_ASSUMPTIONS + CW_ASSUMPTIONS + DRAG_ASSUMPTIONS,
        limitations=CW_LIMITATIONS + CONTROLLER_LIMITATIONS,
        params=(Param("rel_x", REL_POS[0].label, "number", 200.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[0].help),
                Param("rel_y", REL_POS[1].label, "number", -300.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[1].help),
                Param("rel_z", REL_POS[2].label, "number", 80.0, -50_000.0, 50_000.0,
                      10.0, "m", REL_POS[2].help),
                *REL_VEL, ALTITUDE, INCLINATION, DURATION, DT,
                Param("model", "Model to run", "select", "comparison",
                      options=("comparison", "cw", "high_fidelity"),
                      help="'comparison' runs both and differences them."),
                CONTROLLED, OMEGA_N, MAX_ACCEL,
                Param("use_explicit_gains", "Set Kp and Kd directly", "checkbox", False,
                      help="Otherwise a critically damped controller is built from the "
                           "bandwidth above.", advanced=True),
                KP, KD, USE_J2, USE_DRAG, CD, AREA, MASS),
        runner=_run_research,
        result_kind="comparison",
        levels=("professional", "free"),
        reading_guide=(
            "Start from a scenario you already understand and change one thing at a "
            "time. If a result surprises you, check the warnings panel first -- an "
            "under-resolved timestep or an out-of-range separation explains most "
            "surprises."),
    ),
)

SCENARIOS_BY_ID = {scenario.id: scenario for scenario in SCENARIOS}


def scenarios_for(track: str) -> list[Scenario]:
    return [scenario for scenario in SCENARIOS if track in scenario.levels]


def get_scenario(scenario_id: str) -> Scenario | None:
    return SCENARIOS_BY_ID.get(scenario_id)


def default_params(scenario: Scenario) -> dict:
    return {param.name: param.default for param in scenario.params}


__all__ = [
    "Param", "Scenario", "SCENARIOS", "SCENARIOS_BY_ID", "CATEGORY_LABELS",
    "scenarios_for", "get_scenario", "default_params",
]
