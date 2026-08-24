"""Guided educational activities.

*** REQUIRES TEACHER REVIEW BEFORE CLASSROOM USE ***

These activities were written against what the engine genuinely does, and every
expected outcome quoted in them was checked against the simulator. They have
**not** been reviewed by a subject teacher or trialled with students -- reading
level, pacing and curriculum fit are all unverified. See ``KNOWN_GAPS.md``.

Structure of an activity
------------------------
Each activity is a short sequence of steps. A step names a scenario from the
catalogue, supplies starting parameters, asks the learner to **predict** before
running, and then says what to look for. Predict-then-check is deliberate: it
is what turns a simulator from an animation into an experiment.

Nothing here contains physics. The steps reference scenario ids, and the
platform runs them through the same validated path as every other simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActivityStep:
    """One predict-run-observe step."""

    id: str
    title: str
    scenario_id: str
    brief: str
    predict_prompt: str
    predict_options: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    look_for: str = ""
    takeaway: str = ""


@dataclass(frozen=True)
class Activity:
    """A guided sequence of steps on one theme."""

    id: str
    title: str
    level: str                 # introductory | intermediate
    duration_minutes: int
    summary: str
    learning_points: tuple[str, ...]
    steps: tuple[ActivityStep, ...]
    prerequisites: str = "None. Everything you need is explained as you go."
    review_status: str = "DRAFT - not yet reviewed by a subject teacher"


# --------------------------------------------------------------------------
# Activity 1 -- orbits and what disturbs them
# --------------------------------------------------------------------------
ORBITS_AND_PERTURBATIONS = Activity(
    id="orbits_and_perturbations",
    title="What keeps a satellite up, and what pulls it down",
    level="introductory",
    duration_minutes=20,
    summary=(
        "Start from an idealised circular orbit, confirm the simulator conserves "
        "what physics says it should, then switch on the two effects that matter "
        "most in low Earth orbit: Earth's equatorial bulge and atmospheric drag."),
    learning_points=(
        "A circular orbit is free fall: gravity supplies exactly the centripetal "
        "acceleration the orbital speed requires.",
        "Orbital period depends on the size of the orbit and not at all on the "
        "satellite's mass.",
        "Conservation checks are how you decide whether to trust a simulation.",
        "J2 (Earth's bulge) accumulates orbit after orbit -- it does not average out.",
        "Drag removes energy, which lowers the orbit, which makes the satellite "
        "go faster.",
    ),
    steps=(
        ActivityStep(
            id="step1_circular",
            title="An orbit with nothing else going on",
            scenario_id="two_body_orbit",
            brief=(
                "Run a satellite at 400 km with only point-mass gravity acting. "
                "Nothing else -- no bulge, no atmosphere. Physics says energy and "
                "the orbital period must be exactly conserved, so this is really a "
                "test of the simulator rather than of the orbit."),
            predict_prompt="What will the altitude-versus-time plot look like?",
            predict_options=("A flat horizontal line",
                             "A line sloping gradually downwards",
                             "A wave going up and down",
                             "A line sloping upwards"),
            params={"altitude_km": 400.0, "inclination_deg": 0.0, "n_orbits": 2.0},
            look_for=(
                "The altitude plot should be flat. Look at the two numbers above it: "
                "energy drift and period error should both be around 1e-13, which "
                "means about fourteen digits of agreement with theory."),
            takeaway=(
                "The propagator conserves what it should to roughly machine "
                "precision. That is what earns the right to use it for harder "
                "problems where you do not know the answer in advance."),
        ),
        ActivityStep(
            id="step2_period",
            title="Does a higher orbit go round faster or slower?",
            scenario_id="two_body_orbit",
            brief=(
                "Run the same scenario at 400 km, note the orbital period, then run "
                "it again at 1200 km and compare. The satellite has not changed -- "
                "only the size of its orbit."),
            predict_prompt="At 1200 km, compared to 400 km, one orbit will take...",
            predict_options=("Longer", "Less time", "Exactly the same time"),
            params={"altitude_km": 1200.0, "inclination_deg": 0.0, "n_orbits": 1.0},
            look_for=(
                "The period at 400 km is about 92.6 minutes. At 1200 km it should be "
                "noticeably longer, around 109 minutes. Bigger orbit, longer period, "
                "slower speed."),
            takeaway=(
                "T = 2*pi*sqrt(a^3/mu). The satellite's own mass is nowhere in that "
                "expression, which is why a bolt and a space station at the same "
                "altitude orbit together."),
        ),
        ActivityStep(
            id="step3_j2",
            title="Earth is not perfectly spherical",
            scenario_id="j2_effect",
            brief=(
                "Earth bulges at the equator, so its gravity is not quite that of a "
                "point mass. This runs the same starting state twice -- once with the "
                "bulge and once without -- and plots how far apart the two end up."),
            predict_prompt="Over five orbits, how far apart will the two trajectories drift?",
            predict_options=("A few metres", "A few hundred metres",
                             "Several kilometres", "They will not separate at all"),
            params={"altitude_km": 400.0, "inclination_deg": 51.6, "n_orbits": 5.0},
            look_for=(
                "The divergence curve should climb steadily rather than wobble around "
                "zero. That steady climb is what 'secular' means: the effect builds up "
                "rather than cancelling out."),
            takeaway=(
                "Now set inclination to 0 degrees and run again. The effect changes, "
                "because an equatorial orbit sits symmetrically in the bulge. J2 is "
                "the largest gravitational perturbation in low Earth orbit and mission "
                "designers exploit it deliberately -- Sun-synchronous orbits are chosen "
                "so that J2 precession keeps the orbit plane fixed relative to the Sun."),
        ),
        ActivityStep(
            id="step4_drag",
            title="The atmosphere never quite ends",
            scenario_id="drag_effect",
            brief=(
                "At 300 km there is still enough air to matter. Drag always opposes "
                "motion, so it takes energy out of the orbit. Watch what that does to "
                "the altitude -- and then think carefully about what it does to the "
                "speed."),
            predict_prompt="Drag removes energy from the orbit. The satellite therefore...",
            predict_options=("slows down", "speeds up", "keeps the same speed"),
            params={"altitude_km": 300.0, "inclination_deg": 51.6, "n_orbits": 10.0,
                    "area": 2.0, "mass": 50.0, "cd": 2.2},
            look_for=(
                "The altitude trend should be downwards. This is the trap: losing "
                "energy lowers the orbit, and lower orbits are faster, so the "
                "satellite ends up moving faster than it started."),
            takeaway=(
                "Now double the area, or halve the mass, and run again. Both should "
                "roughly double the decay, because drag scales with Cd*A/m. This is "
                "why a dead satellite with big solar panels re-enters years before a "
                "compact one at the same altitude."),
        ),
    ),
)


# --------------------------------------------------------------------------
# Activity 2 -- relative motion
# --------------------------------------------------------------------------
RELATIVE_MOTION = Activity(
    id="relative_motion",
    title="Why you cannot just fly straight at another spacecraft",
    level="introductory",
    duration_minutes=25,
    summary=(
        "Switch to the target's point of view and discover that nearby motion in "
        "orbit is nothing like motion on a road. A spacecraft parked behind another "
        "stays put; one parked above it slides kilometres away."),
    learning_points=(
        "The target's reference frame rotates once per orbit, which introduces "
        "Coriolis and centrifugal effects.",
        "A pure along-track offset is an equilibrium -- a natural parking spot.",
        "A radial offset causes a large along-track drift, because a different "
        "altitude means a different orbital period.",
        "Out-of-plane motion is decoupled and simply oscillates once per orbit.",
    ),
    steps=(
        ActivityStep(
            id="step1_alongtrack",
            title="Parked 1 km behind",
            scenario_id="cw_natural_motion",
            brief=(
                "Put the chaser 1 km behind the target along its direction of travel, "
                "with no relative velocity and no thrust at all, and let it drift for "
                "an orbit."),
            predict_prompt="After one full orbit with no thrust, the chaser will be...",
            predict_options=("still about 1 km behind", "far further away",
                             "much closer to the target", "above the target"),
            params={"rel_x": 0.0, "rel_y": 1000.0, "rel_z": 0.0,
                    "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
                    "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 10.0},
            look_for=(
                "The chaser barely moves. Both spacecraft are on the same orbit at the "
                "same altitude, so they have the same period and keep station "
                "automatically."),
            takeaway=(
                "This is why visiting vehicles hold position along-track from a space "
                "station: it costs nothing to stay there."),
        ),
        ActivityStep(
            id="step2_radial",
            title="Parked 100 m above",
            scenario_id="cw_natural_motion",
            brief=(
                "Now move the same chaser to 100 m directly away from Earth instead -- "
                "a hundred times smaller an offset than the last step. Again: no "
                "relative velocity, no thrust."),
            predict_prompt="After one orbit, roughly where will the chaser be?",
            predict_options=("Still about 100 m above the target",
                            "A few hundred metres away",
                            "Several kilometres away along-track",
                            "It will have fallen onto the target"),
            params={"rel_x": 100.0, "rel_y": 0.0, "rel_z": 0.0,
                    "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
                    "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 10.0},
            look_for=(
                "The along-track position should end near -3770 m. That is -12*pi times "
                "the 100 m radial offset: the standard result. A hundred-metre offset "
                "in the wrong direction turns into nearly four kilometres of drift in "
                "one orbit."),
            takeaway=(
                "Higher orbit, longer period. The chaser is not falling behind because "
                "something pushed it -- it is simply going round on a slightly slower "
                "orbit. Radial offsets are the expensive direction."),
        ),
        ActivityStep(
            id="step3_crosstrack",
            title="Out of the plane",
            scenario_id="cw_natural_motion",
            brief=(
                "Offset the chaser 200 m perpendicular to the orbit plane instead, "
                "and watch the cross-track component of the position plot."),
            predict_prompt="The out-of-plane position over one orbit will...",
            predict_options=("stay constant", "grow steadily",
                             "oscillate like a sine wave and return to where it started",
                             "decay to zero"),
            params={"rel_x": 0.0, "rel_y": 0.0, "rel_z": 200.0,
                    "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
                    "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 10.0},
            look_for=(
                "A clean sine wave completing exactly one cycle per orbit, and the "
                "in-plane components staying at zero throughout."),
            takeaway=(
                "The out-of-plane direction is a simple harmonic oscillator at the "
                "orbital rate, completely decoupled from the in-plane motion. Both "
                "orbits cross the same line of nodes twice per revolution, so the "
                "separation must come back to zero."),
        ),
    ),
)


# --------------------------------------------------------------------------
# Activity 3 -- control and its cost
# --------------------------------------------------------------------------
CONTROL_AND_COST = Activity(
    id="control_and_cost",
    title="Flying the rendezvous, and paying for it",
    level="intermediate",
    duration_minutes=25,
    summary=(
        "Add a controller that thrusts to close the gap, then confront the "
        "engineering reality: every improvement in speed is bought with propellant, "
        "and the model you tuned against is not quite the truth."),
    learning_points=(
        "A PD controller commands thrust from position error and its rate of change.",
        "Delta-v is the currency of spaceflight; it maps directly onto propellant.",
        "Faster convergence costs more delta-v -- there is no free lunch.",
        "Thrust saturation changes the response qualitatively, not just in degree.",
        "A linear model has a validity range, and it can be measured rather than "
        "assumed.",
    ),
    steps=(
        ActivityStep(
            id="step1_rendezvous",
            title="Closing the gap",
            scenario_id="pd_rendezvous",
            brief=(
                "Start the chaser 369 m from the target -- offset in all three axes -- "
                "and let a Proportional-Derivative controller fly it in. Watch the "
                "trajectory curve into the origin and note the delta-v."),
            predict_prompt="Look at the 3D trajectory. Do you expect the chaser to "
                           "approach the target in a straight line?",
            predict_options=("Yes, straight in", "No, it will curve",
                             "It will spiral many times around"),
            params={"rel_x": 200.0, "rel_y": -300.0, "rel_z": 80.0,
                    "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
                    "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 5.0,
                    "omega_n_factor": 6.0, "max_accel": 0.05},
            look_for=(
                "A curved path, not a straight one -- the Coriolis coupling keeps "
                "bending it. Convergence should take around 1595 s for about 2.33 m/s "
                "of delta-v, with peak thrust near 0.017 m/s^2, comfortably under the "
                "0.05 limit."),
            takeaway=(
                "The controller has to fight the orbital dynamics as well as close the "
                "distance, which is why the path curves. Delta-v is the number that "
                "matters: it converts directly into how much propellant you must launch."),
        ),
        ActivityStep(
            id="step2_bandwidth",
            title="Make it faster and see what it costs",
            scenario_id="pd_rendezvous",
            brief=(
                "Same starting point, but push the controller bandwidth from 6 to 12 "
                "times the orbital rate. Compare convergence time, peak thrust and "
                "delta-v against the previous run."),
            predict_prompt="Doubling the bandwidth will make the rendezvous...",
            predict_options=("faster and cheaper", "faster but more expensive",
                             "slower and cheaper", "identical"),
            params={"rel_x": 200.0, "rel_y": -300.0, "rel_z": 80.0,
                    "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
                    "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 5.0,
                    "omega_n_factor": 12.0, "max_accel": 0.05},
            look_for=(
                "Convergence time should drop, and both peak thrust and delta-v should "
                "rise. Check whether peak thrust has hit the 0.05 m/s^2 limit -- if it "
                "has, the controller is saturating and no longer behaving the way its "
                "gains were designed for."),
            takeaway=(
                "This is the trade-off that makes controller design an engineering "
                "decision rather than a maths problem. There is no setting that is "
                "best at everything."),
        ),
        ActivityStep(
            id="step3_gain_sweep",
            title="Map the whole trade-off at once",
            scenario_id="gain_tuning",
            brief=(
                "Rather than trying gains one at a time, sweep a grid of them and look "
                "at the two heatmaps together: convergence time and delta-v."),
            predict_prompt="Do you expect the fastest gains and the cheapest gains to "
                           "be the same cell?",
            predict_options=("Yes", "No", "Only for small starting offsets"),
            params={"rel_x": 200.0, "rel_y": -300.0, "rel_z": 80.0,
                    "altitude_km": 400.0, "duration_s": 5553.62, "dt_s": 10.0},
            look_for=(
                "The fastest and cheapest cells are in different places. Some cells "
                "have a blank convergence time -- those combinations never reached the "
                "1 m tolerance within the orbit. They are reported, not hidden."),
            takeaway=(
                "A blank cell is information, not a bug. Weak gains, or too much "
                "damping relative to stiffness, simply do not get there in time."),
        ),
        ActivityStep(
            id="step4_model_error",
            title="How wrong is the simple model?",
            scenario_id="cw_vs_reference",
            brief=(
                "Everything so far used the linearised Clohessy-Wiltshire model. Now "
                "run the same free drift in both that model and a full non-linear "
                "propagation of both spacecraft, and measure the gap. Start with J2 "
                "switched off, then run it again with J2 on."),
            predict_prompt="Starting 1 km along-track and drifting for one orbit, how "
                           "far apart will the two models be by the end?",
            predict_options=("Under a metre", "A few metres", "Hundreds of metres",
                            "Several kilometres"),
            params={"rel_x": 0.0, "rel_y": 1000.0, "rel_z": 0.0,
                    "rel_vx": 0.0, "rel_vy": 0.0, "rel_vz": 0.0,
                    "altitude_km": 400.0, "inclination_deg": 51.6,
                    "duration_s": 5553.62, "dt_s": 10.0, "controlled": False,
                    "use_j2": False, "use_drag": False},
            look_for=(
                "About 2.8 m with two-body gravity only. Switch J2 on and re-run: it "
                "should roughly double, to about 5.8 m. Over 1 km of separation, a few "
                "metres of model error is small -- which is exactly why the linear "
                "model is useful."),
            takeaway=(
                "Now try the same thing with a 1 km *radial* offset instead of "
                "along-track. The error is far larger, because a radial offset excites "
                "the large along-track drift you found in the previous activity. A "
                "model's validity is not one number -- it depends on what you ask of it."),
        ),
    ),
)


ACTIVITIES: tuple[Activity, ...] = (
    ORBITS_AND_PERTURBATIONS,
    RELATIVE_MOTION,
    CONTROL_AND_COST,
)

ACTIVITIES_BY_ID = {activity.id: activity for activity in ACTIVITIES}


def get_activity(activity_id: str) -> Activity | None:
    return ACTIVITIES_BY_ID.get(activity_id)


__all__ = ["Activity", "ActivityStep", "ACTIVITIES", "ACTIVITIES_BY_ID", "get_activity"]
