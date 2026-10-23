from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    options: tuple[str, ...]
    correct_index: int
    topic: str
    explanation: str = ""


@dataclass(frozen=True)
class Instrument:
    id: str
    version: str
    title: str
    description: str
    items: tuple[Item, ...]
    confidence_question: str = (
        "How confident do you feel about orbital mechanics and spacecraft "
        "rendezvous right now?")
    confidence_labels: tuple[str, ...] = (
        "Not at all confident", "Slightly confident", "Moderately confident",
        "Confident", "Very confident")
    review_status: str = "DRAFT - not yet reviewed by a subject teacher"

    @property
    def max_score(self) -> int:
        return len(self.items)


ORBITAL_BASICS = Instrument(
    id="orbital_basics",
    version="1.0",
    title="Orbital mechanics and rendezvous: what do you already know?",
    description=(
        "Ten quick multiple-choice questions. There is no time limit and no "
        "pass mark -- this measures what the activities teach, not you. You will "
        "answer the same questions again at the end so the change can be "
        "measured. Your answers are anonymous."),
    items=(
        Item(
            id="q1_period_altitude",
            topic="orbits",
            question="A satellite moves from a 400 km circular orbit to an 800 km "
                     "circular orbit. What happens to the time it takes to go once "
                     "around the Earth?",
            options=("It gets longer", "It gets shorter", "It stays the same",
                     "It depends on the satellite's mass"),
            correct_index=0,
            explanation="Period grows with orbit size: T = 2*pi*sqrt(a^3/mu). A higher "
                        "orbit is a longer orbit and a slower one. The satellite's own "
                        "mass does not appear in the formula at all."),
        Item(
            id="q2_conservation",
            topic="orbits",
            question="For a satellite orbiting under point-mass gravity alone, which "
                     "of these stays constant?",
            options=("Its speed", "Its altitude above the ground",
                     "Its total orbital energy", "Its distance travelled per second "
                     "along the ground"),
            correct_index=2,
            explanation="Energy and angular momentum are conserved. Speed and altitude "
                        "are only constant in the special case of a perfectly circular "
                        "orbit; energy is conserved for any orbit shape."),
        Item(
            id="q3_j2",
            topic="perturbations",
            question="Earth's equatorial bulge (the J2 effect) matters because it...",
            options=("slows the satellite down until it re-enters",
                     "makes the orbit plane slowly rotate, orbit after orbit",
                     "only affects satellites below 200 km",
                     "makes the orbit wobble but averages out to nothing over time"),
            correct_index=1,
            explanation="J2 is a secular perturbation: its effect accumulates rather "
                        "than averaging away. The main consequence is precession of the "
                        "orbit plane. It is the dominant gravitational perturbation in "
                        "low Earth orbit."),
        Item(
            id="q4_drag",
            topic="perturbations",
            question="Atmospheric drag removes energy from a low orbit. What happens "
                     "to the satellite's speed?",
            options=("It slows down, because drag opposes its motion",
                     "It speeds up, because it ends up in a lower orbit",
                     "It stays the same; only the altitude changes",
                     "It oscillates but does not change on average"),
            correct_index=1,
            explanation="This is the counter-intuitive one. Drag removes energy, which "
                        "lowers the orbit, and lower orbits are faster. So losing energy "
                        "to drag makes the satellite end up going faster."),
        Item(
            id="q5_ballistic",
            topic="perturbations",
            question="Two satellites are at the same altitude. Satellite A is light "
                     "with a large solar panel; satellite B is heavy and compact. "
                     "Which decays out of orbit faster?",
            options=("Satellite A", "Satellite B",
                     "They decay at the same rate", "It depends only on altitude"),
            correct_index=0,
            explanation="Drag deceleration scales with Cd*A/m -- the ballistic "
                        "coefficient. Large area and small mass both increase it, so "
                        "the light satellite with the big panel decays faster."),
        Item(
            id="q6_lvlh",
            topic="relative_motion",
            question="When describing one spacecraft's position relative to another, "
                     "the 'along-track' direction points...",
            options=("towards the centre of the Earth",
                     "along the target's direction of travel",
                     "perpendicular to the orbit plane",
                     "towards the Sun"),
            correct_index=1,
            explanation="In the LVLH frame: radial (R-bar) points away from Earth, "
                        "along-track (V-bar) points along the velocity direction, and "
                        "cross-track (H-bar) is perpendicular to the orbit plane."),
        Item(
            id="q7_alongtrack_parking",
            topic="relative_motion",
            question="A chaser sits 1 km directly behind a target on the same orbit, "
                     "with no relative velocity and no thrust. Over the next orbit it "
                     "will...",
            options=("drift steadily away from the target",
                     "fall towards the target and collide",
                     "stay roughly 1 km behind",
                     "spiral down towards the Earth"),
            correct_index=2,
            explanation="A pure along-track offset is an equilibrium: both spacecraft "
                        "are on the same orbit at the same altitude, so they have the "
                        "same period. This is why along-track is a natural place to park."),
        Item(
            id="q8_radial_drift",
            topic="relative_motion",
            question="A chaser is placed 100 m directly *above* the target (further "
                     "from Earth), with no relative velocity and no thrust. What "
                     "happens over the next orbit?",
            options=("It stays 100 m above the target",
                     "It drifts a long way along-track, much further than 100 m",
                     "It falls straight down onto the target",
                     "It oscillates up and down by 100 m"),
            correct_index=1,
            explanation="A higher orbit is a slower orbit, so a radial offset means a "
                        "period mismatch and the chaser slides backwards along-track by "
                        "about 12*pi times the offset each orbit -- roughly 3.8 km for a "
                        "100 m offset. This is why radial offsets are the hard case."),
        Item(
            id="q9_gain_tradeoff",
            topic="control",
            question="An engineer increases the gains on a rendezvous controller so the "
                     "chaser reaches the target faster. What is the usual cost?",
            options=("The rendezvous becomes less accurate",
                     "It uses more propellant and needs more thrust",
                     "It takes longer to plan the manoeuvre",
                     "There is no cost; higher gains are simply better"),
            correct_index=1,
            explanation="Speed costs fuel. Higher gains command harder thrusting, which "
                        "raises both peak thrust and total delta-v. Choosing gains is a "
                        "trade-off, not an optimisation of one number."),
        Item(
            id="q10_model_validity",
            topic="modelling",
            question="The Clohessy-Wiltshire equations are a simplified, linear model "
                     "of relative motion. When should you be most careful about "
                     "trusting them?",
            options=("When the two spacecraft are very close together",
                     "When the two spacecraft are far apart",
                     "When the orbit is circular",
                     "They are equally reliable in every situation"),
            correct_index=1,
            explanation="CW is a first-order approximation about the target's orbit, so "
                        "it is most accurate when the separation is small. Error grows "
                        "with separation -- and grows fastest for a radial offset."),
    ),
)


INSTRUMENTS: dict[str, Instrument] = {
    ORBITAL_BASICS.id: ORBITAL_BASICS,
}

DEFAULT_INSTRUMENT_ID = ORBITAL_BASICS.id


def get_instrument(instrument_id: str | None = None) -> Instrument:
    return INSTRUMENTS.get(instrument_id or DEFAULT_INSTRUMENT_ID, ORBITAL_BASICS)


def score_responses(instrument: Instrument, responses: dict[str, int]) -> tuple[int, list]:
    per_item = []
    score = 0
    for item in instrument.items:
        chosen = responses.get(item.id)
        correct = chosen is not None and int(chosen) == item.correct_index
        score += int(correct)
        per_item.append({"item_id": item.id, "topic": item.topic,
                         "chosen_index": None if chosen is None else int(chosen),
                         "correct": correct})
    return score, per_item


__all__ = ["Item", "Instrument", "INSTRUMENTS", "ORBITAL_BASICS",
           "DEFAULT_INSTRUMENT_ID", "get_instrument", "score_responses"]
