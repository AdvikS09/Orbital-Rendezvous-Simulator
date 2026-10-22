"""Participant flow: landing, consent, sign-in, track choice, confirmation.

Implements steps 1-4 and step 10 of the pilot user journey. The simulation
workspace, assessments and feedback form live in their own modules.
"""

from __future__ import annotations

import streamlit as st

import orbital_core
from .. import analytics
from ..analytics import USER_TYPE_LABELS, USER_TYPES
from ..config import get_settings
from ..validation import validate_participant_code, validate_pilot_code
from .common import (
    CONSENT_VERSION,
    PRIVACY_SUMMARY,
    STATUS_STATEMENT,
    VALIDATION_STATEMENT,
    context,
    goto,
    page_header,
)


# 1. Landing
def landing() -> None:
    page_header(
        "Orbital Rendezvous Simulator",
        "An open-source tool for exploring how spacecraft move near one another "
        "in orbit — and how far a simple model can be trusted.")

    st.markdown(
        "This platform runs a real orbital-mechanics simulation engine in your "
        "browser. Nothing to install, no Python required.\n\n"
        "It is built around one question that matters in spaceflight: **when two "
        "spacecraft are close together, how well does the simple, cheap model "
        "engineers actually fly with agree with a more faithful one?** You can "
        "explore that question directly, or work through guided activities that "
        "build up to it.")

    left, right = st.columns(2)
    with left:
        st.subheader("What you can do here")
        st.markdown(
            "- Propagate orbits and check that energy and angular momentum are "
            "conserved\n"
            "- Switch on Earth's equatorial bulge (J2) and atmospheric drag and "
            "see what each does\n"
            "- Explore relative motion in a target-centred frame, where nothing "
            "moves the way intuition expects\n"
            "- Fly a PD-controlled rendezvous and measure what it costs in "
            "propellant\n"
            "- Sweep controller gains to map the speed-versus-fuel trade-off\n"
            "- Compare the linear model against a non-linear reference and "
            "measure the gap\n"
            "- Check keep-out-zone and debris clearance for a trajectory\n"
            "- Download your results and a configuration file that reproduces "
            "the run exactly")
    with right:
        st.subheader("Who it is for")
        st.markdown(
            "- **Students** meeting orbital mechanics for the first time — "
            "guided activities that ask you to predict before you run\n"
            "- **Teachers** wanting a simulator that runs in a browser, with "
            "optional anonymous before-and-after assessments\n"
            "- **Engineers and researchers** who want direct access to the "
            "model, the numerical diagnostics, and reproducible exports")

    st.subheader("What this is not")
    st.warning(
        f"{STATUS_STATEMENT}\n\n{VALIDATION_STATEMENT}", icon="⚠️")

    with st.expander("Known limitations of the physics — worth reading before you start"):
        st.markdown(
            "- **The relative-motion model is linear.** The Clohessy-Wiltshire "
            "equations assume a circular reference orbit and a small separation. "
            "Accuracy degrades as the spacecraft get further apart, and degrades "
            "fastest for an offset directly above or below the target.\n"
            "- **The reference model is better, not perfect.** It adds non-linear "
            "gravity, J2 and drag, but has no higher-order gravity terms, no "
            "third-body effects and no solar radiation pressure.\n"
            "- **The atmosphere is crude.** A single-scale-height exponential "
            "model calibrated at 400 km. Real density varies by an order of "
            "magnitude with solar activity, which is not modelled at all.\n"
            "- **Navigation is perfect.** The controller is given exact, "
            "noise-free knowledge of the state. There is no sensor model and no "
            "estimator, so nothing here says anything about robustness to "
            "measurement error.\n"
            "- **Guidance is one PD controller.** No impulsive targeting, no LQR, "
            "no model-predictive control.\n"
            "- **Nothing avoids anything.** Keep-out zones and debris are "
            "detected and reported, never avoided.\n"
            "- **Identical spacecraft.** Target and chaser share a ballistic "
            "coefficient, so differential drag is not exercised.")

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        if st.button("Start", type="primary", width="stretch"):
            goto("consent")
    with right:
        st.caption(
            f"Engine v{orbital_core.__version__}. Open source under the MIT licence. "
            "The next page explains what is recorded and asks your consent.")


# 2. Privacy notice and consent
def consent() -> None:
    settings = get_settings()
    page_header("Before you begin",
                "A short privacy notice, and your consent to anonymous usage tracking.")

    st.markdown(
        "This platform is being piloted, which means we want to know whether it "
        "is actually useful. To find that out we record **anonymous, aggregate** "
        "usage information. Here is exactly what that means.")

    st.info(PRIVACY_SUMMARY, icon="🔒")

    with st.expander("The full detail"):
        st.markdown(
            "**Why we collect anything at all.** To report how many people used "
            "the platform, which parts they used, whether simulations succeeded, "
            "whether learners' understanding changed, and what people asked for "
            "next. Nothing is used for advertising, sold, or shared with third "
            "parties.\n\n"
            "**How anonymity works.** Your participant code is combined with "
            "your pilot code and a secret key held by the server, and passed "
            "through a one-way cryptographic function. Only the result is "
            "stored. Typing the same code again on another day produces the same "
            "result, which is how return visits are counted — but the stored "
            "value cannot be reversed to recover your code.\n\n"
            "**Your orbital parameters stay yours.** The numbers you type into "
            "simulations are not recorded. The analytics store only broad bands "
            "— for example \"separation between 100 m and 1 km\" — so nobody can "
            "reconstruct what you were working on."
            + ("\n\n**Note for this deployment:** raw parameter logging has been "
               "switched on by the operator of this instance, so the values you "
               "enter *are* being recorded."
               if settings.log_raw_parameters else "") +
            "\n\n**Free text.** Anything you type into the feedback box is stored "
            "as you wrote it, so please do not put personal information there.\n\n"
            f"**Retention.** Pilot data is kept for at most "
            f"{settings.retention_days} days and then deleted.\n\n"
            "**Withdrawing.** Because the data is anonymous, we cannot identify "
            "your records to remove them individually. If you want nothing "
            "recorded at all, please close this page instead of continuing — the "
            "simulator can also be run entirely offline from the source "
            "repository, with no analytics.")

    st.divider()
    agreed = st.checkbox(
        "I have read the above and I consent to anonymous usage information "
        "being recorded.", key="consent_checkbox")

    left, right = st.columns([1, 1])
    with left:
        if st.button("Continue", type="primary", disabled=not agreed,
                     width="stretch"):
            st.session_state.consent_given = True
            goto("entry")
    with right:
        if st.button("Back", width="stretch"):
            goto("landing")

    if not agreed:
        st.caption("You need to tick the box to continue.")


# 3. Participant entry
def entry() -> None:
    settings = get_settings()
    page_header("Sign in anonymously",
                "Three short answers. None of them identify you.")

    if not st.session_state.get("consent_given"):
        st.warning("Please read the privacy notice first.")
        if st.button("Go to the privacy notice"):
            goto("consent")
        return

    with st.form("entry_form"):
        pilot_code_raw = st.text_input(
            "Organisation or pilot code",
            help="The code your teacher or pilot coordinator gave you. If you are "
                 "here on your own, enter SELF.",
            placeholder="e.g. SCHOOL-A or SELF")

        participant_code_raw = st.text_input(
            "Anonymous participant code",
            help="Make one up and write it down — you will need the same code to "
                 "be recognised if you come back. Do NOT use your name or email.",
            placeholder="e.g. BLUE-FALCON-42")

        st.caption(
            "Pick something memorable but not personal. Your code is never "
            "stored — only a one-way hash of it.")

        user_type = st.radio(
            "Which best describes you?",
            USER_TYPES, format_func=lambda value: USER_TYPE_LABELS[value],
            horizontal=False)

        submitted = st.form_submit_button("Continue", type="primary")

    if not submitted:
        st.caption("Nothing is recorded until you press Continue.")
        return

    pilot_code, pilot_error = validate_pilot_code(pilot_code_raw)
    participant_code, participant_error = validate_participant_code(participant_code_raw)

    if pilot_error:
        st.error(pilot_error, icon="🚫")
    if participant_error:
        st.error(participant_error, icon="🚫")
    if pilot_error or participant_error:
        return

    code_row, _ = analytics.resolve_pilot_code(pilot_code, settings)
    if code_row is None and settings.require_known_pilot_code:
        st.error(
            "That pilot code is not registered on this deployment. Check it with "
            "your coordinator.", icon="🚫")
        return

    try:
        participant_context = analytics.start_session(
            pilot_code=pilot_code, participant_code=participant_code,
            user_type=user_type, consent_version=CONSENT_VERSION, settings=settings)
    except Exception:  # noqa: BLE001 - never show a database traceback
        st.error(
            "Sign-in could not be completed because the usage database is not "
            "reachable. You can still use the simulator — press Continue without "
            "tracking below — but your session will not be counted in the pilot.",
            icon="🚫")
        if st.button("Continue without tracking"):
            st.session_state.context = None
            st.session_state.track = None
            goto("track")
        return

    st.session_state.context = participant_context
    analytics.record_event(participant_context, analytics.EventType.PILOT_ACCESSED,
                           settings=settings)

    if code_row is None:
        st.info(
            f"The code **{pilot_code}** is not registered as a pilot on this "
            "deployment, so your session is grouped under \"Unregistered pilot "
            "code\". That is fine — everything works normally.", icon="ℹ️")

    if participant_context.is_return_visit:
        st.success("Welcome back. Your earlier work is recognised.", icon="👋")

    goto("track")


# 4. Track selection
def track_selection() -> None:
    participant_context = context()
    page_header("How would you like to use the simulator?",
                "You can switch tracks at any time.")

    user_type = participant_context.user_type if participant_context else "other"

    cards = [
        ("guided",
         "Guided educational activities",
         "Three short activities that build from 'what keeps a satellite up' to "
         "'how much can you trust a linear model'. Each step asks you to predict "
         "what will happen before you run it.",
         "Best if you are new to orbital mechanics. About 20–25 minutes each. "
         "Includes optional anonymous before-and-after questions so a teacher "
         "can see whether the activities worked."),
        ("professional",
         "Professional / research exploration",
         "Direct access to every scenario, including the model-comparison study, "
         "the integrator convergence check and the external-validation export. "
         "Full parameter control and reproducible configuration downloads.",
         "Best if you already know the subject and want to interrogate the model."),
        ("free",
         "Free simulation mode",
         "Every scenario, no structure, no assessments. Pick something and run it.",
         "Best if you want to explore on your own terms."),
    ]

    # Nudge, do not force: the recommended track is highlighted but all are open.
    recommended = {"student": "guided", "teacher": "guided",
                   "engineer_researcher": "professional"}.get(user_type, "free")

    for track_id, title, description, audience in cards:
        with st.container(border=True):
            heading = f"### {title}"
            if track_id == recommended:
                heading += "  ·  *suggested for you*"
            st.markdown(heading)
            st.markdown(description)
            st.caption(audience)
            if st.button(f"Choose {title.lower()}", key=f"track_{track_id}",
                         type="primary" if track_id == recommended else "secondary"):
                st.session_state.track = track_id
                st.session_state.activity_id = None
                st.session_state.activity_step = 0
                st.session_state.scenario_id = None
                if participant_context is not None:
                    analytics.set_track(participant_context, track_id)
                # Educational users get the pre-assessment offered first.
                if track_id == "guided" and not st.session_state.assessment_pre_done:
                    goto("assessment_pre")
                goto("workspace")


# 10. Confirmation
def confirmation() -> None:
    page_header("Thank you", "Your session is complete.")

    runs = st.session_state.get("completed_runs", 0)
    st.success(
        f"You completed **{runs}** simulation{'s' if runs != 1 else ''} in this "
        "session.", icon="✅")

    st.markdown(
        "### What happens to your feedback\n"
        "This is a pilot, and the point of a pilot is to change what gets built. "
        "Your responses are pooled with everyone else's and reviewed by the "
        "project maintainer. Feedback **may** influence later releases — which "
        "scenarios get added, which explanations get rewritten, what gets fixed "
        "first.\n\n"
        "To be straight with you about it: this is a small open-source project, "
        "so no promise is made that any particular suggestion will be "
        "implemented, or implemented soon. Bug reports are the most likely to "
        "lead to a change, and quickly.\n\n"
        "Nothing you submitted identifies you, so we cannot contact you about "
        "your feedback or tell you what became of it.")

    st.markdown(
        "### Coming back\n"
        "Use the **same participant code** next time and your return visit will "
        "be recognised. That matters for the pilot: repeat use across separate "
        "days is one of the things being measured.")

    st.markdown(
        "### Using this yourself\n"
        "The whole thing is open source under the MIT licence. You can run it "
        "locally with no analytics at all, read the physics, and reuse it in "
        "your own teaching or research.")

    st.divider()
    left, right = st.columns(2)
    with left:
        if st.button("Run more simulations", type="primary", width="stretch"):
            goto("workspace")
    with right:
        if st.button("Finish and start a new session", width="stretch"):
            for key in list(st.session_state.keys()):
                if key != "admin_authenticated":
                    del st.session_state[key]
            goto("landing")


__all__ = ["landing", "consent", "entry", "track_selection", "confirmation"]
