"""The simulation workspace: guided activities, professional and free modes.

Step 5 of the participant journey. All three tracks converge on the same run
path (:func:`pilot.runner.run_scenario`), so validation, safeguards and
analytics behave identically no matter how a participant got here.
"""

from __future__ import annotations

import streamlit as st

from .. import analytics
from ..config import get_settings
from ..content.activities import ACTIVITIES, get_activity
from ..content.scenarios_catalog import (
    CATEGORY_LABELS,
    default_params,
    get_scenario,
    scenarios_for,
)
from ..ratelimit import get_limiter
from ..runner import run_scenario, validate_params
from .common import (
    context,
    goto,
    page_header,
    render_params,
    show_errors,
    show_warnings,
)
from .results import render_result


def _record_download(kind: str) -> None:
    participant_context = context()
    if participant_context is not None:
        analytics.record_event(participant_context,
                               analytics.EventType.RESULTS_DOWNLOADED,
                               feature_category="download",
                               detail={"download_kind": kind})


def _run_and_store(scenario, params: dict) -> None:
    """Run a scenario and hold the outcome in session state."""
    participant_context = context()
    with st.spinner("Running the simulation…"):
        outcome = run_scenario(scenario, params, participant_context)
    st.session_state.last_outcome = outcome
    st.session_state.scenario_id = scenario.id
    st.session_state.params = params
    if outcome.ok:
        st.session_state.completed_runs = st.session_state.get("completed_runs", 0) + 1


def _show_outcome(scenario, outcome, params) -> None:
    settings = get_settings()
    if outcome is None:
        return
    if not outcome.ok:
        show_errors(outcome.errors)
        if outcome.rejected:
            st.caption(
                "Nothing was run. Adjust the values above and try again — the "
                "engine itself has no such limits when you run it locally.")
        return
    show_warnings(outcome.warnings)
    st.caption(f"Completed in {outcome.duration_ms} ms · {outcome.n_steps:,} "
               "integration steps.")
    render_result(scenario, outcome, params, settings.app_version, _record_download)


def _budget_caption() -> None:
    participant_context = context()
    if participant_context is None:
        return
    decision = get_limiter().check(participant_context.session_token)
    if decision.runs_remaining <= 20:
        st.caption(f"{decision.runs_remaining} simulation runs left in this session.")


# Guided track
def _guided() -> None:
    activity_id = st.session_state.get("activity_id")

    if not activity_id:
        st.subheader("Choose an activity")
        st.caption(
            "Work through them in order if you are new to this — each builds on "
            "the one before.")
        for activity in ACTIVITIES:
            with st.container(border=True):
                st.markdown(f"### {activity.title}")
                st.markdown(activity.summary)
                st.caption(
                    f"{activity.level.title()} · about {activity.duration_minutes} "
                    f"minutes · {len(activity.steps)} steps · "
                    f"prerequisites: {activity.prerequisites}")
                with st.expander("What you will learn"):
                    for point in activity.learning_points:
                        st.markdown(f"- {point}")
                if st.button(f"Start: {activity.title}", key=f"act_{activity.id}",
                             type="primary"):
                    st.session_state.activity_id = activity.id
                    st.session_state.activity_step = 0
                    st.session_state.last_outcome = None
                    st.rerun()
        return

    activity = get_activity(activity_id)
    if activity is None:
        st.session_state.activity_id = None
        st.rerun()
        return

    index = int(st.session_state.get("activity_step", 0))
    index = max(0, min(index, len(activity.steps) - 1))
    step = activity.steps[index]
    scenario = get_scenario(step.scenario_id)

    header = st.columns([4, 1])
    with header[0]:
        st.subheader(activity.title)
        st.caption(f"Step {index + 1} of {len(activity.steps)} — {step.title}")
    with header[1]:
        if st.button("Leave activity", width="stretch"):
            st.session_state.activity_id = None
            st.session_state.last_outcome = None
            st.rerun()
    st.progress((index + 1) / len(activity.steps))

    st.markdown(step.brief)

    # predict before running
    prediction_key = f"{activity.id}_{step.id}"
    predictions = st.session_state.setdefault("predictions", {})
    with st.container(border=True):
        st.markdown(f"**Predict first:** {step.predict_prompt}")
        st.caption(
            "There is no penalty for being wrong, and no record is kept of what "
            "you choose. Committing to a prediction is what makes the result "
            "informative.")
        choice = st.radio("Your prediction", step.predict_options,
                          index=None, key=f"pred_{prediction_key}",
                          label_visibility="collapsed")
        if choice is not None:
            predictions[prediction_key] = choice

    made_prediction = predictions.get(prediction_key) is not None

    params = {**default_params(scenario), **step.params}
    with st.expander("Parameters for this step", expanded=False):
        st.caption("Pre-set for the activity. Change them if you want to experiment.")
        params = render_params(scenario, params, f"guided_{prediction_key}",
                               show_advanced=False)

    _budget_caption()
    if st.button("Run this step", type="primary", disabled=not made_prediction,
                 key=f"run_{prediction_key}"):
        _run_and_store(scenario, params)
    if not made_prediction:
        st.caption("Choose a prediction above to unlock the Run button.")

    outcome = st.session_state.get("last_outcome")
    if outcome is not None and st.session_state.get("scenario_id") == scenario.id:
        st.divider()
        st.info(f"**What to look for:** {step.look_for}", icon="🔍")
        _show_outcome(scenario, outcome, st.session_state.get("params", params))
        if outcome.ok:
            st.success(f"**Takeaway:** {step.takeaway}", icon="💡")

    st.divider()
    navigation = st.columns([1, 1, 2])
    with navigation[0]:
        if index > 0 and st.button("Previous step", width="stretch"):
            st.session_state.activity_step = index - 1
            st.session_state.last_outcome = None
            st.rerun()
    with navigation[1]:
        if index < len(activity.steps) - 1:
            if st.button("Next step", type="primary", width="stretch"):
                st.session_state.activity_step = index + 1
                st.session_state.last_outcome = None
                st.rerun()
        elif st.button("Finish activity", type="primary", width="stretch"):
            participant_context = context()
            if participant_context is not None:
                analytics.record_event(
                    participant_context, analytics.EventType.ACTIVITY_COMPLETED,
                    feature_category="guided_activity",
                    detail={"activity_id": activity.id})
            st.session_state.activity_id = None
            st.session_state.last_outcome = None
            st.success("Activity complete.", icon="🎉")
            if not st.session_state.get("assessment_post_done"):
                goto("assessment_post")
            st.rerun()


# Professional and free tracks
def _scenario_picker(track: str) -> None:
    available = scenarios_for(track)
    by_category: dict[str, list] = {}
    for scenario in available:
        by_category.setdefault(scenario.category, []).append(scenario)

    current_id = st.session_state.get("scenario_id")
    if current_id not in {scenario.id for scenario in available}:
        current_id = available[0].id

    labels = {}
    options = []
    for category, scenarios in by_category.items():
        for scenario in scenarios:
            options.append(scenario.id)
            labels[scenario.id] = (
                f"{CATEGORY_LABELS.get(category, category)} — {scenario.title}")

    chosen = st.selectbox(
        "Scenario", options, index=options.index(current_id),
        format_func=lambda value: labels[value], key="scenario_picker")

    if chosen != st.session_state.get("scenario_id"):
        st.session_state.scenario_id = chosen
        st.session_state.params = default_params(get_scenario(chosen))
        st.session_state.last_outcome = None


def _explore(track: str) -> None:
    _scenario_picker(track)
    scenario = get_scenario(st.session_state.scenario_id)

    st.subheader(scenario.title)
    st.markdown(scenario.summary)
    with st.expander("Background", expanded=(track == "free")):
        st.markdown(scenario.explanation)
        if scenario.references:
            st.caption("References: " + "; ".join(scenario.references))

    params = st.session_state.get("params") or default_params(scenario)
    # Guard against a stale parameter dict from a previously selected scenario.
    valid_names = {param.name for param in scenario.params}
    params = {key: value for key, value in params.items() if key in valid_names}
    for param in scenario.params:
        params.setdefault(param.name, param.default)

    st.markdown("**Parameters**")
    params = render_params(scenario, params, f"explore_{scenario.id}",
                           show_advanced=(track == "professional"))
    st.session_state.params = params

    # Live feasibility feedback, before anything is run.
    check = validate_params(scenario, params)
    if check.ok:
        st.caption(f"Estimated {check.n_steps:,} integration steps "
                   f"(~{check.estimated_seconds:.1f} s).")
    else:
        show_errors(check.errors)
    if check.warnings:
        show_warnings(check.warnings)

    _budget_caption()
    buttons = st.columns([1, 1, 3])
    with buttons[0]:
        run = st.button("Run simulation", type="primary", disabled=not check.ok,
                        width="stretch")
    with buttons[1]:
        if st.button("Reset", width="stretch"):
            for param in scenario.params:
                st.session_state.pop(f"explore_{scenario.id}_{param.name}", None)
            st.session_state.params = default_params(scenario)
            st.session_state.last_outcome = None
            st.rerun()

    if run:
        _run_and_store(scenario, params)

    outcome = st.session_state.get("last_outcome")
    if outcome is not None and st.session_state.get("scenario_id") == scenario.id:
        st.divider()
        _show_outcome(scenario, outcome, st.session_state.get("params", params))


# Entry point
def workspace() -> None:
    track = st.session_state.get("track") or "free"

    titles = {
        "guided": ("Guided activities", "Predict, run, and see whether you were right."),
        "professional": ("Research workspace",
                         "Every scenario, full parameter control, reproducible exports."),
        "free": ("Free simulation", "Pick anything and run it."),
    }
    title, subtitle = titles[track]
    page_header(title, subtitle)

    if track == "guided":
        _guided()
    else:
        _explore(track)

    st.divider()
    footer = st.columns([1, 1, 2])
    with footer[0]:
        if st.button("Change track", width="stretch"):
            st.session_state.last_outcome = None
            goto("track")
    with footer[1]:
        if st.button("Give feedback and finish", type="primary",
                     width="stretch"):
            if (track == "guided" and not st.session_state.get("assessment_post_done")
                    and st.session_state.get("assessment_pre_done")):
                goto("assessment_post")
            goto("feedback")


__all__ = ["workspace"]
