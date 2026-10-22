"""Shared Streamlit helpers: session state, layout, disclaimers, widgets.

The single source of truth for the statements this platform makes about itself
lives here (:data:`STATUS_STATEMENT`, :data:`PRIVACY_SUMMARY`). Anything shown
to a participant about validation status or data handling should come from this
module, so those claims cannot drift apart between pages.
"""

from __future__ import annotations

import streamlit as st

import orbital_core
from ..analytics import USER_TYPE_LABELS
from ..config import get_settings

CONSENT_VERSION = "1.0"

#: The one honest description of what this software is. Referenced everywhere.
STATUS_STATEMENT = (
    "This is open-source research and educational software. It is **not** "
    "NASA-approved, NASA-validated, flight-certified, or operational mission "
    "software, and it must not be used for operational flight decisions."
)

VALIDATION_STATEMENT = (
    "The simulation engine is checked against closed-form solutions and "
    "conservation laws, and those checks are part of the automated test suite. "
    "It has **not** been benchmarked against an external tool such as NASA GMAT "
    "or Ansys/AGI STK. The platform provides the workflow to carry out such a "
    "comparison yourself, but no such comparison is shipped with it."
)

PRIVACY_SUMMARY = (
    "**What is recorded:** an irreversible code derived from your pilot and "
    "participant codes, your user type, which scenarios you ran, whether they "
    "succeeded, how long they took, your assessment scores, and your feedback.\n\n"
    "**What is never recorded:** your name, email address, exact age, IP "
    "address, browser details, or the numerical parameters you type into "
    "simulations. Your participant code itself is not stored -- only a one-way "
    "cryptographic hash of it, which cannot be turned back into the code you chose."
)

TRACK_LABELS = {
    "guided": "Guided educational activities",
    "professional": "Professional / research exploration",
    "free": "Free simulation mode",
}

STAGES = ("landing", "consent", "entry", "track", "assessment_pre",
          "workspace", "assessment_post", "feedback", "confirmation")


# Session state
def init_state() -> None:
    """Seed the session-state keys the flow relies on."""
    defaults = {
        "stage": "landing",
        "consent_given": False,
        "context": None,
        "track": None,
        "activity_id": None,
        "activity_step": 0,
        "scenario_id": None,
        "params": {},
        "last_outcome": None,
        "assessment_pre_done": False,
        "assessment_post_done": False,
        "assessment_pre_score": None,
        "feedback_done": False,
        "admin_authenticated": False,
        "completed_runs": 0,
        "predictions": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def goto(stage: str) -> None:
    """Move to a flow stage and rerun."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    st.session_state.stage = stage
    st.rerun()


def context():
    return st.session_state.get("context")


def require_context() -> bool:
    """Send an un-registered visitor back to the start."""
    if context() is None:
        st.warning("Your session has ended. Please start again.")
        if st.button("Back to the start"):
            goto("landing")
        return False
    return True


# Layout
def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def progress_bar(current_stage: str) -> None:
    """Show where the participant is in the flow."""
    steps = [("entry", "Sign in"), ("track", "Choose a track"),
             ("workspace", "Simulate"), ("feedback", "Feedback"),
             ("confirmation", "Done")]
    order = [key for key, _ in steps]
    stage = current_stage
    if stage in ("assessment_pre", "assessment_post"):
        stage = "workspace"
    if stage not in order:
        return
    index = order.index(stage)
    labels = " → ".join(
        f"**{label}**" if position == index else label
        for position, (_, label) in enumerate(steps))
    st.caption(labels)
    st.progress((index + 1) / len(steps))


def status_footer() -> None:
    """The disclaimer that must be reachable from every page."""
    settings = get_settings()
    st.divider()
    with st.expander("About this software, its limits, and your data"):
        st.markdown(f"{STATUS_STATEMENT}\n\n{VALIDATION_STATEMENT}")
        st.markdown("**Privacy**")
        st.markdown(PRIVACY_SUMMARY)
        st.caption(
            f"Engine v{orbital_core.__version__} · platform v{settings.app_version} · "
            "MIT licence · open source.")
    st.caption(
        "Research and educational software — not NASA-approved and not for "
        "operational use.")


def show_assumptions_and_limits(scenario) -> None:
    """Assumptions and limitations shown next to every result."""
    left, right = st.columns(2)
    with left:
        st.markdown("**Assumptions behind this result**")
        for item in scenario.assumptions:
            st.markdown(f"- {item}")
    with right:
        st.markdown("**Limitations you should know about**")
        for item in scenario.limitations:
            st.markdown(f"- {item}")


def show_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning, icon="⚠️")


def show_errors(errors: list[str]) -> None:
    for error in errors:
        st.error(error, icon="🚫")


def metric_row(entries: list[tuple[str, str, str]]) -> None:
    """``entries`` is a list of ``(label, value, help)``."""
    if not entries:
        return
    columns = st.columns(len(entries))
    for column, (label, value, helptext) in zip(columns, entries):
        column.metric(label, value, help=helptext or None)


def format_number(value, unit: str = "", places: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "—"
    if number == 0:
        text = "0"
    elif abs(number) < 1e-3 or abs(number) >= 1e5:
        text = f"{number:.{places - 1}e}"
    else:
        text = f"{number:.{places}g}"
    return f"{text} {unit}".strip()


def user_type_label(user_type: str) -> str:
    return USER_TYPE_LABELS.get(user_type, "Other")


# Parameter widgets
def render_params(scenario, values: dict, key_prefix: str,
                  show_advanced: bool) -> dict:
    """Render a scenario's parameters and return the current values.

    Advanced parameters stay hidden by default so an inexperienced participant
    is not confronted with gain matrices, but they remain one checkbox away for
    someone who wants them.
    """
    updated = dict(values)
    basic = [param for param in scenario.params if not param.advanced]
    advanced = [param for param in scenario.params if param.advanced]

    _render_group(basic, updated, key_prefix)
    if advanced:
        with st.expander(f"Advanced controls ({len(advanced)})", expanded=show_advanced):
            _render_group(advanced, updated, key_prefix)
    return updated


def _render_group(params, values: dict, key_prefix: str) -> None:
    # Checkboxes tile compactly; everything else gets a two-column grid.
    checkboxes = [param for param in params if param.kind == "checkbox"]
    others = [param for param in params if param.kind != "checkbox"]

    for index in range(0, len(others), 2):
        columns = st.columns(2)
        for column, param in zip(columns, others[index:index + 2]):
            with column:
                values[param.name] = _widget(param, values.get(param.name), key_prefix)

    if checkboxes:
        columns = st.columns(min(3, len(checkboxes)))
        for position, param in enumerate(checkboxes):
            with columns[position % len(columns)]:
                values[param.name] = _widget(param, values.get(param.name), key_prefix)


def _widget(param, current, key_prefix: str):
    key = f"{key_prefix}_{param.name}"
    label = f"{param.label} [{param.unit}]" if param.unit else param.label
    helptext = param.help or None

    if param.kind == "checkbox":
        return st.checkbox(label, value=bool(current), key=key, help=helptext)

    if param.kind == "select":
        options = list(param.options)
        index = options.index(current) if current in options else 0
        return st.selectbox(label, options, index=index, key=key, help=helptext)

    if param.kind == "slider":
        return st.slider(label, float(param.minimum), float(param.maximum),
                         float(current), float(param.step or 1.0), key=key,
                         help=helptext)

    # Number input. min/max are deliberately NOT passed to Streamlit: the
    # validator owns the bounds and produces a proper explanation, whereas a
    # widget clamp silently changes what the participant asked for.
    return st.number_input(label, value=float(current), step=float(param.step or 1.0),
                           key=key, help=helptext, format="%g")


__all__ = [
    "CONSENT_VERSION", "STATUS_STATEMENT", "VALIDATION_STATEMENT",
    "PRIVACY_SUMMARY", "TRACK_LABELS", "STAGES", "init_state", "goto", "context",
    "require_context", "page_header", "progress_bar", "status_footer",
    "show_assumptions_and_limits", "show_warnings", "show_errors", "metric_row",
    "format_number", "user_type_label", "render_params",
]
