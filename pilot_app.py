"""Orbital Rendezvous Simulator — pilot platform entry point.

Run with::

    streamlit run pilot_app.py

A thin router. It owns no physics and no analytics logic: it wires session
state to the flow stages in ``pilot/ui/`` and makes sure the database schema
exists before anything tries to write to it.

The administrator area is reached with ``?view=admin`` and is password
protected. It is deliberately not a Streamlit ``pages/`` entry, because that
would advertise it in the navigation to every participant.
"""

from __future__ import annotations

import logging
import os
import sys

import streamlit as st

# Make the project importable when Streamlit is launched from another directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pilot.config import get_settings  # noqa: E402
from pilot.db import get_engine, healthcheck, run_migrations  # noqa: E402
from pilot.ui import admin, assessment, feedback, flow, workspace  # noqa: E402
from pilot.ui.common import (  # noqa: E402
    init_state,
    progress_bar,
    status_footer,
)

logging.basicConfig(
    level=os.environ.get("PILOT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pilot.app")

st.set_page_config(
    page_title="Orbital Rendezvous Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "about": (
            "Open-source orbital mechanics simulator for education and research. "
            "Not NASA-approved, not flight-certified, not for operational use. "
            "MIT licence."),
    },
)


@st.cache_resource(show_spinner=False)
def _prepare_database() -> dict:
    """Migrate on first use. Cached so it runs once per process, not per rerun."""
    settings = get_settings()
    try:
        run_migrations(get_engine(settings))
    except Exception as exc:  # noqa: BLE001 - the simulator must run regardless
        logger.error("database setup failed: %s", type(exc).__name__)
    return healthcheck()


_STAGES = {
    "landing": flow.landing,
    "consent": flow.consent,
    "entry": flow.entry,
    "track": flow.track_selection,
    "assessment_pre": assessment.pre_assessment,
    "workspace": workspace.workspace,
    "assessment_post": assessment.post_assessment,
    "feedback": feedback.feedback_form,
    "confirmation": flow.confirmation,
}

#: Stages a participant can only reach once they have signed in.
_REQUIRE_SESSION = ("assessment_pre", "assessment_post", "feedback", "confirmation")


def _sidebar(health: dict) -> None:
    settings = get_settings()
    with st.sidebar:
        st.markdown("### Orbital Rendezvous Simulator")
        participant_context = st.session_state.get("context")
        if participant_context is not None:
            st.caption(
                f"Signed in anonymously as `{participant_context.participant_ref}` · "
                f"{participant_context.organisation_name}")
            runs = st.session_state.get("completed_runs", 0)
            st.caption(f"{runs} simulation{'s' if runs != 1 else ''} completed "
                       "this session.")
        else:
            st.caption("Not signed in.")

        if not health.get("ok"):
            st.warning(
                "Usage tracking is unavailable — the simulator still works, but "
                "this session will not be counted in the pilot.", icon="⚠️")

        if not settings.is_production:
            st.info(f"Environment: `{settings.environment}`. Records are flagged "
                    "as demo data.", icon="🧪")

        st.divider()
        st.caption("Not NASA-approved · not for operational use · MIT licence")
        st.caption("[Administrator area](?view=admin)")


def main() -> None:
    init_state()
    health = _prepare_database()

    if st.query_params.get("view") == "admin":
        admin.admin_dashboard()
        return

    stage = st.session_state.get("stage", "landing")
    if stage not in _STAGES:
        stage = "landing"
        st.session_state.stage = stage

    # A participant who has not signed in cannot land on a mid-flow stage.
    if stage in _REQUIRE_SESSION and not st.session_state.get("consent_given"):
        st.session_state.stage = stage = "landing"

    if stage not in ("landing", "consent"):
        progress_bar(stage)

    _STAGES[stage]()

    # Rendered last on purpose. Streamlit writes into the sidebar container
    # whenever it is called, but session state (the completed-run count) is only
    # correct once the stage has run -- drawing it first showed a stale count.
    _sidebar(health)
    status_footer()


if __name__ == "__main__":
    main()
