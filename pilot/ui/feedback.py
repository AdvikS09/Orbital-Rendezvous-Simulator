"""The feedback form.

Step 8 of the participant journey. Short by design — a long form gets abandoned
and a pilot with no responses measures nothing. Questions adapt to who is
answering: professionals are asked about usefulness for real work, learners
about whether the explanations made sense.
"""

from __future__ import annotations

import streamlit as st

from .. import analytics
from .common import context, goto, page_header, user_type_label


def feedback_form() -> None:
    participant_context = context()
    user_type = participant_context.user_type if participant_context else "other"
    is_professional = user_type == "engineer_researcher"

    page_header("How was it?",
                "Six questions, about two minutes. All of it is anonymous.")

    if st.session_state.get("feedback_done"):
        st.success("Your feedback has been recorded. Thank you.", icon="✅")
        if st.button("Continue", type="primary"):
            goto("confirmation")
        return

    st.caption(f"Answering as: {user_type_label(user_type)}")

    with st.form("feedback_form"):
        usefulness_label = (
            "How useful was this for real engineering or research work?"
            if is_professional else
            "How useful was this for understanding orbital mechanics?")
        usefulness = st.slider(usefulness_label, 1, 5, 3,
                               help="1 = not useful at all, 5 = very useful")

        usability = st.slider(
            "How easy was the platform to use?", 1, 5, 3,
            help="1 = very difficult, 5 = very easy")

        clarity_label = (
            "How clear were the assumptions and limitations?"
            if is_professional else
            "How clear were the explanations?")
        clarity = st.slider(clarity_label, 1, 5, 3,
                            help="1 = not clear at all, 5 = very clear")

        recommend = st.slider(
            "How likely are you to recommend this to a colleague or classmate?",
            0, 10, 7, help="0 = not at all likely, 10 = extremely likely")

        st.divider()
        most_useful = st.text_area(
            "What was most useful?", height=80,
            placeholder="A scenario, an explanation, a plot — anything.")

        improvement = st.text_area(
            "What would you change?", height=80,
            placeholder=("Anything confusing, missing, or in the way."
                         if not is_professional else
                         "Missing physics, missing diagnostics, workflow friction."))

        bug_report = st.text_area(
            "Did anything go wrong?", height=80,
            placeholder="Errors, wrong-looking results, things that would not load. "
                        "Say which scenario if you can.")

        feature_request = st.text_area(
            "What should be added next?", height=80,
            placeholder=("Impulsive or optimal guidance, differential drag, "
                         "navigation errors, eccentric reference orbits…"
                         if is_professional else
                         "Another activity, a different topic…"))

        st.caption(
            "Please do not put your name, email address or anything else personal "
            "in these boxes — they are stored exactly as written.")

        submitted = st.form_submit_button("Submit feedback", type="primary")

    if submitted:
        recorded = True
        if participant_context is not None:
            recorded = analytics.record_feedback(
                participant_context, usefulness_rating=int(usefulness),
                usability_rating=int(usability), clarity_rating=int(clarity),
                would_recommend=int(recommend), most_useful=most_useful,
                improvement=improvement, bug_report=bug_report,
                feature_request=feature_request)

        st.session_state.feedback_done = True
        if not recorded:
            st.warning(
                "Your feedback could not be saved because the database is not "
                "reachable. Sorry — nothing you can do about that from here.",
                icon="⚠️")
        goto("confirmation")

    st.divider()
    left, right = st.columns(2)
    with left:
        if st.button("Back to the simulator", width="stretch"):
            goto("workspace")
    with right:
        if st.button("Skip feedback", width="stretch"):
            st.session_state.feedback_done = True
            goto("confirmation")


__all__ = ["feedback_form"]
