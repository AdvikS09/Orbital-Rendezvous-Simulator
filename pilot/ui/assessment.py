from __future__ import annotations

import streamlit as st

from .. import analytics
from ..content.assessments import get_instrument, score_responses
from .common import context, goto, page_header


def _run(phase: str) -> None:
    instrument = get_instrument()
    participant_context = context()
    is_pre = phase == "pre"

    page_header(
        "A few questions before you start" if is_pre else "The same questions again",
        f"{len(instrument.items)} multiple-choice questions · anonymous · "
        "no time limit")

    if is_pre:
        st.markdown(
            "These measure what the activities teach, not you. Nobody sees your "
            "individual answers — a teacher only ever sees pooled results for a "
            "whole group. Answer what you think now; you will see the same "
            "questions at the end so the change can be measured.")
    else:
        st.markdown(
            "You saw these at the start. Answering them again is what makes the "
            "change measurable. You will get the explanations afterwards.")
        previous = st.session_state.get("assessment_pre_score")
        if previous is not None:
            st.caption(f"For reference, you scored {previous} out of "
                       f"{instrument.max_score} at the start.")

    if st.session_state.get(f"assessment_{phase}_done"):
        st.success("You have already completed this one.", icon="✅")
        if st.button("Continue", type="primary"):
            goto("workspace" if is_pre else "feedback")
        return

    with st.form(f"assessment_{phase}"):
        responses = {}
        for number, item in enumerate(instrument.items, start=1):
            st.markdown(f"**{number}. {item.question}**")
            choice = st.radio(
                f"Question {number}", item.options, index=None,
                key=f"{phase}_{item.id}", label_visibility="collapsed")
            if choice is not None:
                responses[item.id] = item.options.index(choice)
            st.markdown("")

        st.divider()
        confidence = st.select_slider(
            instrument.confidence_question,
            options=list(range(1, len(instrument.confidence_labels) + 1)),
            value=3,
            format_func=lambda value: instrument.confidence_labels[value - 1],
            key=f"{phase}_confidence")

        submitted = st.form_submit_button("Submit answers", type="primary")

    unanswered = len(instrument.items) - len(responses)
    if submitted:
        if unanswered:
            st.warning(
                f"{unanswered} question(s) are unanswered. You can submit anyway — "
                "they will be marked as incorrect — or go back and answer them.",
                icon="⚠️")

        score, per_item = score_responses(instrument, responses)

        if participant_context is not None:
            analytics.record_assessment(
                participant_context, instrument_id=instrument.id,
                instrument_version=instrument.version, phase=phase, score=score,
                max_score=instrument.max_score, item_responses=per_item,
                confidence_rating=int(confidence))

        st.session_state[f"assessment_{phase}_done"] = True
        if is_pre:
            st.session_state.assessment_pre_score = score
        st.rerun()

    st.divider()
    if st.button("Skip this — I would rather just use the simulator"):
        st.session_state[f"assessment_{phase}_done"] = True
        goto("workspace" if is_pre else "feedback")


def pre_assessment() -> None:
    _run("pre")


def post_assessment() -> None:
    instrument = get_instrument()

    if st.session_state.get("assessment_post_done"):
        page_header("How you did", "Your answers have been recorded anonymously.")

        before = st.session_state.get("assessment_pre_score")
        after = None
        participant_context = context()
        if participant_context is not None:
            rows = analytics.assessment_changes(
                analytics.ReportFilters(include_demo=True))
            mine = [row for row in rows
                    if row["participant_ref"] == participant_context.participant_ref]
            if mine:
                after = mine[0]["post_score"]

        if before is not None and after is not None:
            columns = st.columns(3)
            columns[0].metric("Before", f"{before} / {instrument.max_score}")
            columns[1].metric("After", f"{after} / {instrument.max_score}",
                              delta=int(after) - int(before))
            columns[2].metric("Change", f"{int(after) - int(before):+d}")
            st.caption(
                "One person's before-and-after is not evidence of much on its own — "
                "you have seen the questions twice, which helps by itself. Pooled "
                "across a whole group it becomes more informative.")

        with st.expander("The answers, with explanations", expanded=True):
            for number, item in enumerate(instrument.items, start=1):
                st.markdown(f"**{number}. {item.question}**")
                st.markdown(f"Correct answer: *{item.options[item.correct_index]}*")
                if item.explanation:
                    st.caption(item.explanation)
                st.markdown("")

        if st.button("Continue to feedback", type="primary"):
            goto("feedback")
        return

    _run("post")


__all__ = ["pre_assessment", "post_assessment"]
