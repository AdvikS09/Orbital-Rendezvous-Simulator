"""Password-protected administrator dashboard.

Reports pilot activity using the metric definitions in ``pilot/analytics.py``,
with filters by date, organisation and user type, and CSV export of every table.

Two things it deliberately will not do:

* it cannot show who a participant is, because that information does not exist
  anywhere in the database; and
* it excludes development and demonstration records by default, so a demo run
  can never be mistaken for pilot activity. Including them is possible, but it
  is an explicit switch and the page says loudly when it is on.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .. import analytics
from ..analytics import USER_TYPE_LABELS, USER_TYPES, ReportFilters
from ..auth import get_authenticator
from ..config import get_settings
from ..db import healthcheck, safe_database_label


# Sign-in
def _sign_in() -> None:
    st.title("Administrator sign-in")
    settings = get_settings()
    authenticator = get_authenticator(settings)

    if not authenticator.configured:
        st.error(
            "No administrator credentials are configured on this deployment.\n\n"
            "Set `PILOT_ADMIN_USERNAME` and `PILOT_ADMIN_PASSWORD_HASH` in the "
            "environment (or in Streamlit secrets) and restart. Generate a hash "
            "with `python scripts/hash_password.py`.", icon="🔒")
        return

    with st.form("admin_sign_in"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        result = authenticator.authenticate(username, password)
        if result.ok:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error(result.message, icon="🚫")


# Filters
def _filters() -> ReportFilters:
    settings = get_settings()
    st.sidebar.subheader("Report filters")

    preset = st.sidebar.selectbox(
        "Period", ["Last 7 days", "Last 30 days", "Last 90 days", "All time",
                   "Custom range"], index=1)
    today = date.today()
    if preset == "All time":
        start = end = None
    elif preset == "Custom range":
        chosen = st.sidebar.date_input(
            "Date range", value=(today - timedelta(days=30), today))
        if isinstance(chosen, tuple) and len(chosen) == 2:
            start, end = chosen
        else:
            start, end = None, None
    else:
        days = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}[preset]
        start, end = today - timedelta(days=days), today

    include_demo = st.sidebar.checkbox(
        "Include development / demo records", value=False,
        help="Off by default so demonstration and test runs are never counted as "
             "real pilot activity.")

    organisations = analytics.list_organisations(include_demo=include_demo)
    options = [None] + [org["id"] for org in organisations]
    names = {org["id"]: f"{org['name']} ({org['code']})" for org in organisations}
    organisation_id = st.sidebar.selectbox(
        "Organisation", options, index=0,
        format_func=lambda value: "All organisations" if value is None
        else names.get(value, str(value)))

    user_type = st.sidebar.selectbox(
        "User type", [None, *USER_TYPES], index=0,
        format_func=lambda value: "All user types" if value is None
        else USER_TYPE_LABELS[value])

    if include_demo:
        st.sidebar.warning(
            "Demo records are included. These figures are **not** pilot results.",
            icon="⚠️")

    st.sidebar.caption(
        f"Environment: `{settings.environment}` · database: "
        f"{safe_database_label(settings)}")

    return ReportFilters(start=start, end=end, organisation_id=organisation_id,
                         user_type=user_type, include_demo=include_demo)


# Panels
def _headline(filters: ReportFilters) -> dict:
    metrics = analytics.summary_metrics(filters)

    row = st.columns(4)
    row[0].metric("Organisations", metrics["organisations"],
                  help="Organisations with any recorded activity in this period.")
    row[1].metric("Unique users", metrics["unique_users"],
                  help="Anonymous participants who completed at least one simulation.")
    row[2].metric("Active users", metrics["active_users"],
                  help="Participants who completed a simulation during the selected "
                       "period.")
    row[3].metric("Returning users", metrics["returning_users"],
                  help="Participants who used the platform on at least two separate "
                       "dates.")

    row = st.columns(4)
    row[0].metric("Completed runs", metrics["completed_runs"],
                  help="Simulations that finished successfully and displayed results.")
    row[1].metric("Failed runs", metrics["failed_runs"],
                  help=f"Failure rate {metrics['failure_rate']:.1%}.")
    row[2].metric("Simulations per user", f"{metrics['runs_per_user']:.1f}")
    row[3].metric("Return rate", f"{metrics['return_rate']:.1%}",
                  help="Returning users as a fraction of unique users.")

    row = st.columns(4)
    row[0].metric("Measured-impact users", metrics["measured_impact_users"],
                  help="Participants who completed both the pre- and the "
                       "post-assessment.")
    row[1].metric("Feedback submissions", metrics["feedback_submissions"])
    return metrics


def _table(title: str, rows: list[dict], empty_message: str,
           key: str) -> pd.DataFrame:
    st.subheader(title)
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.caption(empty_message)
        return frame
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(f"Download {title.lower()} (CSV)",
                       frame.to_csv(index=False).encode("utf-8"),
                       f"{key}.csv", "text/csv", key=f"dl_{key}")
    return frame


def _usage_tab(filters: ReportFilters) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    by_date = analytics.usage_by_date(filters)
    st.subheader("Usage by date")
    if by_date:
        frame = pd.DataFrame(by_date).set_index("date")
        st.line_chart(frame[["completed_runs", "participants"]])
        st.dataframe(frame.reset_index(), width="stretch", hide_index=True)
        tables["usage_by_date"] = pd.DataFrame(by_date)
    else:
        st.caption("No activity recorded in this period.")

    tables["usage_by_organisation"] = _table(
        "Participating organisations", analytics.usage_by_organisation(filters),
        "No organisations have recorded activity in this period.",
        "usage_by_organisation")
    st.caption(
        "Codes that were never registered by a coordinator all appear under "
        "\"Unregistered pilot code\". Use the cohort table below to tell them apart.")
    tables["usage_by_pilot_code"] = _table(
        "Cohorts (by pilot code)", analytics.usage_by_pilot_code(filters),
        "No pilot codes have recorded activity in this period.",
        "usage_by_pilot_code")
    tables["usage_by_user_type"] = _table(
        "Usage by user type", analytics.usage_by_user_type(filters),
        "No usage recorded.", "usage_by_user_type")
    tables["usage_by_scenario"] = _table(
        "Usage by scenario", analytics.usage_by_scenario(filters),
        "No scenarios have been run in this period.", "usage_by_scenario")
    tables["usage_by_feature"] = _table(
        "Usage by feature category", analytics.usage_by_feature(filters),
        "No feature usage recorded.", "usage_by_feature")
    tables["simulations_per_user"] = _table(
        "Simulations per user", analytics.simulations_per_user(filters),
        "No participant has completed a simulation in this period.",
        "simulations_per_user")
    return tables


def _learning_tab(filters: ReportFilters) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    st.subheader("Pre- and post-assessment change")
    st.caption(
        "Only participants who completed **both** assessments appear here. The same "
        "instrument is used before and after, so part of any gain comes from having "
        "seen the questions already — treat these as indicative, not as a controlled "
        "measurement.")

    rows = analytics.assessment_changes(filters)
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.caption("No participant has completed both assessments in this period.")
        return tables

    columns = st.columns(4)
    columns[0].metric("Participants measured", len(frame))
    columns[1].metric("Mean before", f"{frame['pre_score'].mean():.2f}")
    columns[2].metric("Mean after", f"{frame['post_score'].mean():.2f}")
    columns[3].metric("Mean change", f"{frame['change'].mean():+.2f}")

    improved = int((frame["change"] > 0).sum())
    unchanged = int((frame["change"] == 0).sum())
    worse = int((frame["change"] < 0).sum())
    st.caption(f"Improved: {improved} · unchanged: {unchanged} · lower: {worse}")

    if "pre_confidence" in frame and frame["pre_confidence"].notna().any():
        st.caption(
            f"Mean self-rated confidence: {frame['pre_confidence'].mean():.2f} before, "
            f"{frame['post_confidence'].mean():.2f} after (1–5).")

    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button("Download assessment changes (CSV)",
                       frame.to_csv(index=False).encode("utf-8"),
                       "assessment_changes.csv", "text/csv", key="dl_assessments")
    tables["assessment_changes"] = frame
    return tables


def _feedback_tab(filters: ReportFilters) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    summary = analytics.feedback_ratings_summary(filters)

    st.subheader("Ratings")
    if not summary or not summary.get("responses"):
        st.caption("No feedback has been submitted in this period.")
        return tables

    def show(value):
        return f"{value:.2f}" if value is not None else "—"

    columns = st.columns(4)
    columns[0].metric("Responses", int(summary["responses"]))
    columns[1].metric("Usefulness (1–5)", show(summary["mean_usefulness"]))
    columns[2].metric("Usability (1–5)", show(summary["mean_usability"]))
    columns[3].metric("Would recommend (0–10)", show(summary["mean_recommend"]))

    st.markdown("**Engineers and researchers specifically**")
    columns = st.columns(3)
    columns[0].metric("Responses", int(summary.get("professional_responses") or 0))
    columns[1].metric("Professional usefulness",
                      show(summary.get("professional_mean_usefulness")))
    columns[2].metric("Professional usability",
                      show(summary.get("professional_mean_usability")))

    rows = analytics.feedback_rows(filters)
    frame = pd.DataFrame(rows)
    tables["feedback"] = frame

    st.subheader("Qualitative feedback")
    for row in rows:
        with st.container(border=True):
            st.caption(
                f"{row['submitted_date']} · {USER_TYPE_LABELS.get(row['user_type'], '—')} "
                f"· {row.get('organisation') or '—'} · participant "
                f"{row.get('participant_ref') or '—'}")
            for label, key in (("Most useful", "most_useful"),
                               ("Would change", "improvement"),
                               ("Bug report", "bug_report"),
                               ("Feature request", "feature_request")):
                if row.get(key):
                    st.markdown(f"**{label}:** {row[key]}")

    st.download_button("Download all feedback (CSV)",
                       frame.to_csv(index=False).encode("utf-8"),
                       "feedback.csv", "text/csv", key="dl_feedback")

    st.subheader("Bugs and feature requests only")
    issues = pd.DataFrame(analytics.bug_and_feature_rows(filters))
    if issues.empty:
        st.caption("No bugs or feature requests reported in this period.")
    else:
        st.dataframe(
            issues[[c for c in ("submitted_date", "user_type", "organisation",
                                "bug_report", "feature_request") if c in issues]],
            width="stretch", hide_index=True)
        st.download_button("Download bugs and requests (CSV)",
                           issues.to_csv(index=False).encode("utf-8"),
                           "bugs_and_requests.csv", "text/csv", key="dl_issues")
        tables["bugs_and_requests"] = issues
    return tables


def _system_tab(filters: ReportFilters) -> None:
    settings = get_settings()
    st.subheader("Deployment")

    health = healthcheck()
    if health["ok"]:
        st.success(f"Database reachable and migrated — schema version "
                   f"{health['schema_version']}. {health['database']}", icon="✅")
    else:
        st.error(f"Database problem: {health['error'] or health['missing_tables']}",
                 icon="🚫")

    warnings = settings.deployment_warnings()
    if warnings:
        st.subheader("Configuration warnings")
        for warning in warnings:
            st.warning(warning, icon="⚠️")
    else:
        st.caption("No configuration warnings.")

    st.subheader("Settings in effect")
    st.dataframe(pd.DataFrame([
        {"setting": "Environment", "value": settings.environment},
        {"setting": "Records marked as demo data", "value": str(settings.is_demo_data)},
        {"setting": "Database", "value": safe_database_label(settings)},
        {"setting": "Raw parameter logging", "value": str(settings.log_raw_parameters)},
        {"setting": "Retention (days)", "value": str(settings.retention_days)},
        {"setting": "Max runs per session", "value": str(settings.max_runs_per_session)},
        {"setting": "Max runs per minute", "value": str(settings.max_runs_per_minute)},
        {"setting": "Max simulation steps", "value": str(settings.max_simulation_steps)},
        {"setting": "Platform version", "value": settings.app_version},
    ]), width="stretch", hide_index=True)
    st.caption("Secrets are never displayed. Database passwords are stripped from "
               "every label shown here and from every log line.")

    st.subheader("Recent events")
    st.caption("Most recent 200 events matching the current filters.")
    events = pd.DataFrame(analytics.recent_events(filters, limit=200))
    if events.empty:
        st.caption("No events recorded.")
    else:
        st.dataframe(events, width="stretch", hide_index=True)


def _bundle_export(tables: dict[str, pd.DataFrame], filters: ReportFilters) -> None:
    """One ZIP containing every aggregate table plus the metric definitions."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            if frame is not None and not frame.empty:
                archive.writestr(f"{name}.csv", frame.to_csv(index=False))
        archive.writestr("METRIC_DEFINITIONS.txt", _definitions_text(filters))
    buffer.seek(0)

    st.download_button(
        "Download every table (ZIP)", buffer.getvalue(),
        f"pilot_export_{date.today().isoformat()}.zip", "application/zip",
        key="dl_bundle", type="primary")


def _definitions_text(filters: ReportFilters) -> str:
    return "\n".join([
        "Pilot metric definitions",
        "========================",
        "",
        f"Exported          : {date.today().isoformat()}",
        f"Period            : {filters.start or 'all time'} to {filters.end or 'today'}",
        f"Organisation      : {filters.organisation_id or 'all'}",
        f"User type         : {filters.user_type or 'all'}",
        f"Demo records      : {'INCLUDED - NOT PILOT DATA' if filters.include_demo else 'excluded'}",
        "",
        "Unique user          an anonymous participant who has completed at least",
        "                     one simulation (not limited to the reporting period)",
        "Active user          an anonymous participant who completed a simulation",
        "                     during the selected reporting period",
        "Returning user       an anonymous participant who used the platform on at",
        "                     least two separate dates",
        "Completed run        a simulation that finished successfully and displayed",
        "                     results (one simulation_completed event)",
        "Failed run           a simulation that raised an error (simulation_failed).",
        "                     Runs rejected by input validation are recorded",
        "                     separately as simulation_rejected and are NOT counted",
        "                     as failures",
        "Measured-impact      a participant who completed both the pre- and the",
        "  user               post-assessment for the same instrument",
        "Return rate          returning users / unique users",
        "Simulations per user completed runs / unique users",
        "",
        "All figures are anonymous and aggregate. No participant can be identified",
        "from this export: the database stores only a one-way hash of the",
        "participant code, and holds no name, email address, age, IP address or",
        "browser information.",
        "",
    ])


# Entry point
def admin_dashboard() -> None:
    if not st.session_state.get("admin_authenticated"):
        _sign_in()
        return

    settings = get_settings()
    header = st.columns([4, 1])
    with header[0]:
        st.title("Pilot administration")
        st.caption("Anonymous aggregate reporting. No participant can be identified "
                   "from anything on this page.")
    with header[1]:
        if st.button("Sign out", width="stretch"):
            st.session_state.admin_authenticated = False
            st.rerun()

    if not settings.is_production:
        st.info(
            f"This deployment is running as `{settings.environment}`, so every "
            "record it creates is flagged as demo data and excluded from reports "
            "by default. Set `PILOT_ENV=production` for a real pilot.", icon="ℹ️")

    filters = _filters()
    _headline(filters)

    tabs = st.tabs(["Usage", "Learning impact", "Feedback", "Export", "System"])
    tables: dict[str, pd.DataFrame] = {}

    with tabs[0]:
        tables.update(_usage_tab(filters))
    with tabs[1]:
        tables.update(_learning_tab(filters))
    with tabs[2]:
        tables.update(_feedback_tab(filters))
    with tabs[3]:
        st.subheader("Export everything")
        st.caption(
            "One ZIP with every table above as CSV, plus a text file recording the "
            "metric definitions and the filters that produced them. Anonymous and "
            "aggregate throughout.")
        _bundle_export(tables, filters)
    with tabs[4]:
        _system_tab(filters)


__all__ = ["admin_dashboard"]
