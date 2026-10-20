"""Analytics access for the API.

The implementation lives in ``pilot.analytics`` and is shared with the legacy
Streamlit application, so both record identical events against identical metric
definitions. This package is the seam for that: if analytics
storage is ever swapped (local SQLite in development, PostgreSQL or Supabase in
production), it is configured behind this boundary rather than at call sites.
"""

from app import bootstrap  # noqa: F401

from pilot.analytics import (  # noqa: E402,F401
    EventType,
    ParticipantContext,
    ReportFilters,
    record_assessment,
    record_event,
    record_feedback,
    set_track,
    start_session,
    summary_metrics,
)
