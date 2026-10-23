"""Delete pilot data past the stated retention period.

The consent page promises participants that data is kept for at most
``PILOT_RETENTION_DAYS`` days. Nothing enforces that automatically -- this
script does, and it needs to be run on a schedule for the promise to be true.

Usage::

    python scripts/purge_expired.py --dry-run    # show what would go
    python scripts/purge_expired.py --confirm    # actually delete

Deletion is permanent and cannot be undone; ``--confirm`` is required.
Participants whose every record has expired are removed too, so nothing is left
that could be correlated across a longer window than the policy allows.
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import delete, func, select  # noqa: E402

from pilot.config import get_settings  # noqa: E402
from pilot.db import get_engine, safe_database_label  # noqa: E402
from pilot.models import (  # noqa: E402
    assessments,
    events,
    feedback,
    participants,
    sessions,
)

TARGETS = (
    (events, events.c.occurred_date),
    (assessments, assessments.c.completed_date),
    (feedback, feedback.c.submitted_date),
    (sessions, sessions.c.started_date),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--retention-days", type=int, default=None,
                        help="override PILOT_RETENTION_DAYS for this run")
    args = parser.parse_args()

    if not (args.dry_run or args.confirm):
        parser.error("pass --dry-run to preview or --confirm to delete")

    settings = get_settings()
    retention = args.retention_days or settings.retention_days
    cutoff = date.today() - timedelta(days=retention)
    engine = get_engine(settings)

    print(f"Database  : {safe_database_label(settings)}")
    print(f"Retention : {retention} days")
    print(f"Cutoff    : anything before {cutoff} is expired")
    print()

    with engine.connect() as connection:
        counts = {
            table.name: int(connection.execute(
                select(func.count()).select_from(table).where(column < cutoff)
            ).scalar() or 0)
            for table, column in TARGETS
        }
        stale_participants = [
            row[0] for row in connection.execute(
                select(participants.c.id).where(participants.c.last_seen_date < cutoff)
            ).all()
        ]

    for name, count in counts.items():
        print(f"  {name:<14} {count:>8} expired row(s)")
    print(f"  {'participants':<14} {len(stale_participants):>8} expired row(s)")

    total = sum(counts.values()) + len(stale_participants)
    if total == 0:
        print("\nNothing to delete.")
        return 0

    if args.dry_run:
        print(f"\nDry run: {total} row(s) would be deleted. Re-run with --confirm.")
        return 0

    with engine.begin() as connection:
        for table, column in TARGETS:
            connection.execute(delete(table).where(column < cutoff))
        if stale_participants:
            connection.execute(delete(participants)
                               .where(participants.c.id.in_(stale_participants)))

    print(f"\nDeleted {total} row(s). This cannot be undone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
