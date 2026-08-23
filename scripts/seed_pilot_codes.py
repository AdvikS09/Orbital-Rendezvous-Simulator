"""Register participating organisations and their pilot codes.

Registering a code is optional -- unregistered codes still work and are grouped
under "Unregistered pilot code". Registering gives an organisation a proper
name in the dashboard and lets you filter reports by it. Setting
``PILOT_REQUIRE_KNOWN_CODE=true`` makes registration mandatory.

Usage::

    python scripts/seed_pilot_codes.py --list

    python scripts/seed_pilot_codes.py \\
        --org-code SCHOOL-A --org-name "Example High School" \\
        --org-type school --code SCHOOL-A-Y12 --label "Year 12 physics"

    python scripts/seed_pilot_codes.py --from-csv organisations.csv

CSV columns: ``org_code, org_name, org_type, pilot_code, label, cohort``.

Only register organisations that have actually agreed to take part. Inventing
institutional relationships would corrupt the pilot's own reporting.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select  # noqa: E402

from pilot.config import get_settings  # noqa: E402
from pilot.db import get_engine, run_migrations, utcnow  # noqa: E402
from pilot.models import organisations, pilot_codes  # noqa: E402

ORG_TYPES = ("school", "university", "company", "research_institute", "other")


def _upsert_organisation(connection, code, name, org_type, is_demo) -> int:
    code = code.strip().upper()
    existing = connection.execute(
        select(organisations.c.id).where(organisations.c.code == code)
    ).scalar_one_or_none()
    if existing is not None:
        connection.execute(organisations.update()
                           .where(organisations.c.id == existing)
                           .values(name=name, org_type=org_type))
        return int(existing)
    return int(connection.execute(organisations.insert().values(
        code=code, name=name, org_type=org_type, is_demo=is_demo,
        active=True, created_at=utcnow(),
    ).returning(organisations.c.id)).scalar_one())


def _upsert_code(connection, organisation_id, code, label, cohort, is_demo) -> str:
    code = code.strip().upper()
    existing = connection.execute(
        select(pilot_codes.c.id).where(pilot_codes.c.code == code)
    ).scalar_one_or_none()
    if existing is not None:
        connection.execute(pilot_codes.update()
                           .where(pilot_codes.c.id == existing)
                           .values(label=label, cohort=cohort,
                                   organisation_id=organisation_id, active=True))
        return "updated"
    connection.execute(pilot_codes.insert().values(
        organisation_id=organisation_id, code=code, label=label, cohort=cohort,
        active=True, is_demo=is_demo, created_at=utcnow()))
    return "created"


def _list(engine) -> int:
    with engine.connect() as connection:
        rows = connection.execute(
            select(organisations.c.code.label("org_code"),
                   organisations.c.name, organisations.c.org_type,
                   organisations.c.is_demo,
                   pilot_codes.c.code.label("pilot_code"),
                   pilot_codes.c.label, pilot_codes.c.cohort,
                   pilot_codes.c.active)
            .select_from(organisations.outerjoin(
                pilot_codes, pilot_codes.c.organisation_id == organisations.c.id))
            .order_by(organisations.c.name)).mappings().all()

    if not rows:
        print("No organisations registered.")
        return 0

    print(f"{'ORG CODE':<18} {'ORGANISATION':<32} {'PILOT CODE':<20} LABEL")
    print("-" * 100)
    for row in rows:
        demo = " [demo]" if row["is_demo"] else ""
        print(f"{row['org_code']:<18} {(row['name'] or '')[:31]:<32} "
              f"{(row['pilot_code'] or '-'):<20} {(row['label'] or '')}{demo}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show what is registered")
    parser.add_argument("--org-code")
    parser.add_argument("--org-name")
    parser.add_argument("--org-type", choices=ORG_TYPES, default="other")
    parser.add_argument("--code", help="the pilot code participants will type")
    parser.add_argument("--label", default=None, help="e.g. 'Year 12 physics'")
    parser.add_argument("--cohort", default=None)
    parser.add_argument("--from-csv", dest="from_csv")
    args = parser.parse_args()

    settings = get_settings()
    engine = get_engine(settings)
    run_migrations(engine)

    if args.list:
        return _list(engine)

    if settings.is_demo_data:
        print(f"NOTE: PILOT_ENV is '{settings.environment}', so anything created "
              "here is flagged as demo data and excluded from reports by default.")

    if args.from_csv:
        with open(args.from_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print("CSV is empty.", file=sys.stderr)
            return 1
        with engine.begin() as connection:
            for row in rows:
                organisation_id = _upsert_organisation(
                    connection, row["org_code"], row["org_name"],
                    row.get("org_type", "other"), settings.is_demo_data)
                action = _upsert_code(
                    connection, organisation_id, row["pilot_code"],
                    row.get("label"), row.get("cohort"), settings.is_demo_data)
                print(f"  {action}: {row['pilot_code']} -> {row['org_name']}")
        print(f"\nProcessed {len(rows)} row(s).")
        return 0

    if not (args.org_code and args.org_name and args.code):
        parser.error("need --org-code, --org-name and --code (or --from-csv, or --list)")

    with engine.begin() as connection:
        organisation_id = _upsert_organisation(
            connection, args.org_code, args.org_name, args.org_type,
            settings.is_demo_data)
        action = _upsert_code(connection, organisation_id, args.code, args.label,
                              args.cohort, settings.is_demo_data)

    print(f"{action}: pilot code {args.code.upper()} for {args.org_name}")
    print(f"\nParticipants enter '{args.code.upper()}' as their organisation or "
          "pilot code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
