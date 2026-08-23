"""Create or migrate the analytics database.

The application migrates on startup as well, so this is for deploying
deliberately -- checking a connection string works before going live, or
running migrations as a separate release step.

Usage::

    python scripts/init_db.py            # migrate, then report
    python scripts/init_db.py --check    # report only, change nothing
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilot.config import get_settings  # noqa: E402
from pilot.db import (  # noqa: E402
    MIGRATIONS,
    get_engine,
    healthcheck,
    run_migrations,
    safe_database_label,
    schema_version,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report status without applying migrations")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Environment : {settings.environment}")
    print(f"Database    : {safe_database_label(settings)}")
    if settings.database_url_is_default:
        print("              (local SQLite fallback -- set DATABASE_URL for a "
              "hosted pilot)")

    engine = get_engine(settings)

    if args.check:
        health = healthcheck(engine)
        print(f"Reachable   : {'yes' if health['error'] is None else 'no'}")
        if health["error"]:
            print(f"Error       : {health['error']}")
            return 1
        print(f"Schema      : version {health['schema_version']} of "
              f"{max(v for v, _, _ in MIGRATIONS)}")
        if health["missing_tables"]:
            print(f"Missing     : {', '.join(health['missing_tables'])}")
            print("\nRun without --check to create them.")
            return 1
        print("Status      : up to date")
        return 0

    try:
        applied = run_migrations(engine)
    except Exception as exc:  # noqa: BLE001 - report cleanly, never traceback
        print(f"\nMigration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check DATABASE_URL and that the database is reachable.",
              file=sys.stderr)
        return 1

    if applied:
        print(f"Applied     : migrations {applied}")
    else:
        print("Applied     : nothing, already up to date")
    print(f"Schema      : version {schema_version(engine)}")

    warnings = settings.deployment_warnings()
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  ! {warning}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
