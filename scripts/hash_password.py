"""Generate a PBKDF2 hash for the administrator password.

Put the output in ``PILOT_ADMIN_PASSWORD_HASH`` so the plaintext never exists
in the deployment environment.

Usage::

    python scripts/hash_password.py                 # prompts, input hidden
    python scripts/hash_password.py --stdin         # read from a pipe

The password is never echoed, never written to a file, and never logged.
"""

import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilot.auth import PBKDF2_ITERATIONS, hash_password, verify_password  # noqa: E402

MIN_LENGTH = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdin", action="store_true",
                        help="read the password from standard input instead of "
                             "prompting (for scripting)")
    parser.add_argument("--iterations", type=int, default=PBKDF2_ITERATIONS)
    args = parser.parse_args()

    if args.stdin:
        password = sys.stdin.readline().rstrip("\n")
    else:
        password = getpass.getpass("Administrator password: ")
        confirmation = getpass.getpass("Confirm: ")
        if password != confirmation:
            print("Passwords do not match.", file=sys.stderr)
            return 1

    if not password:
        print("Empty password refused.", file=sys.stderr)
        return 1
    if len(password) < MIN_LENGTH:
        print(f"Refusing a password shorter than {MIN_LENGTH} characters. This "
              "unlocks the whole pilot dataset.", file=sys.stderr)
        return 1

    encoded = hash_password(password, iterations=args.iterations)
    assert verify_password(password, encoded), "self-check failed"

    print()
    print("Add this to your environment or deployment secrets:")
    print()
    print(f"PILOT_ADMIN_PASSWORD_HASH={encoded}")
    print()
    print("Also set PILOT_ADMIN_USERNAME. Do not commit either value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
