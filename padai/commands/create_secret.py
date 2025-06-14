"""
create_secret  –  generate a cryptographically secure secret string.

Usage
-----
    python -m padai.commands.create_secret               # 40-character secret (default)
    python -m padai.commands.create_secret 64            # 64-character secret
"""

import padai.config.bootstrap  # noqa: F401 always first import in main entry points

import argparse
import secrets
import string


# Everything except the double quote
ALPHABET = (
    string.ascii_letters
    + string.digits
    + string.punctuation.replace('"', "")   # ← removes "
)


def generate_secret(length: int = 40) -> str:
    """Return a random string of *length* characters drawn from ALPHABET."""
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a cryptographically secure random secret string."
    )
    parser.add_argument(
        "length",
        nargs="?",
        type=int,
        default=40,
        help="Length of the secret (default: 40)",
    )
    args = parser.parse_args(argv)
    print(generate_secret(args.length))


if __name__ == "__main__":
    main()
