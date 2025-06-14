"""
text_uuid

Usage
-----
    python -m padai.commands.text_uuid some_text
"""

import padai.config.bootstrap  # noqa: F401 always first import in main entry points

import uuid
import argparse
from padai.config.settings import settings


def text_uuid(text: str, secret: str) -> uuid.UUID:
    """
    Deterministic UUID for *text*, salted by *secret*.

        namespace = UUIDv5(UUID.NAMESPACE_DNS, secret)
        result    = UUIDv5(namespace, text)
    """
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, secret)
    return uuid.uuid5(namespace, text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic UUID from TEXT using settings.secret"
    )
    parser.add_argument("text", help="Text to fingerprint")
    args = parser.parse_args(argv)

    uuid_out = text_uuid(args.text, settings.secret.get_secret_value())
    print(f"{args.text} --> {uuid_out}")


if __name__ == "__main__":
    main()
