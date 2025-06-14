"""
export_dataset – write the psychological-abuse communications dataset
to a Word document.

Examples
--------
# 1)  Raw dataset → default filename in HOME
python -m padai.commands.export_dataset --raw

# 2)  Clean dataset → default filename in HOME
python -m padai.commands.export_dataset --no-raw

# 3)  Raw dataset → explicit path
python -m padai.commands.export_dataset --raw /tmp/raw.docx
"""

import padai.config.bootstrap  # noqa: F401 always first import in main entry points

import argparse
from pathlib import Path
import sys

from padai.datasets.psychological_abuse import (
    get_raw_communications_df,
    get_communications_df,
)
from padai.utils.pandas import write_doc


RAW_NAME = "psychological_abuse_raw_communications_dataset.docx"
CLEAN_NAME = "psychological_abuse_communications_dataset.docx"


def export_dataset(path: Path, raw: bool) -> None:
    """Fetch the required dataframe and export it to *path*."""
    df = get_raw_communications_df() if raw else get_communications_df()
    write_doc(df, path)
    print(f"✔ {'Raw' if raw else 'Clean'} dataset exported to {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the psychological-abuse communications dataset "
            "to a Word document."
        )
    )

    # Boolean flag: --raw (default) / --no-raw
    parser.add_argument(
        "--raw",
        dest="raw",
        action=argparse.BooleanOptionalAction,   # Python ≥3.9
        default=True,
        help="Export the *raw* dataset (default). "
             "Use --no-raw to export the cleaned dataset.",
    )

    # Optional destination file
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Destination .docx file. "
            "If omitted, a filename matching the chosen dataset is created "
            "in your home directory."
        ),
    )
    parser.add_argument(
        "--path",
        dest="path_kw",
        metavar="PATH",
        help="Destination file (overrides positional argument if both given).",
    )

    ns = parser.parse_args(argv)

    # Decide default filename based on --raw choice
    default_name = RAW_NAME if ns.raw else CLEAN_NAME
    default_path = Path.home() / default_name

    # Resolve precedence: --path beats positional, otherwise default
    raw_path = ns.path_kw or ns.path or default_path
    path = Path(raw_path).expanduser().resolve()

    try:
        export_dataset(path, ns.raw)
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
