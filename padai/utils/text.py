import re
from typing import Dict
import textwrap
import re


def substitute_placeholders(text: str, mapping: Dict[str, str]) -> str:
    """
    Return *text* with every exact key from *mapping* replaced by its value.

    Parameters
    ----------
    text : str
        The original text containing placeholders such as "{name:m}".
    mapping : dict[str, str]
        Keys are the exact substrings to search for, values are the
        substitutions.  Keys may contain any characters, including
        regex metacharacters.

    Notes
    -----
    * All keys are escaped with `re.escape`, so they are treated as
      *literal* strings, not regular‑expression patterns.
    * Replacements are done in a single pass with `re.sub`, which is
      faster than calling `str.replace` for each key if the text is
      large or the dictionary has many entries.
    * If a key is not present in *text* it is simply ignored.

    Returns
    -------
    str
        The text with substitutions applied.
    """
    if not mapping:
        return text                      # fast path – nothing to do

    # Build a single pattern that matches any key.  Using sorted keys of
    # decreasing length prevents rare “prefix” ambiguities such as
    # mapping {"{x}": "A", "{x:1}": "B"}.
    alternation = "|".join(
        re.escape(k) for k in sorted(mapping, key=len, reverse=True)
    )
    pattern = re.compile(alternation)

    # `m.group(0)` is the exact key found; look it up in the mapping.
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def make_label(text: str, width: int = 40) -> str:
    """Collapse whitespace and shorten to `width` chars with …"""
    single_line = " ".join(text.split())        # drop \n, multiple spaces ⇢ single space
    return textwrap.shorten(single_line, width=width, placeholder="…")


def strip_text(text: str | None):
    return (text or "").strip()


def process_response_strip(response: str) -> str:
    return response.strip()


def process_response_reasoning(response: str) -> str:
    return re.sub(r"<(reasoning|analysis|think)>.*?</\1>", "", response, flags=re.DOTALL).strip()


def process_response(response: str) -> str:
    response = process_response_strip(response)
    response = process_response_reasoning(response)
    return response
