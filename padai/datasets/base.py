import random
import pandas as pd
from typing import Dict, Tuple, List, Optional, Iterable
from padai.datasets.nombres_por_edad_media import get_nombres_por_edad_media_dataframe
import re

NameFrequency = Tuple[List[str], List[int]]
NameFrequencyCache = Dict[str, NameFrequency]


def _get_names_frequencies(df: pd.DataFrame, gender: str) -> NameFrequency:
    subset = df[df["gender"] == gender]
    return subset["name"].tolist(), subset["frequency"].tolist()


def get_random_name(
    df: pd.DataFrame,
    gender: str,
    cache: NameFrequencyCache | None = None,
) -> str:

    if cache is not None and gender in cache:
        names, frequencies = cache[gender]
    else:
        names, frequencies = _get_names_frequencies(df, gender)
        if cache is not None:
            cache[gender] = (names, frequencies)

    return random.choices(names, frequencies, k=1)[0]


def get_names_pool():
    return {
        "es": get_nombres_por_edad_media_dataframe(),
    }


_NAME_TOKEN_RE = re.compile(
    r"""\{                  # opening brace
        name                # literal 'name'
        (?:\:[^{}:\s]+)*    # 0‑many additional tags, each like ':something'
        :(?P<gender>[fm])   # final tag – captures the required f/m
        \}                  # closing brace
    """,
    re.VERBOSE
)


def build_name_token_dict(
    text: str,
    df: pd.DataFrame,
    *,
    base: Optional[Dict[str, str]] = None,
    cache: Optional[NameFrequencyCache] = None,
) -> Dict[str, str]:

    result: Dict[str, str] = dict(base or {})

    for match in _NAME_TOKEN_RE.finditer(text):
        placeholder = match.group(0)
        gender = match.group('gender')

        if placeholder not in result:
            result[placeholder] = get_random_name(df, gender.upper(), cache)

    return result


def build_name_token_dict_many(
    texts: Iterable[str],
    df: pd.DataFrame,
    *,
    base: Optional[Dict[str, str]] = None,
    cache: Optional[NameFrequencyCache] = None,
) -> Dict[str, str]:

    result: Dict[str, str] = dict(base or {})

    for text in texts:
        result = build_name_token_dict(text, df, base=result, cache=cache)

    return result
