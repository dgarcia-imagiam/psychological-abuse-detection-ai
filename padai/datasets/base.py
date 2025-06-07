import random
import pandas as pd
from typing import Dict, Tuple, List
from padai.datasets.nombres_por_edad_media import get_nombres_por_edad_media_dataframe

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