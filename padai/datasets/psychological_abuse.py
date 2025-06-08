from pathlib import Path
import sqlite3
import pandas as pd
from padai.datasets.base import get_names_pool, get_random_name
import re
from padai.config.language import Language


def _db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "psychological_abuse.sqlite"


def get_raw_communications_df() -> pd.DataFrame:
    db_file = _db_path()
    if not db_file.exists():
        raise FileNotFoundError(f"SQLite file not found: {db_file}")

    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("SELECT * FROM communications", conn, parse_dates=["created_at"])

    df["text"] = df["text"].astype("string")
    df["language"] = df["language"].astype("string")
    df["source_id"] = df["source_id"].astype("string")

    df["translation_of"] = df["translation_of"].astype("Int64")

    assert pd.api.types.is_integer_dtype(df["id"])
    assert pd.api.types.is_string_dtype(df["text"])
    assert pd.api.types.is_string_dtype(df["language"])
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    return df


PLACEHOLDER_RE = re.compile(r"\{f_name}|\{m_name}")


def get_communications_df() -> pd.DataFrame:
    df = get_raw_communications_df()

    names_pool = get_names_pool()

    caches = {lang: {} for lang in names_pool}

    def _replace(row: pd.Series) -> str:
        lang = row["language"]
        names_df = names_pool[lang]
        cache = caches.setdefault(lang, {})

        female = get_random_name(names_df, "F", cache)
        male = get_random_name(names_df, "M", cache)

        def _sub(match: re.Match) -> str:
            return female if match.group(0) == "{f_name}" else male

        return PLACEHOLDER_RE.sub(_sub, row["text"])

    df["text"] = df.apply(_replace, axis=1)
    return df


def get_communications_text_sample(df: pd.DataFrame, language: Language) -> str:
    subset = df[df["language"] == language.value]
    if subset.empty:
        raise ValueError(f"No rows with language = {language!r}")

    return subset["text"].iat[0]
