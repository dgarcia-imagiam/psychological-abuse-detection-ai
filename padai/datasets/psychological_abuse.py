from pathlib import Path
import sqlite3
import pandas as pd
from padai.datasets.base import get_names_pool, get_random_name, NameFrequencyCache, build_name_token_dict_many
from padai.config.language import Language
from typing import Dict, Tuple, Iterable, Optional
from padai.utils.text import substitute_placeholders


def _db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "psychological_abuse.sqlite"


def get_raw_communications_df() -> pd.DataFrame:
    db_file = _db_path()
    if not db_file.exists():
        raise FileNotFoundError(f"SQLite file not found: {db_file}")

    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("SELECT * FROM communications", conn, index_col="id", parse_dates=["created_at"])

    df["text"] = df["text"].astype("string")
    df["context"] = df["context"].astype("string")
    df["language"] = df["language"].astype("string")
    df["source_id"] = df["source_id"].astype("string")

    df["translation_of"] = df["translation_of"].astype("Int64")

    assert pd.api.types.is_integer_dtype(df.index)
    assert pd.api.types.is_string_dtype(df["text"])
    assert pd.api.types.is_string_dtype(df["context"])
    assert pd.api.types.is_string_dtype(df["language"])
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    return df


def get_communications_df() -> pd.DataFrame:
    df = get_raw_communications_df()

    # 1) Names pools:   {"es": es_names_df, "en": en_names_df, …}
    names_pool: Dict[str, pd.DataFrame] = get_names_pool()

    # 2) Per-language cache – either your real object or a plain dict
    caches: Dict[str, NameFrequencyCache] = {
        lang: {} for lang in names_pool
    }

    # 3) Row-wise transformation ------------------------------------------------
    def _process_row(row: pd.Series) -> pd.Series:
        lang = row["language"]
        names_df = names_pool[lang]
        cache = caches[lang]

        # Collect all strings in which placeholders might exist
        texts: Iterable[str] = (
            row["text"],
            row["context"] if pd.notna(row["context"]) else "",
        )

        # Build / extend the mapping for *this* row only
        mapping = build_name_token_dict_many(
            texts=texts,
            df=names_df,
            cache=cache,
        )

        # Substitute in both columns
        row["text"] = substitute_placeholders(row["text"], mapping)
        if pd.notna(row["context"]):
            row["context"] = substitute_placeholders(row["context"], mapping)

        return row

    # Apply the transformation
    df = df.apply(_process_row, axis=1)

    df = df[["text", "context", "language", "source_id", "translation_of", "created_at"]]

    return df


def get_communications_sample(
    df: pd.DataFrame,
    *,
    language: Optional[Language] = None,
    id_: Optional[int] = None,
) -> Tuple[str, str]:

    subset = df
    if language is not None:
        subset = subset[subset["language"] == language.value]

    if subset.empty:
        raise ValueError(
            "No rows match the requested criteria "
            f"(language={language!r}, id={id_!r})"
        )

    if id_ is not None:
        try:
            row = subset.loc[id_]
        except KeyError as exc:
            raise ValueError(
                f"No row with id={id_!r} "
                f"in the dataframe (after language filter)."
            ) from exc
    else:
        row = subset.iloc[0]

    return row["text"], row["context"]
