from pathlib import Path
import sqlite3
import pandas as pd
from padai.datasets.base import get_names_pool, NameFrequencyCache, build_name_token_dict_many
from padai.config.language import Language
from typing import Dict, Tuple, Iterable, Optional
from padai.utils.text import substitute_placeholders
from padai.config.settings import settings
import time
from filelock import FileLock
from padai.utils.sqlite import get_sqlite_row_count, ensure_db, row_to_series
from padai.utils.parquet import get_parquet_row_count


def _db_path() -> Path:
    db_file = Path(__file__).resolve().parents[2] / "datasets" / "psychological_abuse.sqlite"

    if not db_file.exists():
        raise FileNotFoundError(f"SQLite file not found: {db_file}")

    return db_file


def _assert_communications_df(df: pd.DataFrame) -> pd.DataFrame:
    assert pd.api.types.is_integer_dtype(df.index)
    assert pd.api.types.is_string_dtype(df["text"])
    assert pd.api.types.is_string_dtype(df["context"])
    assert pd.api.types.is_string_dtype(df["language"])
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])

    return df


def get_raw_communications_df() -> pd.DataFrame:
    db_file = _db_path()

    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("SELECT * FROM communications", conn, index_col="id", parse_dates=["created_at"])

    df["text"] = df["text"].astype("string")
    df["context"] = df["context"].astype("string")
    df["language"] = df["language"].astype("string")
    df["source_id"] = df["source_id"].astype("string")

    df["translation_of"] = df["translation_of"].astype("Int64")

    return _assert_communications_df(df)


def get_communications_df_no_cache() -> pd.DataFrame:
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


def get_communications_df(*, ttl: int | None = None) -> pd.DataFrame:
    cache_path: Path = settings.path_in_cache("datasets/psychological_abuse_communications.parquet")
    lock_path = cache_path.with_suffix(".lock")

    def cache_is_fresh() -> bool:
        if not cache_path.exists():
            return False
        if ttl == 0:
            return False
        if ttl is not None:
            age = time.time() - cache_path.stat().st_mtime
            if age >= ttl:
                return False
        try:
            return get_parquet_row_count(cache_path) == get_sqlite_row_count(_db_path(), "communications")
        except Exception:
            return False

    if cache_is_fresh():
        return _assert_communications_df(pd.read_parquet(cache_path))

    with FileLock(lock_path):
        if cache_is_fresh():
            return _assert_communications_df(pd.read_parquet(cache_path))

        df = get_communications_df_no_cache()

        tmp_path = cache_path.with_suffix(".tmp")
        df.to_parquet(tmp_path)
        tmp_path.replace(cache_path)

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


def get_or_create_communication(
    id_: int,
    df: pd.DataFrame,
) -> pd.Series:

    _SQL_CREATE = """
        CREATE TABLE IF NOT EXISTS communications (
            id            INTEGER PRIMARY KEY,
            text          TEXT        NOT NULL,
            context       TEXT,
            language      TEXT        NOT NULL
        );
    """

    _SQL_SELECT = """
        SELECT * FROM communications WHERE id = ?
    """

    _SQL_INSERT = """
        INSERT INTO communications (id, text, context, language)
        VALUES (:id, :text, :context, :language)
    """

    path: Path = settings.path_in_home("db/psychological_abuse/communications.sqlite")
    conn = ensure_db(path, _SQL_CREATE)
    cur = conn.cursor()

    cur.execute(_SQL_SELECT, (id_,))
    row = cur.fetchone()

    if row is not None:
        return row_to_series(row, cur)

    try:
        record = df.loc[id_]
    except KeyError as exc:
        raise KeyError(f"id {id_} not found in DataFrame or SQLite db") from exc

    payload = {
        "id": id_,
        "text": str(record["text"]),
        "context": str(record["context"]),
        "language": str(record["language"]),
    }

    cur.execute(_SQL_INSERT, payload)
    conn.commit()

    return pd.Series(payload, name=id_)
