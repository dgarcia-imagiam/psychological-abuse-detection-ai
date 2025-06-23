import sqlite3
from pathlib import Path
import pandas as pd


def get_sqlite_row_count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as conn:
        (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return count


def ensure_db(path: Path, sql_create: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(sql_create)
    conn.commit()
    return conn


def row_to_series(row: tuple, cursor: sqlite3.Cursor) -> pd.Series:
    columns = [col[0] for col in cursor.description]
    return pd.Series(row, index=columns, name=row[0])
