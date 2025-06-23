from pathlib import Path
import pandas as pd
from padai.config.settings import settings
import time
from filelock import FileLock


def _excel_path() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "nombres_por_edad_media.xlsx"


def _assert_df(df: pd.DataFrame) -> pd.DataFrame:
    assert pd.api.types.is_string_dtype(df["name"]), "name column is not string-dtype"
    assert pd.api.types.is_string_dtype(df["gender"]), "gender column is not string-dtype"
    assert pd.api.types.is_integer_dtype(df["frequency"]), "frequency column is not int-dtype"

    return df


def _read_sheet(sheet: str, gender: str) -> pd.DataFrame:
    df = pd.read_excel(_excel_path(), sheet_name=sheet, header=6)

    df = df.rename(
        columns={
            "Nombre": "name",
            "Frecuencia": "frequency",
            "Edad Media (*)": "average_age",
        }
    )

    df["name"] = df["name"].str.title().str.strip().astype("string")
    df["gender"] = gender

    df = df[["name", "gender", "frequency", ]]

    return _assert_df(df)


def get_nombres_por_edad_media_dataframe_no_cache():
    hombres = _read_sheet("Hombres", "M")
    mujeres = _read_sheet("Mujeres", "F")

    df = pd.concat([hombres, mujeres], ignore_index=True)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)

    return df


def get_nombres_por_edad_media_dataframe(*, ttl: int | None = None) -> pd.DataFrame:
    cache_path: Path = settings.path_in_cache("datasets/nombres_por_edad_media.parquet")
    lock_path = cache_path.with_suffix(".lock")

    def cache_is_fresh() -> bool:
        if not cache_path.exists():
            return False
        if ttl is None:
            return True
        if ttl == 0:
            return False
        return (time.time() - cache_path.stat().st_mtime) < ttl

    if cache_is_fresh():
        return _assert_df(pd.read_parquet(cache_path))

    with FileLock(lock_path):
        if cache_is_fresh():
            return _assert_df(pd.read_parquet(cache_path))

        df = get_nombres_por_edad_media_dataframe_no_cache()

        tmp_path = cache_path.with_suffix(".tmp")
        df.to_parquet(tmp_path)
        tmp_path.replace(cache_path)

        return df
