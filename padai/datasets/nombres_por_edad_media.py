from pathlib import Path
import pandas as pd


def _excel_path() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "nombres_por_edad_media.xlsx"


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

    assert pd.api.types.is_string_dtype(df["name"]), "name column is not string-dtype"
    assert pd.api.types.is_string_dtype(df["gender"]), "gender column is not string-dtype"
    assert pd.api.types.is_integer_dtype(df["frequency"]), "frequency column is not int-dtype"

    return df[["name", "gender", "frequency", ]]


def get_nombres_por_edad_media_dataframe():
    hombres = _read_sheet("Hombres", "M")
    mujeres = _read_sheet("Mujeres", "F")

    df = pd.concat([hombres, mujeres], ignore_index=True)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)

    return df

