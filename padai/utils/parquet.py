from pathlib import Path
import pyarrow.parquet as pq


def get_parquet_row_count(parquet_path: Path) -> int:
    return pq.ParquetFile(parquet_path).metadata.num_rows
