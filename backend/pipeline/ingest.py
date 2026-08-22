import csv
import io
import zipfile
from pathlib import Path

REQUIRED_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "asset_name",
    "asset_type",
    "region",
)


def _rows_from_filehandle(handle) -> list[dict]:
    reader = csv.DictReader(handle)
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    return list(reader)


def load_raw_rows(raw_dir: Path, raw_path: Path | None = None) -> tuple[list[dict], Path]:
    """Load the immutable raw dataset from a .zip or .csv file.

    The raw file is only ever read; it is never modified.
    """
    if raw_path is not None:
        candidates = [raw_path]
    else:
        candidates = sorted(raw_dir.glob("*"))
    archives = [p for p in candidates if p.suffix.lower() == ".zip"]
    csvs = [p for p in candidates if p.suffix.lower() == ".csv"]

    if archives:
        source = archives[0]
        with zipfile.ZipFile(source) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV file found inside archive {source.name}")
            with zf.open(members[0]) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                return _rows_from_filehandle(text), source
    if csvs:
        source = csvs[0]
        with open(source, newline="", encoding="utf-8") as fh:
            return _rows_from_filehandle(fh), source
    raise FileNotFoundError(
        f"No .zip or .csv raw dataset found in {raw_dir}. "
        "Place the Kaggle dataset there before running the pipeline."
    )
