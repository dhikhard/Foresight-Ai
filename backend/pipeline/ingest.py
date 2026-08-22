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


SUPPORTED_SUFFIXES = (".zip", ".csv")


def load_raw_rows(raw_dir: Path, raw_path: Path | None = None) -> tuple[list[dict], Path]:
    """Load the immutable raw dataset from a .zip or .csv file.

    The raw file is only ever read; it is never modified. When no explicit
    raw_path is given, the raw directory must contain exactly one supported
    dataset file; ambiguity is an error rather than a silent guess.
    """
    if raw_path is not None:
        if not raw_path.exists():
            raise FileNotFoundError(f"Explicit raw path does not exist: {raw_path}")
        if raw_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported raw file type '{raw_path.name}'. Expected one of: {SUPPORTED_SUFFIXES}"
            )
        source = raw_path
    else:
        candidates = sorted(raw_dir.glob("*"))
        supported = [p for p in candidates if p.suffix.lower() in SUPPORTED_SUFFIXES]
        if len(supported) == 0:
            raise FileNotFoundError(
                f"No .zip or .csv raw dataset found in {raw_dir}. "
                "Place the Kaggle dataset there before running the pipeline."
            )
        if len(supported) > 1:
            names = ", ".join(p.name for p in supported)
            raise ValueError(
                f"Ambiguous raw data directory '{raw_dir}': found {len(supported)} "
                f"candidate datasets ({names}). Remove the extra file or specify "
                "the intended source explicitly with --raw-path."
            )
        source = supported[0]

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV file found inside archive {source.name}")
            with zf.open(members[0]) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                return _rows_from_filehandle(text), source
    with open(source, newline="", encoding="utf-8") as fh:
        return _rows_from_filehandle(fh), source
