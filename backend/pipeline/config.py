from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PipelineConfig:
    symbol: str = "^GSPC"
    raw_dir: Path = DEFAULT_RAW_DIR
    raw_path: Path | None = None
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    ma_windows: tuple[int, ...] = (20, 50, 200)
    volatility_window: int = 20
    annualization_factor: float = float(TRADING_DAYS_PER_YEAR) ** 0.5
    validation_start: str = "2019-01-01"
    test_start: str = "2023-01-01"
    with_targets: bool = False
    target_horizons: tuple[int, ...] = (1, 5, 10)
    future_volatility_window: int = 20
    log: list = field(default_factory=list)

    def note(self, message: str) -> None:
        self.log.append(message)
