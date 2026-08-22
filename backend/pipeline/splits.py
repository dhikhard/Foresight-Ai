def chronological_split(
    rows: list[dict], validation_start: str, test_start: str
) -> dict[str, list[dict]]:
    """Split strictly by date order — never randomly.

    train:      date <  validation_start
    validation: validation_start <= date < test_start
    test:       date >= test_start
    """
    if not validation_start <= test_start:
        raise ValueError("validation_start must be on or before test_start")
    return {
        "train": [r for r in rows if r["date"] < validation_start],
        "validation": [r for r in rows if validation_start <= r["date"] < test_start],
        "test": [r for r in rows if r["date"] >= test_start],
    }
