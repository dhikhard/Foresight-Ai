"""One-command Phase 1 demonstration for ForeSightAI.

Runs the supported market-data workflow end to end, in order, using the
existing CLI entry points (no duplicated pipeline logic):

    1. backend.pipeline.run_preprocess   --with-targets   (raw -> clean/features/targets/splits)
    2. backend.modeling.run_modeling                      (V1 features -> V1 report)
    3. backend.pipeline.run_features_v2                   (clean -> V2 features)
    4. backend.modeling.run_modeling --feature-set v2     (V2 features -> V2 report)

Every artifact is written under a single output directory. The canonical
files in data/processed are never touched. By default the output directory is
a fresh temporary directory; pass --out-dir to choose one. An existing
non-empty --out-dir is refused unless --force is given, so user files are not
silently overwritten.

Usage (from the repository root):
    python run_demo.py
    python run_demo.py --out-dir demo_out
    python run_demo.py --raw-path path/to/raw.csv --out-dir demo_out

Exit status is non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# (label, argv-after-python) for each ordered step. Placeholders
# {out} and {raw} are filled from the resolved options.
_STEP_TEMPLATES = (
    (
        "preprocess + targets",
        ["-m", "backend.pipeline.run_preprocess", "--with-targets",
         "--processed-dir", "{out}"],
    ),
    (
        "V1 model + report",
        ["-m", "backend.modeling.run_modeling",
         "--features", "{out}/sp500_features.csv",
         "--targets", "{out}/sp500_targets.csv",
         "--out", "{out}/modeling_report.json"],
    ),
    (
        "V2 features",
        ["-m", "backend.pipeline.run_features_v2",
         "--clean-path", "{out}/sp500_clean.csv",
         "--processed-dir", "{out}"],
    ),
    (
        "V2 model + report",
        ["-m", "backend.modeling.run_modeling", "--feature-set", "v2",
         "--features", "{out}/sp500_features_v2.csv",
         "--targets", "{out}/sp500_targets.csv",
         "--out", "{out}/modeling_report_v2.json"],
    ),
)


def _fail(message: str, code: int = 1) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def _prepare_out_dir(out_dir: Path | None, force: bool) -> Path | None:
    """Return a usable output directory, or None if writing must be refused.

    Default (out_dir is None): a fresh temporary directory.
    Explicit: refuse a non-empty directory unless force=True.
    """
    if out_dir is None:
        return Path(tempfile.mkdtemp(prefix="foresight_demo_"))
    out_dir = out_dir.resolve()
    if out_dir.exists():
        if not out_dir.is_dir():
            return None
        existing = [p.name for p in out_dir.iterdir()]
        if existing and not force:
            return None
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_step(label: str, argv: list[str]) -> int:
    printable = " ".join(argv)
    print(f"\n=== [{label}] ===")
    print(f"$ {printable}")
    completed = subprocess.run(
        [sys.executable, *argv], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return _fail(
            f"step '{label}' failed with exit code {completed.returncode}",
            completed.returncode,
        )
    if completed.stderr:
        print(completed.stderr, end="")
    return 0


def _render_markdown(report_path: Path, title: str) -> str:
    """Render a concise Markdown results table from one modeling report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lines = [
        f"### {title}",
        "",
        f"- feature_set: `{report.get('feature_set')}`",
        f"- target: `{report.get('target')}`",
        f"- recorded run (generated_at_utc): `{report.get('generated_at_utc')}`",
        f"- rows: {report.get('rows')}",
        f"- date range: {report.get('date_range')}",
        "",
        "| model | validation MAE | test MAE | test R^2 |",
        "|---|---|---|---|",
    ]
    for model_name, result in report.get("results", {}).items():
        by_split = {row["split"]: row for row in result.get("splits", [])}
        val = by_split.get("validation", {})
        test = by_split.get("test", {})
        lines.append(
            "| {m} | {v} | {t} | {r} |".format(
                m=model_name,
                v=_fmt(val.get("mae")),
                t=_fmt(test.get("mae")),
                r=_fmt(test.get("r2")),
            )
        )
    lines.append("")
    lines.append(
        "Metrics above belong to the single recorded run identified by the "
        "`generated_at_utc` value. Selection used validation MAE only; the "
        "test split was scored once, after selection."
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ForeSightAI Phase 1 market-data demonstration."
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="output directory (default: a fresh temporary directory)",
    )
    parser.add_argument(
        "--raw-path", default=None,
        help="explicit raw .csv/.zip dataset path (default: data/raw auto-detect)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="allow writing into an existing non-empty --out-dir",
    )
    args = parser.parse_args(argv)

    out_requested = Path(args.out_dir) if args.out_dir else None
    out_dir = _prepare_out_dir(out_requested, args.force)
    if out_dir is None:
        return _fail(
            f"refusing to write into non-empty directory '{out_requested}'; "
            "pass --force to allow, or choose an empty/new directory",
            code=2,
        )

    raw_path = Path(args.raw_path).resolve() if args.raw_path else None
    if raw_path is not None and not raw_path.exists():
        return _fail(f"--raw-path does not exist: {raw_path}", code=2)

    print(f"ForeSightAI Phase 1 demo")
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  output dir:   {out_dir}")
    if raw_path is not None:
        print(f"  raw path:     {raw_path}")

    substitutions = {"out": str(out_dir), "raw": str(raw_path) if raw_path else ""}
    for label, template in _STEP_TEMPLATES:
        argv_step = [part.format(**substitutions) for part in template]
        if raw_path is not None and "backend.pipeline.run_preprocess" in argv_step:
            argv_step += ["--raw-path", str(raw_path)]
        code = _run_step(label, argv_step)
        if code != 0:
            return code

    results_md = out_dir / "demo_results.md"
    sections = [
        "# ForeSightAI Phase 1 - demo results",
        "",
        f"Generated by `python run_demo.py` into `{out_dir}`.",
        "",
        _render_markdown(out_dir / "modeling_report.json", "V1 feature set (10 features)"),
        _render_markdown(out_dir / "modeling_report_v2.json", "V2 feature set (16 features)"),
    ]
    results_md.write_text("\n".join(sections), encoding="utf-8")

    print("\n=== demo complete ===")
    print(f"  outputs:      {out_dir}")
    print(f"  V1 report:    {out_dir / 'modeling_report.json'}")
    print(f"  V2 report:    {out_dir / 'modeling_report_v2.json'}")
    print(f"  results table:{results_md}")
    print(
        "\nNote: the canonical artifacts in data/processed were NOT modified. "
        "The current models do not beat the train-mean baseline out of sample."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
