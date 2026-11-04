"""External data layer for ForeSightAI.

Snapshot-based acquisition architecture:
  API -> immutable local snapshot -> normalization -> temporal alignment

This package is fully additive and isolated from the frozen V1/V2 pipeline.
All raw snapshots are append-only; existing snapshots are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# HTTP transport (injectable for testing)
# ---------------------------------------------------------------------------

_USER_AGENT = "ForeSightAI-Research/1.0 (academic research)"


def _default_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, bytes]:
    """Real HTTP GET using stdlib urllib. Returns (status_code, body_bytes)."""
    hdrs: dict[str, str] = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.status, resp.read()


TransportFn = Callable[
    [str], tuple[int, bytes]
]
"""Signature for injectable transport: url -> (status, body_bytes).
Headers/timeouts are handled by the caller; the transport is the raw layer."""


def default_transport(url: str) -> tuple[int, bytes]:
    return _default_get(url)


def http_get_json(
    url: str,
    *,
    transport: TransportFn | None = None,
) -> tuple[int, Any]:
    """GET returning (status, parsed_json). Uses default transport if none given."""
    fn = transport or default_transport
    status, body = fn(url)
    return status, json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Snapshot writer (append-only)
# ---------------------------------------------------------------------------

class SnapshotWriter:
    """Manages immutable snapshot directories under a given base path.

    Each call to ``new_snapshot`` creates a fresh directory; an
    ``FileExistsError`` is raised if the directory already exists,
    enforcing the append-only invariant.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    # ---- directory lifecycle ----

    def new_snapshot(
        self, source: str, slug: str, *, stamp: str | None = None
    ) -> Path:
        if stamp is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = self.base_dir / source / f"{stamp}-{slug}"
        if d.exists():
            raise FileExistsError(
                f"Snapshot already exists (append-only invariant): {d}"
            )
        d.mkdir(parents=True, exist_ok=False)
        return d

    # ---- raw file writers ----

    def write_raw(self, snap: Path, filename: str, data: bytes) -> Path:
        out = snap / filename
        out.write_bytes(data)
        return out

    def write_json(self, snap: Path, filename: str, obj: object) -> Path:
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        return self.write_raw(snap, filename, raw)

    # ---- manifest ----

    def write_manifest(
        self,
        snap: Path,
        *,
        source: str,
        endpoint: str,
        request_params: dict[str, Any],
        http_status: int,
        raw_files: list[Path],
        row_counts: dict[str, int],
        code_version: str | None = None,
    ) -> Path:
        if code_version is None:
            code_version = __version__
        hashes = {f.name: sha256_file(f) for f in raw_files}
        manifest = {
            "source": source,
            "endpoint": endpoint,
            "request_params": request_params,
            "http_status": http_status,
            "raw_files": hashes,
            "row_counts": row_counts,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_version": code_version,
            "snapshot_dir": str(snap),
        }
        return self.write_json(snap, "snapshot_manifest.json", manifest)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_dotenv_if_available() -> None:
    """Optionally load a .env file via python-dotenv (no hard dependency)."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        # Look for .env in the repo root (3 levels up from this file)
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass
