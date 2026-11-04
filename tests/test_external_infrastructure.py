"""Tests for external data infrastructure: SnapshotWriter, SHA-256, HTTP transport."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.pipeline.external import (
    SnapshotWriter,
    sha256_bytes,
    sha256_file,
    http_get_json,
)


# --- SHA-256 ---


class TestSHA256:
    def test_bytes_deterministic(self):
        assert sha256_bytes(b"hello") == sha256_bytes(b"hello")

    def test_bytes_differs_on_input_change(self):
        assert sha256_bytes(b"hello") != sha256_bytes(b"world")

    def test_file_sha256_matches_bytes(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"test data here")
        assert sha256_file(f) == sha256_bytes(b"test data here")

    def test_file_sha256_empty(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert sha256_file(f) == sha256_bytes(b"")


# --- SnapshotWriter ---


class TestSnapshotWriter:
    def test_create_snapshot_dir(self, tmp_path):
        sw = SnapshotWriter(tmp_path)
        d = sw.new_snapshot("testsource", "myslug", stamp="20260101T120000Z")
        assert d.exists()
        assert d.is_dir()
        assert "20260101T120000Z-myslug" in str(d)

    def test_append_only_invariant(self, tmp_path):
        sw = SnapshotWriter(tmp_path)
        d = sw.new_snapshot("testsource", "slug", stamp="20260101T000000Z")
        with pytest.raises(FileExistsError, match="append-only"):
            sw.new_snapshot("testsource", "slug", stamp="20260101T000000Z")

    def test_write_raw(self, tmp_path):
        sw = SnapshotWriter(tmp_path)
        d = sw.new_snapshot("src", "s", stamp="20260101T000000Z")
        out = sw.write_raw(d, "data.bin", b"raw bytes")
        assert out.read_bytes() == b"raw bytes"

    def test_write_json(self, tmp_path):
        sw = SnapshotWriter(tmp_path)
        d = sw.new_snapshot("src", "s", stamp="20260101T000000Z")
        out = sw.write_json(d, "data.json", {"key": "val"})
        loaded = json.loads(out.read_text())
        assert loaded == {"key": "val"}

    def test_manifest_structure(self, tmp_path):
        sw = SnapshotWriter(tmp_path)
        d = sw.new_snapshot("src", "s", stamp="20260101T000000Z")
        raw = sw.write_json(d, "raw.json", [1, 2])
        manifest_path = sw.write_manifest(
            d,
            source="src",
            endpoint="https://api.test.com/v1",
            request_params={"limit": 5},
            http_status=200,
            raw_files=[raw],
            row_counts={"rows": 2},
        )
        m = json.loads(manifest_path.read_text())
        assert m["source"] == "src"
        assert m["http_status"] == 200
        assert m["row_counts"] == {"rows": 2}
        assert "raw.json" in m["raw_files"]
        assert len(m["raw_files"]["raw.json"]) == 64  # sha256 hex length
        assert "generated_at_utc" in m

    def test_manifest_code_version(self, tmp_path):
        from backend.pipeline.external import __version__
        sw = SnapshotWriter(tmp_path)
        d = sw.new_snapshot("src", "s", stamp="20260101T000000Z")
        sw.write_manifest(
            d, source="src", endpoint="x", request_params={},
            http_status=200, raw_files=[], row_counts={},
        )
        m = json.loads((d / "snapshot_manifest.json").read_text())
        assert m["code_version"] == __version__


# --- Fake transport ---


def test_http_get_json_with_fake_transport():
    fake_data = [{"id": 1, "name": "test"}]
    body = json.dumps(fake_data).encode()

    def fake_transport(url: str) -> tuple[int, bytes]:
        return 200, body

    status, data = http_get_json("https://fake.api/test", transport=fake_transport)
    assert status == 200
    assert data == fake_data
