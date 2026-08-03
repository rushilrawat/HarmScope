"""Download contracts that do not need the network."""

from __future__ import annotations

import json
import re

import pytest

from src.ingestion.download import MANIFEST_NAME, USER_AGENT, Manifest, extract


def test_user_agent_keeps_its_identification_comment():
    """Not style — the CDN 403s without it.

    `harmscope/0.1`, `harmscope`, and `Mozilla/5.0 (compatible; harmscope/0.1)`
    all fail deterministically against files.consumerfinance.gov; a UA carrying
    a parenthesised comment succeeds. Someone tidying this string into a bare
    token breaks every download, and the failure looks like a server problem.
    """
    assert re.search(r"\(.+\)", USER_AGENT), (
        f"USER_AGENT must carry a parenthesised identification comment; "
        f"got {USER_AGENT!r}"
    )
    assert len(USER_AGENT) > 20


def test_manifest_round_trips(tmp_path):
    m = Manifest(
        url="https://example.test/complaints.csv.zip",
        filename="complaints.csv.zip",
        sha256="a" * 64,
        bytes=123,
        downloaded_at="2026-08-03T00:00:00+00:00",
        last_modified="Sun, 02 Aug 2026 09:30:27 GMT",
    )
    path = tmp_path / MANIFEST_NAME
    m.write(path)
    assert Manifest.read(path) == m
    assert json.loads(path.read_text())["sha256"] == "a" * 64


def test_extract_without_manifest_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="run download first"):
        extract(tmp_path)


def test_extract_pulls_the_single_csv_member(tmp_path):
    import zipfile

    archive = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("complaints.csv", "Complaint ID\n1\n")
    Manifest(
        url="https://example.test/complaints.csv.zip",
        filename=archive.name,
        sha256="b" * 64,
        bytes=archive.stat().st_size,
        downloaded_at="2026-08-03T00:00:00+00:00",
    ).write(tmp_path / MANIFEST_NAME)

    out = extract(tmp_path)
    assert out.name == "complaints.csv"
    assert out.read_text().startswith("Complaint ID")
    # The archive is kept: its sha256 is the reproducibility anchor.
    assert archive.exists()
    assert Manifest.read(tmp_path / MANIFEST_NAME).extracted_csv == "complaints.csv"


def test_extract_rejects_ambiguous_archives(tmp_path):
    import zipfile

    archive = tmp_path / "complaints.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.csv", "x\n")
        zf.writestr("b.csv", "y\n")
    Manifest(
        url="https://example.test/complaints.csv.zip",
        filename=archive.name,
        sha256="c" * 64,
        bytes=archive.stat().st_size,
        downloaded_at="2026-08-03T00:00:00+00:00",
    ).write(tmp_path / MANIFEST_NAME)

    with pytest.raises(ValueError, match="exactly one CSV"):
        extract(tmp_path)
