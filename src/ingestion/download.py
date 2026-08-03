"""Snapshot the CFPB bulk complaint CSV.

docs/DATA.md §1: use the bulk CSV, not the API. Offset pagination is unreliable
at depth, and pulling millions of records over HTTP is slow and fragile.
Download once, snapshot it, treat it as immutable input.

"Immutable" is enforced here rather than assumed: `download()` refuses to
overwrite an existing snapshot without `force=True`. Federal data availability
has been volatile (docs/PROJECT_SPEC.md §5.4) and a snapshot that silently
changes underneath a completed backtest invalidates every number in it.

The manifest records url, timestamp, sha256, byte count, and the server's
Last-Modified — enough to prove which vintage of the database produced a result.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB
MANIFEST_NAME = "manifest.json"

# The CDN in front of files.consumerfinance.gov rejects short bare-token
# User-Agents with a hard 403 — `harmscope/0.1`, `harmscope`, and even
# `Mozilla/5.0 (compatible; harmscope/0.1)` all fail deterministically, while
# urllib's own default and any UA carrying a parenthesised identification
# comment succeed. Measured 2026-08-03, three trials each.
#
# So this string is load-bearing, not decoration: the parenthesised comment is
# what gets the request through. It is also the right thing to send for a bulk
# research download — identify the client and give someone a way to reach you.
# Put a real contact URL here if you fork this.
USER_AGENT = "harmscope/0.1 (research; +https://github.com/harmscope)"


@dataclass
class Manifest:
    url: str
    filename: str
    sha256: str
    bytes: int
    downloaded_at: str
    last_modified: str | None = None
    extracted_csv: str | None = None
    csv_bytes: int | None = None

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> Manifest:
        return cls(**json.loads(path.read_text()))


def _progress(done: int, total: int | None) -> None:
    if total:
        pct = 100 * done / total
        msg = f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB  ({pct:5.1f}%)"
    else:
        msg = f"\r  {done / 1e9:.2f} GB"
    print(msg, end="", file=sys.stderr, flush=True)


def download(url: str, dest_dir: Path, *, force: bool = False) -> Manifest:
    """Stream the bulk archive to `dest_dir`, hashing as we go.

    Returns the manifest, which is also written to `dest_dir/manifest.json`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / MANIFEST_NAME
    filename = url.rsplit("/", 1)[-1]
    target = dest_dir / filename

    if target.exists() and not force:
        if manifest_path.exists():
            existing = Manifest.read(manifest_path)
            print(
                f"snapshot already present: {target} "
                f"({existing.bytes / 1e9:.2f} GB, sha256 {existing.sha256[:12]}…)\n"
                f"the snapshot is immutable input — pass force=True to replace it",
                file=sys.stderr,
            )
            return existing
        raise FileExistsError(
            f"{target} exists but {manifest_path} does not. Refusing to guess "
            f"its provenance; remove it or pass force=True."
        )

    digest = hashlib.sha256()
    tmp = target.with_suffix(target.suffix + ".partial")

    # noqa S310: the URL is CONFIG.data.bulk_csv_url, a fixed https endpoint,
    # not caller-supplied. Guard it anyway so a future config edit cannot turn
    # this into a file:// read.
    if not url.startswith("https://"):
        raise ValueError(f"refusing to download from a non-https URL: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        resp_cm = urllib.request.urlopen(req)  # noqa: S310
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise PermissionError(
                f"403 from {url}.\n"
                f"The CDN rejects short bare-token User-Agents. The UA sent was "
                f"{USER_AGENT!r} — if you edited USER_AGENT in this module, put "
                f"the parenthesised identification comment back. See the note "
                f"above the constant."
            ) from exc
        raise

    with resp_cm as resp:
        total = int(resp.headers.get("Content-Length") or 0) or None
        last_modified = resp.headers.get("Last-Modified")
        done = 0
        with tmp.open("wb") as fh:
            while chunk := resp.read(CHUNK):
                fh.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if done % (64 * CHUNK) < CHUNK:
                    _progress(done, total)
    _progress(done, total)
    print(file=sys.stderr)

    if total is not None and done != total:
        tmp.unlink(missing_ok=True)
        raise OSError(f"truncated download: got {done:,} bytes, expected {total:,}")

    tmp.replace(target)
    manifest = Manifest(
        url=url,
        filename=filename,
        sha256=digest.hexdigest(),
        bytes=done,
        downloaded_at=datetime.now(UTC).isoformat(),
        last_modified=last_modified,
    )
    manifest.write(manifest_path)
    return manifest


def extract(dest_dir: Path, *, force: bool = False) -> Path:
    """Extract the single CSV member from the downloaded archive.

    DuckDB's `read_csv_auto` cannot read a member of a zip archive directly, so
    the CSV is materialised alongside it. The archive is kept: its sha256 is the
    reproducibility anchor recorded in the manifest.
    """
    manifest_path = dest_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}; run download first")
    manifest = Manifest.read(manifest_path)
    archive = dest_dir / manifest.filename

    if not zipfile.is_zipfile(archive):
        return archive  # already a plain CSV

    with zipfile.ZipFile(archive) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV in {archive}, found {members}")
        member = members[0]
        out = dest_dir / Path(member).name
        if out.exists() and not force:
            print(f"already extracted: {out}", file=sys.stderr)
        else:
            print(f"extracting {member} -> {out}", file=sys.stderr)
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)

    manifest.extracted_csv = out.name
    manifest.csv_bytes = out.stat().st_size
    manifest.write(manifest_path)
    return out
