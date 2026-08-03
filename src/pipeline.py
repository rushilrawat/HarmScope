"""Pipeline orchestrator.

The phase registry below is the honest statement of what is built. A phase that
is not implemented raises rather than quietly doing nothing — a stage that
"runs" and produces no output is trap T1, and hiding it behind a no-op entry in
a dispatch table is exactly how that trap gets sprung.

    python -m src.pipeline run --phase init
    python -m src.pipeline run --phase download [--extract] [--force]
    python -m src.pipeline runs [-n 20]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from src import checks, db
from src.config import CONFIG, PATHS


def phase_init(args: argparse.Namespace) -> int:
    """Phase 0 — create the data directories and an empty, schema-valid DB."""
    con = db.bootstrap()
    tables = db.table_names(con)
    with db.run(con, "init", CONFIG) as r:
        checks.expect_rows(con, "runs", min=1)  # this run's own row
        r.finish(output_rows=len(tables))
    print(f"database  : {PATHS.db}")
    print(f"tables    : {len(tables)}")
    print(f"config    : {CONFIG.fingerprint[:16]}…")
    print(f"git sha   : {db.git_sha()}")
    return 0


def phase_download(args: argparse.Namespace) -> int:
    """Phase 1a — snapshot the CFPB bulk CSV. ~1.4 GB compressed."""
    from src.ingestion import download as dl

    PATHS.ensure()
    manifest = dl.download(CONFIG.data.bulk_csv_url, PATHS.raw, force=args.force)
    print(f"snapshot  : {PATHS.raw / manifest.filename}")
    print(f"sha256    : {manifest.sha256}")
    print(f"bytes     : {manifest.bytes:,}")
    print(f"vintage   : {manifest.last_modified}")
    if args.extract:
        csv = dl.extract(PATHS.raw, force=args.force)
        print(f"csv       : {csv} ({csv.stat().st_size:,} bytes)")
    return 0


# Implemented phases only. Everything else is named here so that asking for it
# gives the roadmap phase that would build it, not a KeyError.
PHASES: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": phase_init,
    "download": phase_download,
}

PLANNED: dict[str, str] = {
    "load": "ROADMAP Phase 1 — CSV -> complaints_raw -> complaints, narratives",
    "normalize": "ROADMAP Phase 1 — company canonicalization, taxonomy crosswalk",
    "dedup": "ROADMAP Phase 2 [GATE] — exact + MinHash + campaign detection",
    "embed": "ROADMAP Phase 3 — encode representatives, build FAISS index",
    "cluster": "ROADMAP Phase 4 [GATE] — UMAP + HDBSCAN + novelty scoring",
    "signals": "ROADMAP Phase 5 — disproportionality, changepoint, FDR",
    "backtest": "ROADMAP Phase 6 [GATE] — point-in-time harness",
    "baselines": "ROADMAP Phase 7 — B0 volume, B1 taxonomy, B2 LDA, B3 BERTopic",
    "label": "ROADMAP Phase 8 — LLM cluster labels + evidence retrieval",
    "evaluate": "ROADMAP Phase 9 — metrics, calibration, failure analysis",
}


def cmd_run(args: argparse.Namespace) -> int:
    if args.phase == "all":
        missing = ", ".join(PLANNED)
        raise SystemExit(
            f"--phase all is not runnable yet. Implemented: "
            f"{', '.join(PHASES)}. Not built: {missing}."
        )
    if args.phase in PLANNED:
        raise SystemExit(f"phase '{args.phase}' is not built.\n  {PLANNED[args.phase]}")
    if args.phase not in PHASES:
        known = ", ".join(sorted([*PHASES, *PLANNED, "all"]))
        raise SystemExit(f"unknown phase '{args.phase}'. Known phases: {known}")
    return PHASES[args.phase](args)


def cmd_runs(args: argparse.Namespace) -> int:
    con = db.connect(read_only=True)
    rows = con.execute(
        "SELECT run_id, phase, status, output_rows, started_at, "
        "substr(git_sha, 1, 12), substr(config_hash, 1, 8), error "
        "FROM runs ORDER BY started_at DESC LIMIT ?",
        [args.n],
    ).fetchall()
    if not rows:
        print("no runs recorded")
        return 0
    print(f"{'phase':<12} {'status':<8} {'rows':>10}  {'started':<20} "
          f"{'git':<13} {'cfg':<9} error")
    for _run_id, phase, status, out_rows, started, sha, cfg, err in rows:
        print(f"{phase:<12} {status:<8} {out_rows if out_rows is not None else '-':>10}"
              f"  {str(started)[:19]:<20} {sha:<13} {cfg:<9} {err or ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harmscope", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute a pipeline phase")
    p_run.add_argument("--phase", required=True)
    p_run.add_argument("--force", action="store_true",
                       help="replace an existing raw snapshot (download only)")
    p_run.add_argument("--extract", action="store_true",
                       help="extract the CSV after download")
    p_run.set_defaults(func=cmd_run)

    p_runs = sub.add_parser("runs", help="show the run registry")
    p_runs.add_argument("-n", type=int, default=20)
    p_runs.set_defaults(func=cmd_runs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
