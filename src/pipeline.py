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
from pathlib import Path

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
    if args.gzip:
        csv = dl.recompress_gzip(PATHS.raw, force=args.force)
        print(f"csv.gz    : {csv} ({csv.stat().st_size:,} bytes)")
    elif args.extract:
        csv = dl.extract(PATHS.raw, force=args.force)
        print(f"csv       : {csv} ({csv.stat().st_size:,} bytes)")
    return 0


def phase_load(args: argparse.Namespace) -> int:
    """Phase 1b — CFPB snapshot CSV -> `complaints_raw`, with reconciliation."""
    from src.ingestion import download as dl
    from src.ingestion import load as ld

    if args.csv:
        csv = Path(args.csv)
    else:
        manifest_path = PATHS.raw / dl.MANIFEST_NAME
        if not manifest_path.exists():
            raise SystemExit("no snapshot yet — run `make download` first")
        manifest = dl.Manifest.read(manifest_path)
        if not manifest.extracted_csv:
            raise SystemExit(
                "snapshot is still a zip — run "
                "`python -m src.pipeline run --phase download --extract`"
            )
        csv = PATHS.raw / manifest.extracted_csv

    con = db.bootstrap()
    coverage = 0.0
    dropped = 0

    def acceptance(c) -> None:
        """ROADMAP Phase 1 acceptance. Runs before the load is committed."""
        nonlocal coverage
        nonlocal dropped
        n = c.execute("SELECT count(*) FROM complaints_raw").fetchone()[0]
        dropped = n_csv - n
        if dropped < 0:
            raise checks.CheckFailed(
                f"loaded more rows than the CSV has: {n:,} vs {n_csv:,}"
            )
        frac = dropped / n_csv if n_csv else 0.0
        if frac > CONFIG.expect.max_dropped_fraction:
            raise checks.CheckFailed(
                f"{dropped:,} of {n_csv:,} CSV rows ({frac:.4%}) did not load, "
                f"above the {CONFIG.expect.max_dropped_fraction:.3%} bound. "
                f"Only unkeyed rows are droppable — check whether the export "
                f"shape changed."
            )
        # The corpus-size bound asserts "this is the CFPB corpus", which is
        # simply false when --csv points at something else. Every other check
        # still applies.
        if not args.csv:
            checks.expect_rows(
                c, "complaints_raw",
                min=CONFIG.expect.complaints_raw_min,
                max=CONFIG.expect.complaints_raw_max,
            )
        checks.expect_no_nulls(c, "complaints_raw", ["complaint_id", "date_received"])
        coverage = checks.expect_scalar(
            c,
            "SELECT avg(CAST(has_narrative AS INT)) FROM complaints_raw",
            lo=CONFIG.expect.narrative_fraction_min,
            hi=CONFIG.expect.narrative_fraction_max,
            label="narrative coverage fraction",
        )

    with db.run(con, "load", CONFIG, params={"csv": str(csv)}) as r:
        n_csv = ld.csv_row_count(con, csv)
        n_loaded = ld.load_raw(con, csv, validate=acceptance)
        r.finish(output_rows=n_loaded, input_rows=n_csv)

    n_products, n_issues, lo, hi = con.execute(
        "SELECT count(DISTINCT product), count(DISTINCT issue), "
        "min(date_received), max(date_received) FROM complaints_raw"
    ).fetchone()
    print(f"rows          : {n_loaded:,} of {n_csv:,} CSV rows")
    print(f"dropped       : {dropped:,} unkeyed ({dropped / n_csv:.4%})")
    print(f"date range    : {lo} .. {hi}")
    print(f"narrative frac: {coverage:.4f}  <- record this in docs/DATA.md §5")
    print(f"distinct      : {n_products} products, {n_issues} issues")
    print("\nnext: the taxonomy crosswalk needs those product/issue values —")
    print("      `python -m src.pipeline taxonomy` lists them by volume.")
    return 0


def phase_normalize(args: argparse.Namespace) -> int:
    """Phase 1c — crosswalk, company canonicalization, complaints, narratives."""
    from src.ingestion import build
    from src.ingestion import download as dl
    from src.normalization import company, taxonomy

    manifest = dl.Manifest.read(PATHS.raw / dl.MANIFEST_NAME)
    csv = PATHS.raw / (manifest.extracted_csv or "")
    if not csv.exists():
        raise SystemExit("no readable CSV — run `make gzip` first")

    con = db.bootstrap()
    window = CONFIG.data.window_start

    with db.run(con, "normalize", CONFIG) as r:
        n_cross = taxonomy.load_crosswalk(con)
        uncovered = taxonomy.uncovered_labels(con)
        if uncovered:
            raise checks.CheckFailed(
                "crosswalk does not cover: "
                + "; ".join(f"{p} / {sp} ({n:,})" for p, sp, n in uncovered[:10])
                + "\nproduct_family is NOT NULL — add these to "
                "data/ground_truth/taxonomy_crosswalk.csv rather than defaulting them."
            )

        n_co, n_alias = company.build_canonical(con)
        n_complaints = build.build_complaints(con, window)
        n_narr, hits, docs = build.build_narratives(con, csv, window, r.run_id)

        checks.expect_rows(con, "taxonomy_crosswalk", min=1)
        checks.expect_no_nulls(con, "complaints", ["product_family", "period_month"])
        checks.expect_rows(con, "complaints", min=1)
        checks.expect_scalar(
            con,
            "SELECT count(*) FROM complaints WHERE company_id IS NULL",
            hi=0, label="complaints with unresolved company",
        )
        # narratives must line up exactly with the has_narrative flag
        expected = con.execute(
            "SELECT count(*) FROM complaints WHERE has_narrative"
        ).fetchone()[0]
        if n_narr != expected:
            raise checks.CheckFailed(
                f"narratives: {n_narr:,} rows but {expected:,} complaints are "
                f"flagged has_narrative"
            )
        checks.expect_no_nulls(con, "narratives", ["text_redacted", "text_hash"])
        checks.expect_scalar(
            con,
            "SELECT avg(redaction_count) FROM narratives",
            lo=CONFIG.expect.redaction_rate_min,
            hi=CONFIG.expect.redaction_rate_max,
            label="mean redactions per narrative",
        )
        checks.expect_scalar(
            con,
            "SELECT avg(CASE WHEN redaction_count > 0 THEN 1.0 ELSE 0 END) "
            "FROM narratives",
            lo=CONFIG.expect.redacted_doc_fraction_min,
            hi=CONFIG.expect.redacted_doc_fraction_max,
            label="fraction of narratives with a redaction",
        )
        r.finish(output_rows=n_complaints, input_rows=n_cross)

    print(f"crosswalk     : {n_cross} rules, 0 uncovered labels")
    print(f"companies     : {n_co:,} canonical from {n_alias:,} raw strings")
    print(f"complaints    : {n_complaints:,} (>= {window})")
    print(f"narratives    : {n_narr:,}  ({docs:,} had at least one redaction)")
    print("redactions    : " + ", ".join(f"{k}={v:,}" for k, v in hits.items() if v))

    print("\nfamily volume continuity (ROADMAP Phase 1 acceptance):")
    gaps = con.execute(
        """
        WITH m AS (
          SELECT product_family f, period_month p, count(*) n
          FROM complaints GROUP BY 1, 2
        ), rng AS (
          SELECT f, min(p) lo, max(p) hi FROM m GROUP BY 1
        )
        SELECT r.f, count(*) FILTER (WHERE m.n IS NULL) AS empty_months
        FROM rng r
        LEFT JOIN generate_series(r.lo, r.hi, INTERVAL 1 MONTH) g(p) ON true
        LEFT JOIN m ON m.f = r.f AND m.p = g.p::DATE
        GROUP BY 1 ORDER BY 2 DESC, 1
        """
    ).fetchall()
    for fam, empty in gaps:
        flag = "  <- GAP" if empty else ""
        print(f"  {fam:<18} {empty} empty months in range{flag}")
    if any(e for _, e in gaps):
        raise SystemExit(
            "a product_family has months with no complaints inside its own active "
            "range — that is the signature of a label vanishing at a schema "
            "boundary, i.e. the crosswalk is incomplete."
        )
    return 0


def phase_dedup(args: argparse.Namespace) -> int:
    """Phase 2 [GATE] — exact + MinHash dedup, union-find, campaign detection."""
    import numpy as np

    from src.dedup import campaign, detect

    con = db.bootstrap()
    as_of = con.execute("SELECT max(date_received) FROM complaints").fetchone()[0]

    with db.run(con, "dedup", CONFIG) as r:
        con.execute("DELETE FROM dup_pairs")
        n_exact = detect.exact_pairs(con)
        print(f"tier 1 exact  : {n_exact:,} pairs", flush=True)

        reps = detect.representatives(con)
        ids = np.array([x[0] for x in reps], dtype=np.int64)
        families = [x[1] for x in reps]
        print(f"representatives: {len(reps):,}", flush=True)

        sig = detect.build_signatures([x[2] for x in reps], CONFIG)
        pairs = detect.candidate_pairs(con, ids, families, sig)
        print(f"lsh candidates : {len(pairs):,}", flush=True)

        if len(pairs):
            keep, sims = detect.verify(sig, pairs, CONFIG.dedup.jaccard_threshold)
            good = pairs[keep]
            print(f"verified >= {CONFIG.dedup.jaccard_threshold}: {len(good):,} "
                  f"({len(good) / max(len(pairs), 1):.1%} of candidates)", flush=True)
            if len(good):
                a = np.minimum(ids[good[:, 0]], ids[good[:, 1]])
                b = np.maximum(ids[good[:, 0]], ids[good[:, 1]])
                frame = {"complaint_id_a": a, "complaint_id_b": b,  # noqa: F841
                         "similarity": sims[keep].astype(float),
                         "method": np.array(["minhash"] * len(good), dtype=object)}
                con.execute(
                    "INSERT INTO dup_pairs SELECT * FROM frame "
                    "WHERE (complaint_id_a, complaint_id_b) NOT IN "
                    "(SELECT complaint_id_a, complaint_id_b FROM dup_pairs)"
                )

        n_groups, n_rows = detect.assign_groups(con, r.run_id, as_of)
        n_cand, n_flagged = detect.build_campaigns(con, r.run_id, CONFIG, as_of)
        r.finish(output_rows=n_rows, input_rows=len(reps))

    print(f"\ngroups        : {n_groups:,} over {n_rows:,} narratives")
    print(f"campaigns     : {n_cand:,} candidates, {n_flagged:,} flagged")
    print(f"corpus boilerplate baseline: {campaign.boilerplate_share(con):.4f}")

    print("\ncampaign-flagged share by product family "
          "(METHODOLOGY §2.4: credit reporting must be clearly highest):")
    for fam, tot, flagged in con.execute(
        f"""
        SELECT c.product_family, count(*) AS tot,
               count(*) FILTER (WHERE cm.complaint_id IS NOT NULL) AS flagged
        FROM complaints c
        JOIN narratives n USING (complaint_id)
        LEFT JOIN (SELECT DISTINCT m.complaint_id FROM campaign_members m
                   JOIN campaigns ca USING (campaign_id)
                   WHERE ca.run_id = '{r.run_id}' AND ca.flagged) cm
               USING (complaint_id)
        GROUP BY 1 ORDER BY flagged::DOUBLE / count(*) DESC
        """
    ).fetchall():
        print(f"  {fam:<18} {flagged:>9,} / {tot:>9,}  {flagged / tot:6.2%}")
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    """Print the label vocabulary the crosswalk has to cover, by volume."""
    con = db.connect(read_only=True)
    rows = con.execute(
        "SELECT product, count(*) n, min(date_received), max(date_received) "
        "FROM complaints_raw WHERE product IS NOT NULL "
        "GROUP BY product ORDER BY n DESC"
    ).fetchall()
    if not rows:
        raise SystemExit("complaints_raw is empty — run `--phase load` first")
    print(f"{'n':>12}  {'first':<12} {'last':<12} product")
    for product, n, lo, hi in rows:
        print(f"{n:>12,}  {str(lo):<12} {str(hi):<12} {product}")
    print(f"\n{len(rows)} distinct products. Products whose date range ends near the")
    print("2017 restructuring are the ones the crosswalk has to map forward.")
    return 0


# Implemented phases only. Everything else is named here so that asking for it
# gives the roadmap phase that would build it, not a KeyError.
PHASES: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": phase_init,
    "download": phase_download,
    "load": phase_load,
    "normalize": phase_normalize,
    "dedup": phase_dedup,
}

PLANNED: dict[str, str] = {
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
    p_run.add_argument("--gzip", action="store_true",
                       help="restream the snapshot as .csv.gz (DuckDB reads it directly)")
    p_run.add_argument("--csv", help="load from this CSV instead of the snapshot")
    p_run.set_defaults(func=cmd_run)

    p_tax = sub.add_parser("taxonomy", help="label vocabulary by volume")
    p_tax.set_defaults(func=cmd_taxonomy)

    p_runs = sub.add_parser("runs", help="show the run registry")
    p_runs.add_argument("-n", type=int, default=20)
    p_runs.set_defaults(func=cmd_runs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
