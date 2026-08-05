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
    """Phase 2 [GATE] — exact + MinHash dedup, star clustering, campaigns."""
    import numpy as np

    from src.dedup import campaign, detect, evalset

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
            # Rejected candidates are never persisted; sample them now or the
            # gate's recall has no denominator. See evalset.write_near_misses.
            evalset.write_near_misses(
                ids, pairs[~keep], sims[~keep], CONFIG.dedup.jaccard_threshold
            )
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


def phase_embed(args: argparse.Namespace) -> int:
    """Phase 3 — encode distinct narratives, build the ANN index."""
    from src.embed import encode
    from src.embed import index as faiss_index

    con = db.bootstrap()
    model_name = args.model or CONFIG.embed.model
    memmap = PATHS.artifacts / f"embeddings.{model_name.split('/')[-1]}.npy"
    index_path = PATHS.artifacts / f"faiss.{model_name.split('/')[-1]}.index"

    with db.run(con, "embed", CONFIG, params={"model": model_name}) as r:
        stats = encode.encode_all(
            con, model_name, memmap, batch_size=args.batch or CONFIG.embed.batch_size,
            device=args.device, limit=args.limit,
            checkpoint_every=CONFIG.embed.checkpoint_every,
        )
        n_mapped, n_unmapped = encode.build_map(
            con, model_name, stats["dim"], stats["n_total"]
        )
        if n_unmapped and not args.limit:
            raise checks.CheckFailed(
                f"{n_unmapped:,} narratives have no embedding row after a full "
                f"encode — every narrative's text must be in the memmap"
            )
        checks.expect_rows(con, "embedding_map", min=1)
        # A row of zeros is what a crashed or skipped batch leaves behind, and
        # it is invisible downstream: it clusters happily as noise. Norms are 1
        # by construction (METHODOLOGY §3), so anything else is a gap.
        vectors = encode.np.load(memmap, mmap_mode="r")
        sample = vectors[:: max(1, len(vectors) // 20_000)]
        norms = encode.np.linalg.norm(sample, axis=1)
        if not encode.np.allclose(norms, 1.0, atol=1e-3):
            raise checks.CheckFailed(
                f"{int((~encode.np.isclose(norms, 1.0, atol=1e-3)).sum())} of "
                f"{len(sample):,} sampled vectors are not unit length "
                f"(min {norms.min():.4f}) — a zero row is an unencoded row"
            )
        n_indexed = faiss_index.build(memmap, index_path)
        if n_indexed != stats["n_total"]:
            raise checks.CheckFailed(
                f"index holds {n_indexed:,} vectors, memmap has {stats['n_total']:,}"
            )
        r.finish(output_rows=n_mapped, input_rows=stats["n_total"])

    print(f"model      : {model_name} ({stats['dim']}-d) on {stats['device']}")
    print(f"texts      : {stats['n_total']:,} distinct narratives")
    print(f"encoded    : {stats['encoded']:,} this run "
          f"(resumed at {stats['resumed_at']:,})")
    if stats["encoded"]:
        print(f"throughput : {stats.get('rate', 0):,.0f} texts/s, "
              f"{stats['seconds'] / 60:.1f} min wall")
    print(f"memmap     : {memmap} ({memmap.stat().st_size / 1e9:.2f} GB)")
    print(f"index      : {index_path} ({n_indexed:,} vectors)")
    print(f"map rows   : {n_mapped:,} complaint_ids -> {stats['n_total']:,} rows"
          + (f"  ({n_unmapped:,} unmapped, --limit run)" if n_unmapped else ""))
    return 0


def cmd_neighbours(args: argparse.Namespace) -> int:
    """ROADMAP Phase 3 acceptance: are the 5 nearest neighbours topically right?

    Deliberately prints text rather than a similarity number. The criterion is
    "correct on inspection", and a cosine of 0.94 tells you nothing about whether
    two complaints are about the same thing.
    """
    import random

    from src.embed import encode
    from src.embed import index as faiss_index

    con = db.connect(read_only=True)
    model_name = args.model or CONFIG.embed.model
    memmap = PATHS.artifacts / f"embeddings.{model_name.split('/')[-1]}.npy"
    index_path = PATHS.artifacts / f"faiss.{model_name.split('/')[-1]}.index"
    if not index_path.exists():
        raise SystemExit(f"no index at {index_path} — run `--phase embed` first")

    vectors = encode.np.load(memmap, mmap_mode="r")
    idx = faiss_index.load(index_path)
    rng = random.Random(CONFIG.seed)  # noqa: S311 - sampling, not cryptography

    rows = con.execute(
        """
        SELECT e.row_idx, any_value(n.text_redacted), any_value(c.product_family)
        FROM embedding_map e
        JOIN narratives n USING (complaint_id)
        JOIN complaints c USING (complaint_id)
        WHERE e.model = ? GROUP BY e.row_idx
        """,
        [model_name],
    ).fetchall()
    by_row = {r[0]: (r[1], r[2]) for r in rows}
    picked = rng.sample(sorted(by_row), min(args.n, len(by_row)))

    for row in picked:
        text, family = by_row[row]
        print(f"\n{'=' * 72}\n[{family}] {' '.join(text.split())[:280]}")
        print("-" * 72)
        for neighbour, score in faiss_index.neighbours(idx, vectors, row, args.k):
            ntext, nfamily = by_row.get(neighbour, ("<not mapped>", "?"))
            print(f"  {score:.3f} [{nfamily}] {' '.join(ntext.split())[:220]}")
    return 0


def latest_run(con, phase: str) -> str:
    """The most recent successful run of `phase`. Everything Phase 2 reports is
    run-scoped, so reading the wrong run is silently wrong, not an error."""
    row = con.execute(
        "SELECT run_id FROM runs WHERE phase = ? AND status = 'ok' "
        "ORDER BY started_at DESC LIMIT 1",
        [phase],
    ).fetchone()
    if not row:
        raise SystemExit(f"no successful '{phase}' run — run it first")
    return row[0]


def cmd_gate(args: argparse.Namespace) -> int:
    """ROADMAP Phase 2 [GATE] — precision, recall, campaign share by family.

    Reports; never tunes. If precision is under the bound the correct response
    is to fix the detector, not `jaccard_threshold` (trap T4).
    """
    import csv as _csv

    from src.dedup import evalset

    con = db.connect(read_only=True)
    run_id = args.run_id or latest_run(con, "dedup")
    print(f"run           : {run_id}\n")

    rows = evalset.read()
    m = evalset.score(con, rows, run_id)
    if len(rows) != 300:
        print(f"WARNING: eval set has {len(rows)} pairs, not the specified 300")
    print(f"pairs scored  : {m['tp'] + m['fp'] + m['fn'] + m['tn']}")
    print(f"tp/fp/fn/tn   : {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}")
    verdict = "PASS" if m["precision"] >= CONFIG.dedup.min_precision else "FAIL"
    print(f"precision     : {m['precision']:.4f}  "
          f"(gate >= {CONFIG.dedup.min_precision})  {verdict}")
    print(f"recall        : {m['recall']:.4f}   (reported, not gated)")
    print(f"f1            : {m['f1']:.4f}")

    # The labels were frozen when jaccard_threshold was 0.85; the detector now
    # runs at 0.88. That cannot inflate precision — a pair the detector declines
    # is never a false positive — but it does depress recall with pairs the
    # detector is *right* to reject at its own operating point. Rescore on the
    # restricted population rather than subtracting an estimate: MinHash SE is
    # ~0.088, so a pair at 0.86 clears 0.88 about half the time and is a true
    # positive, not a definitional miss. Only `recall` is meaningful here —
    # every label in the subset is `dup`, so precision is 1.0 by construction.
    thr = CONFIG.dedup.jaccard_threshold
    label_bar = max(
        (r["true_jaccard"] for r in rows if r["label"] == "not_dup"), default=0.0
    )
    at_thr = [r for r in rows if r["true_jaccard"] >= thr]
    below = [r for r in rows if r["label"] == "dup" and r["true_jaccard"] < thr]
    if below:
        print(f"  labels frozen at ~{label_bar:.2f}, detector runs at {thr}: "
              f"{len(below)} dup-labelled pairs sit below the detector's own "
              f"threshold and are not its errors.")
        try:
            r_at = evalset.score(con, at_thr, run_id)["recall"]
            print(f"  recall over the {len(at_thr)} pairs at or above {thr}: "
                  f"{r_at:.4f}")
        except ValueError:
            print(f"  recall over the {len(at_thr)} pairs at or above {thr}: "
                  f"n/a — no true positives in the subset, which is itself the "
                  f"finding")

    # d9ead66's diagnosis: MinHash overestimates near the bar. These merges are
    # only not false positives because the label bar sits at 0.85.
    over = sum(
        1 for r in rows
        if r["true_jaccard"] < thr and con.execute(
            "SELECT count(*) = 2 AND count(DISTINCT group_id) = 1 FROM dup_groups "
            "WHERE run_id = ? AND complaint_id IN (?, ?)",
            [run_id, r["complaint_id_a"], r["complaint_id_b"]],
        ).fetchone()[0]
    )
    print(f"  merged despite true Jaccard < {thr}: {over}  "
          f"(MinHash overestimate near the bar)")

    for stratum in ("obvious", "hard", "unrelated"):
        sub = [r for r in rows if r["stratum"] == stratum]
        if sub:
            s = evalset.score(con, sub, run_id) if any(
                r["label"] == "dup" for r in sub) else None
            wrong = sum(
                1 for r in sub
                if (con.execute(
                    "SELECT count(*) = 2 AND count(DISTINCT group_id) = 1 "
                    "FROM dup_groups WHERE run_id = ? AND complaint_id IN (?, ?)",
                    [run_id, r["complaint_id_a"], r["complaint_id_b"]],
                ).fetchone()[0]) != (r["label"] == "dup")
            )
            note = f"P={s['precision']:.3f} R={s['recall']:.3f}" if s else "negatives"
            print(f"  {stratum:<10} n={len(sub):<4} errors={wrong:<3} {note}")

    near = PATHS.interim / evalset.NEAR_MISS_CSV
    print("\nrejected candidates (recall denominator the eval file cannot see):")
    if not near.exists():
        print("  none captured — re-run `--phase dedup` to sample them")
    else:
        with near.open(encoding="utf-8") as fh:
            sample = list(_csv.DictReader(fh))
        missed = 0
        for r in sample:
            texts = dict(con.execute(
                "SELECT complaint_id, text_redacted FROM narratives "
                "WHERE complaint_id IN (?, ?)",
                [int(r["complaint_id_a"]), int(r["complaint_id_b"])],
            ).fetchall())
            if len(texts) < 2:
                continue
            tj = evalset.true_jaccard(*texts.values(), CONFIG.dedup.shingle_size)
            r["true_jaccard"] = tj
            missed += tj >= CONFIG.dedup.jaccard_threshold
        print(f"  {len(sample)} sampled, {missed} were true duplicates by exact "
              f"Jaccard ({missed / max(len(sample), 1):.1%} of near misses)")

    print("\ngroup sizes (chaining check — one hop from the seed, by construction):")
    for bucket, n_groups, n_rows in con.execute(
        """
        SELECT CASE WHEN group_size = 1 THEN '1'
                    WHEN group_size <= 10 THEN '2-10'
                    WHEN group_size <= 100 THEN '11-100'
                    WHEN group_size <= 1000 THEN '101-1000'
                    ELSE '>1000' END AS b,
               count(DISTINCT group_id), count(*)
        FROM dup_groups WHERE run_id = ? GROUP BY 1
        ORDER BY min(group_size)
        """,
        [run_id],
    ).fetchall():
        print(f"  {bucket:<10} {n_groups:>9,} groups  {n_rows:>10,} narratives")
    print(f"  largest    {con.execute('SELECT max(group_size) FROM dup_groups '
                                      'WHERE run_id = ?', [run_id]).fetchone()[0]:,}"
          " members")

    _merge_audit(con, run_id, args.merge_audit, show=args.read)

    print("\ncampaign-flagged share by product family "
          "(METHODOLOGY §2.4: credit reporting must be clearly highest):")
    for fam, tot, flagged in con.execute(
        """
        SELECT c.product_family, count(*) AS tot,
               count(*) FILTER (WHERE cm.complaint_id IS NOT NULL) AS flagged
        FROM complaints c
        JOIN narratives n USING (complaint_id)
        LEFT JOIN (SELECT DISTINCT m.complaint_id FROM campaign_members m
                   JOIN campaigns ca USING (campaign_id)
                   WHERE ca.run_id = ? AND ca.flagged) cm USING (complaint_id)
        GROUP BY 1 ORDER BY flagged::DOUBLE / count(*) DESC
        """,
        [run_id],
    ).fetchall():
        print(f"  {fam:<18} {flagged:>9,} / {tot:>9,}  {flagged / tot:6.2%}")

    if args.disputed:
        _print_disputed(con, run_id, rows)
    if args.read:
        _print_reading_material(con, run_id, args.read)
    return 0


def _merge_audit(con, run_id: str, n: int, show: int = 0) -> None:
    """What fraction of real merges does the eval set actually cover?

    `dedup_eval_pairs.csv` samples from `dup_pairs`, so every pair in it has a
    verified edge. Star clustering merges by admitting members to a *seed*, so
    two arbitrary members of a group need no edge between them — in a
    49,457-member star only 49,456 of ~1.2 billion member-pairs are edges. The
    eval set therefore measures precision on a population that is almost none of
    the merges the corpus contains, and no number computed from it can say so.
    This samples same-group pairs the way the corpus holds them.
    """
    import random

    rng = random.Random(CONFIG.seed)  # noqa: S311 - sampling, not cryptography
    groups = con.execute(
        "SELECT group_id, any_value(group_size) FROM dup_groups "
        "WHERE run_id = ? AND group_size >= 100 GROUP BY 1 ORDER BY 1",
        [run_id],
    ).fetchall()
    if not groups:
        return
    picked = rng.sample(groups, min(n, len(groups)))

    direct = 0
    two_hop: list[tuple] = []
    for gid, size in picked:
        members = [
            r[0] for r in con.execute(
                "SELECT complaint_id FROM dup_groups WHERE run_id = ? AND group_id = ? "
                "ORDER BY complaint_id", [run_id, gid],
            ).fetchall()
        ]
        a, b = sorted(rng.sample(members, 2))
        if con.execute(
            "SELECT count(*) FROM dup_pairs WHERE complaint_id_a = ? AND complaint_id_b = ?",
            [a, b],
        ).fetchone()[0]:
            direct += 1
        else:
            two_hop.append((gid, size, a, b))

    print(f"\nmerge audit — {len(picked)} random same-group pairs from groups of 100+:")
    print(f"  {direct} have a verified edge, {len(two_hop)} are seed-mediated (2 hops).")
    print(f"  The eval set can only ever sample the first kind, i.e. "
          f"{direct / len(picked):.0%} of merges as the corpus actually holds them.")
    for gid, size, a, b in two_hop[:show]:
        print(f"\n  === {gid} ({size:,} members), no edge between these two")
        for cid in (a, b):
            text = con.execute(
                "SELECT text_redacted FROM narratives WHERE complaint_id = ?", [cid]
            ).fetchone()
            print(f"    [{cid}] " + " ".join((text[0] if text else "").split())[:300])


def _print_disputed(con, run_id: str, rows: list[dict]) -> None:
    """Every pair the detector and the reference label disagree on, with text.

    Precision is the gated quantity and the labels are an exact-Jaccard proxy,
    not a judgement about whether two complaints are the same filing. So the
    false positives get read by a human before the gate is called either way —
    that is the part of ROADMAP Phase 2's "hand-label" the proxy cannot supply.
    """
    print(f"\n{'=' * 72}\nDISPUTED PAIRS — read these before calling the gate\n{'=' * 72}")
    for row in rows:
        merged = con.execute(
            "SELECT count(*) = 2 AND count(DISTINCT group_id) = 1 FROM dup_groups "
            "WHERE run_id = ? AND complaint_id IN (?, ?)",
            [run_id, row["complaint_id_a"], row["complaint_id_b"]],
        ).fetchone()[0]
        if merged == (row["label"] == "dup"):
            continue
        kind = "FALSE POSITIVE (merged, labelled not_dup)" if merged else \
               "false negative (not merged, labelled dup)"
        print(f"\n--- {kind}  tj={row['true_jaccard']:.4f}  {row['stratum']}")
        for key in ("complaint_id_a", "complaint_id_b"):
            text = con.execute(
                "SELECT text_redacted FROM narratives WHERE complaint_id = ?",
                [row[key]],
            ).fetchone()
            print(f"  [{row[key]}] " + " ".join((text[0] if text else "").split())[:600])


def _print_reading_material(con, run_id: str, n: int) -> None:
    """Trap T2's countermeasure: the excerpts have to be read by a human.

    `n` flagged campaigns and `n` unflagged high-volume groups, largest first,
    with one excerpt from the group's representative. No statistic substitutes
    for seeing whether the flagged set is obviously templated.
    """
    print(f"\n{'=' * 72}\nREAD: {n} flagged campaigns, largest first\n{'=' * 72}")
    for cid, nc, fam, sigs, boiler, cv, burst in con.execute(
        "SELECT campaign_id, n_complaints, product_family, n_signals, "
        "boilerplate_score, length_cv, burstiness FROM campaigns "
        "WHERE run_id = ? AND flagged ORDER BY n_complaints DESC LIMIT ?",
        [run_id, n],
    ).fetchall():
        text = con.execute(
            "SELECT n.text_redacted FROM campaign_members m "
            "JOIN narratives n USING (complaint_id) "
            "WHERE m.campaign_id = ? ORDER BY n.complaint_id LIMIT 1",
            [cid],
        ).fetchone()
        print(f"\n[{nc:,} complaints] {fam}  signals={sigs} "
              f"boiler={boiler:.2f} cv={cv:.2f} burst={burst:.1f}")
        print("  " + " ".join((text[0] if text else "").split())[:400])

    print(f"\n{'=' * 72}\nREAD: {n} unflagged groups, largest first\n{'=' * 72}")
    for gid, size, fam in con.execute(
        """
        SELECT d.group_id, any_value(d.group_size), any_value(c.product_family)
        FROM dup_groups d JOIN complaints c USING (complaint_id)
        WHERE d.run_id = ? AND d.group_id NOT IN (
          SELECT DISTINCT m.group_id FROM campaign_members mm
          JOIN campaigns ca USING (campaign_id)
          JOIN dup_groups m ON m.complaint_id = mm.complaint_id AND m.run_id = ?
          WHERE ca.run_id = ? AND ca.flagged)
        GROUP BY 1 ORDER BY any_value(d.group_size) DESC LIMIT ?
        """,
        [run_id, run_id, run_id, n],
    ).fetchall():
        text = con.execute(
            "SELECT n.text_redacted FROM dup_groups d "
            "JOIN narratives n USING (complaint_id) "
            "WHERE d.run_id = ? AND d.group_id = ? AND d.is_representative",
            [run_id, gid],
        ).fetchone()
        print(f"\n[{size:,} complaints] {fam}  {gid}")
        print("  " + " ".join((text[0] if text else "").split())[:400])


def cmd_adjudicate(args: argparse.Namespace) -> int:
    """Print a stratum of eval pairs for blind judgement (METHODOLOGY §2.4.1).

    Withholds the detector's decision, the proxy label, and `true_jaccard`.
    Showing any of them anchors the adjudicator to the answer being checked,
    which is the whole reason the disagreement is worth adjudicating at all.
    Order is shuffled by `CONFIG.seed` so position carries no information.
    """
    import random

    from src.dedup import evalset

    con = db.connect(read_only=True)
    rows = [r for r in evalset.read() if r["stratum"] == args.stratum]
    if not rows:
        raise SystemExit(f"no pairs in stratum {args.stratum!r}")
    random.Random(CONFIG.seed).shuffle(rows)  # noqa: S311 - ordering, not cryptography
    rows = rows[args.offset : args.offset + args.limit]

    print(f"# {len(rows)} pairs from stratum '{args.stratum}', "
          f"offset {args.offset}, seed {CONFIG.seed}")
    print("# blind: detector decision, proxy label and true_jaccard withheld")
    print("# rule: METHODOLOGY §2.4.1\n")
    for row in rows:
        print(f"=== {row['complaint_id_a']} / {row['complaint_id_b']}")
        for key in ("complaint_id_a", "complaint_id_b"):
            text = con.execute(
                "SELECT text_redacted FROM narratives WHERE complaint_id = ?",
                [row[key]],
            ).fetchone()
            body = " ".join((text[0] if text else "").split())
            clipped = body[: args.chars]
            print(f"  [{row[key]}] {clipped}"
                  + (f" …(+{len(body) - args.chars} chars)" if len(body) > args.chars else ""))
        print()
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
    "embed": phase_embed,
}

PLANNED: dict[str, str] = {
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
    p_run.add_argument("--model", help="embedding model (default: config)")
    p_run.add_argument("--batch", type=int, help="encode batch size")
    p_run.add_argument("--device", help="cpu | mps | cuda (default: autodetect)")
    p_run.add_argument("--limit", type=int, help="encode only the first N texts")
    p_run.set_defaults(func=cmd_run)

    p_gate = sub.add_parser("gate", help="Phase 2 gate report: precision, recall, "
                                         "campaign share")
    p_gate.add_argument("--run-id", help="default: latest successful dedup run")
    p_gate.add_argument("--merge-audit", type=int, default=40, metavar="N",
                        help="sample N same-group pairs to measure how much of "
                             "the merge population the eval set can see")
    p_gate.add_argument("--disputed", action="store_true",
                        help="print every pair the detector and the label "
                             "disagree on, with narrative text, for hand reading")
    p_gate.add_argument("--read", type=int, default=0, metavar="N",
                        help="also print N flagged campaigns and N unflagged "
                             "groups for the manual read (trap T2)")
    p_gate.set_defaults(func=cmd_gate)

    p_adj = sub.add_parser("adjudicate", help="print eval pairs for blind judgement")
    p_adj.add_argument("--stratum", default="hard", choices=["obvious", "hard", "unrelated"])
    p_adj.add_argument("--offset", type=int, default=0)
    p_adj.add_argument("--limit", type=int, default=25)
    p_adj.add_argument("--chars", type=int, default=700)
    p_adj.set_defaults(func=cmd_adjudicate)

    p_nn = sub.add_parser("neighbours", help="Phase 3 acceptance: nearest-neighbour read")
    p_nn.add_argument("-n", type=int, default=10, help="how many query narratives")
    p_nn.add_argument("-k", type=int, default=5, help="neighbours per query")
    p_nn.add_argument("--model")
    p_nn.set_defaults(func=cmd_neighbours)

    p_tax = sub.add_parser("taxonomy", help="label vocabulary by volume")
    p_tax.set_defaults(func=cmd_taxonomy)

    p_runs = sub.add_parser("runs", help="show the run registry")
    p_runs.add_argument("-n", type=int, default=20)
    p_runs.set_defaults(func=cmd_runs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
