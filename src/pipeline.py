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
    artifact_paths = encode.embedding_artifact_paths(PATHS.artifacts, model_name)
    memmap = artifact_paths.memmap
    index_path = artifact_paths.index

    # `limit` belongs in params even when it is None. A --limit smoke run wrote a
    # runs row that was indistinguishable from a full encode — same phase, same
    # config_hash, just a smaller output_rows that nothing interprets. Provenance
    # that cannot tell a test from the real thing is the registry lying quietly.
    params = {"model": model_name, "limit": args.limit}
    with db.run(con, "embed", CONFIG, params=params) as r:
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
    artifact_paths = encode.embedding_artifact_paths(PATHS.artifacts, model_name)
    memmap = artifact_paths.memmap
    index_path = artifact_paths.index
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


def _representatives(con, dedup_run: str, model: str, cutoff=None) -> tuple[dict, dict]:
    """`{product_family: memmap row indices}` for this dedup run's representatives.

    Clustering consumes representatives only (METHODOLOGY §2.3) — one document
    per dup group, so a 24,507-member template contributes one point rather than
    24,507. Phase 5 expands back through `dup_groups` for its counts.
    """
    import numpy as np

    rows = con.execute(
        """
        SELECT c.product_family, e.row_idx, d.complaint_id
        FROM dup_groups d
        JOIN complaints c USING (complaint_id)
        JOIN embedding_map e USING (complaint_id)
        WHERE d.run_id = ? AND d.is_representative AND e.model = ?
          AND (? IS NULL OR c.date_received < ?)
        ORDER BY c.product_family, d.complaint_id
        """,
        [dedup_run, model, cutoff, cutoff],
    ).fetchall()
    out: dict[str, list] = {}
    ids: dict[str, list] = {}
    for family, row_idx, complaint_id in rows:
        out.setdefault(family, []).append(row_idx)
        ids.setdefault(family, []).append(complaint_id)
    return (
        {f: np.asarray(v, dtype=np.int64) for f, v in out.items()},
        {f: np.asarray(v, dtype=np.int64) for f, v in ids.items()},
    )


def phase_cluster(args: argparse.Namespace) -> int:
    """Phase 4 [GATE] — UMAP + HDBSCAN per family, assignment, novelty."""
    import numpy as np

    from src.cluster import assign as assign_mod
    from src.cluster import fit as fit_mod
    from src.embed import encode
    from src.ids import cluster_id as make_cluster_id

    con = db.bootstrap()
    model = args.model or CONFIG.embed.model
    memmap = encode.embedding_artifact_paths(PATHS.artifacts, model).memmap
    if not memmap.exists():
        raise SystemExit(f"no embeddings at {memmap} — run `--phase embed` first")
    vectors = np.load(memmap, mmap_mode="r")

    dedup_run = args.dedup_run or latest_run(con, "dedup")
    as_of = _as_of(con, getattr(args, "cutoff", None))
    by_family, ids_by_family = _representatives(
        con, dedup_run, model, getattr(args, "cutoff", None)
    )
    families = [f for f in sorted(by_family, key=lambda f: -len(by_family[f]))
                if not args.family or f == args.family]

    # B3 reuses this whole stage with BERTopic's default hyperparameters and an
    # identity dedup run; `system` is what keeps its clusters, signals and
    # backtest rows separable from HarmScope's.
    cfg = getattr(args, "cluster_cfg", None) or CONFIG.cluster
    system = getattr(args, "system", None)
    params = {"model": model, "dedup_run": dedup_run, "limit": args.limit,
              "family": args.family, "system": system,
              "cutoff": str(getattr(args, "cutoff", None) or "")}
    totals = {"clusters": 0, "assigned": 0, "reps": 0}
    with db.run(con, "cluster", CONFIG, params=params) as r:
        con.execute(f"DELETE FROM cluster_novelty WHERE cluster_id LIKE '{r.run_id}:%'")
        con.execute(f"DELETE FROM cluster_members WHERE cluster_id LIKE '{r.run_id}:%'")
        con.execute("DELETE FROM related_clusters WHERE cluster_id_a LIKE ?",
                    [f"{r.run_id}:%"])
        con.execute("DELETE FROM clusters WHERE run_id = ?", [r.run_id])

        print(f"dedup run  : {dedup_run}")
        print(f"model      : {model}\nas_of      : {as_of}\n")
        centroids_by_family: dict[str, np.ndarray] = {}
        cluster_ids: dict[str, list[str]] = {}

        for family in families:
            rows, complaint_ids = by_family[family], ids_by_family[family]
            totals["reps"] += len(rows)
            result = fit_mod.fit_family(
                vectors, rows, family, cfg, CONFIG.seed,
                sample_size=args.limit or None,
            )
            if result is None or result.n_clusters == 0:
                continue

            # Every representative goes through the same rule, sampled or not —
            # see the note in cluster/fit.py on why that symmetry matters.
            labels, sims = assign_mod.assign(
                vectors, rows, result.centroids, cfg.assign_max_distance
            )
            coherence = assign_mod.coherence(labels, sims, result.n_clusters)
            medoid = assign_mod.medoids(vectors, rows, labels, sims, result.n_clusters)
            assigned = labels != assign_mod.NOISE
            totals["assigned"] += int(assigned.sum())

            local_ids = []
            cluster_rows, member_rows = [], []
            for c in range(result.n_clusters):
                member = labels == c
                n_members = int(member.sum())
                if n_members == 0:
                    local_ids.append(None)
                    continue
                cid = make_cluster_id(r.run_id, family, c)
                local_ids.append(cid)
                cluster_rows.append((
                    cid, r.run_id, family, n_members,
                    float(result.persistence[c]) if c < len(result.persistence) else 0.0,
                    float(coherence[c]), int(medoid[c]), as_of,
                ))
                for pos in np.flatnonzero(member):
                    member_rows.append((
                        cid, int(complaint_ids[pos]), float(sims[pos]),
                        bool(rows[pos] == medoid[c]),
                    ))
            con.executemany(
                "INSERT INTO clusters (cluster_id, run_id, product_family, n_members,"
                " persistence, coherence, centroid_idx, as_of) VALUES (?,?,?,?,?,?,?,?)",
                cluster_rows,
            )
            con.executemany(
                "INSERT INTO cluster_members (cluster_id, complaint_id, "
                "membership_prob, is_exemplar) VALUES (?, ?, ?, ?)",
                member_rows,
            )
            totals["clusters"] += len(cluster_rows)
            centroids_by_family[family] = result.centroids
            cluster_ids[family] = local_ids
            print(f"  {'':<18} {len(cluster_rows):>5} kept  "
                  f"assigned {assigned.mean():5.1%}  "
                  f"mean coherence {coherence[coherence > 0].mean():.3f}")

        # `cluster_novelty` and `related_clusters` are the descriptive layer:
        # the panel, the signals and the backtest never join either, only
        # `pipeline alerts` and METHODOLOGY §4.2 read them. EVALUATION §2 runs
        # B3 with "no novelty scoring", and skipping both changes no B3 number.
        # `related` is also O(k_a x k_b) per family pair, which at BERTopic's
        # min_cluster_size of 10 is tens of thousands of centroids a side.
        if getattr(args, "no_descriptive", False):
            n_novel = n_related = 0
        else:
            n_novel = _write_novelty(con, r.run_id)
            n_related = _write_related(con, centroids_by_family, cluster_ids)
        checks.expect_rows(con, "clusters", min=1)
        r.finish(output_rows=totals["clusters"], input_rows=totals["reps"])

    print(f"\nclusters   : {totals['clusters']:,} over {totals['reps']:,} "
          f"representatives")
    print(f"assigned   : {totals['assigned']:,} "
          f"({totals['assigned'] / max(totals['reps'], 1):.1%}); the rest are noise")
    print(f"novel      : {n_novel:,} pass novelty + coherence + persistence")
    print(f"related    : {n_related:,} cross-family links")
    print("\nGATE: `pipeline stability` and `pipeline ablation` are the two "
          "numbers ROADMAP Phase 4 turns on.")
    return 0


def _write_novelty(con, run_id: str) -> int:
    """Score every cluster against the taxonomy it is supposed to be new to."""
    from src.cluster import novelty as novelty_mod

    members = _cluster_label_members(con, run_id)
    space = dict(con.execute(
        "SELECT product_family, count(DISTINCT (issue_std, sub_issue_std)) "
        "FROM complaints GROUP BY 1"
    ).fetchall())
    meta = dict(con.execute(
        "SELECT cluster_id, (product_family, coherence, persistence) FROM clusters "
        "WHERE run_id = ?", [run_id],
    ).fetchall())

    rows = []
    for cluster_id, pairs in members.items():
        family, coherence, persistence = meta[cluster_id]
        nov = novelty_mod.score_cluster(
            [novelty_mod.label_of(i, s) for i, s in pairs],
            space.get(family, 2), CONFIG.novelty,
        )
        rows.append((
            cluster_id, nov.dominant_label, nov.dominant_share, nov.entropy,
            _family_nmi(con, run_id, family), nov.score,
            novelty_mod.is_novel(nov, coherence or 0.0, persistence or 0.0,
                                 CONFIG.novelty),
        ))
    con.executemany(
        "INSERT INTO cluster_novelty (cluster_id, dominant_label, "
        "dominant_label_share, label_entropy, normalized_mutual_info, "
        "novelty_score, is_novel) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    return sum(1 for r in rows if r[-1])


_NMI_CACHE: dict[tuple[str, str], float] = {}


def _family_nmi(con, run_id: str, family: str) -> float:
    """NMI between cluster assignment and label assignment, within a family.

    METHODOLOGY §5 calls for this "computed globally"; the schema stores it per
    cluster. It is a property of a whole partition, not of one cluster, so the
    family's value is written onto each of its clusters and cached rather than
    recomputed per row.
    """
    key = (run_id, family)
    if key in _NMI_CACHE:
        return _NMI_CACHE[key]
    from sklearn.metrics import normalized_mutual_info_score

    pairs = con.execute(
        """
        SELECT m.cluster_id, coalesce(c.issue_std, '') || '|' || coalesce(c.sub_issue_std, '')
        FROM cluster_members m
        JOIN clusters cl USING (cluster_id)
        JOIN complaints c ON c.complaint_id = m.complaint_id
        WHERE cl.run_id = ? AND cl.product_family = ?
        """,
        [run_id, family],
    ).fetchall()
    value = 0.0
    if len(pairs) > 1:
        value = float(normalized_mutual_info_score([a for a, _ in pairs],
                                                   [b for _, b in pairs]))
    _NMI_CACHE[key] = value
    return value


def _cluster_label_members(
    con, run_id: str, family: str | None = None
) -> dict[str, list[tuple[str | None, str | None]]]:
    rows = con.execute(
        """
        SELECT m.cluster_id, c.issue_std, c.sub_issue_std
        FROM cluster_members m
        JOIN clusters cl USING (cluster_id)
        JOIN complaints c ON c.complaint_id = m.complaint_id
        WHERE cl.run_id = ? AND (? IS NULL OR cl.product_family = ?)
        """,
        [run_id, family, family],
    ).fetchall()
    out: dict[str, list] = {}
    for cluster_id, issue, sub_issue in rows:
        out.setdefault(cluster_id, []).append((issue, sub_issue))
    return out


def _write_related(con, centroids: dict, cluster_ids: dict) -> int:
    from src.cluster import assign as assign_mod

    links = assign_mod.related(centroids, CONFIG.cluster.related_min_similarity)
    rows = []
    for fa, ia, fb, ib, sim in links:
        a, b = cluster_ids[fa][ia], cluster_ids[fb][ib]
        if a and b:
            rows.append((a, b, sim) if a < b else (b, a, sim))
    if rows:
        con.executemany(
            "INSERT INTO related_clusters (cluster_id_a, cluster_id_b, similarity) "
            "VALUES (?, ?, ?) ON CONFLICT DO NOTHING", rows,
        )
    return len(rows)


def latest_run(con, phase: str) -> str:
    """The most recent successful *real* run of `phase`.

    Runs over a deliberately altered input are excluded, and that is not a
    nicety. The Phase 5 negative control permutes cluster labels and writes a
    fully-formed `signals` run; being the newest, it became what
    `latest_run('signals')` returned, so `pipeline alerts` with no arguments
    reported alerts computed on shuffled data with nothing in the output saying
    so. Same for a `--limit` smoke run. Everything downstream is run-scoped, so
    reading the wrong run is silently wrong rather than an error — which is the
    entire failure mode this project keeps rediscovering.
    """
    row = con.execute(
        """
        SELECT run_id FROM runs
        WHERE phase = ? AND status = 'ok'
          AND coalesce(json_extract(params_json, '$.params.shuffle'), '0') = '0'
          AND coalesce(json_extract(params_json, '$.params.limit'), 'null') = 'null'
        ORDER BY started_at DESC LIMIT 1
        """,
        [phase],
    ).fetchone()
    if not row:
        raise SystemExit(
            f"no successful full-input '{phase}' run — run it first "
            f"(runs over --limit or --shuffle inputs do not count)"
        )
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


ALERT_SQL = """
WITH fired AS (
  SELECT cluster_id, company_id,
         max(CASE WHEN method = 'ebgm' THEN statistic END)   AS eb05,
         min(CASE WHEN method = 'ebgm' THEN q_value END)     AS q_value,
         max(CASE WHEN method IN ('ewma', 'pelt') THEN 1 ELSE 0 END) AS changed,
         min(CASE WHEN method IN ('ewma', 'pelt') THEN period_month END) AS change_month,
         max(n_supporting)        AS n_supporting,
         max(n_supporting_groups) AS n_groups
  FROM signals WHERE run_id = ? GROUP BY 1, 2
)
SELECT c.product_family, f.company_id, f.cluster_id, f.eb05, f.q_value,
       f.changed, f.change_month, f.n_supporting, f.n_groups,
       c.coherence, c.persistence, n.novelty_score, n.is_novel, n.dominant_label
FROM fired f
JOIN clusters c USING (cluster_id)
LEFT JOIN cluster_novelty n USING (cluster_id)
WHERE c.coherence >= ?
  AND f.n_groups >= ?
  AND (f.q_value <= ? OR f.changed = 1)
  AND (? = 'all'
       OR (? = 'novel' AND n.novelty_score >= ?)
       OR (? = 'known' AND n.novelty_score <  ?))
ORDER BY f.eb05 DESC NULLS LAST, f.n_groups DESC
LIMIT ?
"""


def cmd_alerts(args: argparse.Namespace) -> int:
    """METHODOLOGY §6.3 — the joint criteria, ranked by EB05.

    An alert is not a significant test. It is a coherent cluster, carrying
    enough distinct dup-groups to not be one filing, that either fires
    disproportionality below the FDR bound or shows a changepoint. Campaign
    -flagged complaints never entered the panel, so that criterion is satisfied
    upstream rather than filtered here.

    Two tracks, labelled separately as §6.3 allows: `novel` is the contribution,
    `known` is the sanity check that the machinery finds things anyone would
    already know about.
    """
    con = db.connect(read_only=True)
    run_id = args.run_id or latest_run(con, "signals")
    track = args.track
    rows = con.execute(ALERT_SQL, [
        run_id, CONFIG.novelty.min_coherence, CONFIG.signals.min_supporting_groups,
        CONFIG.signals.fdr_alpha, track, track, CONFIG.novelty.threshold,
        track, CONFIG.novelty.threshold, args.n,
    ]).fetchall()

    print(f"run    : {run_id}")
    print(f"track  : {track}   (coherence >= {CONFIG.novelty.min_coherence}, "
          f"groups >= {CONFIG.signals.min_supporting_groups}, "
          f"q <= {CONFIG.signals.fdr_alpha} or changepoint)")
    print(f"alerts : {len(rows)} shown\n")
    if not rows:
        print("none — which is a finding, not an error")
        return 0

    for (family, company, cluster, eb05, q, changed, change_month,
         n_sup, n_groups, coh, _pers, novelty, is_novel, dominant) in rows:
        name = con.execute(
            "SELECT canonical_name FROM company_canonical WHERE company_id = ?",
            [company],
        ).fetchone()
        label = (name[0] if name else company)[:38]
        print(f"[{family}] {label}")
        print(f"  EB05 {eb05 if eb05 is None else round(eb05, 2)}  "
              f"q={'—' if q is None else f'{q:.2e}'}  "
              f"{'changepoint ' + str(change_month) if changed else 'no changepoint'}")
        print(f"  {n_groups:,} groups / {n_sup:,} complaints   "
              f"coherence {coh:.2f}  novelty {novelty:.2f}"
              f"{'  NOVEL' if is_novel else ''}")
        print(f"  nearest existing label: {dominant}")
        if args.evidence:
            for (text,) in con.execute(
                """
                SELECT n.text_redacted FROM cluster_members m
                JOIN narratives n USING (complaint_id)
                WHERE m.cluster_id = ? ORDER BY m.is_exemplar DESC LIMIT 2
                """, [cluster],
            ).fetchall():
                print("    • " + " ".join(text.split())[:200])
        print()
    return 0


def cmd_refit(args: argparse.Namespace) -> int:
    """EVALUATION §1.1 — rebuild every date-dependent stage at one cutoff.

    ROADMAP Phase 6 gates on this running end to end with a documented wall
    time. What is refit and what is not comes from §1.1.1 and is not a
    judgement call: it follows from whether a stage's output can depend on when
    a complaint arrived.

        embeddings, dup_pairs   reused   a vector and a pairwise Jaccard do not
                                         change because another complaint exists
        grouping, campaigns     refit    seed selection and time-windowed
                                         features are both date-dependent
        clustering, novelty     refit    trap T3: cluster definitions are THE
                                         leakage vector
        panel, signals          refit    obviously

    Nothing here filters an existing artifact by date. Each stage is recomputed
    from inputs restricted to `date_received < cutoff`, which is the difference
    §1.2 exists to insist on.
    """
    import time
    from datetime import date as _date

    cutoff = _date.fromisoformat(args.cutoff)
    con = db.bootstrap()
    n_before = con.execute(
        "SELECT count(*) FROM complaints WHERE date_received < ? AND has_narrative",
        [cutoff],
    ).fetchone()[0]
    as_of = con.execute(
        "SELECT max(date_received) FROM complaints WHERE date_received < ?", [cutoff]
    ).fetchone()[0]
    print(f"cutoff      : {cutoff}   as_of {as_of}")
    print(f"in scope    : {n_before:,} narrative-bearing complaints\n")

    timings: list[tuple[str, float]] = []
    t_all = time.time()

    # --- grouping + campaigns, from the reused dup_pairs ---------------------
    from src.dedup import detect

    t0 = time.time()
    with db.run(con, "dedup", CONFIG, params={"cutoff": str(cutoff), "limit": None}) as r:
        dedup_run = r.run_id
        n_groups, n_rows = detect.assign_groups(con, dedup_run, as_of, cutoff=cutoff)
        n_cand, n_flag = detect.build_campaigns(con, dedup_run, CONFIG, as_of)
        r.finish(output_rows=n_rows, input_rows=n_before)
    timings.append(("grouping + campaigns", time.time() - t0))
    print(f"  groups    : {n_groups:,} over {n_rows:,}; campaigns {n_cand:,} "
          f"({n_flag:,} flagged)   [{timings[-1][1] / 60:.1f} min]")

    # --- clustering + novelty ------------------------------------------------
    t0 = time.time()
    cluster_args = argparse.Namespace(
        model=args.model, dedup_run=dedup_run, family=None, limit=None,
        cutoff=cutoff,
    )
    phase_cluster(cluster_args)
    timings.append(("clustering + novelty", time.time() - t0))

    # --- panel + signals -----------------------------------------------------
    t0 = time.time()
    cluster_run = latest_run(con, "cluster")
    signal_args = argparse.Namespace(
        run_id=cluster_run, dedup_run=dedup_run, shuffle=0, cutoff=cutoff,
    )
    phase_signals(signal_args)
    timings.append(("panel + signals", time.time() - t0))

    total = time.time() - t_all
    print(f"\n{'=' * 60}\nFULL REFIT AT {cutoff} — wall time")
    for name, seconds in timings:
        print(f"  {name:<24} {seconds / 60:6.1f} min")
    print(f"  {'TOTAL':<24} {total / 60:6.1f} min")
    print(f"\n  8 annual cutoffs at this rate: {8 * total / 3600:.1f} h")
    return 0


def cmd_worklist(args: argparse.Namespace) -> int:
    """Emit blinded adjudication worklists (EVALUATION §1.3 steps 1-2).

    One CSV per action, containing cluster text and nothing that indicates
    whether — or how strongly — anything fired. Decoys from unrelated companies
    are shuffled in, because twenty candidates all drawn from one company would
    itself tell the adjudicator the system fired on that company.
    """
    from src.evaluation import backtest as bt
    from src.evaluation import worklist as wl

    con = db.connect(read_only=True)
    out_dir = PATHS.interim / "adjudication"
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = con.execute(
        """
        SELECT action_id, company_id, filed_date, harm_summary
        FROM enforcement_actions WHERE usable AND company_id IS NOT NULL
        ORDER BY filed_date
        """
    ).fetchall()

    written = skipped = 0
    for action_id, company_id, filed, _summary in actions:
        cutoff = bt.cutoff_for(filed)
        signals_run = bt.run_for_cutoff(con, "signals", cutoff, args.system)
        cluster_run = bt.run_for_cutoff(con, "cluster", cutoff, args.system)
        if not signals_run or not cluster_run:
            skipped += 1
            continue
        candidates, _truth = wl.select(
            con, signals_run, cluster_run, company_id,
            CONFIG.eval.adjudication_top_k, CONFIG.eval.adjudication_decoys,
            CONFIG.seed,
        )
        if not candidates:
            skipped += 1
            continue
        wl.write(action_id, candidates, out_dir / f"{args.system}__{action_id}.csv")
        written += 1

    print(f"system   : {args.system}")
    print(f"worklists: {written} written to {out_dir}")
    print(f"skipped  : {skipped} (no refit, or no candidates)")
    print("\nEach row has match_quality blank. Fill with strong / partial / none")
    print("against the action's own description, then `pipeline adjudicate`.")
    return 0


def cmd_verdicts(args: argparse.Namespace) -> int:
    """Read filled-in worklists into `backtest_links` (§1.3 steps 3-4).

    Named `verdicts`, not `adjudicate`: that name already belongs to the Phase 2
    dedup-pair reader, and two commands with one name is how someone runs the
    wrong one.
    """
    from src.evaluation import adjudicate as adj

    con = db.bootstrap()
    directory = PATHS.interim / "adjudication"
    files = sorted(directory.glob(f"{args.system}__*.csv"))
    if not files:
        raise SystemExit(f"no worklists in {directory} — run `worklist` first")

    total = 0
    for path in files:
        rows = adj.read_worklist(path)
        verdicts = adj.parse(rows)
        if verdicts:
            total += adj.record(con, verdicts, args.adjudicator)
    counts = dict(con.execute(
        "SELECT match_quality, count(*) FROM backtest_links GROUP BY 1"
    ).fetchall())
    print(f"recorded : {total} verdicts by {args.adjudicator!r}")
    print(f"in table : {counts}")
    print("\nOnly 'strong' counts as a detection (§1.3 step 5). Re-run the "
          "backtest with --strong-only.")
    return 0


def phase_label(args: argparse.Namespace) -> int:
    """Phase 8 — label clusters with the LLM. Descriptive only.

    Imported inside the function on purpose: `src/pipeline.py` is imported by
    every detection phase, so a module-level import would make the whole
    pipeline fail once `src/llm/` is deleted — turning LLM_LAYER §1's
    determinism test from a property into a crash. `tests/test_llm.py` pins it.
    """
    from src.llm import run as llm_run

    con = db.bootstrap()
    cluster_run = args.run_id or latest_run(con, "cluster")
    signals_run = args.signals_run or latest_run(con, "signals")
    model = args.model or CONFIG.embed.dev_model

    params = {"cluster_run": cluster_run, "signals_run": signals_run,
              "limit": args.limit, "control_n": args.control_n}
    with db.run(con, "label", CONFIG, params=params) as r:
        stats = llm_run.run(con, cluster_run, signals_run, args.control_n,
                            args.limit, model, run_id=r.run_id)
        r.finish(output_rows=stats.labelled)

    print(f"\nlabelled       : {stats.labelled:,}")
    print(f"cache hits     : {stats.cached:,}")
    print(f"refused        : {stats.refused:,}")
    print(f"failed         : {stats.failed:,}")
    print(f"skipped        : {stats.skipped:,} (no narratives)")
    print(f"input tokens   : {stats.input_tokens:,}")
    print(f"output tokens  : {stats.output_tokens:,}")
    print(f"latency        : {stats.latency_seconds:.2f} s")
    print(f"estimated cost : ${stats.estimated_cost_usd:.6f} "
          "(estimated, not invoice)")
    print("  next: `label-verify export --n 50 --output PATH` for the "
          "LLM_LAYER §2.5 human read")
    return 0


def phase_baselines(args: argparse.Namespace) -> int:
    """Phase 7 — build a baseline's units, then run the identical pipeline.

    Only the definition of a "cluster" changes. Everything after it is the same
    code, which is what EVALUATION §2 means by "the same statistical machinery".
    """
    from src.evaluation import baselines

    system = args.system
    if system != "B3" and system not in baselines.BUILDERS:
        raise SystemExit(
            f"no builder for {system}; have {[*baselines.BUILDERS, 'B3']}"
        )

    con = db.bootstrap()
    cutoff = getattr(args, "cutoff", None)
    as_of = _as_of(con, cutoff)
    if system == "B3":
        return _baseline_b3(con, args, cutoff, as_of)

    dedup_run = args.dedup_run or (
        backtest_run_for(con, cutoff) if cutoff else latest_run(con, "dedup")
    )

    params = {"system": system, "dedup_run": dedup_run, "limit": None,
              "cutoff": str(cutoff or "")}
    with db.run(con, "cluster", CONFIG, params=params) as r:
        n_clusters, n_members = baselines.BUILDERS[system](
            con, r.run_id, dedup_run, as_of, cutoff
        )
        checks.expect_rows(con, "clusters", min=1)
        r.finish(output_rows=n_clusters, input_rows=n_members)

    print(f"{system}: {n_clusters:,} units over {n_members:,} representatives")
    print(f"  cluster run: {r.run_id}")
    print(f"  next: `run --phase signals --run-id {r.run_id} "
          f"--dedup-run {dedup_run}` puts it through the identical detection path")
    return 0


def _baseline_b3(con, args, cutoff, as_of) -> int:
    """B3 — off-the-shelf BERTopic defaults, no dedup, no novelty scoring.

    Two things make B3 different from B0/B1/B2, and both are expressed as
    inputs to the existing stage rather than as a parallel implementation:
    BERTopic's hyperparameters replace HarmScope's, and the dedup run is an
    identity one. `src/evaluation/baselines.py` documents why each falls out of
    the unchanged code path.
    """
    from src.evaluation import baselines

    with db.run(con, "dedup_identity", CONFIG,
                params={"system": "B3", "limit": None,
                        "cutoff": str(cutoff or "")}) as r:
        n_rows = baselines.identity_groups(con, r.run_id, as_of, cutoff)
        r.finish(output_rows=n_rows, input_rows=n_rows)
    identity_run = r.run_id
    print(f"identity dedup: {n_rows:,} singleton groups  ({identity_run})\n")

    rc = phase_cluster(argparse.Namespace(
        model=args.model, dedup_run=identity_run, family=None, limit=None,
        cutoff=cutoff, system="B3", cluster_cfg=baselines.bertopic_cluster_config(),
        no_descriptive=True,
    ))
    cluster_run = latest_run(con, "cluster")
    print(f"  next: `run --phase signals --run-id {cluster_run} "
          f"--dedup-run {identity_run}` puts it through the identical detection path")
    return rc


def backtest_run_for(con, cutoff):
    """The dedup run belonging to a cutoff's refit."""
    from src.evaluation import backtest as bt

    run = bt.run_for_cutoff(con, "dedup", cutoff)
    if run is None:
        raise SystemExit(f"no dedup refit at {cutoff} — run `refit --cutoff` first")
    return run


def phase_backtest(args: argparse.Namespace) -> int:
    """Phase 6 — evaluate every usable action against its own cutoff's refit."""
    from src.evaluation import backtest

    con = db.bootstrap()
    params = {"system": args.system, "strong_only": args.strong_only, "limit": None}
    with db.run(con, "backtest", CONFIG, params=params) as r:
        outcomes, missing = backtest.evaluate(
            con, CONFIG.signals.min_supporting_groups, CONFIG.signals.fdr_alpha,
            strong_only=args.strong_only, system=args.system,
        )
        n = backtest.write(con, r.run_id, args.system, outcomes)
        r.finish(output_rows=n)

    stats = backtest.summarise(outcomes)
    print(f"system      : {args.system}")
    print(f"adjudicated : {'strong links only' if args.strong_only else 'unadjudicated (company-level)'}")
    print(f"actions     : {stats['n_actions']}")
    print(f"detected    : {stats['n_detected']}  ({stats['detect_rate']:.1%})")
    if stats["median_lead_days"] is not None:
        print(f"lead time   : median {stats['median_lead_days']:.0f} d  "
              f"(p25 {stats['lead_p25']} / p75 {stats['lead_p75']})")
    if missing:
        n_skipped = sum(len(v) for v in missing.values())
        print(f"\nCOVERAGE GAP: {n_skipped} actions not evaluated — no refit at "
              f"{', '.join(str(c) for c in sorted(missing))}")
        print("  These are excluded from the denominator, not counted as misses.")
    print("\nby cutoff:")
    by: dict = {}
    for o in outcomes:
        k = o.cutoff.year
        by.setdefault(k, [0, 0])
        by[k][0] += 1
        by[k][1] += int(o.detected)
    for year in sorted(by):
        total, hit = by[year]
        print(f"  {year}  {hit:>3} / {total:<3}  {hit / total:5.0%}")
    if not args.strong_only:
        print(
            "\nUNADJUDICATED — both numbers above are upper bounds, and the lead\n"
            "time is the more inflated of the two. `first_signal` is the earliest\n"
            "month ANY cluster fired for this company, not the cluster that\n"
            "corresponds to this action's harm, so it measures 'when did this\n"
            "company first look unusual at all' — which for a large bank is\n"
            "close to always. EVALUATION §1.3 requires a human to match the\n"
            "cluster to the harm; only then is a lead time a lead time.\n"
            "Re-run with --strong-only once backtest_links is populated."
        )
    return 0


def _as_of(con, cutoff=None):
    """The newest complaint date actually in scope.

    Taken from the corpus *restricted to the cutoff*, never from the whole
    corpus. Getting this from `max(date_received)` over everything stamped a
    2017 refit's clusters with as_of 2026-08-03, which defeats the point of
    carrying as_of at all: EVALUATION §5 item 2 asserts cluster definitions
    used at cutoff C were fit only on < C data, and it asserts it on this field.
    """
    return con.execute(
        "SELECT max(date_received) FROM complaints WHERE (? IS NULL OR date_received < ?)",
        [cutoff, cutoff],
    ).fetchone()[0]


def _months(con, cutoff=None) -> list:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT period_month FROM complaints "
        "WHERE (? IS NULL OR period_month < ?) ORDER BY 1",
        [cutoff, cutoff],
    ).fetchall()]


def phase_signals(args: argparse.Namespace) -> int:
    """Phase 5 — disproportionality, changepoint, FDR, alert construction."""
    from src.ids import signal_id as make_signal_id
    from src.signals import changepoint, disproportionality, timeseries

    con = db.bootstrap()
    cluster_run = args.run_id or latest_run(con, "cluster")
    dedup_run = args.dedup_run or latest_run(con, "dedup")
    cutoff = getattr(args, "cutoff", None)
    as_of = _as_of(con, cutoff)
    months = _months(con, cutoff)

    # The system that produced the clusters is carried onto the signals run, or
    # the backtest cannot tell B1's signals from HarmScope's and every baseline
    # silently reports HarmScope's numbers.
    system = con.execute(
        "SELECT coalesce(json_extract_string(params_json, '$.params.system'), "
        "'harmscope') FROM runs WHERE run_id = ?", [cluster_run],
    ).fetchone()[0]
    params = {"cluster_run": cluster_run, "dedup_run": dedup_run,
              "shuffle": args.shuffle, "limit": None, "system": system,
              "cutoff": str(getattr(args, "cutoff", None) or "")}
    with db.run(con, "signals", CONFIG, params=params) as r:
        n_expanded = timeseries.build_expanded(
            con, cluster_run, dedup_run, getattr(args, 'cutoff', None)
        )
        if args.shuffle:
            n_shuffled = _shuffle_clusters(con, CONFIG.seed + args.shuffle)
            print(f"NEGATIVE CONTROL: cluster labels permuted within family "
                  f"({n_shuffled:,} rows, seed offset {args.shuffle})")
        print(f"expanded   : {n_expanded:,} non-campaign complaints")

        n_total, n_company = timeseries.build_panel(con, r.run_id, cluster_run, as_of)
        print(f"panel      : {n_total:,} cluster-level rows, "
              f"{n_company:,} company-level")

        cells = timeseries.contingency(con)
        scored = disproportionality.analyse(cells, CONFIG.signals.min_a)
        print(f"2x2 tests  : {len(scored):,} pairs with a >= {CONFIG.signals.min_a} "
              f"(of {len(cells):,} company x cluster pairs)")

        series = timeseries.series(con, r.run_id)
        changes = changepoint.detect(
            series, months, CONFIG.signals, CONFIG.signals.min_supporting_groups
        )
        print(f"changepoint: {len(changes):,} series fired of {len(series):,}")

        rows = _build_signals(con, r.run_id, cluster_run, scored, changes,
                              as_of, make_signal_id, months)
        con.execute("DELETE FROM signals WHERE run_id = ?", [r.run_id])
        if rows:
            con.executemany(
                "INSERT INTO signals (signal_id, run_id, cluster_id, company_id, "
                "period_month, method, statistic, ci_low, ci_high, p_value, "
                "q_value, n_supporting, n_supporting_groups, as_of) "
                "VALUES (" + ",".join("?" * 14) + ")", rows,
            )
        checks.expect_no_nulls(con, "signals", ["as_of", "company_id"])
        r.finish(output_rows=len(rows), input_rows=n_expanded)

    alerts = con.execute(
        "SELECT count(*) FROM signals WHERE run_id = ? AND q_value <= ?",
        [r.run_id, CONFIG.signals.fdr_alpha],
    ).fetchone()[0]
    print(f"\nsignals    : {len(rows):,} rows")
    print(f"alerts     : {alerts:,} at q <= {CONFIG.signals.fdr_alpha}")
    by_method = con.execute(
        "SELECT method, count(*) FROM signals WHERE run_id = ? GROUP BY 1 ORDER BY 2 DESC",
        [r.run_id],
    ).fetchall()
    print("by method  : " + ", ".join(f"{m}={n:,}" for m, n in by_method))
    if args.shuffle:
        n_tests = sum(n for m, n in by_method if m == "ebgm")
        rate = alerts / max(n_tests, 1)
        print(f"\n{'=' * 72}\nNEGATIVE CONTROL (ROADMAP Phase 5, mandatory)")
        print(f"  permuted-label alerts : {alerts:,} of {n_tests:,} tests")
        print(f"  false-alert rate      : {rate:.4f}")
        print(f"  FDR alpha             : {CONFIG.signals.fdr_alpha}")
        # The rate, not the count. Shuffling spreads every cluster across every
        # company, so a permuted panel has far more series than the real one and
        # the raw counts are not comparable — which is exactly the mistake that
        # would make a broken detector look fine.
        verdict = "PASS" if rate <= CONFIG.signals.fdr_alpha * 2 else "FAIL"
        print(f"  verdict               : {verdict}")
        if verdict == "FAIL":
            print("\n  A random assignment is producing real alerts. Per ROADMAP "
                  "Phase 5 the statistics are wrong — fix before Phase 6.")
    return 0


def _shuffle_clusters(con, seed: int) -> int:
    """Permute cluster labels among **dup-groups**, within product family.

    The negative control ROADMAP Phase 5 makes mandatory. Permuting *within*
    family preserves every marginal that is not the thing under test — family
    volume, monthly totals, company mix, group sizes — so anything that still
    fires is the machinery inventing signal rather than finding it.

    Groups, not complaints. In production a group's cluster comes from its
    representative, so every complaint in a group shares one cluster; permuting
    per complaint breaks that invariant and scatters each group across many
    clusters, which the first attempt did. That inflates every count, makes a
    unit fall in several cells at once, and produced a 10.7% false-alert rate
    that looked like broken statistics rather than a broken control. A null has
    to preserve the structure of the thing it is a null for.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _shuffled AS
        WITH grp AS (
          SELECT DISTINCT product_family, group_id, cluster_id
          FROM _expanded WHERE cluster_id IS NOT NULL
        ),
        src AS (
          SELECT product_family, group_id,
                 row_number() OVER (PARTITION BY product_family
                                    ORDER BY hash(group_id || '{seed}')) AS pos
          FROM grp
        ),
        dst AS (
          SELECT product_family, cluster_id,
                 row_number() OVER (PARTITION BY product_family
                                    ORDER BY group_id) AS pos
          FROM grp
        )
        SELECT s.group_id, d.cluster_id
        FROM src s JOIN dst d
          ON d.product_family = s.product_family AND d.pos = s.pos
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _expanded AS
        SELECT e.complaint_id, e.company_id, e.period_month, e.product_family,
               e.group_id, s.cluster_id
        FROM _expanded e LEFT JOIN _shuffled s USING (group_id)
    """)
    return con.execute("SELECT count(*) FROM _shuffled").fetchone()[0]


def _build_signals(con, run_id, cluster_run, scored, changes, as_of, make_id, months):
    """Assemble signal rows, carrying both support counts on every one.

    `n_supporting_groups` is **distinct** groups over the whole window, not the
    sum of monthly group counts. Summing the panel counted a group once per
    month it stayed active, so a single template running for two years reported
    24 supporting groups and sailed past the `min_supporting_groups` gate that
    exists precisely to stop one filing from looking like many.
    """
    support = dict(con.execute("""
        SELECT (cluster_id, coalesce(company_id, '__ALL__')), n FROM (
          SELECT cluster_id, company_id, count(DISTINCT group_id) AS n
          FROM _expanded WHERE cluster_id IS NOT NULL
          GROUP BY GROUPING SETS ((cluster_id, company_id), (cluster_id))
        )
    """).fetchall())
    raw = dict(con.execute("""
        SELECT (cluster_id, coalesce(company_id, '__ALL__')), n FROM (
          SELECT cluster_id, company_id, count(*) AS n FROM _expanded
          WHERE cluster_id IS NOT NULL GROUP BY GROUPING SETS ((cluster_id, company_id), (cluster_id))
        )
    """).fetchall())

    rows, i = [], 0
    # The period a disproportionality signal is dated at is the newest month IN
    # SCOPE, not the newest month in the corpus. Reading it from the whole
    # corpus dated every 2017-cutoff signal at 2026-08 — after its own cutoff,
    # which would make any lead time computed from it meaningless.
    latest = months[-1]
    for s in scored:
        key = (s.cluster_id, s.company_id)
        groups = support.get(key, 0)
        rows.append((
            make_id(run_id, i), run_id, s.cluster_id, s.company_id, latest,
            "ebgm", s.eb05, s.prr_low, s.prr_high, s.p_value, s.q_value,
            int(raw.get(key, 0)), int(groups), as_of,
        ))
        i += 1
    for (cluster_id, company_id), fired in changes.items():
        for change in fired:
            key = (cluster_id, company_id)
            groups = support.get(key, 0)
            rows.append((
                make_id(run_id, i), run_id, cluster_id, company_id, change.period,
                change.method, change.statistic, None, None, None, None,
                int(raw.get(key, 0)), int(groups), as_of,
            ))
            i += 1
    return rows


def _family_rows(con, dedup_run: str, model: str, family: str):
    import numpy as np

    rows = con.execute(
        """
        SELECT e.row_idx FROM dup_groups d
        JOIN complaints c USING (complaint_id)
        JOIN embedding_map e USING (complaint_id)
        WHERE d.run_id = ? AND d.is_representative AND e.model = ?
          AND c.product_family = ? ORDER BY d.complaint_id
        """,
        [dedup_run, model, family],
    ).fetchall()
    return np.asarray([r[0] for r in rows], dtype=np.int64)


def cmd_stability(args: argparse.Namespace) -> int:
    """ROADMAP Phase 4 gate — ARI between disjoint halves, and across sample sizes.

    Reports whatever it finds. METHODOLOGY §4.3: "Report the ARI in the README
    regardless of what it says."
    """
    import numpy as np

    from src.cluster import stability
    from src.embed import encode

    con = db.connect(read_only=True)
    model = args.model or CONFIG.embed.model
    memmap = encode.embedding_artifact_paths(PATHS.artifacts, model).memmap
    vectors = np.load(memmap, mmap_mode="r")
    dedup_run = args.dedup_run or latest_run(con, "dedup")

    families = args.family.split(",") if args.family else ["credit_reporting"]
    for family in families:
        rows = _family_rows(con, dedup_run, model, family)
        print(f"\n{'=' * 72}\n{family}: {len(rows):,} representatives\n{'=' * 72}")

        print("\ndisjoint halves (the gate's headline number):")
        halves = stability.disjoint_halves(
            vectors, rows, family, CONFIG.cluster, CONFIG.seed,
            eval_size=args.eval_size,
        )
        if halves.get("ari") is None:
            print("  not computable — a half produced no clusters")
        else:
            print(f"  half sizes      {halves['half_sizes'][0]:,} / "
                  f"{halves['half_sizes'][1]:,}, evaluated on {halves['eval_size']:,} "
                  f"held-out points neither half saw")
            print(f"  clusters        {halves['n_clusters'][0]} / {halves['n_clusters'][1]}")
            print(f"  fit noise       {halves['fit_noise'][0]:.1%} / "
                  f"{halves['fit_noise'][1]:.1%}")
            print(f"  assigned        {halves['assigned_fraction'][0]:.1%} / "
                  f"{halves['assigned_fraction'][1]:.1%}")
            print(f"  ARI             {halves['ari']:.4f}  "
                  f"over {halves['n_compared']:,} points both assigned")

        sizes = tuple(s for s in CONFIG.cluster.stability_sample_sizes
                      if s <= len(rows))
        if len(sizes) < 2:
            print(f"\nsample-size sweep: needs two of "
                  f"{CONFIG.cluster.stability_sample_sizes}, family has "
                  f"{len(rows):,} — skipped")
            continue
        print(f"\nsample-size sweep {sizes}:")
        sweep = stability.sample_size_sweep(
            vectors, rows, family, CONFIG.cluster, CONFIG.seed, sizes,
            eval_size=args.eval_size,
        )
        print(f"  {'size':>9} {'clusters':>9} {'fit noise':>10} {'assigned':>9} "
              f"{'ARI vs largest':>15}")
        for run in sweep["runs"]:
            print(f"  {run['size']:>9,} {run['n_clusters']:>9} "
                  f"{run['fit_noise']:>9.1%} {run['assigned_fraction']:>9.1%} "
                  f"{run['ari_vs_largest']:>15.4f}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """ROADMAP Phase 4 gate — can novelty recover deliberately hidden Issues?

    "If the score cannot recover deliberately hidden categories, it will not
    find real new ones" (METHODOLOGY §5.1).
    """
    from src.cluster import novelty as novelty_mod

    con = db.connect(read_only=True)
    run_id = args.run_id or latest_run(con, "cluster")
    space = dict(con.execute(
        "SELECT product_family, count(DISTINCT (issue_std, sub_issue_std)) "
        "FROM complaints GROUP BY 1"
    ).fetchall())
    families = con.execute(
        "SELECT product_family, count(*) FROM clusters WHERE run_id = ? "
        "GROUP BY 1 ORDER BY 2 DESC", [run_id],
    ).fetchall()

    print(f"run   : {run_id}")
    print(f"bound : AUC >= {CONFIG.novelty.ablation_min_auc} "
          f"over {CONFIG.novelty.ablation_n_issues} hidden issues\n")
    print(f"{'family':<18} {'clusters':>9} {'issues':>7} {'mean AUC':>9} "
          f"{'pooled':>8}  verdict")
    overall = []
    for family, n_clusters in families:
        members = _cluster_label_members(con, run_id, family)
        report = novelty_mod.ablation_auc(members, space.get(family, 2), CONFIG.novelty)
        if report["mean_auc"] is None:
            print(f"{family:<18} {n_clusters:>9} {'—':>7} {'—':>9} {'—':>8}  "
                  f"no issue dominates a cluster")
            continue
        overall.append(report["mean_auc"])
        verdict = "PASS" if report["passes"] else "FAIL"
        print(f"{family:<18} {n_clusters:>9} {report['n_issues']:>7} "
              f"{report['mean_auc']:>9.4f} {report['pooled_auc']:>8.4f}  {verdict}")
        if args.verbose:
            for row in sorted(report["per_issue"], key=lambda r: r["auc"]):
                print(f"    {row['auc']:.4f}  n={row['n_positive']:<4} {row['issue'][:64]}")
    if overall:
        mean = sum(overall) / len(overall)
        print(f"\nacross {len(overall)} families: mean AUC {mean:.4f} — "
              f"{'PASS' if mean >= CONFIG.novelty.ablation_min_auc else 'FAIL'}")
    return 0


def cmd_clusters(args: argparse.Namespace) -> int:
    """ROADMAP Phase 4 gate — read N random clusters. Can you name each one?

    "If not, `min_cluster_size` is wrong." No statistic answers this.
    """
    import random

    con = db.connect(read_only=True)
    run_id = args.run_id or latest_run(con, "cluster")
    rows = con.execute(
        """
        SELECT c.cluster_id, c.product_family, c.n_members, c.coherence,
               c.persistence, n.novelty_score, n.dominant_label, n.is_novel,
               n.dominant_label_share
        FROM clusters c LEFT JOIN cluster_novelty n USING (cluster_id)
        WHERE c.run_id = ? ORDER BY c.cluster_id
        """,
        [run_id],
    ).fetchall()
    if not rows:
        raise SystemExit(f"no clusters for run {run_id}")
    rng = random.Random(CONFIG.seed)  # noqa: S311 - sampling, not cryptography
    picked = rows if args.novel_only else rng.sample(rows, min(args.n, len(rows)))
    if args.novel_only:
        picked = [r for r in rows if r[7]]
        picked = rng.sample(picked, min(args.n, len(picked)))

    for (cid, family, n_members, coh, pers, score, dominant, novel,
         share) in picked:
        print(f"\n{'=' * 72}")
        print(f"[{family}] {n_members:,} members  coherence {coh:.3f}  "
              f"persistence {pers:.3f}")
        print(f"novelty {score:.3f}{'  NOVEL' if novel else ''}   "
              f"dominant label ({share:.0%}): {dominant}")
        print("-" * 72)
        for (text,) in con.execute(
            """
            SELECT n.text_redacted FROM cluster_members m
            JOIN narratives n USING (complaint_id)
            WHERE m.cluster_id = ?
            ORDER BY m.is_exemplar DESC, m.membership_prob DESC LIMIT ?
            """,
            [cid, args.k],
        ).fetchall():
            print("  • " + " ".join(text.split())[:240])
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


DISCLAIMER = (
    "Complaints are consumer allegations. Publication does not indicate the "
    "CFPB verified the allegations or that the company acted unlawfully."
)


def cmd_ask(args: argparse.Namespace) -> int:
    """Retrieve and render one grounded analyst answer for an explicit scope."""
    from src.llm import answer

    con = db.bootstrap()
    result = answer.answer_question(
        con,
        args.cluster_id,
        args.company_id,
        args.question,
        args.model or CONFIG.embed.dev_model,
        include_enforcement_context=args.include_enforcement_context,
    )
    print(answer.render_cli(result, disclaimer=DISCLAIMER))
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
    "cluster": phase_cluster,
    "signals": phase_signals,
    "backtest": phase_backtest,
    "baselines": phase_baselines,
    "label": phase_label,
}

PLANNED: dict[str, str] = {
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
    # A cutoff is a date everywhere it is used internally — `cmd_refit` already
    # parses it, and `backtest.run_for_cutoff` calls `.isoformat()` on it. Doing
    # it once here rather than in each phase keeps the CLI path and the refit
    # path handing the phases the same type.
    if getattr(args, "cutoff", None):
        from datetime import date as _date

        args.cutoff = _date.fromisoformat(args.cutoff)
    return PHASES[args.phase](args)


def cmd_runs(args: argparse.Namespace) -> int:
    con = db.connect(read_only=True)
    # json_extract, not Python's json: params_json is a DuckDB JSON column, and
    # parsing it a second time in Python was how the path bug below survived a
    # passing test. One parser, and the path is visible in the query.
    rows = con.execute(
        "SELECT run_id, phase, status, output_rows, started_at, "
        "substr(git_sha, 1, 12), substr(config_hash, 1, 8), error, "
        "json_extract(params_json, '$.params.limit') "
        "FROM runs ORDER BY started_at DESC LIMIT ?",
        [args.n],
    ).fetchall()
    if not rows:
        print("no runs recorded")
        return 0
    print(f"{'phase':<13} {'status':<8} {'rows':>10}  {'started':<20} "
          f"{'git':<13} {'cfg':<9} error")
    partial = False
    for _run_id, phase, status, out_rows, started, sha, cfg, err, limit in rows:
        # A run over a deliberately truncated input is not a run of the phase.
        # Without this marker a --limit smoke test and the real thing differ
        # only by output_rows, which nothing reads as a warning.
        # json_extract returns JSON text, so an absent key and a stored null
        # arrive as None and the four characters "null" respectively.
        capped = limit not in (None, "null")
        partial |= capped
        label = f"{phase}{'*' if capped else ''}"
        print(f"{label:<13} {status:<8} {out_rows if out_rows is not None else '-':>10}"
              f"  {str(started)[:19]:<20} {sha:<13} {cfg:<9} {err or ''}")
    if partial:
        print("\n* ran over a truncated input (--limit); not a full run of the phase")
    return 0


def cmd_label_verify(args: argparse.Namespace) -> int:
    """Export, ingest, and report the blinded human label-review workflow."""
    from src.llm import verify

    con = db.bootstrap()
    if args.verify_action == "export":
        signals_run = args.signals_run or latest_run(con, "signals")
        path = verify.export_worklist(
            con, signals_run, args.n, CONFIG.llm.verification_seed, Path(args.output),
        )
        metadata = verify.load_worklist_metadata(con, path)
        print(f"worklist  : {path}")
        print(f"sidecar   : {verify.worklist_sidecar_path(path)}")
        print(f"version   : {metadata.worklist_version}")
        return 0
    if args.verify_action == "record":
        # This CLI is the human-review ingestion path. Model-origin reviews
        # remain available to callers of src.llm.verify, never as CLI input.
        count, metadata = verify.record_worklist(
            con, Path(args.input), args.reviewer,
        )
        print(f"recorded  : {count}")
        print(f"version   : {metadata.worklist_version}")
        return 0
    print(verify.report(con, args.worklist_version).render())
    return 0


def build_parser() -> argparse.ArgumentParser:
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
    p_run.add_argument("--limit", type=int,
                       help="encode only the first N texts; cluster: fit sample size")
    p_run.add_argument("--dedup-run", help="dedup run whose representatives to cluster")
    p_run.add_argument("--cutoff", help="ISO date; restricts to complaints before it")
    p_run.add_argument("--family", help="cluster only this product family")
    p_run.add_argument("--system", default="harmscope", help="backtest: system label")
    p_run.add_argument("--strong-only", action="store_true",
                       help="backtest: require an adjudicated strong link")
    p_run.add_argument("--run-id", help="cluster run to build signals from")
    p_run.add_argument("--signals-run", help="label: signals run defining which clusters fired")
    p_run.add_argument("--control-n", type=int, default=50,
                       help="label: random non-firing clusters to also label, so the "
                            "LLM_LAYER §2.5 verification sample is not drawn only from alerts")
    p_run.add_argument("--shuffle", type=int, default=0, metavar="K",
                       help="negative control: permute cluster labels within "
                            "family using seed offset K (ROADMAP Phase 5)")
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

    p_stab = sub.add_parser("stability", help="Phase 4 gate: ARI across refits")
    p_stab.add_argument("--family", help="comma-separated; default credit_reporting")
    p_stab.add_argument("--dedup-run")
    p_stab.add_argument("--model")
    p_stab.add_argument("--eval-size", type=int, default=50_000)
    p_stab.set_defaults(func=cmd_stability)

    p_abl = sub.add_parser("ablation", help="Phase 4 gate: label-ablation AUC")
    p_abl.add_argument("--run-id")
    p_abl.add_argument("-v", "--verbose", action="store_true")
    p_abl.set_defaults(func=cmd_ablation)

    p_cl = sub.add_parser("clusters", help="Phase 4 gate: read N random clusters")
    p_cl.add_argument("-n", type=int, default=15)
    p_cl.add_argument("-k", type=int, default=4, help="narratives per cluster")
    p_cl.add_argument("--run-id")
    p_cl.add_argument("--novel-only", action="store_true")
    p_cl.set_defaults(func=cmd_clusters)

    p_re = sub.add_parser("refit", help="Phase 6: full refit at one cutoff")
    p_re.add_argument("--cutoff", required=True, help="ISO date; uses complaints < this")
    p_re.add_argument("--model")
    p_re.set_defaults(func=cmd_refit)

    p_wl = sub.add_parser("worklist", help="Phase 6: emit blinded adjudication worklists")
    p_wl.add_argument("--system", default="harmscope")
    p_wl.set_defaults(func=cmd_worklist)

    p_vd = sub.add_parser("verdicts", help="Phase 6: read filled worklists")
    p_vd.add_argument("--system", default="harmscope")
    p_vd.add_argument("--adjudicator", default="claude-opus-5")
    p_vd.set_defaults(func=cmd_verdicts)

    p_al = sub.add_parser("alerts", help="Phase 5: the joint alert criteria (§6.3)")
    p_al.add_argument("-n", type=int, default=20)
    p_al.add_argument("--run-id")
    p_al.add_argument("--track", default="novel", choices=["novel", "known", "all"])
    p_al.add_argument("--evidence", action="store_true", help="show exemplar text")
    p_al.set_defaults(func=cmd_alerts)

    p_tax = sub.add_parser("taxonomy", help="label vocabulary by volume")
    p_tax.set_defaults(func=cmd_taxonomy)

    p_ask = sub.add_parser("ask", help="retrieve and render one grounded analyst answer")
    p_ask.add_argument("--cluster-id", required=True)
    p_ask.add_argument("--company-id", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--model", help="embedding model (default: config)")
    p_ask.add_argument("--include-enforcement-context", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_verify = sub.add_parser(
        "label-verify", help="export, ingest, or report blinded human label review",
    )
    verify_sub = p_verify.add_subparsers(dest="verify_action", required=True)
    p_verify_export = verify_sub.add_parser("export", help="write a blinded worklist")
    p_verify_export.add_argument("--n", type=int, default=CONFIG.llm.human_verify_n)
    p_verify_export.add_argument("--output", required=True)
    p_verify_export.add_argument("--signals-run")
    p_verify_export.set_defaults(func=cmd_label_verify)
    p_verify_record = verify_sub.add_parser("record", help="ingest completed human review")
    p_verify_record.add_argument("--input", required=True)
    p_verify_record.add_argument("--reviewer", required=True)
    p_verify_record.set_defaults(func=cmd_label_verify)
    p_verify_report = verify_sub.add_parser("report", help="show human-review agreement")
    p_verify_report.add_argument("--worklist-version")
    p_verify_report.set_defaults(func=cmd_label_verify)

    p_runs = sub.add_parser("runs", help="show the run registry")
    p_runs.add_argument("-n", type=int, default=20)
    p_runs.set_defaults(func=cmd_runs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
