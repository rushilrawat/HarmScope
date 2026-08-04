"""Phase 2 driver: exact grouping, MinHash near-dup, union-find, campaigns.

docs/METHODOLOGY.md §2. The order matters — Tier 1 runs first and collapses
35% of the corpus on the 2026-08-03 snapshot, which is what makes Tier 2
tractable: MinHash only ever sees one representative per exact-duplicate group.

LSH banding is done in DuckDB rather than in a Python dict-of-buckets. At 2.5M
representatives x 16 bands that is 40M entries; DuckDB handles it out-of-core,
a Python dict does not fit in 19 GB.
"""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np

from src.config import Config
from src.dedup import minhash
from src.ids import campaign_id as make_campaign_id

SIG_BATCH = 20_000
# Above this, an LSH bucket is star-linked to its lowest-id member instead of
# expanded to all pairs: a 10k-member bucket is 50M pairs. With 16 bands each
# document gets 16 chances to be bucketed, so a link missed in one oversized
# band is usually recovered in another.
MAX_BUCKET_PAIRS = 200


def exact_pairs(con: duckdb.DuckDBPyConnection) -> int:
    """Tier 1. Star-link every exact-duplicate group to its lowest id.

    Star rather than all-pairs: identical text is transitive by definition, so
    a spanning set is enough for union-find and n-1 edges beats n(n-1)/2.
    """
    con.execute(
        """
        INSERT INTO dup_pairs
        SELECT lo AS complaint_id_a, complaint_id AS complaint_id_b, 1.0, 'exact'
        FROM (
          SELECT complaint_id, text_hash,
                 min(complaint_id) OVER (PARTITION BY text_hash) AS lo
          FROM narratives
        ) WHERE complaint_id <> lo
        """
    )
    return con.execute("SELECT count(*) FROM dup_pairs").fetchone()[0]


def representatives(con: duckdb.DuckDBPyConnection) -> list[tuple[int, str, str]]:
    """One `(complaint_id, product_family, text)` per exact-duplicate group."""
    return con.execute(
        """
        SELECT n.complaint_id, c.product_family, n.text_redacted
        FROM narratives n
        JOIN complaints c USING (complaint_id)
        JOIN (SELECT text_hash, min(complaint_id) AS complaint_id
              FROM narratives GROUP BY 1) k USING (complaint_id)
        ORDER BY n.complaint_id
        """
    ).fetchall()


def build_signatures(
    texts: list[str], cfg: Config, progress: bool = True
) -> np.ndarray:
    a, b = minhash.permutations(cfg.dedup.minhash_perms, cfg.seed)
    out = np.empty((len(texts), cfg.dedup.minhash_perms), dtype=np.uint32)
    for start in range(0, len(texts), SIG_BATCH):
        chunk = texts[start : start + SIG_BATCH]
        out[start : start + len(chunk)] = minhash.signatures(
            chunk, cfg.dedup.shingle_size, a, b
        )
        if progress and start and start % (10 * SIG_BATCH) == 0:
            print(f"  signatures {start:,}/{len(texts):,}", flush=True)
    return out


def candidate_pairs(
    con: duckdb.DuckDBPyConnection,
    ids: np.ndarray,
    families: list[str],
    sig: np.ndarray,
) -> np.ndarray:
    """LSH candidate pairs as an `(m, 2)` array of row indices.

    Blocked by `product_family` per docs/METHODOLOGY.md §2.2 — a mortgage
    narrative and a credit-reporting narrative are never the same filing, and
    blocking keeps the buckets small enough to expand.
    """
    bands = minhash.band_hashes(sig, seed=0)
    n, n_bands = bands.shape

    con.execute("DROP TABLE IF EXISTS _bands")
    con.execute(
        "CREATE TEMP TABLE _bands (row_idx BIGINT, family VARCHAR, "
        "band SMALLINT, h UBIGINT)"
    )
    rows = np.repeat(np.arange(n, dtype=np.int64), n_bands)
    band_idx = np.tile(np.arange(n_bands, dtype=np.int16), n)
    fam = np.repeat(np.array(families, dtype=object), n_bands)
    frame = {  # noqa: F841 - referenced by DuckDB replacement scan
        "row_idx": rows, "family": fam,
        "band": band_idx, "h": bands.reshape(-1),
    }
    con.execute("INSERT INTO _bands SELECT * FROM frame")

    buckets = con.execute(
        "SELECT list(row_idx ORDER BY row_idx) FROM _bands "
        "GROUP BY family, band, h HAVING count(*) > 1"
    ).fetchall()
    con.execute("DROP TABLE IF EXISTS _bands")

    pairs: set[tuple[int, int]] = set()
    for (members,) in buckets:
        if len(members) <= MAX_BUCKET_PAIRS:
            for i, x in enumerate(members):
                for y in members[i + 1 :]:
                    pairs.add((x, y))
        else:
            anchor = members[0]
            pairs.update((anchor, y) for y in members[1:])
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def verify(sig: np.ndarray, pairs: np.ndarray, threshold: float) -> np.ndarray:
    """Keep only candidate pairs whose estimated Jaccard clears the threshold.

    Band collisions cost time, never correctness — this is where that is made
    true. Chunked because `pairs` can be tens of millions of rows.
    """
    keep = np.zeros(len(pairs), dtype=bool)
    sims = np.zeros(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), 1_000_000):
        sl = slice(start, start + 1_000_000)
        s = (sig[pairs[sl, 0]] == sig[pairs[sl, 1]]).mean(axis=1)
        sims[sl] = s
        keep[sl] = s >= threshold
    return keep, sims


def assign_groups(
    con: duckdb.DuckDBPyConnection, run_id: str, as_of: date
) -> tuple[int, int]:
    """Union-find over `dup_pairs`, then write `dup_groups`.

    The representative is the earliest `date_received`, ties on the lower
    `complaint_id` (docs/METHODOLOGY.md §2.3): deterministic, and backward
    looking, so a group that gains members after a backtest cutoff does not
    retroactively change what a pre-cutoff run saw.
    """
    uf = minhash.UnionFind()
    for a, b in con.execute(
        "SELECT complaint_id_a, complaint_id_b FROM dup_pairs "
        "ORDER BY complaint_id_a, complaint_id_b"
    ).fetchall():
        uf.union(a, b)

    groups = uf.groups()
    rows = [
        (run_id, cid, f"g{root}", len(members), as_of)
        for root, members in groups.items()
        for cid in members
    ]
    # Singletons are groups of one; they must still appear so that downstream
    # "distinct groups" counts are correct.
    con.execute(f"DELETE FROM dup_groups WHERE run_id = '{run_id}'")
    if rows:
        con.executemany(
            "INSERT INTO dup_groups (run_id, complaint_id, group_id, "
            "is_representative, group_size, as_of) VALUES (?, ?, ?, false, ?, ?)",
            rows,
        )
    con.execute(
        f"""
        INSERT INTO dup_groups (run_id, complaint_id, group_id,
                                is_representative, group_size, as_of)
        SELECT '{run_id}', n.complaint_id, 'g' || n.complaint_id, false, 1, ?
        FROM narratives n
        WHERE NOT EXISTS (SELECT 1 FROM dup_groups d
                          WHERE d.run_id = '{run_id}' AND d.complaint_id = n.complaint_id)
        """,
        [as_of],
    )
    con.execute(
        f"""
        UPDATE dup_groups SET is_representative = true
        WHERE run_id = '{run_id}' AND complaint_id IN (
          SELECT complaint_id FROM (
            SELECT d.complaint_id,
                   row_number() OVER (PARTITION BY d.group_id
                                      ORDER BY c.date_received, d.complaint_id) rn
            FROM dup_groups d JOIN complaints c USING (complaint_id)
            WHERE d.run_id = '{run_id}'
          ) WHERE rn = 1)
        """
    )
    n_groups = con.execute(
        f"SELECT count(DISTINCT group_id) FROM dup_groups WHERE run_id = '{run_id}'"
    ).fetchone()[0]
    n_rows = con.execute(
        f"SELECT count(*) FROM dup_groups WHERE run_id = '{run_id}'"
    ).fetchone()[0]
    return n_groups, n_rows


def build_campaigns(
    con: duckdb.DuckDBPyConnection, run_id: str, cfg: Config, as_of: date
) -> tuple[int, int]:
    """Score every multi-member group and flag the campaigns."""
    from src.dedup import campaign

    # ponytail: join in SQL and fetchall(), not fetchdf() -> pandas. pandas is
    # a Phase 9 reporting dependency at most; nothing here needs a DataFrame.
    for name, expr in (
        ("_hhi_state", "c.state"),
        ("_hhi_company", "c.company_id"),
        ("_hhi_via", "r.submitted_via"),
    ):
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(
            f"CREATE TEMP TABLE {name} AS {campaign.concentration_sql(run_id, expr)}"
        )

    cols = [
        "group_id", "n_complaints", "first_seen", "last_seen", "product_family",
        "top_company_id", "burstiness", "length_cv", "boilerplate_score",
        "state_concentration", "company_concentration",
        "submitted_via_concentration",
    ]
    records = con.execute(
        f"""
        SELECT f.*, s.hhi AS state_concentration, co.hhi AS company_concentration,
               v.hhi AS submitted_via_concentration
        FROM ({campaign.feature_sql(run_id)}) f
        LEFT JOIN _hhi_state   s  USING (group_id)
        LEFT JOIN _hhi_company co USING (group_id)
        LEFT JOIN _hhi_via     v  USING (group_id)
        WHERE f.n_complaints >= {cfg.dedup.campaign_min_size}
        """
    ).fetchall()
    for name in ("_hhi_state", "_hhi_company", "_hhi_via"):
        con.execute(f"DROP TABLE IF EXISTS {name}")
    if not records:
        return 0, 0

    baseline = campaign.family_baselines(con)
    rows = []
    members = []
    for i, rec in enumerate(dict(zip(cols, r, strict=True)) for r in records):
        n_sig = campaign.count_signals(rec, cfg.dedup, baseline)
        flagged = campaign.is_flagged(rec, cfg.dedup, baseline)
        cid = make_campaign_id(run_id, i)
        rows.append((
            cid, run_id, int(rec["n_complaints"]), 1,
            rec["first_seen"], rec["last_seen"], rec["top_company_id"],
            rec["product_family"], rec["burstiness"],
            rec["state_concentration"], rec["company_concentration"],
            rec["submitted_via_concentration"], rec["length_cv"],
            rec["boilerplate_score"], n_sig, flagged, as_of,
        ))
        members.append((rec["group_id"], cid))

    con.execute(f"DELETE FROM campaign_members WHERE campaign_id LIKE '{run_id}:%'")
    con.execute(f"DELETE FROM campaigns WHERE run_id = '{run_id}'")
    con.executemany(
        "INSERT INTO campaigns (campaign_id, run_id, n_complaints, n_groups, "
        "first_seen, last_seen, top_company_id, product_family, burstiness, "
        "state_concentration, company_concentration, submitted_via_concentration, "
        "length_cv, boilerplate_score, n_signals, flagged, as_of) "
        "VALUES (" + ", ".join("?" * 17) + ")",
        rows,
    )
    con.execute("DROP TABLE IF EXISTS _cmap")
    con.execute("CREATE TEMP TABLE _cmap (group_id VARCHAR, campaign_id VARCHAR)")
    con.executemany("INSERT INTO _cmap VALUES (?, ?)", members)
    con.execute(
        f"""
        INSERT INTO campaign_members
        SELECT d.complaint_id, m.campaign_id
        FROM dup_groups d JOIN _cmap m USING (group_id)
        WHERE d.run_id = '{run_id}'
        """
    )
    con.execute("DROP TABLE IF EXISTS _cmap")
    n_flagged = con.execute(
        f"SELECT count(*) FROM campaigns WHERE run_id = '{run_id}' AND flagged"
    ).fetchone()[0]
    return len(rows), n_flagged
