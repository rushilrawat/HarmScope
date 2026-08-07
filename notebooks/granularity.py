"""Units, support per unit, and what clears the floor — all four systems + HarmScope.

Extends the table in ENGINEERING_NOTES (Phase 9 partial) to B2 and B3. B3's
support column counts complaints, not dup-groups: it runs without dedup, so its
groups are singletons and `n_supporting_groups == n_supporting`. That column is
not comparable across the B3 row and the others.
"""
import duckdb

CUTOFF = "2024-01-01"
FLOOR = 15
con = duckdb.connect("data/harmscope.duckdb", read_only=True)

print(f"cutoff {CUTOFF}, floor {FLOOR}\n")
print(f"{'system':<12} {'units':>7} {'p25':>5} {'med':>5} {'p75':>5} {'max':>7} "
      f"{'clears floor':>13}")
for system in ["harmscope", "B0", "B1", "B2", "B3"]:
    cl = con.execute(
        """SELECT run_id FROM runs WHERE phase='cluster' AND status='ok'
           AND coalesce(json_extract_string(params_json,'$.params.system'),
                        'harmscope') = ?
           AND json_extract_string(params_json,'$.params.cutoff') = ?
           ORDER BY started_at DESC LIMIT 1""", [system, CUTOFF]).fetchone()
    sg = con.execute(
        """SELECT run_id FROM runs WHERE phase='signals' AND status='ok'
           AND coalesce(json_extract_string(params_json,'$.params.system'),
                        'harmscope') = ?
           AND json_extract_string(params_json,'$.params.cutoff') = ?
           ORDER BY started_at DESC LIMIT 1""", [system, CUTOFF]).fetchone()
    if cl is None or sg is None:
        print(f"{system:<12} (no run at this cutoff)")
        continue
    units = con.execute("SELECT count(*) FROM clusters WHERE run_id = ?", cl).fetchone()[0]
    row = con.execute(
        """SELECT quantile_cont(n_supporting_groups, 0.25),
                  median(n_supporting_groups),
                  quantile_cont(n_supporting_groups, 0.75),
                  max(n_supporting_groups),
                  avg((n_supporting_groups >= ?)::INT), count(*)
           FROM signals WHERE run_id = ? AND company_id <> '__ALL__'""",
        [FLOOR, sg[0]]).fetchone()
    p25, med, p75, mx, clears, n = row
    print(f"{system:<12} {units:>7,} {p25:>5.0f} {med:>5.0f} {p75:>5.0f} {mx:>7,} "
          f"{clears:>12.1%}  (of {n:,} company-level signals)")
