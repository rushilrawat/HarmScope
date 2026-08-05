-- 2026-08-05. Add related_clusters.
--
-- METHODOLOGY §4.2 has specified this table since Phase 0 — clusters in
-- different families whose centroids are within a cosine threshold are linked,
-- because a servicing failure that shows up under both mortgage and student
-- loan is exactly the kind of cross-product harm the project exists to find.
-- It was never in db/schema.sql or ARCHITECTURE.md, so the drift test added in
-- 36abc79 could not see it: that test compares the doc against the schema, and
-- the table was missing from both.
--
-- Phase 3's neighbour read made it load-bearing rather than a nicety. Nearest
-- neighbours cross product_family constantly — the same credit-repair template
-- under credit_reporting and debt_collection, title loans split between
-- personal_loan and vehicle_loan — so family-stratified clustering will
-- fragment those harms by construction, and this is what puts them back
-- together.
--
-- Ordered pair (a < b) so a link is stored once and cannot be double-counted.

CREATE TABLE IF NOT EXISTS related_clusters (
  cluster_id_a VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  cluster_id_b VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  similarity   DOUBLE NOT NULL,   -- cosine between the two cluster centroids
  PRIMARY KEY (cluster_id_a, cluster_id_b)
);
