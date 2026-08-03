-- 2026-08-03. Add `sub_product_raw` to taxonomy_crosswalk.
--
-- The real corpus showed two of CFPB's taxonomy changes are splits rather than
-- renames (`Consumer Loan` -> vehicle/personal, `Credit card or prepaid card`
-- -> credit card/prepaid), so Product alone cannot route those rows. See
-- docs/DATA.md §3.4.
--
-- Safe to run as a drop-and-recreate: the table is a projection of the
-- committed CSV at data/ground_truth/taxonomy_crosswalk.csv and holds no state
-- of its own. `--phase normalize` repopulates it.

DROP TABLE IF EXISTS taxonomy_crosswalk;

CREATE TABLE taxonomy_crosswalk (
  product_raw     VARCHAR NOT NULL,
  sub_product_raw VARCHAR NOT NULL,
  product_std     VARCHAR NOT NULL,
  product_family  VARCHAR NOT NULL,
  era             VARCHAR,
  PRIMARY KEY (product_raw, sub_product_raw)
);
