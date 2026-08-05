"""Enforcement-action scraper, against a captured fixture of the real page.

No network. The fixture is a trimmed copy of the live listing markup as of
2026-08-04, so a CFPB layout change shows up as a test failure here rather than
as a quietly smaller ground-truth set.
"""

from __future__ import annotations

import csv
from datetime import date

import pytest

from src.ingestion import enforcement as ea

PAGE = """
<html><body>
<p>386 results</p>
<article class="o-post-preview" lang="en">
  <div class="m-meta-header"><div class="m-meta-header__item">
    <span class="a-date">Date filed:
      <span class="datetime"><time datetime="2024-12-23T00:00:00">DEC 23, 2024</time></span>
    </span></div></div>
  <div class="o-post-preview__content">
    <h3 class="o-post-preview__title">
      <a href="/enforcement/actions/walmart-inc-and-branch-messenger-inc/">Walmart Inc., and Branch Messenger, Inc.</a>
    </h3>
    <div class="o-post-preview__description">
      On December 23, 2024, the Bureau filed a complaint against Walmart Inc.
    </div>
  </div>
</article>
<article class="o-post-preview" lang="en">
  <div class="m-meta-header"><div class="m-meta-header__item">
    <span class="a-date">Date filed:
      <span class="datetime"><time datetime="2017-01-03T00:00:00">JAN 03, 2017</time></span>
    </span></div></div>
  <div class="o-post-preview__content">
    <h3 class="o-post-preview__title">
      <a href="/enforcement/actions/transunion-interactive-inc/">TransUnion Interactive, Inc.</a>
    </h3>
    <div class="o-post-preview__description">Deceptive marketing of credit scores.</div>
  </div>
</article>
<article class="o-post-preview" lang="en">
  <div class="m-meta-header"><div class="m-meta-header__item">
    <span class="a-date">Date filed:
      <span class="datetime"><time datetime="2013-05-30T00:00:00">MAY 30, 2013</time></span>
    </span></div></div>
  <div class="o-post-preview__content">
    <h3 class="o-post-preview__title">
      <a href="/enforcement/actions/old-action-inc/">Old Action, Inc.</a>
    </h3>
    <div class="o-post-preview__description">Before the first backtest cutoff.</div>
  </div>
</article>
</body></html>
"""


def test_parses_every_article():
    actions = ea.parse_page(PAGE)
    assert len(actions) == 3
    assert [a.action_id for a in actions] == [
        "walmart-inc-and-branch-messenger-inc",
        "transunion-interactive-inc",
        "old-action-inc",
    ]


def test_extracts_the_filed_date_not_the_display_date():
    """`filed_date` is the label date the whole backtest keys on, so it comes
    from the machine-readable attribute, never the rendered 'DEC 23, 2024'."""
    first = ea.parse_page(PAGE)[0]
    assert first.filed_date == "2024-12-23"
    assert date.fromisoformat(first.filed_date) == date(2024, 12, 23)


def test_company_and_url_survive_markup():
    first = ea.parse_page(PAGE)[0]
    assert first.company_raw == "Walmart Inc., and Branch Messenger, Inc."
    assert first.source_url == (
        "https://www.consumerfinance.gov/enforcement/actions/"
        "walmart-inc-and-branch-messenger-inc/"
    )
    assert "<" not in first.company_raw and "<" not in first.description


def test_window_filter_excludes_pre_first_cutoff():
    """An action before the first annual cutoff has no cutoff strictly
    preceding it and cannot be evaluated (EVALUATION.md §1.2)."""
    kept = ea.in_window(ea.parse_page(PAGE), date(2017, 1, 1), date(2024, 12, 31))
    assert [a.action_id for a in kept] == [
        "walmart-inc-and-branch-messenger-inc",
        "transunion-interactive-inc",
    ]


def test_advertised_total_is_read():
    assert ea.advertised_total(PAGE) == 386


def test_shortfall_against_advertised_total_raises(monkeypatch):
    """A layout change that halves the yield must fail loudly. Silently
    accepting fewer actions shrinks the ground truth and makes the backtest
    easier without anyone noticing."""
    monkeypatch.setattr(ea, "fetch", lambda url: PAGE)
    monkeypatch.setattr(ea, "PAGE_DELAY_S", 0)
    with pytest.raises(ea.ScrapeMismatch, match="advertises 386"):
        ea.scrape(max_pages=1)


def test_scrape_stops_when_a_page_adds_nothing(monkeypatch):
    page = PAGE.replace("386 results", "3 results")
    monkeypatch.setattr(ea, "fetch", lambda url: page)
    monkeypatch.setattr(ea, "PAGE_DELAY_S", 0)
    actions = ea.scrape(max_pages=5)
    assert len(actions) == 3  # deduped by action_id, not repeated per page


def test_fetch_refuses_non_https():
    for url in ("http://example.test/x", "file:///etc/passwd"):
        with pytest.raises(ValueError, match="non-https"):
            ea.fetch(url)


def test_worklist_leaves_judgement_columns_blank(tmp_path):
    """docs/DATA.md §4 wants harm_summary in the curator's words. The scraper
    must not pre-fill it, or curation becomes rubber-stamping CFPB's wording."""
    actions = ea.parse_page(PAGE)
    path = ea.write_candidates(actions, path=tmp_path / "c.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3
    for row in rows:
        for judgement in ("harm_summary", "product_family", "usable",
                          "harm_keywords", "conduct_start", "exclusion_reason"):
            assert row[judgement] == "", f"{judgement} must be curated by a human"
        assert row["source_url"].startswith("https://")
        assert row["cfpb_description"]  # carried, but in its own column


def test_worklist_is_sorted_by_filed_date(tmp_path):
    path = ea.write_candidates(ea.parse_page(PAGE), path=tmp_path / "c.csv")
    dates = [r["filed_date"] for r in csv.DictReader(path.open())]
    assert dates == sorted(dates)


def test_company_matching_is_exact_normalized_only(con):
    """Trap T6 again: a wrong company match attaches an action to the wrong
    complaint stream, and every lead time computed from it is meaningless."""
    con.execute(
        "INSERT INTO company_canonical (company_id, canonical_name, verified_by) "
        "VALUES ('transunion-interactive', 'TRANSUNION INTERACTIVE', 'manual')"
    )
    actions = ea.parse_page(PAGE)
    matched = ea.match_companies(con, actions)
    assert matched == {"transunion-interactive-inc": "transunion-interactive"}
    # Walmart has no canonical row, so it stays unmatched rather than guessing.
    assert "walmart-inc-and-branch-messenger-inc" not in matched
