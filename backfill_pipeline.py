"""
Unified SEC Filing Backfill Pipeline

Optimized pipeline that fetches SEC filings from EDGAR and populates all
database tables (filings, facts, sections, financial statements, embeddings).

Key optimizations:
- Single EDGAR fetch per filing (shared across all processing stages)
- Parallel ticker processing with shared rate limiter
- Incremental mode: skips already-loaded filings
- Gap analysis: shows what data is missing before running
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from edgar import Company, set_identity

import config
from config import (
    TICKERS,
    YEARS,
    SEC_RATE_LIMIT_DELAY,
    infer_fiscal_year,
    infer_fiscal_quarter,
)
from xbrl_to_postgres import (
    parse_xbrl_to_facts,
    save_filing,
    save_facts,
    get_xbrl_entity_info,
    create_indexes,
    analyze_tables,
    init_db as init_xbrl_db,
)
from filing_sections import (
    init_sections_table,
    save_sections_to_db,
    SECTION_METADATA,
)
from fetch_financials_to_postgres import (
    save_to_postgres as save_financial_statement,
    init_db as init_financials_db,
)
from section_vector_tables import (
    init_vector_tables,
    migrate_from_filing_sections,
)
from chunk_and_embed import (
    process_sections_10k,
    process_sections_10q,
)

load_dotenv()

# ── Shared rate limiter ──────────────────────────────────────────────────────
_rate_lock = threading.Lock()
_last_request_time = 0.0


def rate_limited_sleep():
    """Thread-safe SEC EDGAR rate limiter."""
    global _last_request_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < SEC_RATE_LIMIT_DELAY:
            time.sleep(SEC_RATE_LIMIT_DELAY - elapsed)
        _last_request_time = time.time()


# ── Database connection pool ─────────────────────────────────────────────────
_connection_pool = None


def get_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv("PG_HOST", "localhost"),
            port=os.getenv("PG_PORT", "5432"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            database=os.getenv("PG_DATABASE"),
        )
    return _connection_pool


@contextmanager
def get_db_connection():
    p = get_connection_pool()
    conn = p.getconn()
    try:
        yield conn
    finally:
        p.putconn(conn)


# ── Gap Analysis ─────────────────────────────────────────────────────────────

def get_existing_accessions() -> set[str]:
    """Return set of accession numbers already loaded in the filings table."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT accession_number FROM filings")
            return {row[0] for row in cur.fetchall()}


def analyze_gaps(tickers: list[str], years: range) -> dict[str, dict]:
    """
    Analyze data gaps per ticker. Returns dict:
      {ticker: {"annual_years": set, "quarterly_years": set, "sections_years": set}}
    """
    gaps = {}

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for ticker in tickers:
                # Annual facts coverage
                cur.execute(
                    "SELECT DISTINCT fiscal_year FROM annual_facts WHERE ticker = %s",
                    (ticker,),
                )
                annual_years = {row[0] for row in cur.fetchall()}

                # Quarterly facts coverage
                cur.execute(
                    "SELECT DISTINCT fiscal_year FROM quarterly_facts WHERE ticker = %s",
                    (ticker,),
                )
                quarterly_years = {row[0] for row in cur.fetchall()}

                # Filing sections coverage
                cur.execute(
                    "SELECT DISTINCT fiscal_year FROM filing_sections WHERE ticker = %s",
                    (ticker,),
                )
                sections_years = {row[0] for row in cur.fetchall()}

                # Financial documents coverage
                cur.execute(
                    "SELECT DISTINCT fiscal_year FROM financial_documents WHERE ticker = %s",
                    (ticker,),
                )
                financials_years = {row[0] for row in cur.fetchall()}

                # Vector embeddings coverage (sections_10k)
                cur.execute(
                    "SELECT DISTINCT fiscal_year FROM sections_10k WHERE ticker = %s AND is_chunked = TRUE",
                    (ticker,),
                )
                embedded_years = {row[0] for row in cur.fetchall()}

                all_years = set(years)
                gaps[ticker] = {
                    "annual_present": annual_years & all_years,
                    "annual_missing": all_years - annual_years,
                    "quarterly_present": quarterly_years & all_years,
                    "quarterly_missing": all_years - quarterly_years,
                    "sections_present": sections_years & all_years,
                    "sections_missing": all_years - sections_years,
                    "financials_present": financials_years & all_years,
                    "financials_missing": all_years - financials_years,
                    "embedded_present": embedded_years & all_years,
                    "embedded_missing": all_years - embedded_years,
                }

    return gaps


def print_gap_report(gaps: dict[str, dict], years: range):
    """Print a formatted gap analysis report."""
    year_count = len(years)
    print(f"\n{'='*70}")
    print("DATA GAP ANALYSIS")
    print(f"{'='*70}")
    print(f"Year range: {years.start}-{years.stop - 1} ({year_count} years)\n")

    print(f"{'Ticker':<8} {'Annual':>8} {'Quarterly':>10} {'Sections':>10} {'Financials':>12} {'Embedded':>10}")
    print(f"{'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    total_annual_missing = 0
    total_quarterly_missing = 0

    for ticker, data in gaps.items():
        a_count = len(data["annual_present"])
        q_count = len(data["quarterly_present"])
        s_count = len(data["sections_present"])
        f_count = len(data["financials_present"])
        e_count = len(data["embedded_present"])

        a_str = f"{a_count}/{year_count}"
        q_str = f"{q_count}/{year_count}"
        s_str = f"{s_count}/{year_count}"
        f_str = f"{f_count}/{year_count}"
        e_str = f"{e_count}/{year_count}"

        print(f"{ticker:<8} {a_str:>8} {q_str:>10} {s_str:>10} {f_str:>12} {e_str:>10}")

        total_annual_missing += len(data["annual_missing"])
        total_quarterly_missing += len(data["quarterly_missing"])

    print(f"\nTotal missing: {total_annual_missing} annual, {total_quarterly_missing} quarterly")

    # Show per-ticker missing year details
    print(f"\nDetailed missing years:")
    for ticker, data in gaps.items():
        missing_annual = sorted(data["annual_missing"])
        missing_quarterly = sorted(data["quarterly_missing"])
        if missing_annual or missing_quarterly:
            parts = []
            if missing_annual:
                parts.append(f"annual={_format_year_ranges(missing_annual)}")
            if missing_quarterly:
                parts.append(f"quarterly={_format_year_ranges(missing_quarterly)}")
            print(f"  {ticker}: {', '.join(parts)}")

    print(f"{'='*70}\n")


def _format_year_ranges(years: list[int]) -> str:
    """Format a list of years into compact ranges like '2010-2015, 2018-2020'."""
    if not years:
        return "none"
    ranges = []
    start = years[0]
    end = years[0]
    for y in years[1:]:
        if y == end + 1:
            end = y
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = y
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


# ── Per-Filing Processing ────────────────────────────────────────────────────

def extract_sections_from_obj(obj) -> dict[str, str]:
    """Extract section texts from a filing object (cached from filing.obj())."""
    sections = {}
    try:
        if obj and hasattr(obj, "sections"):
            for key in obj.sections.keys():
                section = obj.sections.get(key)
                if section:
                    try:
                        text = section.text()
                        if text and len(text.strip()) > 0:
                            sections[key] = text
                    except Exception as e:
                        print(f"    Warning: Could not extract section {key}: {e}")
    except Exception as e:
        print(f"    Error extracting sections: {e}")
    return sections


def extract_financial_statements(xbrl) -> list[tuple[str, str]]:
    """Extract financial statement markdown from XBRL. Returns [(type, markdown)]."""
    results = []
    if not xbrl:
        return results

    try:
        financials = xbrl.statements
        statement_funcs = [
            ("income_statement", financials.income_statement),
            ("balance_sheet", financials.balance_sheet),
            ("cash_flow", financials.cashflow_statement),
        ]

        for stmt_type, stmt_func in statement_funcs:
            try:
                statement = stmt_func()
                if statement is None:
                    continue
                md_content = statement.render().to_markdown()
                results.append((stmt_type, md_content))
            except Exception as e:
                print(f"    Warning: Could not extract {stmt_type}: {e}")
    except Exception as e:
        print(f"    Error extracting financial statements: {e}")

    return results


def process_single_filing(
    ticker: str,
    company_name: str,
    filing,
    existing_accessions: set[str],
    skip_sections: bool = False,
    skip_financials: bool = False,
) -> dict:
    """
    Process a single filing through all stages:
      A) XBRL facts -> filings + annual_facts/quarterly_facts
      B) Section extraction -> filing_sections
      C) Financial statements -> financial_documents

    Returns stats dict with counts.
    """
    accession = filing.accession_number
    stats = {"facts": 0, "sections": 0, "statements": 0, "skipped": False}

    # Incremental: skip if already loaded
    if accession in existing_accessions:
        stats["skipped"] = True
        return stats

    form_type = filing.form
    fiscal_year = infer_fiscal_year(ticker, filing)
    fiscal_quarter = infer_fiscal_quarter(ticker, filing)
    fiscal_period = "FY" if form_type == "10-K" else f"Q{fiscal_quarter}"

    print(f"  {form_type} FY{fiscal_year} {fiscal_period} ({accession})")

    # ── Stage A: XBRL Facts ──────────────────────────────────────────────
    xbrl = None
    try:
        rate_limited_sleep()
        xbrl = filing.xbrl()
    except Exception as e:
        print(f"    XBRL fetch failed: {e}")

    filing_id = None
    if xbrl:
        entity_info = get_xbrl_entity_info(xbrl)
        facts = parse_xbrl_to_facts(xbrl)
        filing_id = save_filing(ticker, company_name, filing, xbrl)
        if filing_id and facts:
            saved = save_facts(filing_id, facts, ticker, fiscal_year, form_type, fiscal_quarter,
                               entity_info=entity_info)
            stats["facts"] = saved
            print(f"    Facts: {saved}")
    else:
        # Still save filing metadata even without XBRL
        filing_id = save_filing(ticker, company_name, filing, None)
        print(f"    No XBRL data available")

    # If save_filing returned None, the filing already existed (race condition)
    if filing_id is None:
        # Try to get existing filing_id
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM filings WHERE accession_number = %s",
                    (accession,),
                )
                row = cur.fetchone()
                if row:
                    filing_id = row[0]

    # ── Stage B: Section Extraction ──────────────────────────────────────
    if not skip_sections and filing_id:
        try:
            rate_limited_sleep()
            obj = filing.obj()
            sections = extract_sections_from_obj(obj)
            if sections:
                saved = save_sections_to_db(
                    filing_id, ticker, form_type, fiscal_year, fiscal_period, sections
                )
                stats["sections"] = saved
                print(f"    Sections: {saved}")
        except Exception as e:
            print(f"    Section extraction failed: {e}")

    # ── Stage C: Financial Statements ────────────────────────────────────
    if not skip_financials and xbrl:
        stmts = extract_financial_statements(xbrl)
        for stmt_type, md_content in stmts:
            period_str = f"{fiscal_year}" + (f" Q{fiscal_quarter}" if fiscal_quarter else "")
            md_content += (
                f"\n\n---\n*Source: {form_type} {period_str} | "
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
            )
            save_financial_statement(
                ticker, form_type, stmt_type, fiscal_year, fiscal_quarter, md_content
            )
            stats["statements"] += 1
        if stmts:
            print(f"    Statements: {stats['statements']}")

    return stats


# ── Per-Ticker Processing ────────────────────────────────────────────────────

def process_ticker(
    ticker: str,
    years: range,
    forms: list[str],
    existing_accessions: set[str],
    skip_sections: bool = False,
    skip_financials: bool = False,
) -> dict:
    """
    Process all filings for a single ticker.
    Returns aggregated stats.
    """
    set_identity("Subal Bhattarai (bhattaraisubal@gmail.com)")

    totals = {"filings": 0, "facts": 0, "sections": 0, "statements": 0, "skipped": 0}

    try:
        company = Company(ticker)
        company_name = company.name
        print(f"\n{'#'*60}")
        print(f"# {ticker} - {company_name}")
        print(f"{'#'*60}")
    except Exception as e:
        print(f"\nERROR: Could not load company for {ticker}: {e}")
        return totals

    for form in forms:
        print(f"\n  Fetching {form} filings...")
        try:
            filings = company.get_filings(form=form, amendments=False)
        except Exception as e:
            print(f"  ERROR fetching {form} filing list: {e}")
            continue

        for filing in filings:
            try:
                fy = infer_fiscal_year(ticker, filing)
                if fy not in years:
                    continue

                stats = process_single_filing(
                    ticker,
                    company_name,
                    filing,
                    existing_accessions,
                    skip_sections=skip_sections,
                    skip_financials=skip_financials,
                )

                if stats["skipped"]:
                    totals["skipped"] += 1
                else:
                    totals["filings"] += 1
                    totals["facts"] += stats["facts"]
                    totals["sections"] += stats["sections"]
                    totals["statements"] += stats["statements"]
                    # Track newly added accession for incremental check
                    existing_accessions.add(filing.accession_number)

                rate_limited_sleep()
            except Exception as e:
                acc = getattr(filing, 'accession_number', 'unknown')
                print(f"    ERROR processing filing {acc}: {e}")
                continue

    print(f"\n  {ticker} Summary: {totals['filings']} new filings, "
          f"{totals['facts']} facts, {totals['sections']} sections, "
          f"{totals['statements']} statements, {totals['skipped']} skipped")

    return totals


# ── Post-Processing Stages ───────────────────────────────────────────────────

def run_post_processing(skip_embed: bool = False):
    """Run post-fetch stages: migration, chunking, embedding, indexing."""

    # Stage D: Migrate filing_sections -> sections_10k / sections_10q
    print(f"\n{'='*60}")
    print("Stage D: Migrating sections to vector tables...")
    print(f"{'='*60}")
    try:
        init_vector_tables(reset=False)
        migrate_from_filing_sections()
    except Exception as e:
        print(f"  ERROR in migration: {e}")

    # Stage E: Chunk and embed
    if not skip_embed:
        print(f"\n{'='*60}")
        print("Stage E: Chunking and embedding sections...")
        print(f"{'='*60}")
        try:
            chunks_10k = process_sections_10k()
            chunks_10q = process_sections_10q()
            print(f"\n  Embedding complete: {chunks_10k} 10-K chunks, {chunks_10q} 10-Q chunks")
        except Exception as e:
            print(f"  ERROR in embedding: {e}")
    else:
        print("\n  Skipping embedding (--skip-embed)")

    # Stage F: Rebuild indexes and analyze
    print(f"\n{'='*60}")
    print("Stage F: Rebuilding indexes and analyzing tables...")
    print(f"{'='*60}")
    try:
        create_indexes()
        analyze_tables()
    except Exception as e:
        print(f"  ERROR in index/analyze: {e}")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SEC Filing Backfill Pipeline - Populate all database tables from EDGAR"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Process a single ticker (e.g., AAPL). Default: all 10 tickers.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=YEARS.start,
        help=f"Start year (default: {YEARS.start})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=YEARS.stop - 1,
        help=f"End year inclusive (default: {YEARS.stop - 1})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show gap analysis only, do not fetch data.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip the chunking/embedding stage (fetch + parse only).",
    )
    parser.add_argument(
        "--skip-sections",
        action="store_true",
        help="Skip section extraction (Stage B).",
    )
    parser.add_argument(
        "--skip-financials",
        action="store_true",
        help="Skip financial statement extraction (Stage C).",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q"],
        help="Filing types to process (default: 10-K 10-Q).",
    )
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    # Validate ticker
    if args.ticker and args.ticker.upper() not in TICKERS:
        print(f"Warning: {args.ticker.upper()} is not in the configured TICKERS list.")
        print(f"Configured tickers: {', '.join(TICKERS)}")
        print("Proceeding anyway...\n")

    print(f"{'='*60}")
    print("SEC Filing Backfill Pipeline")
    print(f"{'='*60}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Years: {years.start}-{years.stop - 1}")
    print(f"Forms: {', '.join(args.forms)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'BACKFILL'}")
    if not args.dry_run:
        print(f"Options: embed={'no' if args.skip_embed else 'yes'}, "
              f"sections={'no' if args.skip_sections else 'yes'}, "
              f"financials={'no' if args.skip_financials else 'yes'}")
    print(f"{'='*60}")

    # Ensure tables exist
    try:
        init_xbrl_db(reset=False)
        init_sections_table()
        init_financials_db(reset=False)
    except Exception as e:
        print(f"ERROR initializing database tables: {e}")
        sys.exit(1)

    # Gap analysis
    gaps = analyze_gaps(tickers, years)
    print_gap_report(gaps, years)

    if args.dry_run:
        print("Dry run complete. No data was fetched.")
        return

    # Get existing accessions for incremental mode
    existing_accessions = get_existing_accessions()
    print(f"Existing filings in DB: {len(existing_accessions)}")

    # Process tickers sequentially (EDGAR rate limiting makes parallelism less effective)
    start_time = time.time()
    grand_totals = {"filings": 0, "facts": 0, "sections": 0, "statements": 0, "skipped": 0}

    for ticker in tickers:
        totals = process_ticker(
            ticker,
            years,
            args.forms,
            existing_accessions,
            skip_sections=args.skip_sections,
            skip_financials=args.skip_financials,
        )
        for k in grand_totals:
            grand_totals[k] += totals[k]

    # Post-processing
    run_post_processing(skip_embed=args.skip_embed)

    # Final summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print("BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {minutes}m {seconds}s")
    print(f"New filings processed: {grand_totals['filings']}")
    print(f"Facts saved: {grand_totals['facts']}")
    print(f"Sections saved: {grand_totals['sections']}")
    print(f"Statements saved: {grand_totals['statements']}")
    print(f"Skipped (already loaded): {grand_totals['skipped']}")
    print(f"{'='*60}")

    # Run gap analysis again to confirm
    print("\nPost-backfill gap analysis:")
    gaps = analyze_gaps(tickers, years)
    print_gap_report(gaps, years)


if __name__ == "__main__":
    main()
