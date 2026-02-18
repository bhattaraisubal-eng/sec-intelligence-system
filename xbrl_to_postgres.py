"""
XBRL Filing -> JSON -> PostgreSQL Pipeline
Optimized for high-performance bulk inserts and fast queries.
"""

import os
import io
import time
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from edgar import Company, set_identity
from config import infer_fiscal_year, infer_fiscal_quarter, YEARS

load_dotenv()

# Connection pool for better performance
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
            database=os.getenv("PG_DATABASE")
        )
    return _connection_pool

@contextmanager
def get_db_connection():
    """Get connection from pool with automatic return."""
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

# Schema setup with optimized indexes
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS filings (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    form_type VARCHAR(20) NOT NULL,
    filing_date DATE NOT NULL,
    accession_number VARCHAR(50) UNIQUE NOT NULL,
    fiscal_year INT,
    fiscal_period VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS annual_facts (
    id SERIAL PRIMARY KEY,
    filing_id INT REFERENCES filings(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INT NOT NULL,
    concept VARCHAR(255) NOT NULL,
    value NUMERIC,
    unit VARCHAR(50),
    start_date DATE,
    end_date DATE,
    instant_date DATE,
    decimals INT,
    raw_value TEXT,
    dimension VARCHAR(255),
    member VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quarterly_facts (
    id SERIAL PRIMARY KEY,
    filing_id INT REFERENCES filings(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INT NOT NULL,
    fiscal_quarter INT NOT NULL,
    concept VARCHAR(255) NOT NULL,
    value NUMERIC,
    unit VARCHAR(50),
    start_date DATE,
    end_date DATE,
    instant_date DATE,
    decimals INT,
    raw_value TEXT,
    dimension VARCHAR(255),
    member VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Indexes created after bulk load for better insert performance
INDEX_SQL = """
-- Filings indexes
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_form_date ON filings(form_type, filing_date DESC);

-- Annual facts indexes (optimized for common queries)
CREATE INDEX IF NOT EXISTS idx_annual_filing ON annual_facts(filing_id);
CREATE INDEX IF NOT EXISTS idx_annual_ticker_year ON annual_facts(ticker, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_annual_concept_value ON annual_facts(concept, value) WHERE value IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_annual_concept_hash ON annual_facts USING hash(concept);

-- Quarterly facts indexes
CREATE INDEX IF NOT EXISTS idx_quarterly_filing ON quarterly_facts(filing_id);
CREATE INDEX IF NOT EXISTS idx_quarterly_ticker_year_q ON quarterly_facts(ticker, fiscal_year, fiscal_quarter);
CREATE INDEX IF NOT EXISTS idx_quarterly_concept_value ON quarterly_facts(concept, value) WHERE value IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_quarterly_concept_hash ON quarterly_facts USING hash(concept);

-- Covering indexes for common RAG queries (includes value to avoid table lookup)
CREATE INDEX IF NOT EXISTS idx_annual_rag ON annual_facts(ticker, concept, fiscal_year) INCLUDE (value, unit);
CREATE INDEX IF NOT EXISTS idx_quarterly_rag ON quarterly_facts(ticker, concept, fiscal_year, fiscal_quarter) INCLUDE (value, unit);

-- Partial indexes for consolidated-total queries (dimension IS NULL = totals)
CREATE INDEX IF NOT EXISTS idx_annual_consolidated ON annual_facts(ticker, concept, fiscal_year) INCLUDE (value, unit) WHERE dimension IS NULL;
CREATE INDEX IF NOT EXISTS idx_quarterly_consolidated ON quarterly_facts(ticker, concept, fiscal_year, fiscal_quarter) INCLUDE (value, unit) WHERE dimension IS NULL;
"""

def migrate_add_dimensions():
    """Add dimension/member columns to existing tables (idempotent)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE annual_facts ADD COLUMN IF NOT EXISTS dimension VARCHAR(255);
                ALTER TABLE annual_facts ADD COLUMN IF NOT EXISTS member VARCHAR(255);
                ALTER TABLE quarterly_facts ADD COLUMN IF NOT EXISTS dimension VARCHAR(255);
                ALTER TABLE quarterly_facts ADD COLUMN IF NOT EXISTS member VARCHAR(255);
            """)
        conn.commit()

def init_db(reset=False):
    """Create tables if they don't exist. Use reset=True to drop and recreate."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if reset:
                cur.execute("""
                    DROP TABLE IF EXISTS annual_facts CASCADE;
                    DROP TABLE IF EXISTS quarterly_facts CASCADE;
                    DROP TABLE IF EXISTS filings CASCADE;
                """)
            cur.execute(SCHEMA_SQL)
        conn.commit()
    # Migrate existing DBs to add dimension/member columns
    migrate_add_dimensions()
    print("Database initialized.")

def create_indexes():
    """Create indexes after bulk data load for better performance."""
    print("Creating indexes...")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(INDEX_SQL)
        conn.commit()
    print("Indexes created.")

def analyze_tables():
    """Update table statistics for query optimizer."""
    print("Analyzing tables...")
    with get_db_connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("ANALYZE filings;")
            cur.execute("ANALYZE annual_facts;")
            cur.execute("ANALYZE quarterly_facts;")
    print("Tables analyzed.")

def parse_xbrl_to_facts(xbrl) -> list[dict]:
    """Extract facts from XBRL object using vectorized pandas operations."""
    import pandas as pd
    import numpy as np

    try:
        df = xbrl.facts.to_dataframe()

        # Handle decimals - replace inf with NaN before converting
        decimals_col = pd.to_numeric(df['decimals'], errors='coerce')
        decimals_col = decimals_col.replace([np.inf, -np.inf], np.nan)

        # Vectorized column selection and renaming
        result_cols = {
            'concept': df['concept'],
            'value': pd.to_numeric(df['numeric_value'], errors='coerce'),
            'unit': df['unit_ref'].where(df['unit_ref'].notna(), None),
            'start_date': df['period_start'].astype(str).replace({'nan': None, 'NaT': None, 'None': None}),
            'end_date': df['period_end'].astype(str).replace({'nan': None, 'NaT': None, 'None': None}),
            'instant_date': df['period_instant'].astype(str).replace({'nan': None, 'NaT': None, 'None': None}),
            'decimals': decimals_col,
            'raw_value': df['value'].astype(str).where(df['value'].notna(), None),
        }

        # Extract dimension/member if present in the DataFrame
        if 'dimension' in df.columns:
            result_cols['dimension'] = df['dimension'].where(df['dimension'].notna(), None)
        else:
            result_cols['dimension'] = None
        if 'member' in df.columns:
            result_cols['member'] = df['member'].where(df['member'].notna(), None)
        else:
            result_cols['member'] = None

        result = pd.DataFrame(result_cols)

        # Filter rows with valid concept
        result = result[result['concept'].notna()]

        # Convert decimals to int where possible, cap to valid INT range
        def safe_int(x):
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return None
            try:
                val = int(x)
                # PostgreSQL INT range: -2147483648 to 2147483647
                if -2147483648 <= val <= 2147483647:
                    return val
                return None
            except (ValueError, OverflowError, TypeError):
                return None

        result['decimals'] = result['decimals'].apply(safe_int)

        # Convert to list of dicts and clean up NaN values
        records = result.to_dict('records')
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    r[k] = None

        return records

    except Exception as e:
        print(f"  Warning parsing facts: {e}")
        return []

def get_xbrl_entity_info(xbrl) -> dict:
    """Safely extract entity_info dict from XBRL object (DEI tags).

    Returns dict with keys like 'fiscal_period' ('FY'/'Q1'-'Q4'),
    'fiscal_year', 'fiscal_year_end_month', 'fiscal_year_end_day'.
    Returns empty dict on failure.
    """
    try:
        if hasattr(xbrl, 'entity_info'):
            info = xbrl.entity_info
            if isinstance(info, dict):
                return info
    except Exception:
        pass
    return {}


def _infer_fiscal_year_from_date(d, fy_end_month: int) -> int:
    """Infer fiscal year from a date and FY-end month.

    Same logic as config.infer_fiscal_year but operates on a plain date.
    """
    from datetime import date, datetime
    if isinstance(d, str):
        try:
            d = datetime.strptime(d, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return d.year if hasattr(d, 'year') else 0
    if fy_end_month == 12:
        return d.year
    # If the date's month is after FY-end month, it belongs to next FY
    if d.month > fy_end_month:
        return d.year + 1
    return d.year


def _infer_quarter_from_date(d, fy_end_month: int) -> int:
    """Infer fiscal quarter (1-4) from a date and FY-end month.

    Same logic as config.infer_fiscal_quarter but operates on a plain date.
    """
    from datetime import date, datetime
    if isinstance(d, str):
        try:
            d = datetime.strptime(d, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return (d.month - 1) // 3 + 1 if hasattr(d, 'month') else 1
    fy_start_month = (fy_end_month % 12) + 1
    month = d.month
    # Handle 52/53-week fiscal calendars (e.g., AAPL, AVGO) where period
    # end dates can spill 1-7 days into the next month.
    if d.day <= 7:
        month = 12 if month == 1 else month - 1
    if month >= fy_start_month:
        months_into_fy = month - fy_start_month
    else:
        months_into_fy = (12 - fy_start_month) + month
    quarter = (months_into_fy // 3) + 1
    return max(1, min(4, quarter))


# Duration ranges from edgartools TTMCalculator
_QUARTER_MIN, _QUARTER_MAX = 70, 120
_YTD_6M_MIN, _YTD_6M_MAX = 140, 229
_YTD_9M_MIN, _YTD_9M_MAX = 230, 329
_ANNUAL_MIN, _ANNUAL_MAX = 330, 420


def classify_fact_period(fact: dict, entity_info: dict,
                         filing_fiscal_year: int, filing_fiscal_quarter: int | None,
                         fy_end_month: int = 12) -> tuple:
    """Classify a single fact into (target_table, fiscal_year, fiscal_quarter).

    Returns:
        (target_table, fiscal_year, fiscal_quarter) where:
        - target_table is 'annual_facts', 'quarterly_facts', or None (skip)
        - fiscal_year is the inferred FY for this fact
        - fiscal_quarter is the inferred quarter (None for annual)
    """
    from datetime import datetime

    start_str = fact.get("start_date")
    end_str = fact.get("end_date")
    instant_str = fact.get("instant_date")

    # --- Duration facts (have start_date and end_date) ---
    if start_str and end_str and start_str != 'None' and end_str != 'None':
        try:
            start_d = datetime.strptime(start_str, "%Y-%m-%d").date()
            end_d = datetime.strptime(end_str, "%Y-%m-%d").date()
            days = (end_d - start_d).days

            if _QUARTER_MIN <= days <= _QUARTER_MAX:
                fy = _infer_fiscal_year_from_date(end_d, fy_end_month)
                q = _infer_quarter_from_date(end_d, fy_end_month)
                return ('quarterly_facts', fy, q)
            elif _ANNUAL_MIN <= days <= _ANNUAL_MAX:
                fy = _infer_fiscal_year_from_date(end_d, fy_end_month)
                return ('annual_facts', fy, None)
            else:
                # YTD (6M/9M) or out-of-range — skip
                return (None, None, None)
        except (ValueError, TypeError):
            pass  # Fall through to fallback

    # --- Instant facts (have instant_date only) ---
    if instant_str and instant_str != 'None':
        filing_period = entity_info.get('fiscal_period', '') if entity_info else ''
        instant_d = None
        try:
            instant_d = datetime.strptime(instant_str, "%Y-%m-%d").date()
            fy = _infer_fiscal_year_from_date(instant_d, fy_end_month)
        except (ValueError, TypeError):
            fy = filing_fiscal_year

        if filing_period == 'FY' or filing_fiscal_quarter is None:
            return ('annual_facts', fy, None)
        else:
            if instant_d is not None:
                q = _infer_quarter_from_date(instant_d, fy_end_month)
            else:
                q = filing_fiscal_quarter
            return ('quarterly_facts', fy, q)

    # --- No dates: fallback to filing-level classification ---
    if filing_fiscal_quarter is None:
        return ('annual_facts', filing_fiscal_year, None)
    else:
        return ('quarterly_facts', filing_fiscal_year, filing_fiscal_quarter)


def save_filing(ticker: str, company_name: str, filing, xbrl) -> int | None:
    """Save filing metadata and return filing_id."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check if already exists
            cur.execute(
                "SELECT id FROM filings WHERE accession_number = %s",
                (filing.accession_number,)
            )
            existing = cur.fetchone()
            if existing:
                print(f"  Filing {filing.accession_number} already exists, skipping.")
                return None

            # Determine fiscal period
            fiscal_year = infer_fiscal_year(ticker, filing)
            quarter = infer_fiscal_quarter(ticker, filing)
            fiscal_period = "FY" if filing.form == "10-K" else f"Q{quarter}"

            cur.execute("""
                INSERT INTO filings (ticker, company_name, form_type, filing_date,
                                     accession_number, fiscal_year, fiscal_period)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (ticker, company_name, filing.form, filing.filing_date,
                  filing.accession_number, fiscal_year, fiscal_period))

            filing_id = cur.fetchone()[0]
        conn.commit()
    return filing_id

def save_facts_copy(conn, filing_id: int, facts: list[dict], ticker: str,
                    fiscal_year: int, form_type: str, fiscal_quarter: int = None,
                    entity_info: dict = None):
    """Ultra-fast bulk insert using PostgreSQL COPY command.

    Routes each fact individually to annual_facts or quarterly_facts based on
    period duration when entity_info is provided.
    """
    if not facts:
        return 0

    from config import FISCAL_YEAR_END_MONTH
    fy_end_month = FISCAL_YEAR_END_MONTH.get(ticker, 12)

    annual_buffer = io.StringIO()
    quarterly_buffer = io.StringIO()
    annual_count = 0
    quarterly_count = 0

    def _make_row(f, fact_fy, fact_q, is_annual):
        raw = (f["raw_value"] or '').replace('\t', ' ').replace('\n', ' ') or '\\N'
        common = [
            str(f["value"]) if f["value"] is not None else '\\N',
            f["unit"] or '\\N',
            f["start_date"] or '\\N', f["end_date"] or '\\N', f["instant_date"] or '\\N',
            str(int(f["decimals"])) if f["decimals"] is not None else '\\N',
            raw,
            f["dimension"] or '\\N',
            f["member"] or '\\N',
        ]
        if is_annual:
            return [str(filing_id), ticker, str(fact_fy), f["concept"] or ''] + common
        else:
            return [str(filing_id), ticker, str(fact_fy), str(fact_q), f["concept"] or ''] + common

    for f in facts:
        if entity_info is not None:
            table, fact_fy, fact_q = classify_fact_period(
                f, entity_info, fiscal_year, fiscal_quarter, fy_end_month
            )
        else:
            if form_type == "10-K":
                table, fact_fy, fact_q = 'annual_facts', fiscal_year, None
            else:
                table, fact_fy, fact_q = 'quarterly_facts', fiscal_year, fiscal_quarter

        if table is None:
            continue

        if table == 'annual_facts':
            row = _make_row(f, fact_fy, fact_q, True)
            annual_buffer.write('\t'.join(row) + '\n')
            annual_count += 1
        else:
            row = _make_row(f, fact_fy, fact_q, False)
            quarterly_buffer.write('\t'.join(row) + '\n')
            quarterly_count += 1

    with conn.cursor() as cur:
        if annual_count > 0:
            annual_buffer.seek(0)
            cur.copy_from(annual_buffer, 'annual_facts',
                columns=('filing_id', 'ticker', 'fiscal_year', 'concept', 'value',
                        'unit', 'start_date', 'end_date', 'instant_date', 'decimals', 'raw_value',
                        'dimension', 'member'))
        if quarterly_count > 0:
            quarterly_buffer.seek(0)
            cur.copy_from(quarterly_buffer, 'quarterly_facts',
                columns=('filing_id', 'ticker', 'fiscal_year', 'fiscal_quarter', 'concept', 'value',
                        'unit', 'start_date', 'end_date', 'instant_date', 'decimals', 'raw_value',
                        'dimension', 'member'))

    return annual_count + quarterly_count

def save_facts(filing_id: int, facts: list[dict], ticker: str, fiscal_year: int,
               form_type: str, fiscal_quarter: int = None, entity_info: dict = None):
    """Bulk insert facts, routing each fact to annual_facts or quarterly_facts
    based on its actual period duration rather than the filing's form type.

    When entity_info is provided, uses per-fact classification. Otherwise falls
    back to the legacy form_type-based routing for backward compatibility.
    """
    if not facts:
        return 0

    from config import FISCAL_YEAR_END_MONTH
    fy_end_month = FISCAL_YEAR_END_MONTH.get(ticker, 12)

    annual_values = []
    quarterly_values = []
    skipped = 0

    for f in facts:
        if entity_info is not None:
            table, fact_fy, fact_q = classify_fact_period(
                f, entity_info, fiscal_year, fiscal_quarter, fy_end_month
            )
        else:
            # Legacy fallback: route all facts by form_type
            if form_type == "10-K":
                table, fact_fy, fact_q = 'annual_facts', fiscal_year, None
            else:
                table, fact_fy, fact_q = 'quarterly_facts', fiscal_year, fiscal_quarter

        if table is None:
            skipped += 1
            continue

        if table == 'annual_facts':
            annual_values.append(
                (filing_id, ticker, fact_fy, f["concept"], f["value"],
                 f["unit"], f["start_date"], f["end_date"], f["instant_date"],
                 f["decimals"], f["raw_value"], f["dimension"], f["member"])
            )
        else:
            quarterly_values.append(
                (filing_id, ticker, fact_fy, fact_q, f["concept"],
                 f["value"], f["unit"], f["start_date"], f["end_date"],
                 f["instant_date"], f["decimals"], f["raw_value"],
                 f["dimension"], f["member"])
            )

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if annual_values:
                execute_values(cur, """
                    INSERT INTO annual_facts
                    (filing_id, ticker, fiscal_year, concept, value, unit,
                     start_date, end_date, instant_date, decimals, raw_value,
                     dimension, member)
                    VALUES %s
                """, annual_values, page_size=1000)
            if quarterly_values:
                execute_values(cur, """
                    INSERT INTO quarterly_facts
                    (filing_id, ticker, fiscal_year, fiscal_quarter, concept,
                     value, unit, start_date, end_date, instant_date, decimals, raw_value,
                     dimension, member)
                    VALUES %s
                """, quarterly_values, page_size=1000)
        conn.commit()

    saved = len(annual_values) + len(quarterly_values)
    if entity_info is not None:
        print(f"    Routed: {len(annual_values)} annual, {len(quarterly_values)} quarterly, {skipped} skipped (YTD/other)")
    return saved

def process_filing(ticker: str, company_name: str, filing):
    """Full pipeline: fetch XBRL -> parse -> classify per-fact -> save."""
    print(f"\nProcessing {filing.form} {filing.filing_date} ({filing.accession_number})")

    # Fetch XBRL
    xbrl = filing.xbrl()
    if not xbrl:
        print("  No XBRL data available.")
        return

    # Extract entity info for per-fact classification
    entity_info = get_xbrl_entity_info(xbrl)

    # Parse to facts (JSON-like dicts)
    facts = parse_xbrl_to_facts(xbrl)
    print(f"  Parsed {len(facts)} facts")

    # Determine fiscal info
    fiscal_year = infer_fiscal_year(ticker, filing)
    fiscal_quarter = infer_fiscal_quarter(ticker, filing)

    # Save filing metadata
    filing_id = save_filing(ticker, company_name, filing, xbrl)
    if filing_id is None:
        return

    # Save facts — each fact routed individually by period duration
    saved = save_facts(filing_id, facts, ticker, fiscal_year, filing.form, fiscal_quarter,
                       entity_info=entity_info)
    print(f"  Saved {saved} facts (filing_id={filing_id})")

def fetch_and_store(ticker: str, forms: list[str] = ["10-K", "10-Q"], years: range = YEARS):
    """Main entry point: fetch filings and store in database."""
    set_identity("Subal Bhattarai (bhattaraisubal@gmail.com)")

    company = Company(ticker)
    print(f"Company: {company.name}")

    for form in forms:
        print(f"\n{'='*60}\nFetching {form} filings...\n{'='*60}")

        for filing in company.get_filings(form=form, amendments=False):
            fy = infer_fiscal_year(ticker, filing)
            if fy in years:
                process_filing(ticker, company.name, filing)
                time.sleep(0.15)

def show_stats():
    """Display summary statistics of saved data."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Total filings
            cur.execute("SELECT COUNT(*) FROM filings")
            total_filings = cur.fetchone()[0]

            # Annual facts
            cur.execute("SELECT COUNT(*) FROM annual_facts")
            annual_count = cur.fetchone()[0]

            # Quarterly facts
            cur.execute("SELECT COUNT(*) FROM quarterly_facts")
            quarterly_count = cur.fetchone()[0]

            # Annual breakdown by ticker/year
            cur.execute("""
                SELECT ticker, fiscal_year, COUNT(*) as facts
                FROM annual_facts
                GROUP BY ticker, fiscal_year
                ORDER BY ticker, fiscal_year
            """)
            annual_breakdown = cur.fetchall()

            # Quarterly breakdown by ticker/year/quarter
            cur.execute("""
                SELECT ticker, fiscal_year, fiscal_quarter, COUNT(*) as facts
                FROM quarterly_facts
                GROUP BY ticker, fiscal_year, fiscal_quarter
                ORDER BY ticker, fiscal_year, fiscal_quarter
            """)
            quarterly_breakdown = cur.fetchall()

            # Top concepts in annual
            cur.execute("""
                SELECT concept, COUNT(*) as cnt FROM annual_facts
                GROUP BY concept ORDER BY cnt DESC LIMIT 5
            """)
            top_annual = cur.fetchall()

            # Top concepts in quarterly
            cur.execute("""
                SELECT concept, COUNT(*) as cnt FROM quarterly_facts
                GROUP BY concept ORDER BY cnt DESC LIMIT 5
            """)
            top_quarterly = cur.fetchall()

    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    print(f"Total Filings:      {total_filings}")
    print(f"Annual Facts:       {annual_count}")
    print(f"Quarterly Facts:    {quarterly_count}")

    if annual_breakdown:
        print("\n[ANNUAL_FACTS] by Ticker/Year:")
        print(f"  {'Ticker':<8} {'Year':<6} {'Facts':>10}")
        print(f"  {'-'*8} {'-'*6} {'-'*10}")
        for row in annual_breakdown:
            print(f"  {row[0]:<8} {row[1]:<6} {row[2]:>10}")

    if quarterly_breakdown:
        print("\n[QUARTERLY_FACTS] by Ticker/Year/Quarter:")
        print(f"  {'Ticker':<8} {'Year':<6} {'Q':<3} {'Facts':>10}")
        print(f"  {'-'*8} {'-'*6} {'-'*3} {'-'*10}")
        for row in quarterly_breakdown:
            print(f"  {row[0]:<8} {row[1]:<6} Q{row[2]:<2} {row[3]:>10}")

    if top_annual:
        print("\nTop 5 Annual Concepts:")
        for concept, cnt in top_annual:
            print(f"  {cnt:>6}x  {concept[:50]}")

    if top_quarterly:
        print("\nTop 5 Quarterly Concepts:")
        for concept, cnt in top_quarterly:
            print(f"  {cnt:>6}x  {concept[:50]}")

    print("=" * 60)


# =============================================================================
# QUERY HELPERS - Optimized for RAG retrieval
# =============================================================================

def query_annual(ticker: str = None, concept: str = None, fiscal_year: int = None,
                 limit: int = 100, include_segments: bool = False) -> list[dict]:
    """Fast query for annual facts with optional filters.

    Uses DISTINCT ON to return one row per concept, picking the fact with the
    latest period end-date and longest duration.  This filters out prior-year
    comparatives that XBRL embeds in the same filing.

    By default, filters to consolidated totals (dimension IS NULL).
    Set include_segments=True to also return segment-level breakdowns.
    """
    conditions = []
    params = []

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker)
    if concept:
        conditions.append("concept = %s")
        params.append(concept)
    if fiscal_year:
        conditions.append("fiscal_year = %s")
        params.append(fiscal_year)
    if not include_segments:
        conditions.append("dimension IS NULL")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT ON (concept, dimension, member)
                    ticker, fiscal_year, concept, value, unit, start_date, end_date,
                    dimension, member
                FROM annual_facts
                {where}
                ORDER BY concept, dimension, member,
                         COALESCE(end_date, instant_date) DESC NULLS LAST,
                         (end_date - start_date) DESC NULLS LAST
                LIMIT %s
            """, params + [limit])

            columns = ['ticker', 'fiscal_year', 'concept', 'value', 'unit',
                        'start_date', 'end_date', 'dimension', 'member']
            return [dict(zip(columns, row)) for row in cur.fetchall()]

def query_quarterly(ticker: str = None, concept: str = None, fiscal_year: int = None,
                    fiscal_quarter: int = None, limit: int = 100,
                    include_segments: bool = False) -> list[dict]:
    """Fast query for quarterly facts with optional filters.

    Uses DISTINCT ON to return one row per concept, picking the fact with the
    latest period end-date and shortest duration to prefer standalone quarterly
    values over cumulative YTD figures.

    By default, filters to consolidated totals (dimension IS NULL).
    Set include_segments=True to also return segment-level breakdowns.
    """
    conditions = []
    params = []

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker)
    if concept:
        conditions.append("concept = %s")
        params.append(concept)
    if fiscal_year:
        conditions.append("fiscal_year = %s")
        params.append(fiscal_year)
    if fiscal_quarter:
        conditions.append("fiscal_quarter = %s")
        params.append(fiscal_quarter)
    if not include_segments:
        conditions.append("dimension IS NULL")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT ON (concept, dimension, member)
                    ticker, fiscal_year, fiscal_quarter, concept, value, unit,
                    dimension, member
                FROM quarterly_facts
                {where}
                ORDER BY concept, dimension, member,
                         COALESCE(end_date, instant_date) DESC NULLS LAST,
                         (end_date - start_date) ASC NULLS LAST
                LIMIT %s
            """, params + [limit])

            columns = ['ticker', 'fiscal_year', 'fiscal_quarter', 'concept', 'value', 'unit',
                        'dimension', 'member']
            return [dict(zip(columns, row)) for row in cur.fetchall()]

def search_concepts(pattern: str, table: str = 'annual_facts', limit: int = 50) -> list[str]:
    """Search for concepts matching a pattern (case-insensitive)."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT concept
                FROM {table}
                WHERE concept ILIKE %s
                ORDER BY concept
                LIMIT %s
            """, (f'%{pattern}%', limit))
            return [row[0] for row in cur.fetchall()]

def get_metric_timeseries(ticker: str, concept: str, table: str = 'quarterly_facts',
                          concepts: list[str] | None = None) -> list[dict]:
    """Get time series data for a specific metric (consolidated totals only).

    When *concepts* (a list of XBRL concept aliases) is provided, queries all
    of them and merges into a single deduplicated timeseries.  This handles
    XBRL concept name changes across filing years (e.g. SalesRevenueNet →
    Revenues → RevenueFromContractWithCustomerExcludingAssessedTax).
    """
    concept_list = concepts if concepts else [concept]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if table == 'quarterly_facts':
                cur.execute(f"""
                    SELECT DISTINCT ON (fiscal_year, fiscal_quarter)
                        fiscal_year, fiscal_quarter, value, unit, end_date
                    FROM quarterly_facts
                    WHERE ticker = %s AND concept = ANY(%s) AND value IS NOT NULL
                          AND dimension IS NULL
                    ORDER BY fiscal_year, fiscal_quarter,
                             COALESCE(end_date, instant_date) DESC NULLS LAST,
                             (end_date - start_date) ASC NULLS LAST
                """, (ticker, concept_list))
                columns = ['fiscal_year', 'fiscal_quarter', 'value', 'unit', 'end_date']
            else:
                cur.execute(f"""
                    SELECT DISTINCT ON (fiscal_year)
                        fiscal_year, value, unit, end_date
                    FROM annual_facts
                    WHERE ticker = %s AND concept = ANY(%s) AND value IS NOT NULL
                          AND dimension IS NULL
                    ORDER BY fiscal_year,
                             COALESCE(end_date, instant_date) DESC NULLS LAST,
                             (end_date - start_date) DESC NULLS LAST
                """, (ticker, concept_list))
                columns = ['fiscal_year', 'value', 'unit', 'end_date']

            return [dict(zip(columns, row)) for row in cur.fetchall()]

def explain_query(sql: str, params: tuple = None):
    """Run EXPLAIN ANALYZE on a query to check performance."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN ANALYZE {sql}", params or ())
            for row in cur.fetchall():
                print(row[0])


def derive_q4(ticker: str, concept: str, fiscal_year: int) -> dict | None:
    """Derive Q4 value from DB: Annual - Q1 - Q2 - Q3.

    SEC does not require a separate 10-Q for Q4, so Q4 must be derived
    from the annual (10-K) value minus the three quarterly (10-Q) values.
    """
    # Get annual value
    annual_facts = query_annual(ticker=ticker, concept=concept,
                                fiscal_year=fiscal_year, limit=1)
    if not annual_facts or annual_facts[0].get("value") is None:
        return None
    annual_val = annual_facts[0]["value"]

    # Get Q1, Q2, Q3
    q_sum = 0
    quarters_found = 0
    for q in (1, 2, 3):
        q_facts = query_quarterly(ticker=ticker, concept=concept,
                                  fiscal_year=fiscal_year, fiscal_quarter=q, limit=1)
        if q_facts and q_facts[0].get("value") is not None:
            q_sum += q_facts[0]["value"]
            quarters_found += 1

    if quarters_found < 3:
        return None

    q4_val = annual_val - q_sum
    return {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": 4,
        "concept": concept,
        "value": q4_val,
        "unit": annual_facts[0].get("unit", "USD"),
        "calculation": "Annual - Q1 - Q2 - Q3",
        "dimension": None,
        "member": None,
    }


def query_q4(ticker: str, concept: str, fiscal_year: int) -> dict | None:
    """
    Query a Q4 value for a specific concept using edgartools' built-in quarterization.

    SEC does not require a separate 10-Q for Q4, so Q4 values are calculated by
    edgartools using its TTMCalculator:
    - Income Statement concepts: Q4 = Annual - Q1 - Q2 - Q3
    - Balance Sheet concepts: Q4 = Annual (point-in-time snapshot)
    - Cash Flow concepts: Q4 = Annual - Q3_YTD

    Returns a dict with calculated value and metadata, or None if not calculable.
    """
    from edgar import Company, set_identity
    from edgar.ttm.calculator import TTMCalculator

    try:
        set_identity("SEC RAG System sec-rag@example.com")
        company = Company(ticker)
        facts = company.facts
        if not facts:
            return None

        # Get all facts and filter to the specific concept
        all_facts = facts.get_all_facts()
        concept_facts = [f for f in all_facts if concept in f.concept]
        if not concept_facts:
            return None

        # Use TTMCalculator to get quarterized facts (includes derived Q4)
        calc = TTMCalculator(concept_facts)
        quarterly_facts = calc._quarterize_facts()

        # Find Q4 for the requested fiscal year
        q4_fact = None
        for fact in quarterly_facts:
            if (fact.fiscal_period == 'Q4' and
                fact.fiscal_year == fiscal_year):
                q4_fact = fact
                break

        if q4_fact is None:
            return None

        # Determine calculation method from calculation_context
        calc_method = 'Reported'
        if q4_fact.calculation_context:
            if 'derived_q4_fy_minus_ytd9' in q4_fact.calculation_context:
                calc_method = 'Annual - Q3_YTD'
            elif 'derived_q4_fy_minus_q1q2q3' in q4_fact.calculation_context:
                calc_method = 'Annual - Q1 - Q2 - Q3'

        return {
            'ticker': ticker,
            'fiscal_year': fiscal_year,
            'fiscal_quarter': 4,
            'concept': concept,
            'value': float(q4_fact.numeric_value) if q4_fact.numeric_value else None,
            'unit': q4_fact.unit,
            'calculation': calc_method,
            'period_start': str(q4_fact.period_start) if q4_fact.period_start else None,
            'period_end': str(q4_fact.period_end) if q4_fact.period_end else None,
        }

    except Exception as e:
        print(f"Error querying Q4 for {ticker}/{concept}/{fiscal_year}: {e}")
        return None


if __name__ == "__main__":
    import config

    init_db(reset=True)  # Fresh start with new schema

    for ticker in config.TICKERS:
        print(f"\n{'#'*60}")
        print(f"# Processing {ticker}")
        print(f"{'#'*60}")
        fetch_and_store(ticker, forms=["10-K", "10-Q"], years=config.YEARS)
        time.sleep(config.SEC_RATE_LIMIT_DELAY)

    create_indexes()  # Create indexes after bulk load
    analyze_tables()  # Update statistics
    show_stats()
