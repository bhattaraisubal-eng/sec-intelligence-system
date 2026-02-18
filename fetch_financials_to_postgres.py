"""
Fetch SEC financial statements using edgartools and save directly to PostgreSQL.
No intermediate markdown files on disk.
"""

import os
import time
from datetime import datetime
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from edgar import Company, set_identity
from config import infer_fiscal_year, infer_fiscal_quarter, YEARS

load_dotenv()

# Database connection pool
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
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS financial_documents (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    form_type VARCHAR(10) NOT NULL,
    statement_type VARCHAR(50) NOT NULL,
    fiscal_year INT NOT NULL,
    fiscal_quarter INT,
    markdown_content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, form_type, statement_type, fiscal_year, fiscal_quarter)
);

CREATE INDEX IF NOT EXISTS idx_fd_lookup
ON financial_documents(ticker, statement_type, fiscal_year, fiscal_quarter);
"""


def init_db(reset=False):
    """Create table. Use reset=True to drop and recreate."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if reset:
                cur.execute("DROP TABLE IF EXISTS financial_documents CASCADE;")
            cur.execute(SCHEMA_SQL)
        conn.commit()
    print("Table 'financial_documents' initialized.")


def save_to_postgres(ticker: str, form_type: str, statement_type: str,
                     fiscal_year: int, fiscal_quarter: int | None,
                     markdown_content: str):
    """Save markdown content directly to PostgreSQL."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO financial_documents
                (ticker, form_type, statement_type, fiscal_year, fiscal_quarter, markdown_content)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, form_type, statement_type, fiscal_year, fiscal_quarter)
                DO UPDATE SET markdown_content = EXCLUDED.markdown_content,
                              created_at = CURRENT_TIMESTAMP
            """, (ticker, form_type, statement_type, fiscal_year, fiscal_quarter, markdown_content))
        conn.commit()


def process_filing(ticker: str, filing, form_type: str, year: int, quarter: int = None):
    """Process a filing and save statements directly to PostgreSQL."""
    try:
        q_str = f"Q{quarter}" if quarter else "Annual"
        print(f"\nProcessing {form_type} {year} {q_str}")

        xbrl = filing.xbrl()
        if xbrl is None:
            print("  No XBRL data available")
            return

        financials = xbrl.statements

        statements = [
            ("income_statement", financials.income_statement),
            ("balance_sheet", financials.balance_sheet),
            ("cash_flow", financials.cashflow_statement),
        ]

        for stmt_type, stmt_func in statements:
            try:
                statement = stmt_func()
                if statement is None:
                    print(f"  Warning: No {stmt_type} data")
                    continue

                # Use edgartools built-in markdown conversion
                md_content = statement.render().to_markdown()

                # Add metadata footer
                period_str = f"{year}" + (f" Q{quarter}" if quarter else "")
                md_content += f"\n\n---\n*Source: {form_type} {period_str} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

                # Save directly to PostgreSQL
                save_to_postgres(ticker, form_type, stmt_type, year, quarter, md_content)
                print(f"  Saved: {stmt_type}")

            except Exception as e:
                print(f"  Error extracting {stmt_type}: {e}")

    except Exception as e:
        print(f"  Error processing filing: {e}")


def fetch_and_store(ticker: str, years: range = YEARS):
    """Fetch filings and store directly in PostgreSQL."""
    set_identity("Subal Bhattarai (bhattaraisubal@gmail.com)")

    company = Company(ticker)
    print(f"Company: {company.name}")

    # Process 10-K (annual) filings
    print("\n" + "=" * 50)
    print("Fetching 10-K (Annual) filings...")
    print("=" * 50)

    for filing in company.get_filings(form="10-K", amendments=False):
        year = infer_fiscal_year(ticker, filing)
        if year in years:
            process_filing(ticker, filing, "10-K", year)
            time.sleep(0.15)

    # Process 10-Q (quarterly) filings
    print("\n" + "=" * 50)
    print("Fetching 10-Q (Quarterly) filings...")
    print("=" * 50)

    for filing in company.get_filings(form="10-Q", amendments=False):
        year = infer_fiscal_year(ticker, filing)
        if year in years:
            quarter = infer_fiscal_quarter(ticker, filing)
            if quarter is None:
                print(f"  Warning: Could not infer quarter, skipping")
                continue
            process_filing(ticker, filing, "10-Q", year, quarter)
            time.sleep(0.15)


def _get_q4_statement_from_edgartools(ticker: str, fiscal_year: int,
                                       statement_type: str) -> str | None:
    """Generate Q4 statement using edgartools' built-in quarterization.

    Edgartools automatically derives Q4 values from annual and YTD data:
    - Income Statement: Q4 = Annual - Q1 - Q2 - Q3
    - Balance Sheet: Q4 = Annual (point-in-time)
    - Cash Flow: Q4 = Annual - Q3_YTD
    """
    try:
        set_identity("SEC RAG System sec-rag@example.com")
        company = Company(ticker)

        # Get quarterly statement with derived Q4 values
        statement_methods = {
            'income_statement': lambda: company.income_statement(period='quarterly', periods=8),
            'balance_sheet': lambda: company.balance_sheet(period='quarterly', periods=8),
            'cash_flow': lambda: company.cash_flow_statement(period='quarterly', periods=8),
        }

        if statement_type not in statement_methods:
            return None

        stmt = statement_methods[statement_type]()
        if stmt is None:
            return None

        # Filter to Q4 of the requested fiscal year
        df = stmt.to_dataframe() if hasattr(stmt, 'to_dataframe') else None
        if df is None or df.empty:
            return None

        # Look for Q4 column matching the fiscal year
        q4_col = None
        for col in df.columns:
            col_str = str(col)
            if f'Q4' in col_str and str(fiscal_year) in col_str:
                q4_col = col
                break

        if q4_col is None:
            # Try alternate format
            for col in df.columns:
                col_str = str(col)
                if 'Q4' in col_str or (str(fiscal_year) in col_str and '4' in col_str):
                    q4_col = col
                    break

        if q4_col is None:
            return None

        # Build markdown output
        statement_titles = {
            'income_statement': 'Income Statement',
            'balance_sheet': 'Balance Sheet',
            'cash_flow': 'Cash Flow Statement',
        }
        title = statement_titles.get(statement_type, statement_type)

        lines = [
            f"# {ticker} Q4 {fiscal_year} {title} (Calculated)",
            "",
            "> **Note:** The SEC does not require a separate 10-Q filing for Q4. These values are",
            "> calculated from the 10-K annual report minus Q1-Q3 data. While the metrics are accurate,",
            "> the table format differs from official 10-Q filings.",
            "",
            "| Concept | Q4 Value |",
            "|---------|----------|",
        ]

        for idx, value in df[q4_col].items():
            if value is not None and str(value) != 'nan':
                # Format value based on concept type
                try:
                    num_val = float(value)
                    idx_str = str(idx).lower()

                    # Per-share metrics (EPS, dividends per share, etc.)
                    if 'pershare' in idx_str or 'earningspershare' in idx_str:
                        formatted = f"${num_val:.2f}"
                    # Share counts
                    elif 'shares' in idx_str or 'sharesoutstanding' in idx_str:
                        if abs(num_val) >= 1e9:
                            formatted = f"{num_val/1e9:.2f}B"
                        elif abs(num_val) >= 1e6:
                            formatted = f"{num_val/1e6:.2f}M"
                        else:
                            formatted = f"{num_val:,.0f}"
                    # Dollar amounts
                    elif abs(num_val) >= 1e9:
                        formatted = f"${num_val/1e9:.2f}B"
                    elif abs(num_val) >= 1e6:
                        formatted = f"${num_val/1e6:.2f}M"
                    elif abs(num_val) >= 1e3:
                        formatted = f"${num_val/1e3:.2f}K"
                    elif abs(num_val) < 100:
                        # Small values - likely per-share or ratio
                        formatted = f"${num_val:.2f}"
                    else:
                        formatted = f"${num_val:,.0f}"
                except (ValueError, TypeError):
                    formatted = str(value)

                lines.append(f"| {idx} | {formatted} |")

        lines.extend([
            "",
            "---",
            f"*Source: Derived from 10-K and 10-Q filings via edgartools | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ])

        return "\n".join(lines)

    except Exception as e:
        print(f"Error generating Q4 statement: {e}")
        return None


def get_statement(ticker: str, statement_type: str, fiscal_year: int,
                  fiscal_quarter: int = None) -> str | None:
    """Retrieve a financial statement markdown.

    For Q4 (fiscal_quarter=4), uses edgartools to calculate values on-demand
    since SEC does not require a separate 10-Q for Q4.
    """
    # Q4 is calculated on-demand using edgartools' built-in quarterization
    if fiscal_quarter == 4:
        return _get_q4_statement_from_edgartools(ticker, fiscal_year, statement_type)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if fiscal_quarter:
                cur.execute("""
                    SELECT markdown_content FROM financial_documents
                    WHERE ticker = %s AND statement_type = %s
                      AND fiscal_year = %s AND fiscal_quarter = %s
                """, (ticker, statement_type, fiscal_year, fiscal_quarter))
            else:
                cur.execute("""
                    SELECT markdown_content FROM financial_documents
                    WHERE ticker = %s AND statement_type = %s
                      AND fiscal_year = %s AND fiscal_quarter IS NULL
                """, (ticker, statement_type, fiscal_year))

            result = cur.fetchone()
            return result[0] if result else None


def show_stats():
    """Display summary of stored documents."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM financial_documents")
            total = cur.fetchone()[0]

            cur.execute("""
                SELECT ticker, form_type, statement_type, COUNT(*)
                FROM financial_documents
                GROUP BY ticker, form_type, statement_type
                ORDER BY ticker, form_type, statement_type
            """)
            breakdown = cur.fetchall()

    print("\n" + "=" * 50)
    print("STORED FINANCIAL DOCUMENTS")
    print("=" * 50)
    print(f"Total Documents: {total}\n")

    if breakdown:
        print(f"{'Ticker':<8} {'Form':<6} {'Statement Type':<20} {'Count':>5}")
        print("-" * 50)
        for ticker, form, stmt, count in breakdown:
            print(f"{ticker:<8} {form:<6} {stmt:<20} {count:>5}")

    print("=" * 50)


if __name__ == "__main__":
    import config

    init_db(reset=True)

    for ticker in config.TICKERS:
        print(f"\n{'#'*60}")
        print(f"# Processing {ticker}")
        print(f"{'#'*60}")
        fetch_and_store(ticker, years=config.YEARS)
        time.sleep(config.SEC_RATE_LIMIT_DELAY)

    show_stats()
