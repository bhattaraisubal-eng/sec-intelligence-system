"""
Store Markdown Financial Tables in PostgreSQL
Simple document storage for full markdown retrieval.
"""

import os
from pathlib import Path
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

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


def parse_file_info(filepath: str) -> dict:
    """Extract metadata from file path."""
    path = Path(filepath)
    filename = path.stem  # e.g., "2024" or "2024_Q3"
    statement_type = path.parent.name  # e.g., "income_statement"
    form_type = path.parent.parent.name  # e.g., "10-K" or "10-Q"

    if '_Q' in filename:
        year, quarter = filename.split('_Q')
        fiscal_year = int(year)
        fiscal_quarter = int(quarter)
    else:
        fiscal_year = int(filename)
        fiscal_quarter = None

    return {
        'ticker': 'AAPL',
        'form_type': form_type,
        'statement_type': statement_type,
        'fiscal_year': fiscal_year,
        'fiscal_quarter': fiscal_quarter,
    }


def save_document(info: dict, content: str):
    """Upsert a financial document."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO financial_documents
                (ticker, form_type, statement_type, fiscal_year, fiscal_quarter, markdown_content)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, form_type, statement_type, fiscal_year, fiscal_quarter)
                DO UPDATE SET markdown_content = EXCLUDED.markdown_content,
                              created_at = CURRENT_TIMESTAMP
            """, (info['ticker'], info['form_type'], info['statement_type'],
                  info['fiscal_year'], info['fiscal_quarter'], content))
        conn.commit()


def process_all_files(base_dir: str = "apple_financials"):
    """Process all markdown files and store in database."""
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = Path(os.path.dirname(__file__)) / base_dir

    md_files = list(base_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files\n")

    for filepath in sorted(md_files):
        rel_path = filepath.relative_to(base_path)
        info = parse_file_info(str(filepath))

        with open(filepath, 'r') as f:
            content = f.read()

        save_document(info, content)

        q_str = f"Q{info['fiscal_quarter']}" if info['fiscal_quarter'] else "Annual"
        print(f"  Saved: {info['ticker']} {info['statement_type']} {info['fiscal_year']} {q_str}")

    print(f"\nDone! Stored {len(md_files)} documents.")


def get_statement(ticker: str, statement_type: str, fiscal_year: int,
                  fiscal_quarter: int = None) -> str | None:
    """Retrieve a financial statement markdown."""
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
    init_db(reset=True)
    process_all_files()
    show_stats()

    # Example: Retrieve a statement
    print("\n--- Example Query ---")
    md = get_statement("AAPL", "income_statement", 2024, 3)
    if md:
        print("Retrieved AAPL Income Statement Q3 2024:")
        print(md[:500] + "..." if len(md) > 500 else md)
