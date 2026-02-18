"""
SEC Filing Section Vector Tables
Creates separate vector database tables for 10-K and 10-Q filing sections.
Designed for chunking and embedding with pgvector.
"""

import os
from contextlib import contextmanager
from datetime import datetime
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
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


# 10-K Section Metadata (Annual Report - More Comprehensive)
SECTIONS_10K = {
    # Part I - Business Overview
    "item_1_business": {
        "title": "Business",
        "part": "I",
        "item": "1",
        "description": "Company overview, products, services, and competitive landscape",
        "section_key": "part_i_item_1"
    },
    "item_1a_risk_factors": {
        "title": "Risk Factors",
        "part": "I",
        "item": "1A",
        "description": "Key risks and uncertainties facing the company",
        "section_key": "part_i_item_1a"
    },
    "item_1b_unresolved_comments": {
        "title": "Unresolved Staff Comments",
        "part": "I",
        "item": "1B",
        "description": "SEC staff comments that remain unresolved",
        "section_key": "part_i_item_1b"
    },
    "item_1c_cybersecurity": {
        "title": "Cybersecurity",
        "part": "I",
        "item": "1C",
        "description": "Cybersecurity risk management and governance",
        "section_key": "part_i_item_1c"
    },
    "item_2_properties": {
        "title": "Properties",
        "part": "I",
        "item": "2",
        "description": "Physical properties and facilities",
        "section_key": "part_i_item_2"
    },
    "item_3_legal_proceedings": {
        "title": "Legal Proceedings",
        "part": "I",
        "item": "3",
        "description": "Pending material legal proceedings",
        "section_key": "part_i_item_3"
    },
    "item_4_mine_safety": {
        "title": "Mine Safety Disclosures",
        "part": "I",
        "item": "4",
        "description": "Mine safety information (if applicable)",
        "section_key": "part_i_item_4"
    },
    # Part II - Financial Information
    "item_5_market_equity": {
        "title": "Market for Common Equity",
        "part": "II",
        "item": "5",
        "description": "Stock market information, dividends, repurchases",
        "section_key": "part_ii_item_5"
    },
    "item_6_selected_financial": {
        "title": "Selected Financial Data",
        "part": "II",
        "item": "6",
        "description": "Historical financial highlights (5-year summary)",
        "section_key": "part_ii_item_6"
    },
    "item_7_mda": {
        "title": "Management Discussion and Analysis",
        "part": "II",
        "item": "7",
        "description": "MD&A - Management's perspective on financial condition and results",
        "section_key": "part_ii_item_7"
    },
    "item_7a_market_risk": {
        "title": "Quantitative Market Risk Disclosures",
        "part": "II",
        "item": "7A",
        "description": "Interest rate, foreign exchange, and other market risks",
        "section_key": "part_ii_item_7a"
    },
    "item_8_financial_statements": {
        "title": "Financial Statements",
        "part": "II",
        "item": "8",
        "description": "Audited financial statements and notes",
        "section_key": "part_ii_item_8"
    },
    "item_9_accountant_disagreements": {
        "title": "Disagreements with Accountants",
        "part": "II",
        "item": "9",
        "description": "Changes in and disagreements with accountants",
        "section_key": "part_ii_item_9"
    },
    "item_9a_controls": {
        "title": "Controls and Procedures",
        "part": "II",
        "item": "9A",
        "description": "Disclosure controls and internal control over financial reporting",
        "section_key": "part_ii_item_9a"
    },
    "item_9b_other_info": {
        "title": "Other Information",
        "part": "II",
        "item": "9B",
        "description": "Other material information",
        "section_key": "part_ii_item_9b"
    },
    "item_9c_foreign_jurisdictions": {
        "title": "Foreign Jurisdictions",
        "part": "II",
        "item": "9C",
        "description": "Disclosure regarding foreign jurisdictions that prevent inspections",
        "section_key": "part_ii_item_9c"
    },
    # Part III - Governance
    "item_10_directors_officers": {
        "title": "Directors and Executive Officers",
        "part": "III",
        "item": "10",
        "description": "Corporate governance, board of directors",
        "section_key": "part_iii_item_10"
    },
    "item_11_exec_compensation": {
        "title": "Executive Compensation",
        "part": "III",
        "item": "11",
        "description": "Compensation discussion and analysis",
        "section_key": "part_iii_item_11"
    },
    "item_12_security_ownership": {
        "title": "Security Ownership",
        "part": "III",
        "item": "12",
        "description": "Beneficial ownership of securities",
        "section_key": "part_iii_item_12"
    },
    "item_13_related_transactions": {
        "title": "Related Transactions",
        "part": "III",
        "item": "13",
        "description": "Related party transactions and director independence",
        "section_key": "part_iii_item_13"
    },
    "item_14_accountant_fees": {
        "title": "Principal Accountant Fees",
        "part": "III",
        "item": "14",
        "description": "Audit and non-audit fees",
        "section_key": "part_iii_item_14"
    },
    # Part IV - Exhibits
    "item_15_exhibits": {
        "title": "Exhibits and Financial Statement Schedules",
        "part": "IV",
        "item": "15",
        "description": "List of exhibits and schedules",
        "section_key": "part_iv_item_15"
    },
    "item_16_summary": {
        "title": "Form 10-K Summary",
        "part": "IV",
        "item": "16",
        "description": "Optional summary of the annual report",
        "section_key": "part_iv_item_16"
    },
}

# 10-Q Section Metadata (Quarterly Report - More Concise)
SECTIONS_10Q = {
    # Part I - Financial Information
    "item_1_financial_statements": {
        "title": "Financial Statements",
        "part": "I",
        "item": "1",
        "description": "Unaudited condensed financial statements",
        "section_key": "part_i_item_1"
    },
    "item_2_mda": {
        "title": "Management Discussion and Analysis",
        "part": "I",
        "item": "2",
        "description": "MD&A - Quarterly financial condition and results",
        "section_key": "part_i_item_2"
    },
    "item_3_market_risk": {
        "title": "Quantitative Market Risk Disclosures",
        "part": "I",
        "item": "3",
        "description": "Changes in market risk from prior quarter",
        "section_key": "part_i_item_3"
    },
    "item_4_controls": {
        "title": "Controls and Procedures",
        "part": "I",
        "item": "4",
        "description": "Quarterly evaluation of disclosure controls",
        "section_key": "part_i_item_4"
    },
    # Part II - Other Information
    "item_1_legal_proceedings": {
        "title": "Legal Proceedings",
        "part": "II",
        "item": "1",
        "description": "Updates to pending legal matters",
        "section_key": "part_ii_item_1"
    },
    "item_1a_risk_factors": {
        "title": "Risk Factors",
        "part": "II",
        "item": "1A",
        "description": "Updates to risk factors from 10-K",
        "section_key": "part_ii_item_1a"
    },
    "item_2_unregistered_sales": {
        "title": "Unregistered Sales of Equity",
        "part": "II",
        "item": "2",
        "description": "Equity sales and repurchase activity",
        "section_key": "part_ii_item_2"
    },
    "item_3_defaults": {
        "title": "Defaults on Senior Securities",
        "part": "II",
        "item": "3",
        "description": "Material defaults (if any)",
        "section_key": "part_ii_item_3"
    },
    "item_4_mine_safety": {
        "title": "Mine Safety Disclosures",
        "part": "II",
        "item": "4",
        "description": "Mine safety information (if applicable)",
        "section_key": "part_ii_item_4"
    },
    "item_5_other_info": {
        "title": "Other Information",
        "part": "II",
        "item": "5",
        "description": "Other material information",
        "section_key": "part_ii_item_5"
    },
    "item_6_exhibits": {
        "title": "Exhibits",
        "part": "II",
        "item": "6",
        "description": "Quarterly filing exhibits",
        "section_key": "part_ii_item_6"
    },
}


# Schema for 10-K sections (Annual - Comprehensive)
SCHEMA_10K = """
-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for 10-K filing sections (Annual Reports)
CREATE TABLE IF NOT EXISTS sections_10k (
    id SERIAL PRIMARY KEY,

    -- Filing identification
    ticker VARCHAR(10) NOT NULL,
    cik VARCHAR(20),
    company_name VARCHAR(255),
    accession_number VARCHAR(30),
    filing_date DATE,
    fiscal_year INT NOT NULL,
    fiscal_period VARCHAR(10) DEFAULT 'FY',

    -- Section identification
    section_id VARCHAR(50) NOT NULL,
    section_title VARCHAR(100) NOT NULL,
    part_number VARCHAR(5),
    item_number VARCHAR(5),
    section_description TEXT,

    -- Content
    section_text TEXT,
    char_count INT,
    word_count INT,

    -- Vector embedding (for semantic search)
    -- Will be populated during chunking/embedding phase
    embedding vector(1536),

    -- Chunking metadata (for RAG)
    is_chunked BOOLEAN DEFAULT FALSE,
    parent_section_id INT REFERENCES sections_10k(id),
    chunk_index INT,
    chunk_start_char INT,
    chunk_end_char INT,

    -- Subsection metadata (for section-aware chunking)
    subsection_heading TEXT,
    subsection_index INT,
    subsection_start_char INT,
    subsection_end_char INT,

    -- Audit
    source_file_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    UNIQUE(ticker, fiscal_year, section_id, chunk_index)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_10k_ticker ON sections_10k(ticker);
CREATE INDEX IF NOT EXISTS idx_10k_fiscal_year ON sections_10k(fiscal_year);
CREATE INDEX IF NOT EXISTS idx_10k_section ON sections_10k(section_id);
CREATE INDEX IF NOT EXISTS idx_10k_ticker_year ON sections_10k(ticker, fiscal_year);
CREATE INDEX IF NOT EXISTS idx_10k_part ON sections_10k(part_number);
CREATE INDEX IF NOT EXISTS idx_10k_item ON sections_10k(item_number);
CREATE INDEX IF NOT EXISTS idx_10k_chunked ON sections_10k(is_chunked);
CREATE INDEX IF NOT EXISTS idx_10k_parent ON sections_10k(parent_section_id);

-- Vector similarity search index (IVFFlat for approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_10k_embedding ON sections_10k
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_10k_text_search ON sections_10k
USING gin(to_tsvector('english', section_text));
"""


# Schema for 10-Q sections (Quarterly - Concise)
SCHEMA_10Q = """
-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for 10-Q filing sections (Quarterly Reports)
CREATE TABLE IF NOT EXISTS sections_10q (
    id SERIAL PRIMARY KEY,

    -- Filing identification
    ticker VARCHAR(10) NOT NULL,
    cik VARCHAR(20),
    company_name VARCHAR(255),
    accession_number VARCHAR(30),
    filing_date DATE,
    fiscal_year INT NOT NULL,
    fiscal_quarter INT NOT NULL,  -- 1, 2, 3 (Q4 is covered by 10-K)
    fiscal_period VARCHAR(10),    -- Q1, Q2, Q3

    -- Section identification
    section_id VARCHAR(50) NOT NULL,
    section_title VARCHAR(100) NOT NULL,
    part_number VARCHAR(5),
    item_number VARCHAR(5),
    section_description TEXT,

    -- Content
    section_text TEXT,
    char_count INT,
    word_count INT,

    -- Vector embedding (for semantic search)
    -- Will be populated during chunking/embedding phase
    embedding vector(1536),

    -- Chunking metadata (for RAG)
    is_chunked BOOLEAN DEFAULT FALSE,
    parent_section_id INT REFERENCES sections_10q(id),
    chunk_index INT,
    chunk_start_char INT,
    chunk_end_char INT,

    -- Subsection metadata (for section-aware chunking)
    subsection_heading TEXT,
    subsection_index INT,
    subsection_start_char INT,
    subsection_end_char INT,

    -- Audit
    source_file_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    UNIQUE(ticker, fiscal_year, fiscal_quarter, section_id, chunk_index)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_10q_ticker ON sections_10q(ticker);
CREATE INDEX IF NOT EXISTS idx_10q_fiscal_year ON sections_10q(fiscal_year);
CREATE INDEX IF NOT EXISTS idx_10q_fiscal_quarter ON sections_10q(fiscal_quarter);
CREATE INDEX IF NOT EXISTS idx_10q_section ON sections_10q(section_id);
CREATE INDEX IF NOT EXISTS idx_10q_ticker_year_qtr ON sections_10q(ticker, fiscal_year, fiscal_quarter);
CREATE INDEX IF NOT EXISTS idx_10q_part ON sections_10q(part_number);
CREATE INDEX IF NOT EXISTS idx_10q_item ON sections_10q(item_number);
CREATE INDEX IF NOT EXISTS idx_10q_chunked ON sections_10q(is_chunked);
CREATE INDEX IF NOT EXISTS idx_10q_parent ON sections_10q(parent_section_id);

-- Vector similarity search index (IVFFlat for approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_10q_embedding ON sections_10q
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_10q_text_search ON sections_10q
USING gin(to_tsvector('english', section_text));
"""


def migrate_add_subsection_columns():
    """Add subsection columns to existing tables (idempotent)."""
    columns = [
        ("subsection_heading", "TEXT"),
        ("subsection_index", "INT"),
        ("subsection_start_char", "INT"),
        ("subsection_end_char", "INT"),
    ]
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for table in ("sections_10k", "sections_10q"):
                for col_name, col_type in columns:
                    cur.execute(f"""
                        ALTER TABLE {table}
                        ADD COLUMN IF NOT EXISTS {col_name} {col_type}
                    """)
        conn.commit()
    print("Subsection columns migration complete.")


def init_vector_tables(reset=False):
    """Create vector tables for 10-K and 10-Q sections."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if reset:
                print("Dropping existing tables...")
                cur.execute("DROP TABLE IF EXISTS sections_10k CASCADE;")
                cur.execute("DROP TABLE IF EXISTS sections_10q CASCADE;")

            print("Creating sections_10k table...")
            cur.execute(SCHEMA_10K)

            print("Creating sections_10q table...")
            cur.execute(SCHEMA_10Q)

        conn.commit()

    # Ensure subsection columns exist on pre-existing tables
    migrate_add_subsection_columns()

    print("\nVector tables initialized successfully!")
    print(f"  - sections_10k: {len(SECTIONS_10K)} section types (Annual Report)")
    print(f"  - sections_10q: {len(SECTIONS_10Q)} section types (Quarterly Report)")


def get_section_metadata_10k(section_key: str) -> dict | None:
    """Get metadata for a 10-K section by its key."""
    for sec_id, meta in SECTIONS_10K.items():
        if meta.get("section_key") == section_key:
            return {"section_id": sec_id, **meta}
    return None


def get_section_metadata_10q(section_key: str) -> dict | None:
    """Get metadata for a 10-Q section by its key."""
    for sec_id, meta in SECTIONS_10Q.items():
        if meta.get("section_key") == section_key:
            return {"section_id": sec_id, **meta}
    return None


def save_section_10k(
    ticker: str,
    fiscal_year: int,
    section_id: str,
    section_text: str,
    cik: str = None,
    company_name: str = None,
    accession_number: str = None,
    filing_date: str = None,
    source_file_path: str = None
):
    """Save a 10-K section to the database."""
    meta = SECTIONS_10K.get(section_id, {})

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sections_10k
                (ticker, cik, company_name, accession_number, filing_date,
                 fiscal_year, section_id, section_title, part_number, item_number,
                 section_description, section_text, char_count, word_count, source_file_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, fiscal_year, section_id, chunk_index)
                DO UPDATE SET
                    section_text = EXCLUDED.section_text,
                    char_count = EXCLUDED.char_count,
                    word_count = EXCLUDED.word_count,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (
                ticker, cik, company_name, accession_number, filing_date,
                fiscal_year, section_id, meta.get("title", section_id),
                meta.get("part"), meta.get("item"), meta.get("description"),
                section_text, len(section_text), len(section_text.split()),
                source_file_path
            ))
            result = cur.fetchone()
        conn.commit()

    return result[0] if result else None


def save_section_10q(
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: int,
    section_id: str,
    section_text: str,
    cik: str = None,
    company_name: str = None,
    accession_number: str = None,
    filing_date: str = None,
    source_file_path: str = None
):
    """Save a 10-Q section to the database."""
    meta = SECTIONS_10Q.get(section_id, {})
    fiscal_period = f"Q{fiscal_quarter}"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sections_10q
                (ticker, cik, company_name, accession_number, filing_date,
                 fiscal_year, fiscal_quarter, fiscal_period, section_id, section_title,
                 part_number, item_number, section_description, section_text,
                 char_count, word_count, source_file_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, fiscal_year, fiscal_quarter, section_id, chunk_index)
                DO UPDATE SET
                    section_text = EXCLUDED.section_text,
                    char_count = EXCLUDED.char_count,
                    word_count = EXCLUDED.word_count,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (
                ticker, cik, company_name, accession_number, filing_date,
                fiscal_year, fiscal_quarter, fiscal_period, section_id,
                meta.get("title", section_id), meta.get("part"), meta.get("item"),
                meta.get("description"), section_text, len(section_text),
                len(section_text.split()), source_file_path
            ))
            result = cur.fetchone()
        conn.commit()

    return result[0] if result else None


def get_section_10k(ticker: str, fiscal_year: int, section_id: str) -> dict | None:
    """Retrieve a specific 10-K section."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, fiscal_year, section_id, section_title,
                       section_text, char_count, word_count, created_at
                FROM sections_10k
                WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
                  AND is_chunked = FALSE
            """, (ticker, fiscal_year, section_id))

            result = cur.fetchone()
            if result:
                return {
                    "id": result[0], "ticker": result[1], "fiscal_year": result[2],
                    "section_id": result[3], "section_title": result[4],
                    "section_text": result[5], "char_count": result[6],
                    "word_count": result[7], "created_at": result[8]
                }
    return None


def get_section_10q(ticker: str, fiscal_year: int, fiscal_quarter: int,
                    section_id: str) -> dict | None:
    """Retrieve a specific 10-Q section."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, fiscal_year, fiscal_quarter, section_id,
                       section_title, section_text, char_count, word_count, created_at
                FROM sections_10q
                WHERE ticker = %s AND fiscal_year = %s AND fiscal_quarter = %s
                  AND section_id = %s AND is_chunked = FALSE
            """, (ticker, fiscal_year, fiscal_quarter, section_id))

            result = cur.fetchone()
            if result:
                return {
                    "id": result[0], "ticker": result[1], "fiscal_year": result[2],
                    "fiscal_quarter": result[3], "section_id": result[4],
                    "section_title": result[5], "section_text": result[6],
                    "char_count": result[7], "word_count": result[8],
                    "created_at": result[9]
                }
    return None


def list_sections_10k(ticker: str = None, fiscal_year: int = None) -> list:
    """List all 10-K sections with optional filters."""
    conditions = ["is_chunked = FALSE"]
    params = []

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker)
    if fiscal_year:
        conditions.append("fiscal_year = %s")
        params.append(fiscal_year)

    where = " AND ".join(conditions)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT ticker, fiscal_year, section_id, section_title,
                       char_count, word_count
                FROM sections_10k
                WHERE {where}
                ORDER BY ticker, fiscal_year, part_number, item_number
            """, params)

            return [
                {"ticker": r[0], "fiscal_year": r[1], "section_id": r[2],
                 "section_title": r[3], "char_count": r[4], "word_count": r[5]}
                for r in cur.fetchall()
            ]


def list_sections_10q(ticker: str = None, fiscal_year: int = None,
                      fiscal_quarter: int = None) -> list:
    """List all 10-Q sections with optional filters."""
    conditions = ["is_chunked = FALSE"]
    params = []

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker)
    if fiscal_year:
        conditions.append("fiscal_year = %s")
        params.append(fiscal_year)
    if fiscal_quarter:
        conditions.append("fiscal_quarter = %s")
        params.append(fiscal_quarter)

    where = " AND ".join(conditions)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT ticker, fiscal_year, fiscal_quarter, section_id,
                       section_title, char_count, word_count
                FROM sections_10q
                WHERE {where}
                ORDER BY ticker, fiscal_year, fiscal_quarter, part_number, item_number
            """, params)

            return [
                {"ticker": r[0], "fiscal_year": r[1], "fiscal_quarter": r[2],
                 "section_id": r[3], "section_title": r[4],
                 "char_count": r[5], "word_count": r[6]}
                for r in cur.fetchall()
            ]


def show_table_stats():
    """Display statistics for both section tables."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 10-K stats
            cur.execute("SELECT COUNT(*) FROM sections_10k WHERE is_chunked = FALSE")
            count_10k = cur.fetchone()[0]

            cur.execute("""
                SELECT ticker, fiscal_year, COUNT(*) as sections,
                       SUM(char_count) as total_chars
                FROM sections_10k WHERE is_chunked = FALSE
                GROUP BY ticker, fiscal_year
                ORDER BY ticker, fiscal_year
            """)
            breakdown_10k = cur.fetchall()

            # 10-Q stats
            cur.execute("SELECT COUNT(*) FROM sections_10q WHERE is_chunked = FALSE")
            count_10q = cur.fetchone()[0]

            cur.execute("""
                SELECT ticker, fiscal_year, fiscal_quarter, COUNT(*) as sections,
                       SUM(char_count) as total_chars
                FROM sections_10q WHERE is_chunked = FALSE
                GROUP BY ticker, fiscal_year, fiscal_quarter
                ORDER BY ticker, fiscal_year, fiscal_quarter
            """)
            breakdown_10q = cur.fetchall()

    print("\n" + "=" * 70)
    print("SECTION VECTOR TABLES SUMMARY")
    print("=" * 70)

    print(f"\n📊 SECTIONS_10K (Annual Reports)")
    print(f"   Total Sections: {count_10k}")
    print(f"   Available Section Types: {len(SECTIONS_10K)}")
    if breakdown_10k:
        print(f"\n   {'Ticker':<8} {'Year':<6} {'Sections':>10} {'Total Chars':>15}")
        print(f"   {'-'*8} {'-'*6} {'-'*10} {'-'*15}")
        for row in breakdown_10k:
            print(f"   {row[0]:<8} {row[1]:<6} {row[2]:>10} {row[3]:>15,}")

    print(f"\n📊 SECTIONS_10Q (Quarterly Reports)")
    print(f"   Total Sections: {count_10q}")
    print(f"   Available Section Types: {len(SECTIONS_10Q)}")
    if breakdown_10q:
        print(f"\n   {'Ticker':<8} {'Year':<6} {'Qtr':<4} {'Sections':>10} {'Total Chars':>15}")
        print(f"   {'-'*8} {'-'*6} {'-'*4} {'-'*10} {'-'*15}")
        for row in breakdown_10q:
            print(f"   {row[0]:<8} {row[1]:<6} Q{row[2]:<3} {row[3]:>10} {row[4]:>15,}")

    print("\n" + "=" * 70)
    print("Section Types Available:")
    print("-" * 70)
    print("\n10-K Sections:")
    for sec_id, meta in SECTIONS_10K.items():
        print(f"  • {sec_id}: {meta['title']} (Part {meta['part']}, Item {meta['item']})")

    print("\n10-Q Sections:")
    for sec_id, meta in SECTIONS_10Q.items():
        print(f"  • {sec_id}: {meta['title']} (Part {meta['part']}, Item {meta['item']})")

    print("=" * 70)


# Mapping from filing_sections.section_key to our section_id
SECTION_KEY_TO_10K_ID = {
    # Standard part_X_item_Y format
    "part_i_item_1": "item_1_business",
    "part_i_item_1a": "item_1a_risk_factors",
    "part_i_item_1b": "item_1b_unresolved_comments",
    "part_i_item_1c": "item_1c_cybersecurity",
    "part_i_item_2": "item_2_properties",
    "part_i_item_3": "item_3_legal_proceedings",
    "part_i_item_4": "item_4_mine_safety",
    "part_ii_item_5": "item_5_market_equity",
    "part_ii_item_6": "item_6_selected_financial",
    "part_ii_item_7": "item_7_mda",
    "part_ii_item_7a": "item_7a_market_risk",
    "part_ii_item_8": "item_8_financial_statements",
    "part_ii_item_9": "item_9_accountant_disagreements",
    "part_ii_item_9a": "item_9a_controls",
    "part_ii_item_9b": "item_9b_other_info",
    "part_ii_item_9c": "item_9c_foreign_jurisdictions",
    "part_iii_item_10": "item_10_directors_officers",
    "part_iii_item_11": "item_11_exec_compensation",
    "part_iii_item_12": "item_12_security_ownership",
    "part_iii_item_13": "item_13_related_transactions",
    "part_iii_item_14": "item_14_accountant_fees",
    "part_iv_item_15": "item_15_exhibits",
    "part_iv_item_16": "item_16_summary",
    "part_iv_item_8": "item_8_financial_statements",  # Alternate location
    # "Item X" format (edgartools section names)
    "Item 1": "item_1_business",
    "Item 1A": "item_1a_risk_factors",
    "Item 1B": "item_1b_unresolved_comments",
    "Item 1C": "item_1c_cybersecurity",
    "Item 2": "item_2_properties",
    "Item 3": "item_3_legal_proceedings",
    "Item 4": "item_4_mine_safety",
    "Item 5": "item_5_market_equity",
    "Item 6": "item_6_selected_financial",
    "Item 7": "item_7_mda",
    "Item 7A": "item_7a_market_risk",
    "Item 8": "item_8_financial_statements",
    "Item 9": "item_9_accountant_disagreements",
    "Item 9A": "item_9a_controls",
    "Item 9B": "item_9b_other_info",
    "Item 9C": "item_9c_foreign_jurisdictions",
    "Item 10": "item_10_directors_officers",
    "Item 11": "item_11_exec_compensation",
    "Item 12": "item_12_security_ownership",
    "Item 13": "item_13_related_transactions",
    "Item 14": "item_14_accountant_fees",
    "Item 15": "item_15_exhibits",
    "Item 16": "item_16_summary",
    # Short name format (legacy section extraction)
    "business": "item_1_business",
    "risk_factors": "item_1a_risk_factors",
    "properties": "item_2_properties",
    "legal_proceedings": "item_3_legal_proceedings",
    "mda": "item_7_mda",
    "market_risk": "item_7a_market_risk",
    "financial_statements": "item_8_financial_statements",
    "controls_procedures": "item_9a_controls",
}

SECTION_KEY_TO_10Q_ID = {
    # Standard part_X_item_Y format
    "part_i_item_1": "item_1_financial_statements",
    "part_i_item_2": "item_2_mda",
    "part_i_item_3": "item_3_market_risk",
    "part_i_item_4": "item_4_controls",
    "part_ii_item_1": "item_1_legal_proceedings",
    "part_ii_item_1a": "item_1a_risk_factors",
    "part_ii_item_2": "item_2_unregistered_sales",
    "part_ii_item_3": "item_3_defaults",
    "part_ii_item_4": "item_4_mine_safety",
    "part_ii_item_5": "item_5_other_info",
    "part_ii_item_6": "item_6_exhibits",
    # "Item X" format (edgartools section names, 10-Q Part I)
    "Item 1": "item_1_financial_statements",
    "Item 2": "item_2_mda",
    "Item 3": "item_3_market_risk",
    "Item 4": "item_4_controls",
    # "Item X" format (edgartools section names, 10-Q Part II)
    "Item 5": "item_5_other_info",
    "Item 8": "item_6_exhibits",
}


def migrate_from_filing_sections():
    """
    Migrate data from the existing filing_sections table to the new
    sections_10k and sections_10q vector tables.
    """
    print("\nMigrating data from filing_sections table...")
    print("-" * 50)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get all 10-K sections
            cur.execute("""
                SELECT fs.ticker, fs.form_type, fs.fiscal_year, fs.fiscal_period,
                       fs.section_key, fs.section_title, fs.section_text,
                       fs.char_count, fs.word_count, fs.file_path,
                       f.accession_number, f.filing_date, f.company_name
                FROM filing_sections fs
                LEFT JOIN filings f ON fs.filing_id = f.id
                WHERE fs.form_type = '10-K'
                ORDER BY fs.fiscal_year, fs.section_key
            """)
            rows_10k = cur.fetchall()

            # Get all 10-Q sections
            cur.execute("""
                SELECT fs.ticker, fs.form_type, fs.fiscal_year, fs.fiscal_period,
                       fs.section_key, fs.section_title, fs.section_text,
                       fs.char_count, fs.word_count, fs.file_path,
                       f.accession_number, f.filing_date, f.company_name
                FROM filing_sections fs
                LEFT JOIN filings f ON fs.filing_id = f.id
                WHERE fs.form_type = '10-Q'
                ORDER BY fs.fiscal_year, fs.fiscal_period, fs.section_key
            """)
            rows_10q = cur.fetchall()

    # Process 10-K sections
    count_10k = 0
    print(f"\nProcessing {len(rows_10k)} 10-K sections...")

    for row in rows_10k:
        ticker, form_type, fiscal_year, fiscal_period, section_key, section_title, \
            section_text, char_count, word_count, file_path, accession_number, \
            filing_date, company_name = row

        # Map section_key to our section_id
        section_id = SECTION_KEY_TO_10K_ID.get(section_key)
        if not section_id:
            print(f"  Warning: Unknown 10-K section_key '{section_key}', skipping")
            continue

        if not section_text:
            continue

        meta = SECTIONS_10K.get(section_id, {})

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sections_10k
                    (ticker, company_name, accession_number, filing_date, fiscal_year,
                     section_id, section_title, part_number, item_number,
                     section_description, section_text, char_count, word_count,
                     source_file_path, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, fiscal_year, section_id, chunk_index)
                    DO UPDATE SET
                        section_text = EXCLUDED.section_text,
                        char_count = EXCLUDED.char_count,
                        word_count = EXCLUDED.word_count,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    ticker, company_name, accession_number, filing_date, fiscal_year,
                    section_id, meta.get("title", section_title),
                    meta.get("part"), meta.get("item"), meta.get("description"),
                    section_text, char_count, word_count, file_path, None
                ))
            conn.commit()
        count_10k += 1

    print(f"  Migrated {count_10k} 10-K sections")

    # Process 10-Q sections
    count_10q = 0
    print(f"\nProcessing {len(rows_10q)} 10-Q sections...")

    for row in rows_10q:
        ticker, form_type, fiscal_year, fiscal_period, section_key, section_title, \
            section_text, char_count, word_count, file_path, accession_number, \
            filing_date, company_name = row

        # Map section_key to our section_id
        section_id = SECTION_KEY_TO_10Q_ID.get(section_key)
        if not section_id:
            print(f"  Warning: Unknown 10-Q section_key '{section_key}', skipping")
            continue

        if not section_text:
            continue

        # Extract quarter from fiscal_period (e.g., "Q1" -> 1)
        fiscal_quarter = int(fiscal_period[1]) if fiscal_period else None
        if not fiscal_quarter:
            print(f"  Warning: Invalid fiscal_period '{fiscal_period}', skipping")
            continue

        meta = SECTIONS_10Q.get(section_id, {})

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sections_10q
                    (ticker, company_name, accession_number, filing_date, fiscal_year,
                     fiscal_quarter, fiscal_period, section_id, section_title,
                     part_number, item_number, section_description, section_text,
                     char_count, word_count, source_file_path, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, fiscal_year, fiscal_quarter, section_id, chunk_index)
                    DO UPDATE SET
                        section_text = EXCLUDED.section_text,
                        char_count = EXCLUDED.char_count,
                        word_count = EXCLUDED.word_count,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    ticker, company_name, accession_number, filing_date, fiscal_year,
                    fiscal_quarter, fiscal_period, section_id,
                    meta.get("title", section_title), meta.get("part"),
                    meta.get("item"), meta.get("description"), section_text,
                    char_count, word_count, file_path, None
                ))
            conn.commit()
        count_10q += 1

    print(f"  Migrated {count_10q} 10-Q sections")
    print(f"\nMigration complete! Total: {count_10k + count_10q} sections")

    return count_10k, count_10q


if __name__ == "__main__":
    print("Initializing SEC Filing Section Vector Tables")
    print("-" * 50)

    init_vector_tables(reset=True)
    migrate_from_filing_sections()  # Migrates all tickers present in filing_sections
    show_table_stats()
