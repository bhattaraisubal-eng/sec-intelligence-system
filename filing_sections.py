"""
SEC Filing Section Extractor
Extracts 10-K and 10-Q section texts, saves to folder structure and PostgreSQL.
"""

import os
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from edgar import Company, set_identity
from config import infer_fiscal_year, infer_fiscal_quarter, YEARS

load_dotenv()

# Section metadata for better naming
SECTION_METADATA = {
    # 10-K sections
    "part_i_item_1": {"title": "Business", "description": "Company overview, products, services"},
    "part_i_item_1a": {"title": "Risk Factors", "description": "Key risks and uncertainties"},
    "part_i_item_1b": {"title": "Unresolved Staff Comments", "description": "SEC staff comments"},
    "part_i_item_1c": {"title": "Cybersecurity", "description": "Cybersecurity risk management"},
    "part_i_item_2": {"title": "Properties", "description": "Physical properties and facilities"},
    "part_i_item_3": {"title": "Legal Proceedings", "description": "Pending legal matters"},
    "part_i_item_4": {"title": "Mine Safety Disclosures", "description": "Mine safety information"},
    "part_ii_item_5": {"title": "Market for Common Equity", "description": "Stock market information"},
    "part_ii_item_6": {"title": "Selected Financial Data", "description": "Historical financial highlights"},
    "part_ii_item_7": {"title": "MD&A", "description": "Management Discussion and Analysis"},
    "part_ii_item_7a": {"title": "Market Risk Disclosures", "description": "Quantitative market risk"},
    "part_ii_item_8": {"title": "Financial Statements", "description": "Audited financial statements"},
    "part_ii_item_9": {"title": "Disagreements with Accountants", "description": "Accountant changes"},
    "part_ii_item_9a": {"title": "Controls and Procedures", "description": "Internal controls"},
    "part_ii_item_9b": {"title": "Other Information", "description": "Other material information"},
    "part_ii_item_9c": {"title": "Foreign Jurisdictions", "description": "Foreign inspection prevention"},
    "part_iii_item_10": {"title": "Directors and Officers", "description": "Corporate governance"},
    "part_iii_item_11": {"title": "Executive Compensation", "description": "Compensation details"},
    "part_iii_item_12": {"title": "Security Ownership", "description": "Beneficial ownership"},
    "part_iii_item_13": {"title": "Related Transactions", "description": "Related party transactions"},
    "part_iv_item_15": {"title": "Exhibits and Schedules", "description": "Filing exhibits"},
    "part_iv_item_16": {"title": "Form 10-K Summary", "description": "Optional summary"},
    # 10-Q sections
    "part_i_item_2": {"title": "MD&A", "description": "Management Discussion and Analysis"},
    "part_ii_item_1": {"title": "Legal Proceedings", "description": "Pending legal matters"},
    "part_ii_item_1a": {"title": "Risk Factors", "description": "Updated risk factors"},
    "part_ii_item_2": {"title": "Unregistered Sales", "description": "Equity sales and repurchases"},
    "part_ii_item_3": {"title": "Defaults", "description": "Senior securities defaults"},
    "part_ii_item_4": {"title": "Mine Safety", "description": "Mine safety disclosures"},
    "part_ii_item_5": {"title": "Other Information", "description": "Other material information"},
    "part_ii_item_6": {"title": "Exhibits", "description": "Filing exhibits"},
}

# Database schema for sections
SECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS filing_sections (
    id SERIAL PRIMARY KEY,
    filing_id INT REFERENCES filings(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    form_type VARCHAR(20) NOT NULL,
    fiscal_year INT NOT NULL,
    fiscal_period VARCHAR(10),
    section_key VARCHAR(50) NOT NULL,
    section_title VARCHAR(100),
    section_text TEXT,
    char_count INT,
    word_count INT,
    file_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(filing_id, section_key)
);

CREATE INDEX IF NOT EXISTS idx_sections_ticker ON filing_sections(ticker);
CREATE INDEX IF NOT EXISTS idx_sections_form ON filing_sections(form_type);
CREATE INDEX IF NOT EXISTS idx_sections_key ON filing_sections(section_key);
CREATE INDEX IF NOT EXISTS idx_sections_ticker_year ON filing_sections(ticker, fiscal_year);

-- Full-text search index for RAG
CREATE INDEX IF NOT EXISTS idx_sections_text_search
ON filing_sections USING gin(to_tsvector('english', section_text));
"""


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        database=os.getenv("PG_DATABASE")
    )


def init_sections_table():
    """Create sections table if it doesn't exist."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SECTIONS_SCHEMA)
        conn.commit()
    print("Sections table initialized.")


def get_filing_id(accession_number: str) -> int | None:
    """Get filing_id from existing filings table."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM filings WHERE accession_number = %s",
                (accession_number,)
            )
            result = cur.fetchone()
            return result[0] if result else None


def extract_sections(filing) -> dict[str, str]:
    """Extract all section texts from a filing."""
    sections = {}
    try:
        obj = filing.obj()
        if obj and hasattr(obj, 'sections'):
            for key in obj.sections.keys():
                section = obj.sections.get(key)
                if section:
                    try:
                        text = section.text()
                        if text and len(text.strip()) > 0:
                            sections[key] = text
                    except Exception as e:
                        print(f"  Warning: Could not extract {key}: {e}")
    except Exception as e:
        print(f"  Error extracting sections: {e}")
    return sections


def save_sections_to_db(filing_id: int, ticker: str, form_type: str,
                        fiscal_year: int, fiscal_period: str,
                        sections: dict[str, str]):
    """Bulk save sections to database."""
    if not sections:
        return 0

    values = []
    for key, text in sections.items():
        meta = SECTION_METADATA.get(key, {})
        values.append((
            filing_id, ticker, form_type, fiscal_year, fiscal_period,
            key, meta.get("title", key), text,
            len(text), len(text.split()),
            None
        ))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO filing_sections
                (filing_id, ticker, form_type, fiscal_year, fiscal_period,
                 section_key, section_title, section_text, char_count, word_count, file_path)
                VALUES %s
                ON CONFLICT (filing_id, section_key) DO UPDATE SET
                    section_text = EXCLUDED.section_text,
                    char_count = EXCLUDED.char_count,
                    word_count = EXCLUDED.word_count,
                    file_path = EXCLUDED.file_path
            """, values, page_size=100)
        conn.commit()

    return len(values)


def process_filing(ticker: str, filing):
    """Process a single filing: extract sections and save to DB."""
    form_type = filing.form
    fiscal_year = infer_fiscal_year(ticker, filing)
    fiscal_period = None

    if form_type == "10-Q":
        quarter = infer_fiscal_quarter(ticker, filing)
        fiscal_period = f"Q{quarter}"

    print(f"\nProcessing {form_type} {fiscal_year} {fiscal_period or ''} ({filing.accession_number})")

    # Get filing_id from database
    filing_id = get_filing_id(filing.accession_number)
    if not filing_id:
        print(f"  Warning: Filing not found in database, skipping DB save")

    # Extract sections
    sections = extract_sections(filing)
    print(f"  Extracted {len(sections)} sections")

    if not sections:
        return

    # Save to database
    if filing_id:
        saved = save_sections_to_db(filing_id, ticker, form_type, fiscal_year, fiscal_period, sections)
        print(f"  Saved {saved} sections to database")


def fetch_and_store_sections(ticker: str, forms: list[str] = ["10-K", "10-Q"],
                             years: range = YEARS):
    """Main entry point: fetch filings and store sections."""
    set_identity("Subal Bhattarai (bhattaraisubal@gmail.com)")

    company = Company(ticker)
    print(f"Company: {company.name}")

    for form in forms:
        print(f"\n{'='*60}\nProcessing {form} filings...\n{'='*60}")

        for filing in company.get_filings(form=form, amendments=False):
            fy = infer_fiscal_year(ticker, filing)
            if fy in years:
                process_filing(ticker, filing)
                time.sleep(0.15)


def show_sections_stats():
    """Display summary of stored sections."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM filing_sections")
            total = cur.fetchone()[0]

            cur.execute("""
                SELECT ticker, form_type, fiscal_year, COUNT(*) as sections,
                       SUM(char_count) as total_chars, SUM(word_count) as total_words
                FROM filing_sections
                GROUP BY ticker, form_type, fiscal_year
                ORDER BY ticker, form_type, fiscal_year
            """)
            breakdown = cur.fetchall()

            cur.execute("""
                SELECT section_key, COUNT(*) as cnt, AVG(char_count)::int as avg_chars
                FROM filing_sections
                GROUP BY section_key
                ORDER BY cnt DESC
                LIMIT 10
            """)
            top_sections = cur.fetchall()

    print("\n" + "="*70)
    print("FILING SECTIONS SUMMARY")
    print("="*70)
    print(f"Total Sections: {total}")

    if breakdown:
        print("\nBreakdown by Filing:")
        print(f"  {'Ticker':<8} {'Form':<8} {'Year':<6} {'Sections':>10} {'Chars':>12} {'Words':>10}")
        print(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*12} {'-'*10}")
        for row in breakdown:
            print(f"  {row[0]:<8} {row[1]:<8} {row[2]:<6} {row[3]:>10} {row[4]:>12,} {row[5]:>10,}")

    if top_sections:
        print("\nTop 10 Section Types:")
        for key, cnt, avg_chars in top_sections:
            meta = SECTION_METADATA.get(key, {})
            title = meta.get("title", key)[:30]
            print(f"  {cnt:>4}x  {key:<25} ({title}) ~{avg_chars:,} chars")

    print("="*70)


# Query helpers for RAG
def search_sections(query: str, ticker: str = None, form_type: str = None, limit: int = 10):
    """Full-text search across section texts."""
    conditions = ["to_tsvector('english', section_text) @@ plainto_tsquery('english', %s)"]
    params = [query]

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker)
    if form_type:
        conditions.append("form_type = %s")
        params.append(form_type)

    where = " AND ".join(conditions)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT ticker, form_type, fiscal_year, fiscal_period,
                       section_key, section_title,
                       ts_rank(to_tsvector('english', section_text),
                               plainto_tsquery('english', %s)) as rank,
                       LEFT(section_text, 500) as preview
                FROM filing_sections
                WHERE {where}
                ORDER BY rank DESC
                LIMIT %s
            """, [query] + params + [limit])

            columns = ['ticker', 'form_type', 'fiscal_year', 'fiscal_period',
                      'section_key', 'section_title', 'rank', 'preview']
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_section(ticker: str, form_type: str, fiscal_year: int,
                section_key: str, fiscal_period: str = None) -> str | None:
    """Get a specific section's full text."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if fiscal_period:
                cur.execute("""
                    SELECT section_text FROM filing_sections
                    WHERE ticker = %s AND form_type = %s AND fiscal_year = %s
                      AND section_key = %s AND fiscal_period = %s
                """, (ticker, form_type, fiscal_year, section_key, fiscal_period))
            else:
                cur.execute("""
                    SELECT section_text FROM filing_sections
                    WHERE ticker = %s AND form_type = %s AND fiscal_year = %s
                      AND section_key = %s
                """, (ticker, form_type, fiscal_year, section_key))

            result = cur.fetchone()
            return result[0] if result else None


if __name__ == "__main__":
    import config

    init_sections_table()

    for ticker in config.TICKERS:
        print(f"\n{'#'*60}")
        print(f"# Processing {ticker}")
        print(f"{'#'*60}")
        fetch_and_store_sections(ticker, forms=["10-K", "10-Q"], years=config.YEARS)
        time.sleep(config.SEC_RATE_LIMIT_DELAY)

    show_sections_stats()
