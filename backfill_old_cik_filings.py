"""
Backfill filings from old/predecessor CIKs and missing individual quarters.

Phase 1: GOOGL FY2010-2014 from Google Inc. (CIK 1288776)
         + GOOGL FY2015 Q1/Q2 (filed under old CIK before Alphabet restructure)
Phase 2: AVGO FY2010-2018 from Avago Technologies (CIK 1441634) and Broadcom Ltd (CIK 1649338)
Phase 3: Missing individual 10-Qs (MSFT FY2012 Q2, NVDA FY2010 Q3)

Usage: .venv/bin/python3 backfill_old_cik_filings.py
"""

import os
import re
import time
import warnings
import logging
from datetime import date

import psycopg2
from dotenv import load_dotenv

load_dotenv(".env")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import edgar
edgar.set_identity("sec_rag_system sec_rag@example.com")

from config import infer_fiscal_year, infer_fiscal_quarter, FISCAL_YEAR_END_MONTH
from chunk_and_embed import chunk_section_aware, generate_embeddings

# --- Config ---
MIN_TEXT_LEN = 1000
SEC_DELAY = 0.3

# Old CIK mappings
OLD_CIKS = {
    "GOOGL": [
        {"cik": 1288776, "name": "Google Inc.", "fy_range": (2010, 2015)},
    ],
    "AVGO": [
        {"cik": 1441634, "name": "Avago Technologies", "fy_range": (2010, 2016)},
        {"cik": 1649338, "name": "Broadcom Ltd", "fy_range": (2016, 2019)},
    ],
}

# HTML patterns for section extraction
_RISK_FACTORS_PATTERNS = {
    "start": r'Item\s*(?:&nbsp;|&#160;|\s)*1A[\.\s:]*(?:&nbsp;|\s)*(?:RIS\s*K\s*FACTOR|RISK\s*FACTOR|Risk\s*Factor)?',
    "end": r'(?:Item\s*(?:&nbsp;|&#160;|\s)*1B|Item\s*(?:&nbsp;|&#160;|\s)*2[\.\s]|PART\s+II)',
}

_MDA_10K_PATTERNS = {
    "start": r'Item\s*(?:&nbsp;|&#160;|\s)*7[\.\s:]*(?:&nbsp;|\s)*(?:MANAGEMENT|Management)',
    "end": r'(?:Item\s*(?:&nbsp;|&#160;|\s)*7A|Item\s*(?:&nbsp;|&#160;|\s)*8[\.\s])',
}

_MDA_10Q_PATTERNS = {
    "start": r'Item\s*(?:&nbsp;|&#160;|\s)*2[\.\s:]*(?:&nbsp;|\s)*(?:MANAGEMENT|Management)',
    "end": r'(?:Item\s*(?:&nbsp;|&#160;|\s)*3[\.\s]|Item\s*(?:&nbsp;|&#160;|\s)*4[\.\s])',
}

_RISK_10Q_PATTERNS = {
    "start": r'Item\s*(?:&nbsp;|&#160;|\s)*1A[\.\s:]*(?:&nbsp;|\s)*(?:RIS\s*K|Risk)',
    "end": r'(?:Item\s*(?:&nbsp;|&#160;|\s)*2[\.\s]|Item\s*(?:&nbsp;|&#160;|\s)*5[\.\s]|PART\s+II)',
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        database=os.getenv("PG_DATABASE"),
    )


def _strip_html(html_text):
    """Strip HTML tags and entities from text."""
    text = re.sub(r'<[^>]+>', ' ', html_text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_section_from_html(html, start_pattern, end_pattern):
    """Extract a section from raw HTML using regex boundaries."""
    threshold = min(len(html) // 20, 100000)
    for m in re.finditer(start_pattern, html, re.IGNORECASE):
        if m.start() < threshold:
            continue
        rest = html[m.start() + 50:]
        for end_match in re.finditer(end_pattern, rest, re.IGNORECASE):
            section_html = html[m.start():m.start() + 50 + end_match.start()]
            text = _strip_html(section_html)
            if len(text) >= MIN_TEXT_LEN:
                return text
    return None


def _try_legacy_extraction(filing_obj, prop_name):
    """Try legacy edgartools parser for a section."""
    try:
        text = getattr(filing_obj, prop_name, None)
        if text and len(text) >= MIN_TEXT_LEN:
            return text
    except Exception:
        pass
    return None


def extract_10k_sections(filing, ticker):
    """Extract Risk Factors and MD&A from a 10-K filing."""
    sections = {}

    # Try legacy parser first
    try:
        obj = filing.obj()
    except Exception:
        obj = None

    # Risk Factors
    text = None
    if obj:
        text = _try_legacy_extraction(obj, "risk_factors")
        if text:
            sections["risk_factors"] = ("legacy", text)

    if "risk_factors" not in sections:
        try:
            html = filing.html()
            text = _extract_section_from_html(html, _RISK_FACTORS_PATTERNS["start"], _RISK_FACTORS_PATTERNS["end"])
            if text:
                sections["risk_factors"] = ("html", text)
        except Exception:
            pass

    # MD&A
    text = None
    if obj:
        text = _try_legacy_extraction(obj, "management_discussion")
        if text:
            sections["mda"] = ("legacy", text)

    if "mda" not in sections:
        try:
            html = filing.html() if 'html' not in dir() else html
            text = _extract_section_from_html(html, _MDA_10K_PATTERNS["start"], _MDA_10K_PATTERNS["end"])
            if text:
                sections["mda"] = ("html", text)
        except Exception:
            pass

    return sections


def extract_10q_sections(filing, ticker):
    """Extract Risk Factors and MD&A from a 10-Q filing."""
    sections = {}

    try:
        obj = filing.obj()
    except Exception:
        obj = None

    # MD&A (Item 2 in 10-Q)
    text = None
    if obj:
        text = _try_legacy_extraction(obj, "management_discussion")
        if text:
            sections["mda"] = ("legacy", text)

    if "mda" not in sections:
        try:
            html = filing.html()
            text = _extract_section_from_html(html, _MDA_10Q_PATTERNS["start"], _MDA_10Q_PATTERNS["end"])
            if text:
                sections["mda"] = ("html", text)
        except Exception:
            pass

    # Risk Factors (Item 1A in 10-Q)
    text = None
    if obj:
        text = _try_legacy_extraction(obj, "risk_factors")
        if text:
            sections["risk_factors"] = ("legacy", text)

    if "risk_factors" not in sections:
        try:
            html = filing.html() if 'html' not in dir() else html
            text = _extract_section_from_html(html, _RISK_10Q_PATTERNS["start"], _RISK_10Q_PATTERNS["end"])
            if text:
                sections["risk_factors"] = ("html", text)
        except Exception:
            pass

    return sections


def insert_filing(conn, ticker, company_name, form_type, filing_date, accession, fiscal_year, fiscal_period):
    """Insert filing metadata. Returns filing_id or None if duplicate."""
    cur = conn.cursor()
    cur.execute("SELECT id FROM filings WHERE accession_number = %s", (accession,))
    existing = cur.fetchone()
    if existing:
        return existing[0]  # Return existing ID

    cur.execute("""
        INSERT INTO filings (ticker, company_name, form_type, filing_date,
                             accession_number, fiscal_year, fiscal_period)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (ticker, company_name, form_type, filing_date, accession, fiscal_year, fiscal_period))
    filing_id = cur.fetchone()[0]
    conn.commit()
    return filing_id


def save_section_10k(conn, ticker, fiscal_year, section_id, section_title, text, accession, filing_date):
    """Save a 10-K section: parent row + chunks + embeddings."""
    cur = conn.cursor()

    # Check if already populated
    cur.execute("""
        SELECT COUNT(*) FROM sections_10k
        WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
        AND is_chunked = true AND embedding IS NOT NULL
    """, (ticker, fiscal_year, section_id))
    if cur.fetchone()[0] >= 5:
        return 0  # Already populated

    # Remove existing stubs
    cur.execute("""
        DELETE FROM sections_10k
        WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
    """, (ticker, fiscal_year, section_id))

    # Insert parent row
    cur.execute("""
        INSERT INTO sections_10k
        (ticker, fiscal_year, section_id, section_title, section_text,
         accession_number, filing_date, char_count, word_count, is_chunked)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, false)
        RETURNING id
    """, (ticker, fiscal_year, section_id, section_title, text,
          accession, filing_date, len(text), len(text.split())))
    parent_id = cur.fetchone()[0]
    conn.commit()

    # Chunk
    chunks = chunk_section_aware(text)
    if not chunks:
        return 0

    # Embed
    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(texts)

    # Insert chunks
    for chunk, emb in zip(chunks, embeddings):
        heading = chunk.subsection_heading or section_title
        cur.execute("""
            INSERT INTO sections_10k
            (ticker, fiscal_year, section_id, section_title, section_text,
             is_chunked, parent_section_id, chunk_index, chunk_start_char, chunk_end_char,
             embedding, char_count, word_count, subsection_heading)
            VALUES (%s, %s, %s, %s, %s, true, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (ticker, fiscal_year, section_id, section_title, chunk.text,
              parent_id, chunk.chunk_index, chunk.start_char, chunk.end_char,
              emb, len(chunk.text), len(chunk.text.split()), heading))

    cur.execute("UPDATE sections_10k SET is_chunked = true WHERE id = %s", (parent_id,))
    conn.commit()
    return len(chunks)


def save_section_10q(conn, ticker, fiscal_year, fiscal_quarter, section_id, section_title, text, accession, filing_date):
    """Save a 10-Q section: parent row + chunks + embeddings."""
    cur = conn.cursor()

    # Check if already populated
    cur.execute("""
        SELECT COUNT(*) FROM sections_10q
        WHERE ticker = %s AND fiscal_year = %s AND fiscal_quarter = %s AND section_id = %s
        AND is_chunked = true AND embedding IS NOT NULL
    """, (ticker, fiscal_year, fiscal_quarter, section_id))
    if cur.fetchone()[0] >= 3:
        return 0  # Already populated

    # Remove existing stubs
    cur.execute("""
        DELETE FROM sections_10q
        WHERE ticker = %s AND fiscal_year = %s AND fiscal_quarter = %s AND section_id = %s
    """, (ticker, fiscal_year, fiscal_quarter, section_id))

    fiscal_period = f"Q{fiscal_quarter}"

    # Insert parent row
    cur.execute("""
        INSERT INTO sections_10q
        (ticker, fiscal_year, fiscal_quarter, fiscal_period, section_id, section_title, section_text,
         accession_number, filing_date, char_count, word_count, is_chunked)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, false)
        RETURNING id
    """, (ticker, fiscal_year, fiscal_quarter, fiscal_period, section_id, section_title, text,
          accession, filing_date, len(text), len(text.split())))
    parent_id = cur.fetchone()[0]
    conn.commit()

    # Chunk
    chunks = chunk_section_aware(text)
    if not chunks:
        return 0

    # Embed
    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(texts)

    # Insert chunks
    for chunk, emb in zip(chunks, embeddings):
        heading = chunk.subsection_heading or section_title
        cur.execute("""
            INSERT INTO sections_10q
            (ticker, fiscal_year, fiscal_quarter, fiscal_period, section_id, section_title, section_text,
             is_chunked, parent_section_id, chunk_index, chunk_start_char, chunk_end_char,
             embedding, char_count, word_count, subsection_heading)
            VALUES (%s, %s, %s, %s, %s, %s, %s, true, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (ticker, fiscal_year, fiscal_quarter, fiscal_period, section_id, section_title, chunk.text,
              parent_id, chunk.chunk_index, chunk.start_char, chunk.end_char,
              emb, len(chunk.text), len(chunk.text.split()), heading))

    cur.execute("UPDATE sections_10q SET is_chunked = true WHERE id = %s", (parent_id,))
    conn.commit()
    return len(chunks)


def process_10k(conn, ticker, filing, company_name):
    """Process a single 10-K filing: insert metadata, extract sections, chunk, embed."""
    accession = filing.accession_no
    filing_date = str(filing.filing_date)

    # Create a simple object for fiscal year inference
    class _FakeFiling:
        def __init__(self, f):
            self.form = f.form
            self.filing_date = f.filing_date
            self.period_of_report = getattr(f, 'period_of_report', f.filing_date)
            self.accession_number = f.accession_no

    fake = _FakeFiling(filing)
    fiscal_year = infer_fiscal_year(ticker, fake)

    # Check if this filing already exists
    cur = conn.cursor()
    cur.execute("SELECT id FROM filings WHERE accession_number = %s", (accession,))
    existing = cur.fetchone()

    if not existing:
        filing_id = insert_filing(conn, ticker, company_name, "10-K", filing_date,
                                  accession, fiscal_year, "FY")
        print(f"    Inserted filing: FY{fiscal_year} 10-K (acc={accession})")
    else:
        filing_id = existing[0]

    # Extract sections
    sections = extract_10k_sections(filing, ticker)
    total_chunks = 0

    if "risk_factors" in sections:
        method, text = sections["risk_factors"]
        n = save_section_10k(conn, ticker, fiscal_year, "item_1a_risk_factors",
                             "Risk Factors", text, accession, filing_date)
        if n > 0:
            print(f"    Risk Factors [{method}]: {len(text):,} chars -> {n} chunks")
            total_chunks += n

    if "mda" in sections:
        method, text = sections["mda"]
        n = save_section_10k(conn, ticker, fiscal_year, "item_7_mda",
                             "Management Discussion and Analysis", text, accession, filing_date)
        if n > 0:
            print(f"    MD&A [{method}]: {len(text):,} chars -> {n} chunks")
            total_chunks += n

    return fiscal_year, total_chunks


def process_10q(conn, ticker, filing, company_name):
    """Process a single 10-Q filing: insert metadata, extract sections, chunk, embed."""
    accession = filing.accession_no
    filing_date = str(filing.filing_date)

    class _FakeFiling:
        def __init__(self, f):
            self.form = f.form
            self.filing_date = f.filing_date
            self.period_of_report = getattr(f, 'period_of_report', f.filing_date)
            self.accession_number = f.accession_no

    fake = _FakeFiling(filing)
    fiscal_year = infer_fiscal_year(ticker, fake)
    fiscal_quarter = infer_fiscal_quarter(ticker, fake)

    if fiscal_quarter is None:
        fiscal_quarter = 1  # Default

    # Insert filing metadata
    cur = conn.cursor()
    cur.execute("SELECT id FROM filings WHERE accession_number = %s", (accession,))
    existing = cur.fetchone()

    if not existing:
        fiscal_period = f"Q{fiscal_quarter}"
        filing_id = insert_filing(conn, ticker, company_name, "10-Q", filing_date,
                                  accession, fiscal_year, fiscal_period)
        print(f"    Inserted filing: FY{fiscal_year} Q{fiscal_quarter} 10-Q (acc={accession})")
    else:
        filing_id = existing[0]

    # Extract sections
    sections = extract_10q_sections(filing, ticker)
    total_chunks = 0

    if "mda" in sections:
        method, text = sections["mda"]
        n = save_section_10q(conn, ticker, fiscal_year, fiscal_quarter, "item_2_mda",
                             "Management Discussion and Analysis", text, accession, filing_date)
        if n > 0:
            print(f"    MD&A [{method}]: {len(text):,} chars -> {n} chunks")
            total_chunks += n

    if "risk_factors" in sections:
        method, text = sections["risk_factors"]
        n = save_section_10q(conn, ticker, fiscal_year, fiscal_quarter, "item_1a_risk_factors",
                             "Risk Factors", text, accession, filing_date)
        if n > 0:
            print(f"    Risk Factors [{method}]: {len(text):,} chars -> {n} chunks")
            total_chunks += n

    return fiscal_year, fiscal_quarter, total_chunks


def phase_1_googl(conn):
    """GOOGL FY2010-2014 from old CIK 1288776, plus FY2015 Q1/Q2."""
    print("=" * 70)
    print("Phase 1: GOOGL from Google Inc. (CIK 1288776)")
    print("=" * 70)

    company = edgar.Company(1288776)
    total_chunks = 0
    successes = 0

    # 10-K filings: FY2010-2014
    print("\n--- 10-K filings ---")
    filings_10k = list(company.get_filings(form="10-K"))
    print(f"Found {len(filings_10k)} total 10-K filings from CIK 1288776")

    for f in filings_10k:
        try:
            # Filter to FY2010-2014 by filing date
            fd = f.filing_date
            if hasattr(fd, 'year'):
                # 10-K for FY2014 filed in early 2015, FY2010 filed in early 2011
                if fd.year < 2011 or fd.year > 2015:
                    continue
            print(f"\n  Processing: filed {f.filing_date}, acc={f.accession_no}")
            fy, n = process_10k(conn, "GOOGL", f, "Google Inc.")
            if fy and 2010 <= fy <= 2014:
                total_chunks += n
                if n > 0:
                    successes += 1
            time.sleep(SEC_DELAY)
        except Exception as e:
            print(f"    ERROR: {e}")
            conn.rollback()

    # 10-Q filings: FY2010-2015 (includes Q1/Q2 of FY2015 filed before restructure)
    print("\n--- 10-Q filings ---")
    filings_10q = list(company.get_filings(form="10-Q"))
    print(f"Found {len(filings_10q)} total 10-Q filings from CIK 1288776")

    for f in filings_10q:
        try:
            fd = f.filing_date
            if hasattr(fd, 'year'):
                if fd.year < 2010 or fd.year > 2015:
                    continue
            print(f"\n  Processing: filed {f.filing_date}, acc={f.accession_no}")
            fy, fq, n = process_10q(conn, "GOOGL", f, "Google Inc.")
            total_chunks += n
            if n > 0:
                successes += 1
            time.sleep(SEC_DELAY)
        except Exception as e:
            print(f"    ERROR: {e}")
            conn.rollback()

    print(f"\n  Phase 1 complete: {successes} sections, {total_chunks} chunks")
    return total_chunks


def phase_2_avgo(conn):
    """AVGO FY2010-2018 from old CIKs."""
    print("\n" + "=" * 70)
    print("Phase 2: AVGO from Avago Technologies (CIK 1441634) + Broadcom Ltd (CIK 1649338)")
    print("=" * 70)

    total_chunks = 0
    successes = 0

    # Check what we already have for AVGO
    cur = conn.cursor()
    cur.execute("SELECT fiscal_year FROM filings WHERE ticker = 'AVGO' AND form_type = '10-K' ORDER BY fiscal_year")
    existing_fy = {r[0] for r in cur.fetchall()}
    print(f"Existing AVGO 10-K fiscal years: {sorted(existing_fy)}")

    for cik_info in OLD_CIKS["AVGO"]:
        cik = cik_info["cik"]
        name = cik_info["name"]
        fy_min, fy_max = cik_info["fy_range"]

        print(f"\n--- {name} (CIK {cik}) ---")
        try:
            company = edgar.Company(cik)
        except Exception as e:
            print(f"  ERROR creating Company: {e}")
            continue

        # 10-K filings
        try:
            filings_10k = list(company.get_filings(form="10-K"))
            print(f"  Found {len(filings_10k)} total 10-K filings")

            for f in filings_10k:
                try:
                    fd = f.filing_date
                    # AVGO fiscal year ends ~October, so 10-K filed in Dec
                    # FY2015 -> filed Dec 2015, FY2010 -> filed Dec 2010
                    if hasattr(fd, 'year'):
                        if fd.year < fy_min or fd.year > fy_max:
                            continue

                    print(f"\n  Processing 10-K: filed {f.filing_date}, acc={f.accession_no}")
                    fy, n = process_10k(conn, "AVGO", f, name)
                    total_chunks += n
                    if n > 0:
                        successes += 1
                    time.sleep(SEC_DELAY)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    conn.rollback()
        except Exception as e:
            print(f"  ERROR fetching 10-K filings: {e}")

        # 10-Q filings
        try:
            filings_10q = list(company.get_filings(form="10-Q"))
            print(f"\n  Found {len(filings_10q)} total 10-Q filings")

            for f in filings_10q:
                try:
                    fd = f.filing_date
                    if hasattr(fd, 'year'):
                        if fd.year < fy_min or fd.year > fy_max:
                            continue

                    print(f"\n  Processing 10-Q: filed {f.filing_date}, acc={f.accession_no}")
                    fy, fq, n = process_10q(conn, "AVGO", f, name)
                    total_chunks += n
                    if n > 0:
                        successes += 1
                    time.sleep(SEC_DELAY)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    conn.rollback()
        except Exception as e:
            print(f"  ERROR fetching 10-Q filings: {e}")

    print(f"\n  Phase 2 complete: {successes} sections, {total_chunks} chunks")
    return total_chunks


def phase_3_missing_quarters(conn):
    """Missing individual 10-Qs: MSFT FY2012 Q2, NVDA FY2010 Q3."""
    print("\n" + "=" * 70)
    print("Phase 3: Missing individual 10-Q filings")
    print("=" * 70)

    total_chunks = 0
    successes = 0

    # MSFT FY2012 Q2: MSFT FY ends June, Q2 = Oct-Dec, filed ~Jan 2012
    # Have Q1 (filed 2011-10-20) and Q3 (filed 2012-04-19), missing Q2
    # NVDA FY2010 Q3: NVDA FY ends January, Q3 = Aug-Oct, filed ~Nov/Feb
    # Have Q1 (filed 2009-08-20) and Q2 (filed 2009-11-19), missing Q3

    missing_quarters = [
        {
            "ticker": "MSFT",
            "cik": 789019,
            "company": "Microsoft Corp",
            "expected_filing_range": (date(2012, 1, 1), date(2012, 3, 31)),
            "fiscal_year": 2012,
            "expected_quarter": 2,
        },
        {
            "ticker": "NVDA",
            "cik": 1045810,
            "company": "NVIDIA Corp",
            "expected_filing_range": (date(2009, 11, 20), date(2010, 3, 31)),
            "fiscal_year": 2010,
            "expected_quarter": 3,
        },
    ]

    for mq in missing_quarters:
        ticker = mq["ticker"]
        print(f"\n  {ticker} FY{mq['fiscal_year']} Q{mq['expected_quarter']}")

        try:
            company = edgar.Company(mq["cik"])
            filings_10q = list(company.get_filings(form="10-Q"))

            # Find the filing in the expected date range
            start, end = mq["expected_filing_range"]
            candidates = []
            for f in filings_10q:
                fd = f.filing_date
                if hasattr(fd, 'year') and start <= fd <= end:
                    candidates.append(f)

            if not candidates:
                print(f"    No 10-Q filing found in range {start} to {end}")
                # Try broader search
                for f in filings_10q:
                    fd = f.filing_date
                    if hasattr(fd, 'year') and fd.year in (start.year, end.year):
                        print(f"    Available: filed {fd}, acc={f.accession_no}")
                continue

            # Check if already in DB
            for f in candidates:
                cur = conn.cursor()
                cur.execute("SELECT id FROM filings WHERE accession_number = %s", (f.accession_no,))
                if cur.fetchone():
                    print(f"    Already in DB: {f.accession_no}")
                    continue

                print(f"    Processing: filed {f.filing_date}, acc={f.accession_no}")
                fy, fq, n = process_10q(conn, ticker, f, mq["company"])
                total_chunks += n
                if n > 0:
                    successes += 1
                time.sleep(SEC_DELAY)

        except Exception as e:
            print(f"    ERROR: {e}")
            conn.rollback()

    print(f"\n  Phase 3 complete: {successes} sections, {total_chunks} chunks")
    return total_chunks


def main():
    conn = get_connection()

    total = 0
    total += phase_1_googl(conn)
    total += phase_2_avgo(conn)
    total += phase_3_missing_quarters(conn)

    print("\n" + "=" * 70)
    print(f"ALL PHASES COMPLETE: {total} total chunks embedded")
    print("=" * 70)

    # Show updated coverage
    cur = conn.cursor()
    print("\nUpdated filing coverage:")
    for ticker in ["GOOGL", "AVGO", "MSFT", "NVDA"]:
        cur.execute("""
            SELECT form_type, COUNT(*), MIN(fiscal_year), MAX(fiscal_year)
            FROM filings WHERE ticker = %s GROUP BY form_type ORDER BY form_type
        """, (ticker,))
        for r in cur.fetchall():
            print(f"  {ticker} {r[0]}: {r[1]} filings (FY{r[2]}-FY{r[3]})")

    conn.close()


if __name__ == "__main__":
    main()
