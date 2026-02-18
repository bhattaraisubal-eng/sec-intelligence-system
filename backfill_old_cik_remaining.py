"""
Fix remaining gaps from backfill_old_cik_filings.py:
1. GOOGL 10-Q MD&A (FY2010-2015) — HTML entity &#8217; in heading
2. AVGO FY2017 10-K MD&A — Broadcom Ltd legacy parser issue
3. NVDA FY2010 Q3 sections — filing exists but no sections extracted

Usage: .venv/bin/python3 backfill_old_cik_remaining.py
"""

import os
import re
import time
import warnings
import logging

import psycopg2
from dotenv import load_dotenv

load_dotenv(".env")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import edgar
edgar.set_identity("sec_rag_system sec_rag@example.com")

from config import infer_fiscal_year, infer_fiscal_quarter
from chunk_and_embed import chunk_section_aware, generate_embeddings

MIN_TEXT_LEN = 1000
SEC_DELAY = 0.3


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        database=os.getenv("PG_DATABASE"),
    )


def _strip_html(html_text):
    text = re.sub(r'<[^>]+>', ' ', html_text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_section_from_html(html, start_pattern, end_pattern, dotall=False):
    threshold = min(len(html) // 20, 100000)
    flags = re.IGNORECASE | (re.DOTALL if dotall else 0)
    for m in re.finditer(start_pattern, html, flags):
        if m.start() < threshold:
            continue
        rest = html[m.start() + 50:]
        for end_match in re.finditer(end_pattern, rest, re.IGNORECASE):
            section_html = html[m.start():m.start() + 50 + end_match.start()]
            text = _strip_html(section_html)
            if len(text) >= MIN_TEXT_LEN:
                return text
    return None


def save_section_10q(conn, ticker, fiscal_year, fiscal_quarter, section_id, section_title, text, accession, filing_date):
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM sections_10q
        WHERE ticker = %s AND fiscal_year = %s AND fiscal_quarter = %s AND section_id = %s
        AND is_chunked = true AND embedding IS NOT NULL
    """, (ticker, fiscal_year, fiscal_quarter, section_id))
    if cur.fetchone()[0] >= 3:
        return 0

    cur.execute("""
        DELETE FROM sections_10q
        WHERE ticker = %s AND fiscal_year = %s AND fiscal_quarter = %s AND section_id = %s
    """, (ticker, fiscal_year, fiscal_quarter, section_id))

    fiscal_period = f"Q{fiscal_quarter}"
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

    chunks = chunk_section_aware(text)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(texts)

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


def save_section_10k(conn, ticker, fiscal_year, section_id, section_title, text, accession, filing_date):
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM sections_10k
        WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
        AND is_chunked = true AND embedding IS NOT NULL
    """, (ticker, fiscal_year, section_id))
    if cur.fetchone()[0] >= 5:
        return 0

    cur.execute("""
        DELETE FROM sections_10k
        WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
    """, (ticker, fiscal_year, section_id))

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

    chunks = chunk_section_aware(text)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(texts)

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


def main():
    conn = get_connection()
    cur = conn.cursor()
    total_chunks = 0

    # --- Fix 1: GOOGL 10-Q MD&A ---
    print("=" * 70)
    print("Fix 1: GOOGL 10-Q MD&A (FY2010-2015)")
    print("=" * 70)

    # Broadened pattern for GOOGL: handles &#8217; entity and ITEM 2.
    # Patterns: "ITEM 2. MANAGEMENT&#8217;S DISCUSSION" or "Item 2. Management's Discussion"
    _sp = r'(?:\s|&nbsp;|&#160;)*'
    mda_start = rf'ITEM{_sp}2\.{_sp}MANAGEMENT.{{0,15}}DISCUSSION'
    _sp2 = r'(?:\s|&nbsp;|&#160;)*'
    mda_end = rf'(?:ITEM{_sp2}3|Item{_sp2}3|ITEM{_sp2}4|Item{_sp2}4)'

    # Also try broader: "Management's Discussion" anywhere (handles &#146; &#8217; entities)
    mda_start_broad = r'Management.{0,15}Discussion\s+and\s+Analysis'
    # Ultra-broad: ITEM 2 ... DISCUSSION (handles heading split across HTML tags/table cells)
    mda_start_ultra = r'ITEM(?:\s|&nbsp;|&#160;)*2\..*?DISCUSSION'

    company = edgar.Company(1288776)
    filings_10q = list(company.get_filings(form="10-Q"))

    for f in filings_10q:
        fd = f.filing_date
        if fd.year < 2010 or fd.year > 2015:
            continue

        acc = f.accession_no

        # Infer FY/Q
        class _Fake:
            def __init__(self, f):
                self.form = f.form
                self.filing_date = f.filing_date
                self.period_of_report = getattr(f, 'period_of_report', f.filing_date)
        fake = _Fake(f)
        fy = infer_fiscal_year("GOOGL", fake)
        fq = infer_fiscal_quarter("GOOGL", fake)
        if fq is None:
            continue

        # Check if already has MD&A
        cur.execute("""
            SELECT COUNT(*) FROM sections_10q
            WHERE ticker = 'GOOGL' AND fiscal_year = %s AND fiscal_quarter = %s
            AND section_id = 'item_2_mda' AND is_chunked = true AND embedding IS NOT NULL
        """, (fy, fq))
        if cur.fetchone()[0] >= 3:
            continue

        print(f"\n  GOOGL FY{fy} Q{fq} (acc={acc})")
        try:
            html = f.html()
            print(f"    HTML: {len(html):,} chars")

            text = _extract_section_from_html(html, mda_start, mda_end)
            if not text:
                text = _extract_section_from_html(html, mda_start_broad, mda_end)
            if not text:
                text = _extract_section_from_html(html, mda_start_ultra, mda_end, dotall=True)

            if text:
                print(f"    MD&A: {len(text):,} chars")
                n = save_section_10q(conn, "GOOGL", fy, fq, "item_2_mda",
                                     "Management Discussion and Analysis", text, acc, str(fd))
                print(f"    => {n} chunks embedded")
                total_chunks += n
            else:
                print(f"    => No MD&A extracted")
        except Exception as e:
            print(f"    => ERROR: {e}")
            conn.rollback()

        time.sleep(SEC_DELAY)

    # --- Fix 2: AVGO FY2017 10-K MD&A ---
    print(f"\n{'=' * 70}")
    print("Fix 2: AVGO FY2017 10-K MD&A (Broadcom Ltd)")
    print("=" * 70)

    cur.execute("""
        SELECT accession_number, filing_date FROM filings
        WHERE ticker = 'AVGO' AND fiscal_year = 2017 AND form_type = '10-K' LIMIT 1
    """)
    row = cur.fetchone()
    if row:
        acc, filing_date = row[0], str(row[1])
        print(f"  AVGO FY2017 (acc={acc})")

        try:
            company_brcm = edgar.Company(1649338)
            api_filings = list(company_brcm.get_filings(form="10-K"))
            api_filing = None
            for af in api_filings:
                if af.accession_no == acc:
                    api_filing = af
                    break

            if api_filing:
                html = api_filing.html()
                print(f"    HTML: {len(html):,} chars")

                # Try generic MD&A extraction with broadened patterns
                mda_patterns = [
                    (r'Item\s*(?:&nbsp;|&#160;|\s)*7[\.\s:]*(?:&nbsp;|\s)*(?:MANAGEMENT|Management)',
                     r'(?:Item\s*(?:&nbsp;|&#160;|\s)*7A|Item\s*(?:&nbsp;|&#160;|\s)*8[\.\s])'),
                    (r'MANAGEMENT.{0,15}DISCUSSION\s+AND\s+ANALYSIS',
                     r'(?:Item\s*7A|Item\s*8[\.\s]|QUANTITATIVE\s+AND\s+QUALITATIVE)'),
                    (r'Management.{0,15}Discussion\s+and\s+Analysis',
                     r'(?:Item\s*7A|Item\s*8|QUANTITATIVE)'),
                ]

                text = None
                for start_p, end_p in mda_patterns:
                    text = _extract_section_from_html(html, start_p, end_p)
                    if text:
                        break

                if text:
                    print(f"    MD&A: {len(text):,} chars")
                    n = save_section_10k(conn, "AVGO", 2017, "item_7_mda",
                                         "Management Discussion and Analysis", text, acc, filing_date)
                    print(f"    => {n} chunks embedded")
                    total_chunks += n
                else:
                    print(f"    => No MD&A extracted")
            else:
                print(f"    => Filing not found in Company API")
        except Exception as e:
            print(f"    => ERROR: {e}")
            conn.rollback()

    # --- Fix 3: NVDA FY2010 Q3 sections ---
    print(f"\n{'=' * 70}")
    print("Fix 3: NVDA FY2010 Q3 sections")
    print("=" * 70)

    cur.execute("""
        SELECT accession_number, filing_date FROM filings
        WHERE ticker = 'NVDA' AND fiscal_year = 2010 AND fiscal_period = 'Q3' AND form_type = '10-Q'
    """)
    row = cur.fetchone()
    if row:
        acc, filing_date = row[0], str(row[1])
        print(f"  NVDA FY2010 Q3 (acc={acc})")

        try:
            company_nvda = edgar.Company(1045810)
            api_filings = list(company_nvda.get_filings(form="10-Q"))
            api_filing = None
            for af in api_filings:
                if af.accession_no == acc:
                    api_filing = af
                    break

            if api_filing:
                html = api_filing.html()
                print(f"    HTML: {len(html):,} chars")

                # MD&A
                mda_text = _extract_section_from_html(
                    html,
                    r'Item\s*(?:&nbsp;|&#160;|\s)*2[\.\s:]*(?:&nbsp;|\s)*(?:MANAGEMENT|Management)',
                    r'(?:Item\s*(?:&nbsp;|&#160;|\s)*3|Item\s*(?:&nbsp;|&#160;|\s)*4)'
                )
                if not mda_text:
                    mda_text = _extract_section_from_html(
                        html, r'Management.{0,15}Discussion', r'(?:Item\s*3|Item\s*4)')

                if mda_text:
                    print(f"    MD&A: {len(mda_text):,} chars")
                    n = save_section_10q(conn, "NVDA", 2010, 3, "item_2_mda",
                                         "Management Discussion and Analysis", mda_text, acc, filing_date)
                    print(f"    => {n} chunks embedded")
                    total_chunks += n

                # Risk Factors
                rf_text = _extract_section_from_html(
                    html,
                    r'Item\s*(?:&nbsp;|&#160;|\s)*1A[\.\s:]*(?:&nbsp;|\s)*(?:RISK|Risk)',
                    r'(?:Item\s*(?:&nbsp;|&#160;|\s)*2[\.\s]|Item\s*(?:&nbsp;|&#160;|\s)*5[\.\s]|PART\s+II)'
                )
                if rf_text:
                    print(f"    Risk Factors: {len(rf_text):,} chars")
                    n = save_section_10q(conn, "NVDA", 2010, 3, "item_1a_risk_factors",
                                         "Risk Factors", rf_text, acc, filing_date)
                    print(f"    => {n} chunks embedded")
                    total_chunks += n
            else:
                print(f"    => Filing not found in Company API")
        except Exception as e:
            print(f"    => ERROR: {e}")
            conn.rollback()

    print(f"\n{'=' * 70}")
    print(f"ALL FIXES COMPLETE: {total_chunks} total chunks embedded")
    print("=" * 70)
    conn.close()


if __name__ == "__main__":
    main()
