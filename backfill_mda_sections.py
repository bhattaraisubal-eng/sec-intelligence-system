"""
Backfill missing MD&A (Item 7) sections for all tickers
using edgartools legacy parser with HTML regex fallback for JPM.

Usage: .venv/bin/python3 backfill_mda_sections.py
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
from edgar import Filing

from chunk_and_embed import chunk_section_aware, generate_embeddings

# --- Config ---
MIN_TEXT_LEN = 1000
SEC_DELAY = 0.3

TICKER_CIK = {
    "AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "AMZN": 1018724,
    "GOOGL": 1652044, "META": 1326801, "BRK-B": 1067983, "LLY": 59478,
    "AVGO": 1730168, "JPM": 19617,
}

# HTML fallback patterns for MD&A extraction
_MDA_HTML_PATTERNS = {
    "start": r'Item\s*7[\.\s:]*(?:&nbsp;|\s)*(?:MANAGEMENT.{0,10}DISCUSSION|Management.{0,10}Discussion)',
    "end": r'(?:Item\s*7A|Item\s*8[\.\s])',
}

# JPM-specific: MD&A is labeled "EXECUTIVE OVERVIEW" through "NOTES TO CONSOLIDATED FINANCIAL"
_JPM_MDA_PATTERNS = {
    "start": r'(?:EXECUTIVE|Executive)\s+(?:OVERVIEW|Overview)',
    "end": r'NOTES?\s+TO\s+(?:THE\s+)?CONSOLIDATED\s+FINANCIAL',
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        database=os.getenv("PG_DATABASE"),
    )


def find_mda_gaps(conn):
    """Find all ticker/year combos missing MD&A in sections_10k."""
    cur = conn.cursor()
    tickers = list(TICKER_CIK.keys())
    gaps = []

    for t in tickers:
        for y in range(2010, 2026):
            # Check if substantial MD&A chunks already exist
            cur.execute("""
                SELECT COUNT(*) FROM sections_10k
                WHERE ticker = %s AND fiscal_year = %s
                AND section_id = 'item_7_mda'
                AND is_chunked = true AND embedding IS NOT NULL
            """, (t, y))
            chunk_count = cur.fetchone()[0]
            if chunk_count >= 5:  # At least 5 real chunks = populated
                continue

            # Get filing metadata
            cur.execute("""
                SELECT accession_number, filing_date, company_name
                FROM filings WHERE ticker = %s AND fiscal_year = %s AND form_type = '10-K'
                LIMIT 1
            """, (t, y))
            row = cur.fetchone()
            if row:
                gaps.append({
                    "ticker": t,
                    "fiscal_year": y,
                    "accession": row[0],
                    "filing_date": str(row[1]),
                    "company_name": row[2],
                    "existing_chunks": chunk_count,
                })
    return gaps


def _extract_section_from_html(html, start_pattern, end_pattern):
    """Extract a section from raw HTML using regex boundaries."""
    threshold = min(len(html) // 20, 100000)
    for m in re.finditer(start_pattern, html, re.IGNORECASE):
        if m.start() < threshold:
            continue
        rest = html[m.start() + 50:]
        for end_match in re.finditer(end_pattern, rest, re.IGNORECASE):
            section_html = html[m.start():m.start() + 50 + end_match.start()]
            text = re.sub(r'<[^>]+>', ' ', section_html)
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)
            text = re.sub(r'&#\d+;', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) >= MIN_TEXT_LEN:
                return text
    return None


def _extract_jpm_mda(html):
    """JPM-specific: extract MD&A from 'EXECUTIVE OVERVIEW' to 'NOTES TO CONSOLIDATED FINANCIAL'."""
    # Skip TOC area (first ~14% of HTML)
    threshold = len(html) // 7

    start_matches = [m for m in re.finditer(_JPM_MDA_PATTERNS["start"], html)
                     if m.start() > threshold]
    end_matches = [m for m in re.finditer(_JPM_MDA_PATTERNS["end"], html, re.IGNORECASE)
                   if m.start() > threshold]

    if not start_matches:
        return None

    start = start_matches[0].start()

    # Find end: first "NOTES TO CONSOLIDATED FINANCIAL" that's at least 100KB after start
    end = None
    for m in end_matches:
        if m.start() > start + 100000:
            end = m.start()
            break

    if not end:
        return None

    section_html = html[start:end]
    text = re.sub(r'<[^>]+>', ' ', section_html)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= MIN_TEXT_LEN else None


def extract_mda(gap):
    """Extract MD&A via legacy parser, with HTML fallback."""
    cik = TICKER_CIK.get(gap["ticker"], 0)
    f = Filing(
        company=gap["company_name"],
        cik=cik,
        form="10-K",
        filing_date=gap["filing_date"],
        accession_no=gap["accession"],
    )
    obj = f.obj()

    # Try legacy parser first (property name is 'management_discussion')
    text = None
    try:
        mda = getattr(obj, 'management_discussion', None)
        if mda and len(mda) >= MIN_TEXT_LEN:
            text = mda
            print(f"  [legacy] MD&A: {len(text):,} chars")
    except Exception as e:
        print(f"  Warning: legacy mda extraction failed: {e}")

    # HTML fallback if legacy parser returned nothing useful
    if not text or len(text) < MIN_TEXT_LEN:
        try:
            html = f.html()

            # JPM-specific extraction
            if gap["ticker"] == "JPM":
                text = _extract_jpm_mda(html)
                if text:
                    print(f"  [JPM HTML] MD&A: {len(text):,} chars")
            else:
                # Generic HTML fallback: Item 7 to Item 7A/8
                text = _extract_section_from_html(
                    html, _MDA_HTML_PATTERNS["start"], _MDA_HTML_PATTERNS["end"]
                )
                if text:
                    print(f"  [HTML fallback] MD&A: {len(text):,} chars")
        except Exception as e:
            print(f"  Warning: HTML fallback failed: {e}")

    return text


def save_and_embed(conn, gap, text):
    """Save MD&A to sections_10k, chunk, embed, and insert chunked rows."""
    cur = conn.cursor()
    ticker = gap["ticker"]
    fy = gap["fiscal_year"]
    section_id = "item_7_mda"
    section_title = "Management Discussion and Analysis"

    # Remove existing stubs/incomplete data
    cur.execute("""
        DELETE FROM sections_10k
        WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
    """, (ticker, fy, section_id))

    # Insert parent row
    cur.execute("""
        INSERT INTO sections_10k
        (ticker, fiscal_year, section_id, section_title, section_text,
         accession_number, filing_date, char_count, word_count, is_chunked)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, false)
        RETURNING id
    """, (ticker, fy, section_id, section_title, text,
          gap["accession"], gap["filing_date"], len(text), len(text.split())))
    parent_id = cur.fetchone()[0]
    conn.commit()

    # Chunk
    chunks = chunk_section_aware(text)
    if not chunks:
        print(f"  => No chunks produced")
        return 0

    # Generate embeddings
    texts = [c.text for c in chunks]
    embeddings = generate_embeddings(texts)

    # Insert chunked rows
    for chunk, emb in zip(chunks, embeddings):
        heading = chunk.subsection_heading or section_title
        cur.execute("""
            INSERT INTO sections_10k
            (ticker, fiscal_year, section_id, section_title, section_text,
             is_chunked, parent_section_id, chunk_index, chunk_start_char, chunk_end_char,
             embedding, char_count, word_count, subsection_heading)
            VALUES (%s, %s, %s, %s, %s, true, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (ticker, fy, section_id, section_title, chunk.text,
              parent_id, chunk.chunk_index, chunk.start_char, chunk.end_char,
              emb, len(chunk.text), len(chunk.text.split()), heading))

    # Mark parent as chunked
    cur.execute("UPDATE sections_10k SET is_chunked = true WHERE id = %s", (parent_id,))
    conn.commit()

    # Also save to filing_sections table
    cur.execute("SELECT id FROM filings WHERE accession_number = %s", (gap["accession"],))
    filing_row = cur.fetchone()
    if filing_row:
        cur.execute("""
            INSERT INTO filing_sections
            (filing_id, ticker, form_type, fiscal_year, fiscal_period,
             section_key, section_title, section_text, char_count, word_count, file_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filing_id, section_key) DO UPDATE SET
                section_text = EXCLUDED.section_text,
                char_count = EXCLUDED.char_count,
                word_count = EXCLUDED.word_count
        """, (filing_row[0], ticker, "10-K", fy, "FY",
              "part_i_item_7", section_title, text, len(text), len(text.split()), None))
        conn.commit()

    return len(chunks)


def main():
    conn = get_connection()
    gaps = find_mda_gaps(conn)
    print(f"{'=' * 60}")
    print(f"MD&A Backfill — {len(gaps)} ticker/year gaps")
    print(f"{'=' * 60}\n")

    total_chunks = 0
    successes = 0
    failures = []

    for i, gap in enumerate(gaps, 1):
        ticker = gap["ticker"]
        fy = gap["fiscal_year"]
        existing = gap["existing_chunks"]
        print(f"[{i}/{len(gaps)}] {ticker} FY{fy} (existing: {existing} chunks)")

        try:
            text = extract_mda(gap)
            if not text:
                print(f"  => No MD&A extracted")
                failures.append((ticker, fy, "no text extracted"))
                time.sleep(SEC_DELAY)
                continue

            n_chunks = save_and_embed(conn, gap, text)
            print(f"  => {n_chunks} chunks embedded")
            total_chunks += n_chunks
            successes += 1

        except Exception as e:
            print(f"  => ERROR: {e}")
            failures.append((ticker, fy, str(e)))
            conn.rollback()

        time.sleep(SEC_DELAY)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {successes} sections backfilled, {total_chunks} chunks embedded")
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for t, y, reason in failures:
            print(f"  {t} FY{y}: {reason}")
    print(f"{'=' * 60}")
    conn.close()


if __name__ == "__main__":
    main()
