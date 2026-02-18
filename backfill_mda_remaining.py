"""
Backfill remaining MD&A gaps: JPM (all years), BRK-B FY2018, AVGO FY2019.
Uses Company API for JPM (to get the full 12M+ char HTML).

Usage: .venv/bin/python3 backfill_mda_remaining.py
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

from chunk_and_embed import chunk_section_aware, generate_embeddings

MIN_TEXT_LEN = 5000  # MD&A should be substantial


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


def extract_jpm_mda_from_html(html):
    """Extract JPM MD&A using multiple start patterns, ending at 'NOTES TO CONSOLIDATED FINANCIAL'.

    JPM's 10-K structure varies by year:
    - FY2018+: "Executive Overview" marks MD&A start
    - FY2011-2017: "Management's discussion and analysis of the financial condition" marks start
    - All years: "NOTES TO CONSOLIDATED FINANCIAL" marks end
    """
    # Try multiple start patterns in priority order
    start_patterns = [
        # Newer filings (2018+)
        r'(?:EXECUTIVE|Executive)\s+(?:OVERVIEW|Overview)',
        # Older filings: "of the financial condition" (with or without "the")
        r'Management.{0,15}s?\s+discussion\s+and\s+analysis\s+of\s+(?:the\s+)?financial\s+condition',
        # Fallback: look for "MD&A" or "MD&amp;A" followed by content
        r'MD(?:&amp;|&)A.{0,10}\)\s+of\s+(?:the\s+)?financial\s+condition',
        # Last resort: "MD&A" content block
        r'(?:MD&amp;A|MD.A).{0,50}of\s+(?:the\s+)?financial\s+condition\s+and\s+results',
    ]

    end_matches = list(re.finditer(
        r'NOTES?\s+TO\s+(?:THE\s+)?CONSOLIDATED\s+FINANCIAL', html, re.IGNORECASE))

    for pattern in start_patterns:
        matches = list(re.finditer(pattern, html, re.IGNORECASE))
        if not matches:
            continue

        # Use the first match (skip TOC-area ones only for "Executive Overview")
        if 'EXECUTIVE' in pattern.upper():
            threshold = len(html) // 7
            matches = [m for m in matches if m.start() > threshold]
            if not matches:
                continue

        start = matches[0].start()

        # Find end: first "NOTES TO CONSOLIDATED FINANCIAL" at least 100KB after start
        end = None
        for m in end_matches:
            if m.start() > start + 100000:
                end = m.start()
                break

        if not end:
            continue

        text = _strip_html(html[start:end])
        if len(text) >= MIN_TEXT_LEN:
            return text

    return None


def extract_generic_mda_from_html(html):
    """Generic: extract between Management's Discussion heading and Item 7A/8.

    Handles HTML entities like &#146; (curly apostrophe) in 'Management's'.
    """
    threshold = min(len(html) // 20, 100000)

    # Multiple start patterns to handle entity variations
    # Note: BRK-B uses &nbsp; (or &#160;) between "Item" and number
    _sp = r'(?:\s|&nbsp;|&#160;)*'  # whitespace OR &nbsp; entity
    start_patterns = [
        rf'Item{_sp}7[\.\s:]*(?:&nbsp;|\s)*(?:MANAGEMENT|Management)',
        r'Management.{0,15}Discussion\s+and\s+Analysis\s+of\s*\n?\s*Financial\s+Condition',
        r'Management.{0,15}Discussion\s+and\s+Analysis',
    ]
    end_pattern = rf'(?:Item{_sp}7A|Item{_sp}8[\.\s])'

    for start_pattern in start_patterns:
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


def save_and_embed(conn, ticker, fy, accession, filing_date, text):
    """Save MD&A to sections_10k, chunk, embed."""
    cur = conn.cursor()
    section_id = "item_7_mda"
    section_title = "Management Discussion and Analysis"

    # Remove existing stubs
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

    # Insert
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

    cur.execute("UPDATE sections_10k SET is_chunked = true WHERE id = %s", (parent_id,))
    conn.commit()

    # Also save to filing_sections
    cur.execute("SELECT id FROM filings WHERE accession_number = %s", (accession,))
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
    cur = conn.cursor()

    # --- JPM: Use Company API to get full HTML ---
    print("=" * 60)
    print("Phase 1: JPM MD&A via Company API")
    print("=" * 60)

    company = edgar.Company("JPM")
    jpm_filings = list(company.get_filings(form="10-K"))

    # Get our DB fiscal years for JPM
    cur.execute("""
        SELECT fiscal_year, accession_number, filing_date
        FROM filings WHERE ticker = 'JPM' AND form_type = '10-K'
        ORDER BY fiscal_year
    """)
    jpm_db = {row[0]: (row[1], str(row[2])) for row in cur.fetchall()}

    # Find which need backfill
    cur.execute("""
        SELECT fiscal_year, COUNT(*) FROM sections_10k
        WHERE ticker = 'JPM' AND section_id = 'item_7_mda'
        AND is_chunked = true AND embedding IS NOT NULL
        GROUP BY fiscal_year HAVING COUNT(*) >= 5
    """)
    jpm_done = {row[0] for row in cur.fetchall()}

    jpm_gaps = sorted(set(jpm_db.keys()) - jpm_done)
    print(f"JPM gaps: {jpm_gaps}\n")

    total_chunks = 0
    successes = 0
    failures = []

    for fy in jpm_gaps:
        acc, filing_date = jpm_db[fy]
        print(f"  JPM FY{fy} (acc={acc})")

        # Find matching filing from Company API
        api_filing = None
        for f in jpm_filings:
            if f.accession_no == acc:
                api_filing = f
                break

        if not api_filing:
            print(f"    => Not found in Company API filings, skipping")
            failures.append(("JPM", fy, "not in Company API"))
            continue

        try:
            html = api_filing.html()
            print(f"    HTML: {len(html):,} chars")

            text = extract_jpm_mda_from_html(html)
            if text:
                print(f"    MD&A: {len(text):,} chars")
                n = save_and_embed(conn, "JPM", fy, acc, filing_date, text)
                print(f"    => {n} chunks embedded")
                total_chunks += n
                successes += 1
            else:
                print(f"    => No MD&A extracted from HTML")
                failures.append(("JPM", fy, "extraction failed"))
        except Exception as e:
            print(f"    => ERROR: {e}")
            failures.append(("JPM", fy, str(e)))
            conn.rollback()

        time.sleep(0.5)

    # --- BRK-B FY2018 and AVGO FY2019: Use Company API with generic HTML fallback ---
    print(f"\n{'=' * 60}")
    print("Phase 2: BRK-B FY2018, AVGO FY2019")
    print("=" * 60)

    other_gaps = [("BRK-B", 2018, "BRK.B"), ("AVGO", 2019, "AVGO")]

    for ticker, fy, api_ticker in other_gaps:
        cur.execute("""
            SELECT accession_number, filing_date FROM filings
            WHERE ticker = %s AND fiscal_year = %s AND form_type = '10-K' LIMIT 1
        """, (ticker, fy))
        row = cur.fetchone()
        if not row:
            print(f"  {ticker} FY{fy}: No filing in DB")
            failures.append((ticker, fy, "no filing"))
            continue

        acc, filing_date = row[0], str(row[1])
        print(f"  {ticker} FY{fy} (acc={acc})")

        try:
            company = edgar.Company(api_ticker)
            api_filings = list(company.get_filings(form="10-K"))

            api_filing = None
            for f in api_filings:
                if f.accession_no == acc:
                    api_filing = f
                    break

            if not api_filing:
                print(f"    => Not in Company API, trying direct Filing()")
                from edgar import Filing
                TICKER_CIK = {"BRK-B": 1067983, "AVGO": 1730168}
                api_filing = Filing(company="", cik=TICKER_CIK[ticker], form="10-K",
                                    filing_date=filing_date, accession_no=acc)

            html = api_filing.html()
            print(f"    HTML: {len(html):,} chars")

            text = extract_generic_mda_from_html(html)
            if text:
                print(f"    MD&A: {len(text):,} chars")
                n = save_and_embed(conn, ticker, fy, acc, filing_date, text)
                print(f"    => {n} chunks embedded")
                total_chunks += n
                successes += 1
            else:
                print(f"    => No MD&A extracted")
                failures.append((ticker, fy, "extraction failed"))
        except Exception as e:
            print(f"    => ERROR: {e}")
            failures.append((ticker, fy, str(e)))
            conn.rollback()

        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {successes} sections, {total_chunks} chunks embedded")
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for t, y, reason in failures:
            print(f"  {t} FY{y}: {reason}")
    print("=" * 60)
    conn.close()


if __name__ == "__main__":
    main()
