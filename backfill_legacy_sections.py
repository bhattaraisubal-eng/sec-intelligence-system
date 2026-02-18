"""
Backfill missing 10-K sections (risk_factors + business) for all tickers
using edgartools legacy parser fallback.

Usage: .venv/bin/python3 backfill_legacy_sections.py
"""

import os
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
SECTION_PROPERTIES = [
    # (edgartools property, filing_sections key, sections_10k id, display title)
    ("risk_factors", "part_i_item_1a", "item_1a_risk_factors", "Risk Factors"),
    ("business", "part_i_item_1", "item_1_business", "Business"),
]
MIN_TEXT_LEN = 1000  # Skip sections shorter than this
SEC_DELAY = 0.3  # Delay between SEC requests

# Ticker → CIK mapping (required for edgartools Filing constructor)
TICKER_CIK = {
    "AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "AMZN": 1018724,
    "GOOGL": 1652044, "META": 1326801, "BRK-B": 1067983, "LLY": 59478,
    "AVGO": 1730168, "JPM": 19617,
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        database=os.getenv("PG_DATABASE"),
    )


def find_gaps(conn):
    """Find all ticker/year combos missing risk_factors in sections_10k."""
    cur = conn.cursor()
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM"]
    gaps = []

    for t in tickers:
        for y in range(2010, 2025):
            # Check if substantial risk_factors already exist
            cur.execute("""
                SELECT COALESCE(SUM(length(section_text)), 0)
                FROM sections_10k
                WHERE ticker = %s AND fiscal_year = %s
                AND section_id = 'item_1a_risk_factors'
                AND length(section_text) > %s
            """, (t, y, MIN_TEXT_LEN))
            if cur.fetchone()[0] >= MIN_TEXT_LEN:
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
                })
    return gaps


def _extract_section_from_html(html, start_pattern, end_patterns):
    """Fallback: extract a section from raw HTML using regex boundaries."""
    import re
    # Skip TOC matches in first ~5% or 100KB
    threshold = min(len(html) // 20, 100000)
    for m in re.finditer(start_pattern, html, re.IGNORECASE):
        if m.start() < threshold:
            continue
        rest = html[m.start() + 50:]
        # Try each end boundary match, skip ones that produce too-short sections
        for end_match in re.finditer(end_patterns, rest, re.IGNORECASE):
            section_html = html[m.start():m.start() + 50 + end_match.start()]
            text = re.sub(r'<[^>]+>', ' ', section_html)
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)
            text = re.sub(r'&#\d+;', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) >= MIN_TEXT_LEN:
                return text
            # Too short — likely hit an inline reference, try next end match
    return None


# HTML fallback patterns for sections the legacy parser can't extract
_HTML_SECTION_PATTERNS = {
    "risk_factors": {
        "start": r'Item\s*1A[\.\s:]*(?:&nbsp;|\s)*(?:RIS\s*K\s*FACTOR|RISK\s*FACTOR|Risk\s*Factor)?',
        "end": r'(?:Item\s*1B|Item\s*2[\.\s]|PART\s+II)',
    },
    "business": {
        "start": r'Item\s*1[\.\s:]*(?:&nbsp;|\s)*(?:BUSINESS|Business)\b',
        "end": r'(?:Item\s*1A|Item\s*2[\.\s])',
    },
}


def extract_legacy_sections(gap):
    """Fetch filing and extract sections via legacy parser, with HTML fallback."""
    import re
    cik = TICKER_CIK.get(gap["ticker"], 0)
    f = Filing(
        company=gap["company_name"],
        cik=cik,
        form="10-K",
        filing_date=gap["filing_date"],
        accession_no=gap["accession"],
    )
    obj = f.obj()

    extracted = {}
    needs_html_fallback = []

    for prop_name, fs_key, s10k_id, title in SECTION_PROPERTIES:
        try:
            text = getattr(obj, prop_name, None)
            if text and len(text) >= MIN_TEXT_LEN:
                extracted[fs_key] = {
                    "text": text,
                    "s10k_id": s10k_id,
                    "title": title,
                }
            else:
                needs_html_fallback.append((prop_name, fs_key, s10k_id, title))
        except Exception as e:
            print(f"    Warning: {prop_name} extraction failed: {e}")
            needs_html_fallback.append((prop_name, fs_key, s10k_id, title))

    # HTML fallback for sections the legacy parser couldn't get
    if needs_html_fallback:
        try:
            html = f.html()
            for prop_name, fs_key, s10k_id, title in needs_html_fallback:
                patterns = _HTML_SECTION_PATTERNS.get(prop_name)
                if not patterns:
                    continue
                text = _extract_section_from_html(html, patterns["start"], patterns["end"])
                if text:
                    extracted[fs_key] = {
                        "text": text,
                        "s10k_id": s10k_id,
                        "title": title,
                    }
                    print(f"    [HTML fallback] {prop_name}: {len(text)} chars")
        except Exception as e:
            print(f"    Warning: HTML fallback failed: {e}")

    return extracted


def save_to_filing_sections(conn, gap, sections):
    """Insert extracted sections into filing_sections table."""
    cur = conn.cursor()
    # Get filing_id
    cur.execute("SELECT id FROM filings WHERE accession_number = %s", (gap["accession"],))
    row = cur.fetchone()
    if not row:
        return 0
    filing_id = row[0]

    saved = 0
    for fs_key, sec in sections.items():
        text = sec["text"]
        cur.execute("""
            INSERT INTO filing_sections
            (filing_id, ticker, form_type, fiscal_year, fiscal_period,
             section_key, section_title, section_text, char_count, word_count, file_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filing_id, section_key) DO UPDATE SET
                section_text = EXCLUDED.section_text,
                char_count = EXCLUDED.char_count,
                word_count = EXCLUDED.word_count
        """, (filing_id, gap["ticker"], "10-K", gap["fiscal_year"], "FY",
              fs_key, sec["title"], text, len(text), len(text.split()), None))
        saved += 1
    conn.commit()
    return saved


def save_to_sections_10k(conn, gap, sections):
    """Insert parent rows into sections_10k table."""
    cur = conn.cursor()
    inserted = []

    for fs_key, sec in sections.items():
        s10k_id = sec["s10k_id"]
        text = sec["text"]

        # Skip if already exists with substantial content
        cur.execute("""
            SELECT COUNT(*) FROM sections_10k
            WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
            AND length(section_text) > %s
        """, (gap["ticker"], gap["fiscal_year"], s10k_id, MIN_TEXT_LEN))
        if cur.fetchone()[0] > 0:
            continue

        # Remove stubs
        cur.execute("""
            DELETE FROM sections_10k
            WHERE ticker = %s AND fiscal_year = %s AND section_id = %s
        """, (gap["ticker"], gap["fiscal_year"], s10k_id))

        # Insert parent row
        cur.execute("""
            INSERT INTO sections_10k
            (ticker, fiscal_year, section_id, section_title, section_text,
             accession_number, filing_date, char_count, word_count, is_chunked)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, false)
            RETURNING id
        """, (gap["ticker"], gap["fiscal_year"], s10k_id, sec["title"], text,
              gap["accession"], gap["filing_date"], len(text), len(text.split())))
        parent_id = cur.fetchone()[0]
        inserted.append((parent_id, s10k_id, sec["title"], text))

    conn.commit()
    return inserted


def chunk_and_embed_sections(conn, gap, parent_rows):
    """Chunk parent sections, generate embeddings, insert chunked rows."""
    all_chunks = []

    for parent_id, section_id, section_title, text in parent_rows:
        chunks = chunk_section_aware(text)
        for chunk in chunks:
            all_chunks.append({
                "parent_id": parent_id,
                "ticker": gap["ticker"],
                "fiscal_year": gap["fiscal_year"],
                "section_id": section_id,
                "section_title": section_title,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "heading": chunk.subsection_heading or section_title,
            })

    if not all_chunks:
        return 0

    # Generate embeddings
    texts = [c["text"] for c in all_chunks]
    embeddings = generate_embeddings(texts)

    # Insert
    cur = conn.cursor()
    for chunk, emb in zip(all_chunks, embeddings):
        cur.execute("""
            INSERT INTO sections_10k
            (ticker, fiscal_year, section_id, section_title, section_text,
             is_chunked, parent_section_id, chunk_index, chunk_start_char, chunk_end_char,
             embedding, char_count, word_count, subsection_heading)
            VALUES (%s, %s, %s, %s, %s, true, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (chunk["ticker"], chunk["fiscal_year"], chunk["section_id"],
              chunk["section_title"], chunk["text"],
              chunk["parent_id"], chunk["chunk_index"],
              chunk["start_char"], chunk["end_char"],
              emb, len(chunk["text"]), len(chunk["text"].split()),
              chunk["heading"]))

    # Mark parents as chunked
    for parent_id, _, _, _ in parent_rows:
        cur.execute("UPDATE sections_10k SET is_chunked = true WHERE id = %s", (parent_id,))

    conn.commit()
    return len(all_chunks)


def main():
    conn = get_connection()
    gaps = find_gaps(conn)
    print(f"{'=' * 60}")
    print(f"Legacy Section Backfill — {len(gaps)} ticker/year gaps")
    print(f"{'=' * 60}\n")

    total_sections = 0
    total_chunks = 0
    failures = []

    for i, gap in enumerate(gaps, 1):
        ticker = gap["ticker"]
        fy = gap["fiscal_year"]
        print(f"[{i}/{len(gaps)}] {ticker} FY{fy} ({gap['accession']})")

        try:
            # Step 1: Extract via legacy parser
            sections = extract_legacy_sections(gap)
            if not sections:
                print(f"  => No sections extracted (legacy parser returned nothing)")
                failures.append((ticker, fy, "no sections"))
                time.sleep(SEC_DELAY)
                continue

            sec_summary = ", ".join(
                f"{s['title']}={len(s['text'])//1000}k" for s in sections.values()
            )
            print(f"  Extracted: {sec_summary}")

            # Step 2: Save to filing_sections
            save_to_filing_sections(conn, gap, sections)

            # Step 3: Save parent rows to sections_10k
            parent_rows = save_to_sections_10k(conn, gap, sections)
            if not parent_rows:
                print(f"  => Already in sections_10k, skipping")
                time.sleep(SEC_DELAY)
                continue

            # Step 4: Chunk and embed
            n_chunks = chunk_and_embed_sections(conn, gap, parent_rows)
            print(f"  => {n_chunks} chunks embedded")

            total_sections += len(parent_rows)
            total_chunks += n_chunks

        except Exception as e:
            print(f"  => ERROR: {e}")
            failures.append((ticker, fy, str(e)))
            conn.rollback()

        time.sleep(SEC_DELAY)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {total_sections} sections, {total_chunks} chunks embedded")
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for t, y, reason in failures:
            print(f"  {t} FY{y}: {reason}")
    print(f"{'=' * 60}")
    conn.close()


if __name__ == "__main__":
    main()
