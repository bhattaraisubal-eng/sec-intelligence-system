"""
Section-Aware Chunking and Embedding for SEC Filings

Creates semantically meaningful chunks from risk_factors and MD&A sections,
preserving heading context and using overlap for continuity.
Generates embeddings using OpenAI text-embedding-3-small.
Uses cross-encoder reranking for improved retrieval accuracy.
"""

import os

# Suppress noisy HF tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import io
import logging
import re
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from openai import OpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Configuration from .env
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
MIN_CHUNK_SIZE = int(os.getenv("RAG_MIN_CHUNK_SIZE", "300"))
MAX_CHUNK_SIZE = int(os.getenv("RAG_MAX_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "2048"))
PARENT_CONTEXT_WINDOW = int(os.getenv("RAG_PARENT_CONTEXT_WINDOW", "1000"))

# Sections to process (only risk_factors and MD&A)
TARGET_SECTIONS_10K = ["item_1a_risk_factors", "item_7_mda"]
TARGET_SECTIONS_10Q = ["item_1a_risk_factors", "item_2_mda"]

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cross-encoder for reranking (lazy loaded)
_reranker = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker():
    """Lazy load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        # Suppress noisy safetensors/tqdm output (position_ids UNEXPECTED, progress bar)
        _old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*position_ids.*")
                warnings.filterwarnings("ignore", message=".*Tokenizer.*sequence length.*")
                logging.disable(logging.WARNING)
                _reranker = CrossEncoder(RERANKER_MODEL)
        finally:
            sys.stderr = _old_stderr
            logging.disable(logging.NOTSET)
    return _reranker


def rerank(query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank documents using cross-encoder model.

    Args:
        query: The search query
        documents: List of dicts with 'text' field and other metadata
        top_k: Number of top results to return after reranking

    Returns:
        Reranked list of documents with added 'rerank_score' field
    """
    if not documents:
        return []

    reranker = get_reranker()

    # Prepare query-document pairs for cross-encoder
    pairs = [(query, doc["text"]) for doc in documents]

    # Get cross-encoder scores
    scores = reranker.predict(pairs)

    # Add scores to documents
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    # Sort by rerank score (descending) and return top_k
    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# Connection pool
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
    db_pool = get_connection_pool()
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


@dataclass
class Subsection:
    """Represents a detected subsection within a filing section."""
    heading: str
    start_char: int
    end_char: int
    index: int


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    heading: str  # Current section heading for context
    chunk_index: int
    start_char: int
    end_char: int
    char_count: int
    word_count: int
    subsection_heading: str = ""
    subsection_index: int = 0
    subsection_start_char: int = 0
    subsection_end_char: int = 0


def detect_subsections(text: str) -> list[Subsection]:
    """Detect subsection boundaries in SEC filing text.

    Looks for title-case lines and ALL-CAPS headings that serve as
    subsection dividers within a filing section (e.g. within MD&A).

    Returns a list of Subsection objects. Falls back to a single
    subsection spanning the entire text if no headings are detected.
    """
    if not text or not text.strip():
        return []

    # Patterns for subsection headings:
    # 1. Title-case lines (5-200 chars, mostly alpha + common punctuation)
    title_case_re = re.compile(r'^[A-Z][a-zA-Z\s,&\-/()]+$')
    # 2. ALL-CAPS lines (3+ alpha chars, 80%+ uppercase, <200 chars)

    headings = []  # list of (char_pos, heading_text)
    lines = text.split('\n')
    current_pos = 0

    for line in lines:
        stripped = line.strip()
        if stripped and 5 <= len(stripped) <= 200:
            alpha_chars = [c for c in stripped if c.isalpha()]
            if len(alpha_chars) >= 3:
                upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

                # ALL-CAPS heading
                if upper_ratio > 0.8:
                    headings.append((current_pos, stripped))
                # Title-case heading
                elif title_case_re.match(stripped):
                    headings.append((current_pos, stripped))

        current_pos += len(line) + 1  # +1 for newline

    text_len = len(text)

    # Fallback: single subsection spanning entire text
    if not headings:
        return [Subsection(heading="", start_char=0, end_char=text_len, index=0)]

    # Build subsections from heading positions
    subsections = []
    for i, (pos, heading) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else text_len
        subsections.append(Subsection(
            heading=heading,
            start_char=pos,
            end_char=end,
            index=i,
        ))

    # If first heading doesn't start at 0, prepend a preamble subsection
    if subsections and subsections[0].start_char > 0:
        preamble = Subsection(
            heading="",
            start_char=0,
            end_char=subsections[0].start_char,
            index=0,
        )
        # Re-index
        for s in subsections:
            s.index += 1
        subsections.insert(0, preamble)

    return subsections


def chunk_section_aware(
    text: str,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Create subsection-aware chunks that never cross subsection boundaries.

    Strategy:
    1. Detect subsections within the text
    2. For each subsection:
       - If it fits in max_chunk_size, emit as a single chunk
       - Otherwise, split with sliding window + sentence-boundary snapping
    3. Prepend subsection heading for semantic context
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    subsections = detect_subsections(text)

    # Sentence boundary pattern
    sentence_end_re = re.compile(r'[.!?]\s+(?=[A-Z])|[.!?]\s*\n')

    all_chunks: list[Chunk] = []
    global_chunk_index = 0

    for sub in subsections:
        sub_text = text[sub.start_char:sub.end_char].strip()
        if not sub_text:
            continue

        # Prepend heading for semantic context
        heading_prefix = f"[{sub.heading}]\n\n" if sub.heading else ""

        sub_len = len(sub_text)

        if sub_len <= max_chunk_size:
            # Subsection fits in one chunk
            chunk_text = heading_prefix + sub_text
            all_chunks.append(Chunk(
                text=chunk_text,
                heading=sub.heading,
                chunk_index=global_chunk_index,
                start_char=sub.start_char,
                end_char=sub.end_char,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                subsection_heading=sub.heading,
                subsection_index=sub.index,
                subsection_start_char=sub.start_char,
                subsection_end_char=sub.end_char,
            ))
            global_chunk_index += 1
        else:
            # Split subsection with sliding window
            local_start = 0

            while local_start < sub_len:
                local_end = min(local_start + max_chunk_size, sub_len)

                # Snap to sentence boundary if not at the end
                if local_end < sub_len:
                    search_start = max(local_start, local_end - int(max_chunk_size * 0.2))
                    search_region = sub_text[search_start:local_end]
                    matches = list(sentence_end_re.finditer(search_region))
                    if matches:
                        local_end = search_start + matches[-1].end()

                fragment = sub_text[local_start:local_end].strip()
                if not fragment or len(fragment) < min_chunk_size // 2:
                    # Too small to stand alone — skip (absorbed by overlap)
                    break

                chunk_text = heading_prefix + fragment
                abs_start = sub.start_char + local_start
                abs_end = sub.start_char + local_end

                all_chunks.append(Chunk(
                    text=chunk_text,
                    heading=sub.heading,
                    chunk_index=global_chunk_index,
                    start_char=abs_start,
                    end_char=abs_end,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    subsection_heading=sub.heading,
                    subsection_index=sub.index,
                    subsection_start_char=sub.start_char,
                    subsection_end_char=sub.end_char,
                ))
                global_chunk_index += 1

                # Advance with overlap, but never cross subsection boundary
                local_start = local_end - overlap if local_end < sub_len else sub_len

    return all_chunks


def generate_embeddings(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE, return_usage: bool = False):
    """Generate embeddings for a list of texts using OpenAI API.

    Processes in batches (default 2048, the OpenAI max for text-embedding-3-small).
    The OpenAI SDK handles 429 retries automatically, so no manual sleep needed.

    When return_usage=True, returns (embeddings, usage_dict) tuple.
    """
    all_embeddings = []
    total_prompt_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )

            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Accumulate usage
            if return_usage and response.usage:
                total_prompt_tokens += response.usage.prompt_tokens

        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size}: {e}")
            # Return empty embeddings for failed batch
            all_embeddings.extend([None] * len(batch))

    if return_usage:
        return all_embeddings, {"prompt_tokens": total_prompt_tokens}
    return all_embeddings


def process_sections_10k():
    """Process 10-K risk_factors and MD&A sections (batch-first)."""
    print("\n" + "=" * 60)
    print("Processing 10-K Sections (Risk Factors & MD&A)")
    print("=" * 60)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, company_name, accession_number, filing_date,
                       fiscal_year, section_id, section_title, part_number,
                       item_number, section_description, section_text,
                       source_file_path
                FROM sections_10k
                WHERE section_id = ANY(%s)
                  AND is_chunked = FALSE
                  AND section_text IS NOT NULL
                ORDER BY ticker, fiscal_year, section_id
            """, (TARGET_SECTIONS_10K,))
            sections = cur.fetchall()

    print(f"Found {len(sections)} sections to process")

    # Phase 1: Chunk all sections (CPU-only)
    all_chunk_records = []  # list of (section_meta, chunk)
    for section in sections:
        (parent_id, ticker, company_name, accession_number, filing_date,
         fiscal_year, section_id, section_title, part_number, item_number,
         section_description, section_text, source_file_path) = section

        chunks = chunk_section_aware(section_text)
        print(f"  {ticker} {fiscal_year} {section_id}: {len(chunks)} chunks")

        meta = (parent_id, ticker, company_name, accession_number, filing_date,
                fiscal_year, section_id, section_title, part_number, item_number,
                section_description, source_file_path)
        for chunk in chunks:
            all_chunk_records.append((meta, chunk))

    if not all_chunk_records:
        print("No chunks to embed.")
        return 0

    # Phase 2: Embed all chunks in one batch
    all_texts = [rec[1].text for rec in all_chunk_records]
    print(f"\nEmbedding {len(all_texts)} chunks in batch...")
    all_embeddings = generate_embeddings(all_texts)

    # Phase 3: Write to DB
    print("Writing to database...")
    # Group by (ticker, fiscal_year, section_id) to delete old chunks per section
    from collections import defaultdict
    groups = defaultdict(list)
    for (meta, chunk), embedding in zip(all_chunk_records, all_embeddings):
        if embedding is None:
            continue
        key = (meta[1], meta[5], meta[6])  # ticker, fiscal_year, section_id
        groups[key].append((meta, chunk, embedding))

    total_chunks = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for (ticker, fiscal_year, section_id), records in groups.items():
                cur.execute("""
                    DELETE FROM sections_10k
                    WHERE ticker = %s AND fiscal_year = %s
                      AND section_id = %s AND is_chunked = TRUE
                """, (ticker, fiscal_year, section_id))

                for meta, chunk, embedding in records:
                    (parent_id, _, company_name, accession_number, filing_date,
                     _, _, section_title, part_number, item_number,
                     section_description, source_file_path) = meta

                    cur.execute("""
                        INSERT INTO sections_10k
                        (ticker, company_name, accession_number, filing_date,
                         fiscal_year, section_id, section_title, part_number,
                         item_number, section_description, section_text,
                         char_count, word_count, embedding, is_chunked,
                         parent_section_id, chunk_index, chunk_start_char,
                         chunk_end_char, subsection_heading, subsection_index,
                         subsection_start_char, subsection_end_char,
                         source_file_path)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, TRUE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        ticker, company_name, accession_number, filing_date,
                        fiscal_year, section_id, section_title, part_number,
                        item_number, section_description, chunk.text,
                        chunk.char_count, chunk.word_count, embedding,
                        parent_id, chunk.chunk_index, chunk.start_char,
                        chunk.end_char, chunk.subsection_heading or None,
                        chunk.subsection_index, chunk.subsection_start_char,
                        chunk.subsection_end_char, source_file_path
                    ))

                total_chunks += len(records)
            conn.commit()

    print(f"\n10-K Processing Complete: {total_chunks} total chunks created")
    return total_chunks


def process_sections_10q():
    """Process 10-Q risk_factors and MD&A sections (batch-first)."""
    print("\n" + "=" * 60)
    print("Processing 10-Q Sections (Risk Factors & MD&A)")
    print("=" * 60)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, company_name, accession_number, filing_date,
                       fiscal_year, fiscal_quarter, fiscal_period, section_id,
                       section_title, part_number, item_number, section_description,
                       section_text, source_file_path
                FROM sections_10q
                WHERE section_id = ANY(%s)
                  AND is_chunked = FALSE
                  AND section_text IS NOT NULL
                ORDER BY ticker, fiscal_year, fiscal_quarter, section_id
            """, (TARGET_SECTIONS_10Q,))
            sections = cur.fetchall()

    print(f"Found {len(sections)} sections to process")

    # Phase 1: Chunk all sections (CPU-only)
    all_chunk_records = []
    for section in sections:
        (parent_id, ticker, company_name, accession_number, filing_date,
         fiscal_year, fiscal_quarter, fiscal_period, section_id, section_title,
         part_number, item_number, section_description, section_text,
         source_file_path) = section

        chunks = chunk_section_aware(section_text)
        print(f"  {ticker} {fiscal_year} Q{fiscal_quarter} {section_id}: {len(chunks)} chunks")

        meta = (parent_id, ticker, company_name, accession_number, filing_date,
                fiscal_year, fiscal_quarter, fiscal_period, section_id, section_title,
                part_number, item_number, section_description, source_file_path)
        for chunk in chunks:
            all_chunk_records.append((meta, chunk))

    if not all_chunk_records:
        print("No chunks to embed.")
        return 0

    # Phase 2: Embed all chunks in one batch
    all_texts = [rec[1].text for rec in all_chunk_records]
    print(f"\nEmbedding {len(all_texts)} chunks in batch...")
    all_embeddings = generate_embeddings(all_texts)

    # Phase 3: Write to DB
    print("Writing to database...")
    from collections import defaultdict
    groups = defaultdict(list)
    for (meta, chunk), embedding in zip(all_chunk_records, all_embeddings):
        if embedding is None:
            continue
        key = (meta[1], meta[5], meta[6], meta[8])  # ticker, fiscal_year, fiscal_quarter, section_id
        groups[key].append((meta, chunk, embedding))

    total_chunks = 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for (ticker, fiscal_year, fiscal_quarter, section_id), records in groups.items():
                cur.execute("""
                    DELETE FROM sections_10q
                    WHERE ticker = %s AND fiscal_year = %s
                      AND fiscal_quarter = %s AND section_id = %s
                      AND is_chunked = TRUE
                """, (ticker, fiscal_year, fiscal_quarter, section_id))

                for meta, chunk, embedding in records:
                    (parent_id, _, company_name, accession_number, filing_date,
                     _, _, fiscal_period, _, section_title,
                     part_number, item_number, section_description,
                     source_file_path) = meta

                    cur.execute("""
                        INSERT INTO sections_10q
                        (ticker, company_name, accession_number, filing_date,
                         fiscal_year, fiscal_quarter, fiscal_period, section_id,
                         section_title, part_number, item_number, section_description,
                         section_text, char_count, word_count, embedding, is_chunked,
                         parent_section_id, chunk_index, chunk_start_char,
                         chunk_end_char, subsection_heading, subsection_index,
                         subsection_start_char, subsection_end_char,
                         source_file_path)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, TRUE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        ticker, company_name, accession_number, filing_date,
                        fiscal_year, fiscal_quarter, fiscal_period, section_id,
                        section_title, part_number, item_number, section_description,
                        chunk.text, chunk.char_count, chunk.word_count, embedding,
                        parent_id, chunk.chunk_index, chunk.start_char,
                        chunk.end_char, chunk.subsection_heading or None,
                        chunk.subsection_index, chunk.subsection_start_char,
                        chunk.subsection_end_char, source_file_path
                    ))

                total_chunks += len(records)
            conn.commit()

    print(f"\n10-Q Processing Complete: {total_chunks} total chunks created")
    return total_chunks


def show_chunk_stats():
    """Display statistics about chunked sections."""
    print("\n" + "=" * 60)
    print("CHUNK STATISTICS")
    print("=" * 60)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 10-K chunk stats
            cur.execute("""
                SELECT ticker, fiscal_year, section_id,
                       COUNT(*) as chunk_count,
                       SUM(char_count) as total_chars,
                       AVG(char_count)::int as avg_chars
                FROM sections_10k
                WHERE is_chunked = TRUE
                GROUP BY ticker, fiscal_year, section_id
                ORDER BY ticker, fiscal_year, section_id
            """)
            stats_10k = cur.fetchall()

            # 10-Q chunk stats
            cur.execute("""
                SELECT ticker, fiscal_year, fiscal_quarter, section_id,
                       COUNT(*) as chunk_count,
                       SUM(char_count) as total_chars,
                       AVG(char_count)::int as avg_chars
                FROM sections_10q
                WHERE is_chunked = TRUE
                GROUP BY ticker, fiscal_year, fiscal_quarter, section_id
                ORDER BY ticker, fiscal_year, fiscal_quarter, section_id
            """)
            stats_10q = cur.fetchall()

            # Count chunks with embeddings
            cur.execute("""
                SELECT COUNT(*) FROM sections_10k
                WHERE is_chunked = TRUE AND embedding IS NOT NULL
            """)
            embedded_10k = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM sections_10q
                WHERE is_chunked = TRUE AND embedding IS NOT NULL
            """)
            embedded_10q = cur.fetchone()[0]

    print("\n10-K Chunks:")
    print(f"  {'Ticker':<8} {'Year':<6} {'Section':<25} {'Chunks':>8} {'Total Chars':>12} {'Avg Chars':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*25} {'-'*8} {'-'*12} {'-'*10}")
    for row in stats_10k:
        print(f"  {row[0]:<8} {row[1]:<6} {row[2]:<25} {row[3]:>8} {row[4]:>12,} {row[5]:>10}")
    print(f"\n  Total chunks with embeddings: {embedded_10k}")

    print("\n10-Q Chunks:")
    print(f"  {'Ticker':<8} {'Year':<6} {'Qtr':<4} {'Section':<20} {'Chunks':>8} {'Total Chars':>12} {'Avg Chars':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*4} {'-'*20} {'-'*8} {'-'*12} {'-'*10}")
    for row in stats_10q:
        print(f"  {row[0]:<8} {row[1]:<6} Q{row[2]:<3} {row[3]:<20} {row[4]:>8} {row[5]:>12,} {row[6]:>10}")
    print(f"\n  Total chunks with embeddings: {embedded_10q}")


def _expand_with_parent_context(
    chunks: list[dict],
    table: str,
    window: int = PARENT_CONTEXT_WINDOW,
) -> list[dict]:
    """Expand chunk text with surrounding context from the parent section.

    Implements small-to-big retrieval: embeddings are matched on small chunks
    for precision, but a larger window from the parent section is passed to
    the LLM for richer context.

    Args:
        chunks: List of chunk dicts from semantic search (must include
            parent_section_id, chunk_start_char, chunk_end_char).
        table: DB table name ('sections_10k' or 'sections_10q').
        window: Number of characters to expand on each side of the chunk.

    Returns:
        The same list with an 'expanded_text' field added to each chunk.
    """
    if not chunks or window <= 0:
        return chunks

    # Collect unique parent IDs
    parent_ids = set()
    for c in chunks:
        pid = c.get("parent_section_id")
        if pid is not None:
            parent_ids.add(pid)

    if not parent_ids:
        return chunks

    # Batch-fetch parent section texts
    parent_texts = {}
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, section_text FROM {table}
                WHERE id = ANY(%s) AND is_chunked = FALSE
            """, (list(parent_ids),))
            for row in cur.fetchall():
                parent_texts[row[0]] = row[1]

    # Sentence boundary pattern for snapping
    sentence_start_re = re.compile(r'\.\s+')

    for chunk in chunks:
        pid = chunk.get("parent_section_id")
        start = chunk.get("chunk_start_char")
        end = chunk.get("chunk_end_char")

        if pid and pid in parent_texts and start is not None and end is not None:
            parent_text = parent_texts[pid]
            parent_len = len(parent_text)

            # Clamp expansion to subsection bounds when available
            sub_start = chunk.get("subsection_start_char")
            sub_end = chunk.get("subsection_end_char")
            if sub_start is not None and sub_end is not None:
                bound_start = sub_start
                bound_end = sub_end
            else:
                bound_start = 0
                bound_end = parent_len

            # Expand window within bounds
            exp_start = max(bound_start, start - window)
            exp_end = min(bound_end, end + window)

            # Snap to sentence boundaries to avoid mid-sentence cuts
            if exp_start > bound_start:
                match = sentence_start_re.search(parent_text, exp_start)
                if match and match.end() < start:
                    exp_start = match.end()

            if exp_end < bound_end:
                search_region = parent_text[end:exp_end]
                match = None
                for m in sentence_start_re.finditer(search_region):
                    match = m
                if match:
                    exp_end = end + match.end()

            expanded = parent_text[exp_start:exp_end].strip()

            # Prepend subsection heading (more specific) or section title
            heading = chunk.get("subsection_heading") or chunk.get("section_title") or chunk.get("section_id", "")
            if heading and not expanded[:100].upper().startswith(heading.upper()[:20]):
                expanded = f"[{heading}]\n\n{expanded}"

            chunk["expanded_text"] = expanded

    return chunks


def semantic_search_10k(
    query: str,
    ticker: str = None,
    fiscal_year: int = None,
    top_k: int = 5,
    use_reranking: bool = True,
    initial_candidates: int = 20,
    section_ids: list[str] | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """
    Perform semantic search on 10-K chunks with optional cross-encoder reranking.

    Args:
        query: Search query
        ticker: Filter by ticker symbol (optional)
        fiscal_year: Filter by fiscal year (optional)
        top_k: Number of final results to return
        use_reranking: Whether to apply cross-encoder reranking
        initial_candidates: Number of candidates to retrieve before reranking
        section_ids: Filter by section_id values (e.g. ["item_1a_risk_factors"])
        query_embedding: Pre-computed embedding vector (skips OpenAI API call if provided)

    Returns:
        List of matching chunks with similarity/rerank scores
    """
    # Use pre-computed embedding or generate one
    if query_embedding is None:
        embeddings = generate_embeddings([query])
        if not embeddings or embeddings[0] is None:
            return []
        query_embedding = embeddings[0]

    # Retrieve more candidates if reranking
    retrieve_count = initial_candidates if use_reranking else top_k

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Build dynamic WHERE clause based on filters
            conditions = ["is_chunked = TRUE", "embedding IS NOT NULL"]
            params = [query_embedding]

            if ticker:
                conditions.append("ticker = %s")
                params.append(ticker)
            if fiscal_year:
                conditions.append("fiscal_year = %s")
                params.append(fiscal_year)
            if section_ids:
                conditions.append("section_id = ANY(%s)")
                params.append(section_ids)

            where_clause = " AND ".join(conditions)
            params.extend([query_embedding, retrieve_count])

            cur.execute(f"""
                SELECT id, ticker, fiscal_year, section_id, section_title,
                       section_text, 1 - (embedding <=> %s::vector) as similarity,
                       parent_section_id, chunk_start_char, chunk_end_char,
                       accession_number
                FROM sections_10k
                WHERE {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, params)

            results = cur.fetchall()

    documents = [
        {
            "id": r[0],
            "ticker": r[1],
            "fiscal_year": r[2],
            "section_id": r[3],
            "section_title": r[4],
            "text": r[5],
            "similarity": float(r[6]),
            "parent_section_id": r[7],
            "chunk_start_char": r[8],
            "chunk_end_char": r[9],
            "accession_number": r[10],
        }
        for r in results
    ]

    # Apply cross-encoder reranking (scores on small chunk text)
    if use_reranking and documents:
        documents = rerank(query, documents, top_k=top_k)

    documents = documents[:top_k]

    # Expand to parent context window (small-to-big retrieval)
    documents = _expand_with_parent_context(documents, "sections_10k")

    return documents


def semantic_search_10q(
    query: str,
    ticker: str = None,
    fiscal_year: int = None,
    fiscal_quarter: int = None,
    top_k: int = 5,
    use_reranking: bool = True,
    initial_candidates: int = 20,
    section_ids: list[str] | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """
    Perform semantic search on 10-Q chunks with optional cross-encoder reranking.

    Args:
        query: Search query
        ticker: Filter by ticker symbol (optional)
        fiscal_year: Filter by fiscal year (optional)
        fiscal_quarter: Filter by fiscal quarter (optional)
        top_k: Number of final results to return
        use_reranking: Whether to apply cross-encoder reranking
        initial_candidates: Number of candidates to retrieve before reranking
        section_ids: Filter by section_id values (e.g. ["item_1a_risk_factors"])
        query_embedding: Pre-computed embedding vector (skips OpenAI API call if provided)

    Returns:
        List of matching chunks with similarity/rerank scores
    """
    # Use pre-computed embedding or generate one
    if query_embedding is None:
        embeddings = generate_embeddings([query])
        if not embeddings or embeddings[0] is None:
            return []
        query_embedding = embeddings[0]

    # Retrieve more candidates if reranking
    retrieve_count = initial_candidates if use_reranking else top_k

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Build dynamic WHERE clause based on filters
            conditions = ["is_chunked = TRUE", "embedding IS NOT NULL"]
            params = [query_embedding]

            if ticker:
                conditions.append("ticker = %s")
                params.append(ticker)
            if fiscal_year:
                conditions.append("fiscal_year = %s")
                params.append(fiscal_year)
            if fiscal_quarter:
                conditions.append("fiscal_quarter = %s")
                params.append(fiscal_quarter)
            if section_ids:
                conditions.append("section_id = ANY(%s)")
                params.append(section_ids)

            where_clause = " AND ".join(conditions)
            params.extend([query_embedding, retrieve_count])

            cur.execute(f"""
                SELECT id, ticker, fiscal_year, fiscal_quarter, section_id,
                       section_title, section_text,
                       1 - (embedding <=> %s::vector) as similarity,
                       parent_section_id, chunk_start_char, chunk_end_char,
                       accession_number
                FROM sections_10q
                WHERE {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, params)

            results = cur.fetchall()

    documents = [
        {
            "id": r[0],
            "ticker": r[1],
            "fiscal_year": r[2],
            "fiscal_quarter": r[3],
            "section_id": r[4],
            "section_title": r[5],
            "text": r[6],
            "similarity": float(r[7]),
            "parent_section_id": r[8],
            "chunk_start_char": r[9],
            "chunk_end_char": r[10],
            "accession_number": r[11],
        }
        for r in results
    ]

    # Apply cross-encoder reranking (scores on small chunk text)
    if use_reranking and documents:
        documents = rerank(query, documents, top_k=top_k)

    documents = documents[:top_k]

    # Expand to parent context window (small-to-big retrieval)
    documents = _expand_with_parent_context(documents, "sections_10q")

    return documents


if __name__ == "__main__":
    print("=" * 60)
    print("SEC Filing Chunk & Embed Pipeline")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"  Chunk Size: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} chars (variable)")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP} chars")
    print(f"  Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
    print(f"\nTarget Sections:")
    print(f"  10-K: {TARGET_SECTIONS_10K}")
    print(f"  10-Q: {TARGET_SECTIONS_10Q}")

    # Process 10-K sections
    chunks_10k = process_sections_10k()

    # Process 10-Q sections
    chunks_10q = process_sections_10q()

    # Show statistics
    show_chunk_stats()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total chunks created: {chunks_10k + chunks_10q}")
    print(f"  10-K: {chunks_10k}")
    print(f"  10-Q: {chunks_10q}")
