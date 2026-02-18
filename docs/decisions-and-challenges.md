# Key Decisions & Challenges

This document captures the engineering decisions, technical challenges, and domain-specific complexities encountered while building the SEC RAG system — from raw data ingestion through answer generation. It reflects the iterative problem-solving process behind a production-grade financial AI system.

---

## Table of Contents

1. [Data Ingestion Challenges](#1-data-ingestion-challenges)
   - [SEC EDGAR Data Quality](#11-sec-edgar-data-quality)
   - [Custom Parser vs. edgartools](#12-custom-parser-vs-edgartools)
   - [Ticker-Specific Failures](#13-ticker-specific-failures)
2. [Domain-Specific Challenges](#2-domain-specific-challenges)
   - [Non-Calendar Fiscal Years](#21-non-calendar-fiscal-years)
   - [52/53-Week Fiscal Calendars](#22-5253-week-fiscal-calendars)
   - [Q4 Derivation Problem](#23-the-q4-derivation-problem)
   - [XBRL Concept Renames](#24-xbrl-concept-renames-across-years)
   - [IPOs, Mergers, and Corporate Events](#25-ipos-mergers-and-corporate-events)
   - [YTD Fact Contamination](#26-ytd-fact-contamination)
   - [Consolidated vs. Segment Data](#27-consolidated-vs-segment-data)
3. [Database Design Decisions](#3-database-design-decisions)
   - [Why PostgreSQL + pgvector](#31-why-postgresql--pgvector)
   - [Table Schema Rationale](#32-table-schema-rationale)
   - [Indexing Strategy](#33-indexing-strategy)
   - [Connection Pool Pattern](#34-connection-pool-pattern)
4. [RAG Architecture Design](#4-rag-architecture-design)
   - [Chunking Strategy](#41-chunking-strategy)
   - [Embedding Model Selection](#42-embedding-model-selection)
   - [Reranking with Cross-Encoders](#43-reranking-with-cross-encoders)
   - [Why 5 Retrieval Routes](#44-why-5-retrieval-routes)
5. [Query Routing & Edge Cases](#5-query-routing--edge-cases)
   - [LLM-Based Classification](#51-llm-based-classification)
   - [Route Override Heuristics](#52-route-override-heuristics)
   - [Multi-Ticker Query Decomposition](#53-multi-ticker-query-decomposition)
   - [Latest Year Resolution](#54-latest-year-resolution)
   - [Multi-Ticker Reranking Fairness](#55-multi-ticker-reranking-fairness)
   - [Year-Quarter Mapping](#56-year-quarter-mapping)
6. [Answer Generation & Evaluation](#6-answer-generation--evaluation)
   - [Context Formatting by Route](#61-context-formatting-by-route)
   - [Prompt Engineering for Financial Accuracy](#62-prompt-engineering-for-financial-accuracy)
   - [Citation System](#63-citation-system)
   - [Contradiction Detection](#64-contradiction-detection)
   - [Confidence Scoring System](#65-confidence-scoring-system)
   - [Cost Tracking & Efficiency Grading](#66-cost-tracking--efficiency-grading)
7. [Caching Architecture](#7-caching-architecture)
8. [Lessons Learned](#8-lessons-learned)

---

## 1. Data Ingestion Challenges

### 1.1 SEC EDGAR Data Quality

SEC EDGAR is the authoritative source for public company filings, but working with it at scale revealed significant data quality issues:

- **Inconsistent filing formats**: Filings span decades of format changes. Older filings use HTML with embedded formatting; newer ones use Inline XBRL. There is no single parsing strategy that works across all years.
- **Missing or malformed XBRL tags**: Some filings have incomplete XBRL tagging. A company might report revenue in their financial statements but not tag it properly in XBRL, meaning programmatic extraction fails silently.
- **Rate limiting**: SEC EDGAR enforces strict rate limits (10 requests/second with a declared User-Agent). The pipeline uses a 0.15-second delay between API calls, which means ingesting 15+ years of filings across 10 tickers takes hours, not minutes.
- **Filing amendments**: Companies file amendments (10-K/A, 10-Q/A) that supersede original filings. The pipeline must handle deduplication by accession number to avoid double-counting.

### 1.2 Custom Parser vs. edgartools

**Initial approach**: Built a custom XBRL parser to extract financial facts from SEC EDGAR JSON endpoints. The parser worked for straightforward filings but quickly ran into problems:

- **Fact period classification** was unreliable — a single 10-K filing contains annual facts, quarterly comparatives, and year-to-date facts, all interleaved. Sorting these into the correct buckets required understanding each fact's duration, not just the filing type.
- **Missing data** was rampant. The custom parser would silently drop facts it couldn't classify, leading to gaps in the database that only surfaced when users asked questions and got empty results.
- **Section extraction** was fragile — HTML parsing of filing documents broke across different formatting styles.

**Decision**: Switched to [edgartools](https://github.com/dgunning/edgartools), an open-source library specifically designed for SEC EDGAR data. This provided:
- Reliable XBRL fact extraction with proper period handling
- Built-in support for quarterly/annual classification
- Section extraction from filing documents
- TTM (Trailing Twelve Months) calculation utilities for Q4 derivation

However, edgartools was not a silver bullet — it still struggled with certain filings (see below).

### 1.3 Ticker-Specific Failures

**JPMorgan (JPM)**: Bank filings are structurally different from tech companies. JPM's XBRL taxonomy includes banking-specific concepts (loan loss provisions, risk-weighted assets, tier 1 capital) that don't map to standard `us-gaap:` tags. Even edgartools' parser sometimes failed on JPM's complex filing structure. The legacy custom parser also couldn't handle JPM's multi-entity reporting structure.

**Solution**: For concepts not in the standard CONCEPT_ALIASES mapping, the system falls back to fuzzy database search using `ILIKE` pattern matching against actual concepts stored in the DB:

```python
def search_concepts(term, table="annual_facts", limit=5):
    """Fuzzy ILIKE search against actual DB concepts."""
    cur.execute(
        f"SELECT DISTINCT concept FROM {table} WHERE concept ILIKE %s LIMIT %s",
        (f"%{term}%", limit),
    )
```

**Berkshire Hathaway (BRK-B)**: The ticker format with a hyphen required careful handling throughout the pipeline — some APIs don't accept hyphens, SEC EDGAR uses a different format, and the LLM sometimes outputs "BRK.B" or "BRKB". The company-name-to-ticker mapping handles these aliases:

```python
_COMPANY_NAME_TO_TICKER = {
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpmorgan chase": "JPM", "chase": "JPM",
}
```

---

## 2. Domain-Specific Challenges

### 2.1 Non-Calendar Fiscal Years

Only 4 of the 10 covered companies use a December fiscal year-end. The others have fiscal years that don't align with the calendar:

| Ticker | FY End Month | Implication |
|--------|-------------|-------------|
| AAPL | September | FY2024 covers Oct 2023 - Sep 2024 |
| MSFT | June | FY2024 covers Jul 2023 - Jun 2024 |
| NVDA | January | FY2025 covers Feb 2024 - Jan 2025 |
| AVGO | October | FY2024 covers Nov 2023 - Oct 2024 |

This means a user asking "What was NVIDIA's revenue in 2024?" is actually asking about FY2025 in NVIDIA's filing terminology, because NVIDIA's fiscal year ending in January 2025 covers most of calendar year 2024.

**Solution**: A `calendar_to_fiscal_year()` mapping and `infer_fiscal_year()` function that uses the filing's `period_of_report` (not just the filing date) combined with the company's fiscal year-end month:

```python
def infer_fiscal_year(ticker, filing):
    fy_end = FISCAL_YEAR_END_MONTH[ticker]  # e.g., 1 for NVDA
    period = filing.period_of_report
    # If period month > FY end month, this period belongs to FY = year + 1
    if period.month > fy_end:
        return period.year + 1
    return period.year
```

The LLM classifier is instructed about these mappings in its system prompt, but the real safeguard is the programmatic mapping that runs regardless of LLM output.

### 2.2 52/53-Week Fiscal Calendars

Apple and Broadcom use 52/53-week fiscal calendars rather than exact month-end dates. This means their fiscal quarter endings can spill 1-7 days into the next calendar month. For example, Apple's Q1 FY2024 ended on December 30, 2023 — not December 31.

This created quarter-misclassification bugs: a period ending January 2 would be classified as Q2 instead of Q1 because the raw month was January.

**Solution**: A day-of-month rollback heuristic:

```python
def infer_fiscal_quarter(ticker, filing):
    period = filing.period_of_report
    month = period.month
    # Handle 52/53-week fiscal calendars where period end dates
    # can spill 1-7 days into the next month
    if period.day <= 7:
        month = 12 if month == 1 else month - 1
```

This correctly handles the 4-7 day spillover that affects ~5% of Apple's and Broadcom's quarterly filings.

### 2.3 The Q4 Derivation Problem

The SEC does not require companies to file a separate 10-Q for Q4. Instead, Q4 data is embedded in the annual 10-K filing. This means you cannot simply query `quarterly_facts WHERE fiscal_quarter = 4` — that row often doesn't exist.

**Solution**: Three complementary approaches:

1. **Arithmetic derivation**: `Q4 = Annual - Q1 - Q2 - Q3`. This requires all three quarterly filings to be present. If any quarter is missing, the derivation fails gracefully.

2. **edgartools TTM calculator**: Uses the library's built-in quarterization logic, which handles edge cases like balance sheet items (which are point-in-time, not duration-based — for these, Q4 = Annual value, not a subtraction).

3. **Income statement vs. balance sheet distinction**: Revenue is a flow metric (Q4 = Annual - YTD Q3), but total assets is a stock metric (Q4 = Annual period-end value). The derivation logic branches on concept type:

```python
# Income Statement: Q4 = Annual - Q1 - Q2 - Q3
# Balance Sheet: Q4 = Annual (point-in-time snapshot)
# Cash Flow: Q4 = Annual - Q3 YTD
```

### 2.4 XBRL Concept Renames Across Years

The XBRL taxonomy evolves. What was `us-gaap:SalesRevenueNet` in 2015 became `us-gaap:Revenues` in 2018 and then `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` after ASC 606 adoption. A timeseries query for "Apple's revenue 2015-2024" must search across all three concept names.

**Solution**: A multi-layer concept resolution chain:

```python
CONCEPT_ALIASES = {
    "revenue": [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",  # Post-ASC 606
        "us-gaap:Revenues",                                             # Intermediate
        "us-gaap:SalesRevenueNet",                                      # Legacy
        "us-gaap:SalesRevenueGoodsNet",                                 # Some retailers
    ],
}
```

The `get_metric_timeseries()` function accepts a list of concept aliases and merges results, using `DISTINCT ON (fiscal_year, fiscal_quarter)` with ordering by `end_date DESC` to pick the most relevant value when multiple concepts match the same period.

Some companies also use custom namespace concepts. For example, NVIDIA uses `nvda:PurchasesOfPropertyAndEquipmentAndIntangibleAssets` for capital expenditures instead of the standard `us-gaap:PaymentsToAcquirePropertyPlantAndEquipment`. These ticker-specific concepts are included in the alias mappings.

### 2.5 IPOs, Mergers, and Corporate Events

Not all companies have data spanning the full 2010-present range:

- **Meta (Facebook)**: IPO was May 2012. Queries for "Meta's revenue in 2010" must return an explicit explanation, not an empty result.
- **Broadcom (AVGO)**: IPO as Avago Technologies in August 2009, then merged with Broadcom Corporation in 2016. Pre-merger data uses different entity structures.

**Solution**: An `_IPO_YEAR` dictionary with human-readable explanations:

```python
_IPO_YEAR = {
    "META": (2012, "Meta (Facebook) had its IPO in May 2012. No SEC filings exist before 2012."),
    "AVGO": (2009, "Broadcom (as Avago Technologies) had its IPO in August 2009."),
}
```

The `check_data_availability()` function checks requested years against IPO dates and injects warnings into the LLM context, with explicit instructions: "Do NOT silently omit a company from the comparison — explain WHY data is unavailable."

### 2.6 YTD Fact Contamination

A single 10-K filing contains not just annual facts but also quarterly comparatives and year-to-date (YTD) aggregations. Without proper filtering, a query for "FY2023 revenue" might return the YTD-9-month figure instead of the full-year figure.

**Solution**: Duration-based fact classification:

```python
def classify_fact_period(fact):
    duration_days = (end_date - start_date).days
    if 70 <= duration_days <= 120:    # ~3 months → quarterly
        return "quarterly", infer_quarter()
    elif 330 <= duration_days <= 420:  # ~12 months → annual
        return "annual", None
    else:                              # 6-month, 9-month YTD → skip
        return None, None              # Deliberately excluded
```

Facts with durations between 120 and 330 days (YTD periods) are dropped entirely. This prevents silent data contamination while preserving the correct annual and quarterly values.

### 2.7 Consolidated vs. Segment Data

XBRL filings report the same metric multiple times with different dimensional breakdowns: consolidated total, by geographic segment, by product line, by operating segment. Without filtering, a revenue query returns 5-10 values instead of one.

**Solution**: Default to consolidated totals by filtering `WHERE dimension IS NULL`:

```sql
CREATE INDEX idx_annual_consolidated ON annual_facts(ticker, concept, fiscal_year)
    INCLUDE (value, unit) WHERE dimension IS NULL;
```

This partial index serves double duty: it enforces the filtering logic at the query level and dramatically speeds up the most common query pattern. Segment data is available via `include_segments=True` for queries that specifically request geographic or product breakdowns.

---

## 3. Database Design Decisions

### 3.1 Why PostgreSQL + pgvector

**Considered alternatives**:
- **Pinecone/Weaviate** (dedicated vector DB): Would require a separate database for structured data, adding operational complexity and cost. Financial queries are 60%+ relational (XBRL lookups, timeseries), not just vector search.
- **SQLite + FAISS**: No concurrent access, no full-text search, harder to deploy.
- **MongoDB**: No native vector search (at the time), weaker relational query support for the structured XBRL data that dominates the workload.

**Why PostgreSQL + pgvector won**:
1. **Single database** for both structured (XBRL facts, financial statements) and unstructured (embeddings) data. This eliminates cross-database joins and simplifies the deployment.
2. **pgvector** provides cosine similarity search with IVFFlat indexing, which is sufficient for ~130K embeddings (sections_10k + sections_10q).
3. **Full-text search** via GIN indexes supplements vector search for keyword-heavy queries.
4. **Mature tooling**: `pg_dump`/`pg_restore` for migrations, connection pooling, ACID transactions for idempotent ingestion.
5. **Cost**: A single PostgreSQL instance on Railway costs ~$2-3/month for this dataset. Pinecone's starter plan would cost more for equivalent vector storage alone, and you'd still need a relational DB.

### 3.2 Table Schema Rationale

**8 tables** serve distinct purposes in the retrieval pipeline:

| Table | Purpose | Why Separate |
|-------|---------|-------------|
| `filings` | Filing metadata (accession numbers, dates) | Central reference for all data; FK target for deduplication |
| `annual_facts` | XBRL facts from 10-K | Separated from quarterly for query performance — annual queries never scan quarterly data and vice versa |
| `quarterly_facts` | XBRL facts from 10-Q | Same concept schema but with `fiscal_quarter` column; different query patterns |
| `earnings_reports` | 8-K earnings summaries | Structured earnings data with pre-extracted metrics (revenue, EPS, etc.); avoids re-parsing 8-K text |
| `filing_sections` | Raw section text | Intermediate storage from section extraction; used as source for chunking pipeline |
| `sections_10k` | 10-K chunks + embeddings | Vector search table with pgvector; includes parent-child chunk relationships |
| `sections_10q` | 10-Q chunks + embeddings | Separate from 10-K because 10-Q sections have `fiscal_quarter` and different section structure |
| `financial_documents` | Full financial statements as markdown | Used by the `full_statement` route; stored as pre-formatted markdown tables for direct LLM context injection |

**Why separate annual/quarterly tables instead of one table with a nullable quarter column?**

Performance. The `annual_facts` table has 360K rows and `quarterly_facts` has 734K rows. Annual queries (the majority) would waste time scanning quarterly rows behind a `WHERE fiscal_quarter IS NULL` filter. Separate tables with dedicated indexes give cleaner query plans and faster lookups. The covering indexes (`INCLUDE (value, unit)`) enable index-only scans for the most common access pattern.

**Why separate 10-K/10-Q vector tables?**

The section structures are different (10-K has Items 1-15, 10-Q has Parts I-II with different items), they have different temporal columns (`fiscal_period` vs `fiscal_quarter`), and they have different uniqueness constraints. Combining them into one table would require complex conditional logic throughout the retrieval code.

### 3.3 Indexing Strategy

The indexing strategy reflects actual query patterns, not theoretical optimization:

- **Hash indexes** on `concept` columns: XBRL concept lookups are always exact-match (`WHERE concept = 'us-gaap:Revenues'`). Hash indexes are faster than B-tree for equality checks.
- **Covering indexes** with `INCLUDE (value, unit)`: The most common query pattern is "get value and unit for ticker + concept + year." Covering indexes serve this entirely from the index without touching the heap.
- **Partial indexes** with `WHERE dimension IS NULL`: Consolidated-only queries (the default) skip segment data at the index level.
- **IVFFlat vector indexes** with `lists=50`: Originally set to 100 lists, reduced to 50 for Railway's memory constraints (64 MB `maintenance_work_mem`). With ~42K 10-K chunks, 50 lists gives ~840 vectors per list, which is within the recommended range (rows/lists between 100-1000).
- **GIN full-text indexes**: Supplement vector search for queries with specific terminology that benefits from keyword matching.

### 3.4 Connection Pool Pattern

All database modules share a lazy singleton `ThreadedConnectionPool` pattern:

```python
_connection_pool = None

def get_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=1, maxconn=10, ...)
    return _connection_pool
```

**Why this pattern**:
- **Lazy initialization**: The pool isn't created until the first database query, avoiding startup failures if the DB is temporarily unavailable.
- **Thread-safe**: FastAPI uses async workers that may share connections across coroutines.
- **minconn=1, maxconn=10**: One idle connection for responsiveness; up to 10 for concurrent requests. Railway's PostgreSQL default allows 100 connections, so 10 is conservative.
- **No connection timeout**: Connections are returned to the pool via context managers, never left dangling.

---

## 4. RAG Architecture Design

### 4.1 Chunking Strategy

**Decision: Section-aware chunking with subsection detection**

Naive fixed-size chunking (e.g., 512 tokens) destroys the logical structure of SEC filings. A risk factor that spans 3 paragraphs gets split mid-sentence, and the chunk boundary falls in the middle of a quantitative claim.

**Approach**:

1. **Subsection detection**: Regex-based heading detection identifies logical boundaries within sections (title-case lines, ALL-CAPS headings). Each subsection becomes a chunk candidate.

2. **Size-based splitting with sentence boundary snapping**: If a subsection exceeds `MAX_CHUNK_SIZE` (1500 chars), it's split using a sliding window. Critically, the split point snaps to the nearest sentence boundary:

```python
# Search for the last sentence ending in the region
# Region: [80% of chunk_end, chunk_end]
pattern = r'[.!?]\s+(?=[A-Z])|[.!?]\s*\n'
snap_region = text[max(start, end - 0.2 * max_size):end]
match = re.search(pattern, snap_region)
```

3. **Overlap**: 150-character overlap between consecutive chunks ensures that context isn't lost at boundaries.

4. **Minimum fragment filtering**: Chunks smaller than `MIN_CHUNK_SIZE / 2` (150 chars) are discarded — these are typically table headers or whitespace remnants.

**Which sections are embedded**:

Only **Risk Factors** (Item 1A) and **MD&A** (Item 7 for 10-K, Item 2 for 10-Q) are chunked and embedded. These sections contain the narrative content that benefits from semantic search. Financial statements (Item 8) are stored as structured markdown in `financial_documents` for direct retrieval — embedding a table of numbers doesn't improve semantic search relevance.

### 4.2 Embedding Model Selection

**Selected**: OpenAI `text-embedding-3-small` (1536 dimensions)

**Why not larger models** (`text-embedding-3-large`, 3072 dims):
- Storage doubles (each row in `sections_10k` grows by 6 KB).
- IVFFlat index build time increases significantly.
- Marginal relevance improvement for domain-specific financial text doesn't justify the cost at ~130K embeddings.

**Why not open-source models** (e.g., `all-MiniLM-L6-v2`):
- OpenAI's model showed better out-of-the-box performance on financial terminology (tested informally on revenue/risk queries).
- The cross-encoder reranking step compensates for any embedding quality gaps.

### 4.3 Reranking with Cross-Encoders

**Selected**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Vector search retrieves the top ~100 candidates per (ticker, year) pair. These are then reranked using a cross-encoder that scores each (query, chunk) pair directly, accounting for fine-grained semantic relevance that bi-encoder embeddings miss.

**Why reranking matters for financial text**: A query about "revenue growth drivers" might have high cosine similarity with chunks discussing "revenue recognition policies" (shares the word "revenue") but low relevance. The cross-encoder correctly scores the first higher because it processes query and chunk together, understanding the relationship.

**Performance trade-off**: Cross-encoder inference on 100 chunks takes 1-2 seconds. This is acceptable because the LLM classification step (3-4 seconds) runs first, and the reranking runs concurrently with retrieval plan streaming.

### 4.4 Why 5 Retrieval Routes

A single retrieval pipeline cannot optimally serve the diversity of financial queries:

| Query Type | Optimal Data Source | Why Separate Route |
|-----------|-------|---------|
| "What was Apple's revenue in 2023?" | XBRL annual_facts | Direct DB lookup; no embeddings needed; sub-100ms retrieval |
| "Show AAPL revenue 2020-2024" | XBRL timeseries | Needs multi-year aggregation with concept alias merging |
| "Apple's balance sheet Q3 2024" | financial_documents | Full statement retrieval; pre-formatted markdown |
| "What are Meta's key risks?" | sections_10k/10q | Semantic search over narrative text |
| "Compare AAPL vs MSFT profitability and explain drivers" | All of the above | Structured numbers + narrative context |

A unified pipeline would either over-fetch (running vector search for a simple metric lookup) or under-fetch (skipping XBRL data for a hybrid question). The 5-route architecture lets each query type use the fastest, most relevant retrieval path.

---

## 5. Query Routing & Edge Cases

### 5.1 LLM-Based Classification

The classifier uses GPT-4o-mini with **function calling** to extract structured metadata from natural language queries. The function schema defines 15+ fields including route, tickers, fiscal years, XBRL concepts, retrieval intent, query type, and target sections.

**Why function calling instead of prompt-based classification**: Function calling returns structured JSON with typed fields, eliminating parsing failures. A prompt-based approach ("respond with JSON") frequently produced malformed output, especially for complex multi-ticker queries with multiple years.

**Why GPT-4o-mini instead of GPT-4o**: Classification doesn't require deep reasoning — it's pattern matching on financial query structure. GPT-4o-mini handles this at 10x lower cost with comparable accuracy. The savings compound: classification runs on every query, including cache misses.

### 5.2 Route Override Heuristics

The LLM classifier occasionally misroutes queries. Two post-classification heuristics catch the most common errors:

**1. Risk query override**: When the LLM classifies "What are Apple's risk factors?" as `metric_lookup` (because it sees "Apple" and defaults to financial data), but the query contains risk-related keywords and no XBRL concepts:

```python
_RISK_KEYWORDS = r'\brisk(?:s| factors?)\b|\bitem\s*1a\b|\brisk management\b'
if route in ("metric_lookup", "timeseries", "full_statement") \
   and re.search(_RISK_KEYWORDS, query, re.IGNORECASE) \
   and not classification.get("xbrl_concepts"):
    classification["route"] = "narrative"  # Override to narrative search
```

**2. Misroute correction**: When the LLM routes to `narrative` but its own extracted metadata includes XBRL concepts and a quantitative retrieval intent:

```python
if route == "narrative" \
   and classification.get("xbrl_concepts") \
   and classification["retrieval_intent"] in ("specific_metric", "comparison", "timeseries"):
    classification["route"] = "metric_lookup"  # Correct to relational
```

These heuristics were added after observing real query misclassifications. They act as safety nets — the LLM is right ~95% of the time, but the 5% of misroutes cause visible user-facing errors.

### 5.3 Multi-Ticker Query Decomposition

Queries like "Compare Apple and Microsoft's revenue growth" require data from both companies, but the retrieval functions are optimized for single-ticker queries. The decomposition step uses an LLM to break the parent query into per-ticker sub-queries:

```
Parent: "Compare AAPL and MSFT revenue growth 2020-2024"
  → Sub-query 1: AAPL revenue 2020-2024 (purpose: "Apple's revenue data for comparison")
  → Sub-query 2: MSFT revenue 2020-2024 (purpose: "Microsoft's revenue data for comparison")
```

Sub-queries execute in parallel using `ThreadPoolExecutor(max_workers=4)`, and results are merged. This was a significant performance improvement — sequential execution of multi-ticker queries added 5-10 seconds of latency.

**Edge case**: For growth comparisons, each sub-query includes `years - 1` as well (e.g., if the user asks about 2024 growth, 2023 is also fetched) to enable YoY calculation.

### 5.4 Latest Year Resolution

Users frequently ask "What's Apple's latest revenue?" without specifying a year. The LLM is instructed to set `fiscal_year = null` for these queries, and the system auto-resolves to the latest available data:

```python
def get_latest_fiscal_year(ticker):
    """Query DB for the most recent fiscal year available.

    Checks quarterly_facts BEFORE annual_facts — quarterly data
    may be newer (e.g., Q1 2026 before FY2025 10-K is filed).
    """
    max_year = max(
        max_year_from_annual_facts,
        max_year_from_quarterly_facts,
    )
```

**Why this matters**: If the pipeline just ingested Q1 2026 data but the FY2025 10-K hasn't been filed yet, checking only `annual_facts` would return FY2024 as "latest." Checking `quarterly_facts` first catches the newer data.

### 5.5 Multi-Ticker Reranking Fairness

A subtle but critical edge case: when comparing Apple and Microsoft, the narrative retrieval runs separate vector searches for each ticker. The cross-encoder reranking scores are query-dependent — "Apple's risks" chunks are scored against the full comparison query, and so are "Microsoft's risks" chunks. But these scores are **not comparable across tickers** because the cross-encoder's scoring distribution depends on the corpus.

Without intervention, one company's chunks could dominate the results simply because its filing text happens to score higher against the comparison query phrasing.

**Solution**: Per-ticker fair allocation in guardrails:

```python
if is_multi_ticker:
    per_ticker_max = max(max_chunks // len(tickers), 3)
    by_ticker = group_chunks_by_ticker(chunks)
    for ticker, ticker_chunks in by_ticker.items():
        ticker_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        kept.extend(ticker_chunks[:per_ticker_max])
```

Each ticker gets an equal share of the chunk budget (e.g., 25 chunks each for a 2-ticker query with max_chunks=50), sorted by rerank score within each ticker's group.

### 5.6 Year-Quarter Mapping

Queries like "Compare Q3 2023 and Q2 2016" require different quarters for different years. A single `fiscal_quarter` field can't represent this. The classification schema includes a `year_quarters` mapping:

```python
"year_quarters": {
    "2023": 3,  # Q3 2023
    "2016": 2,  # Q2 2016
}
```

This was added after discovering that the LLM would arbitrarily pick one quarter (usually the first mentioned) and apply it to all years, producing incorrect results for the second year.

---

## 6. Answer Generation & Evaluation

### 6.1 Context Formatting by Route

The context injected into the answer generation prompt is formatted differently for each route:

- **Relational** (metric_lookup, timeseries): Structured text with labels — `"AAPL FY2024 us-gaap:Revenues: $391,035M (USD)"`
- **Narrative**: Section chunks with metadata headers — `"[10-K, AAPL, FY2024, Item 1A: Risk Factors, Chunk 3]: ...text..."`
- **Hybrid**: Both sections separated by headers, with explicit instructions to use XBRL for numbers and narrative for qualitative context.

**Relational context also includes a "Financial Snapshot"** for risk queries — key metrics (revenue, total assets, debt, equity) prepended to the context so the LLM can reference them when discussing risks:

```
Financial Snapshot for JPM:
  - Revenue: $128.7B (FY2024)
  - Total Assets: $3.9T
  - Long-term Debt: $268B
  - Net Income: $58.5B
```

### 6.2 Prompt Engineering for Financial Accuracy

The system prompt for answer generation includes route-specific and query-type-specific instructions:

**Hybrid queries**: "Use XBRL data for top-line metrics. Use narrative sections for SPECIFIC, QUANTITATIVE breakdowns. When narrative text mentions segment growth (e.g., 'Azure grew 28%'), cite these EXACT figures."

**Risk analysis**: "The context includes a Financial Snapshot with key metrics. You MUST reference these metrics when discussing risks. For EACH company, open with a brief financial profile using the snapshot data before listing risk factors."

**Multi-company comparisons**: "Note if companies have different fiscal year ends (e.g., Apple=Sep, Microsoft=Jun). Explicitly inform the user about these differences."

**Data availability warnings**: "The context contains Data Availability Notices about missing data or pre-IPO limitations. You MUST explicitly inform the user about these limitations. Do NOT silently omit a company from the comparison."

**Temperature**: Set to 0.1 (near-deterministic) for financial accuracy. Higher temperatures introduced numerical hallucinations in early testing.

### 6.3 Citation System

Every factual claim in the generated answer must include an inline citation:

```
[Source: 10-K, Ticker: AAPL, Year: 2024, Section: Income Statement]
```

The citation format is enforced through the system prompt with explicit examples. Post-generation, the API server enriches citations with SEC EDGAR URLs by looking up accession numbers:

```python
def _enrich_relational_sources(sources):
    """Look up accession numbers from the filings table for sources lacking URLs."""
    for ticker, fiscal_year in lookups:
        cur.execute(
            "SELECT accession_number, form_type, fiscal_period FROM filings "
            "WHERE ticker = %s AND fiscal_year = %s",
            (ticker, fiscal_year),
        )
        # Build URL: sec.gov/Archives/edgar/data/{CIK}/{accession}/index.htm
```

This allows the frontend to render clickable links directly to the SEC EDGAR filing index for every source cited.

### 6.4 Contradiction Detection

The guardrails system cross-references narrative claims against XBRL data to detect inconsistencies:

**Direction contradictions**: If a narrative chunk says "revenue increased" but the XBRL YoY comparison shows a decrease greater than 2%, this is flagged:

```python
if narrative_direction == "increase" and data_direction == "decrease":
    contradictions.append({
        "type": "direction_mismatch",
        "severity": "high" if abs(pct_change) > 10 else "medium",
        "detail": f"Narrative says revenue increased, but data shows {pct_change:+.1f}%",
    })
```

**Magnitude contradictions**: If the narrative claims "revenue grew approximately 20%" but XBRL data shows 5.3% growth, the discrepancy is flagged when the gap exceeds 5 percentage points.

**Watched concepts** are configured in `guardrails.yaml`: revenue, net income, operating income, EPS, gross profit, total assets, and earnings. Each maps to multiple XBRL concept aliases to handle the rename issue.

**Why this matters**: SEC filing narratives are written by management and can contain optimistic framing. Cross-referencing against hard XBRL numbers catches cases where the narrative tone doesn't match the actual financial performance.

### 6.5 Confidence Scoring System

Every answer receives a confidence score (0-100) computed from 5 weighted signals:

| Signal | What It Measures | Typical Impact |
|--------|-----------------|---------------|
| **Retrieval Quality** | Source reliability | XBRL data scores 90; statement fallback scores 72; narrative varies by rerank scores |
| **Source Coverage** | Completeness | What fraction of requested (ticker, year, concept) tuples had data |
| **Cross-Source Agreement** | Consistency | Penalty per contradiction: -25 (high severity), -15 (medium) |
| **Citation Density** | Groundedness | Citations per sentence vs. target (0.5); under-cited answers score lower |
| **Data Recency** | Staleness | -20 points per year of staleness between requested and available data |

**Route-specific weight overrides**: Relational routes weight retrieval quality at 40% (high confidence in XBRL data), while narrative routes weight it at 25% (inherently less precise). Timeseries routes weight source coverage at 35% (incomplete timeseries data is a clear quality issue).

**Confidence tiers**: High (75+, green), Moderate (50-74, yellow), Low (25-49, red), Insufficient (<25, dim). These are surfaced in the UI with color coding and plain-English descriptions.

All thresholds, weights, and tier boundaries are configured in `guardrails.yaml` — no hardcoded values. This enables tuning without code changes.

### 6.6 Cost Tracking & Efficiency Grading

Every query tracks its OpenAI API cost across all phases:

```python
class CostTracker:
    def record(phase, model, usage, elapsed_ms):
        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        throughput = total_tokens / (elapsed_ms / 1000)
```

**Pricing table** (current as of deployment):
- GPT-4o-mini: $0.15/1M input, $0.60/1M output
- text-embedding-3-small: $0.02/1M input

**Efficiency grading**: Each query receives a grade based on cost per token:
- **S** (cached): Zero LLM cost
- **A+** (< $0.0002/token): "Hyper Efficient"
- **A** (< $0.0005/token): "Efficient"
- **B** (< $0.001/token): "Good"
- **C** (≥ $0.001/token): "Moderate"

The frontend displays a "cost card" with per-phase breakdown (classify, generate, embed), total cost, token counts, throughput (tokens/second), and the efficiency grade. A typical query costs $0.0003-$0.0007 — well under a penny.

---

## 7. Caching Architecture

The three-layer caching strategy was designed around the observation that different parts of the pipeline have different cache invalidation requirements:

| Layer | What's Cached | TTL | Rationale |
|-------|--------------|-----|-----------|
| **Query Results** | Full answer + sources + confidence | 1 hour | Answers change when data is ingested; 1h balances freshness and cost |
| **Classification** | Route + metadata extraction | 2 hours | Classification is deterministic for the same query; longer TTL reduces LLM calls |
| **Retrieval** | Raw retrieval results per (route + tickers + years + concepts) | 1 hour | Different queries can hit the same underlying data; this layer captures that overlap |

**Cache key construction**: Retrieval cache keys are hashed from a composite of `(route, sorted(tickers), sorted(years), fiscal_quarter, temporal_granularity, sorted(xbrl_concepts), sorted(concepts), sorted(statement_types))`. This ensures that "Apple's revenue in 2024" and "AAPL revenue FY2024" — which produce different query strings but identical retrieval parameters — share the same cache entry.

**Version bumping**: All keys are namespaced with `CACHE_VERSION` (currently `v3`). When code changes affect cache structure, bumping the version string invalidates all old entries without a manual cache flush.

**Graceful degradation**: If Redis is unavailable, the system functions without caching. The `_get_redis()` function sets `_redis_available = False` on connection failure and never retries, avoiding per-request connection timeouts.

---

## 8. Lessons Learned

1. **Domain complexity dominates technical complexity.** The hardest problems weren't about vector search or LLM prompting — they were about understanding SEC filing structures, fiscal year conventions, and XBRL taxonomy evolution. A naive RAG system that ignores these domain details produces confidently wrong answers.

2. **Structured data should stay structured.** Early iterations tried to embed everything (including financial tables). This was worse in every dimension: higher storage cost, slower retrieval, and lower answer quality. Financial numbers should be queried relationally; only narrative text benefits from semantic search.

3. **LLM classification needs guardrails.** The 95% accuracy of GPT-4o-mini classification sounds high, but the 5% of misrouted queries cause the worst user experiences. Rule-based overrides for known failure patterns (risk queries misrouted to metric_lookup, quantitative queries misrouted to narrative) catch these reliably.

4. **Multi-entity queries are fundamentally different.** Comparing two companies isn't "run single-company pipeline twice." Reranking scores are incomparable across tickers, chunk budgets need fair allocation, and the answer prompt needs explicit instructions about fiscal year differences.

5. **Cost observability drives trust.** Showing users exactly what each query costs (in tokens and dollars) with an efficiency grade builds confidence that the system isn't wasteful. It also helps during development — a query that suddenly costs 5x more than similar queries usually indicates a retrieval bug.

6. **Config-driven guardrails enable iteration.** Moving all thresholds, weights, and keywords into `guardrails.yaml` was one of the best architectural decisions. Tuning confidence scoring weights or contradiction detection sensitivity doesn't require code changes, testing, or deployment — just a YAML edit.
