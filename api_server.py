"""
FastAPI server wrapping the SEC Filing RAG query engine.
Run: .venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import json
import os
import re
import time
import traceback
from collections import defaultdict
from datetime import date
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# Railway (and similar PaaS) provide DATABASE_URL as a single connection string.
# Parse it into individual PG_* env vars that the rest of the codebase expects.
_database_url = os.environ.get("DATABASE_URL")
if _database_url and not os.environ.get("PG_HOST"):
    _parsed = urlparse(_database_url)
    os.environ.setdefault("PG_HOST", _parsed.hostname or "localhost")
    os.environ.setdefault("PG_PORT", str(_parsed.port or 5432))
    os.environ.setdefault("PG_USER", _parsed.username or "")
    os.environ.setdefault("PG_PASSWORD", _parsed.password or "")
    os.environ.setdefault("PG_DATABASE", _parsed.path.lstrip("/") or "sec_filings")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from rag_query import rag_query, classify_query, build_retrieval_plan, CostTracker
from cache import cache_stats, cache_clear, get_cached_query_result

app = FastAPI(title="SEC Filing RAG API")

# CORS: allow localhost for dev + production frontend URL from env
_cors_origins = ["http://localhost:3000"]
_frontend_url = os.environ.get("FRONTEND_URL", "")
if _frontend_url:
    url = _frontend_url.rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url
    _cors_origins.append(url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate Limiting (IP-based + global daily cap) ---
DAILY_QUERY_LIMIT = int(os.environ.get("DAILY_QUERY_LIMIT", "10"))
GLOBAL_DAILY_LIMIT = int(os.environ.get("GLOBAL_DAILY_LIMIT", "50"))
_rate_limit_store: dict[str, dict] = defaultdict(lambda: {"date": date.today(), "count": 0})
_global_counter: dict[str, int | object] = {"date": date.today(), "count": 0}


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> tuple[bool, int, str | None]:
    """Check if IP is within daily limit and global cap. Returns (allowed, remaining, reason)."""
    today = date.today()

    # Reset global counter on new day
    if _global_counter["date"] != today:
        _global_counter["date"] = today
        _global_counter["count"] = 0

    # Check global limit first
    if _global_counter["count"] >= GLOBAL_DAILY_LIMIT:
        return False, 0, "global"

    # Check per-IP limit
    entry = _rate_limit_store[ip]
    if entry["date"] != today:
        entry["date"] = today
        entry["count"] = 0

    if entry["count"] >= DAILY_QUERY_LIMIT:
        remaining = 0
        return False, remaining, "ip"

    remaining = min(
        DAILY_QUERY_LIMIT - entry["count"],
        GLOBAL_DAILY_LIMIT - int(_global_counter["count"]),
    )
    return True, remaining, None


def _increment_rate_limit(ip: str):
    """Increment query count for IP and global counter."""
    _rate_limit_store[ip]["count"] += 1
    _global_counter["count"] = int(_global_counter["count"]) + 1


ROUTE_NAME_MAP = {
    "metric_lookup": "Metric Lookup",
    "timeseries": "Timeseries",
    "full_statement": "Full Statement",
    "narrative": "Narrative Search",
    "hybrid": "Hybrid",
}

# CIK numbers for supported tickers (used to build SEC EDGAR filing URLs)
TICKER_CIK = {
    "AAPL": "320193",
    "MSFT": "789019",
    "NVDA": "1045810",
    "AMZN": "1018724",
    "GOOGL": "1652044",
    "META": "1326801",
    "BRK-B": "1067983",
    "LLY": "59478",
    "AVGO": "1649338",
    "JPM": "19617",
}


def _build_filing_url(accession_number, ticker):
    """Build a SEC EDGAR filing index URL from an accession number and ticker.

    URL format: https://www.sec.gov/Archives/edgar/data/{CIK}/{accession_nodash}/{accession}-index.htm
    """
    if not accession_number or not ticker:
        return None
    cik = TICKER_CIK.get(ticker.upper())
    if not cik:
        return None
    accession_clean = accession_number.strip()
    accession_nodash = accession_clean.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_clean}-index.htm"


class QueryRequest(BaseModel):
    query: str


def _build_sources(result):
    """Build source list from rag_query result, with SEC EDGAR filing URLs."""
    sources = []
    cls = result.get("classification", {})
    route = result.get("route")
    ticker = cls.get("ticker", "")
    fiscal_year = cls.get("fiscal_year", "")
    fiscal_quarter = cls.get("fiscal_quarter")

    if result.get("sources"):
        for s in result["sources"]:
            if isinstance(s, dict):
                src_ticker = s.get("ticker") or ticker
                src = {
                    "filing": s.get("filing", ""),
                    "reference": s.get("reference", ""),
                    "filing_date": s.get("filing_date", ""),
                    "ticker": src_ticker,
                    "_ticker": src_ticker,  # kept for enrichment, stripped before response
                }
                # Build EDGAR URL from accession_number
                accession = s.get("accession_number")
                url = _build_filing_url(accession, src_ticker)
                if url:
                    src["filing_url"] = url
                sources.append(src)
            else:
                sources.append({"filing": str(s), "reference": "", "filing_date": ""})
    else:
        filing_type = "10-K" if not fiscal_quarter else "10-Q"
        if route == "narrative":
            filing_type = "10-K / 10-Q"
        q_str = f" Q{fiscal_quarter}" if fiscal_quarter else ""
        if ticker:
            sources.append({
                "filing": f"{ticker} {filing_type} FY{fiscal_year}{q_str}",
                "reference": f"Route: {route}",
                "filing_date": "",
            })

    # For relational sources without accession_number, try to look up from DB
    _enrich_relational_sources(sources)

    # Strip internal fields before returning
    for s in sources:
        s.pop("_ticker", None)

    return sources


def _enrich_relational_sources(sources):
    """Look up accession numbers from the filings table for sources that lack URLs."""
    needs_lookup = [s for s in sources if not s.get("filing_url")]
    if not needs_lookup:
        return

    # Extract ticker + fiscal_year pairs that need lookup (using _ticker field)
    lookups = set()
    for s in needs_lookup:
        ticker = s.get("_ticker")
        filing = s.get("filing", "")
        year_match = re.search(r"FY\s?(\d{4})", filing)
        if ticker and year_match:
            lookups.add((ticker, int(year_match.group(1))))

    if not lookups:
        return

    try:
        from xbrl_to_postgres import get_connection_pool
        db_pool = get_connection_pool()
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                for ticker, fy in lookups:
                    cur.execute(
                        "SELECT accession_number, form_type, fiscal_period FROM filings "
                        "WHERE ticker = %s AND fiscal_year = %s "
                        "ORDER BY filing_date DESC LIMIT 10",
                        (ticker, fy),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        continue
                    # Build a map of (form_type, fiscal_period) -> accession
                    # e.g. {('10-Q','Q1'): acc1, ('10-Q','Q2'): acc2, ('10-K','FY'): acc3}
                    acc_map = {}
                    form_map = {}  # fallback: form_type -> accession (first seen)
                    for acc, form, period in rows:
                        ft = form.strip().upper()
                        fp = (period or "").strip().upper()
                        if (ft, fp) not in acc_map:
                            acc_map[(ft, fp)] = acc
                        if ft not in form_map:
                            form_map[ft] = acc

                    # Match sources to accession numbers
                    for s in needs_lookup:
                        if s.get("filing_url"):
                            continue
                        src_ticker = s.get("_ticker")
                        if src_ticker != ticker:
                            continue
                        filing = s.get("filing", "")
                        if f"FY {fy}" not in filing and f"FY{fy}" not in filing:
                            continue

                        # Extract quarter from filing text (e.g. "Q1", "Q2", "Q3", "Q4")
                        q_match = re.search(r"\bQ([1-4])\b", filing)
                        quarter_str = f"Q{q_match.group(1)}" if q_match else None

                        # Pick the right accession based on filing type + quarter
                        acc = None
                        if "10-Q" in filing and quarter_str:
                            # Try exact quarter match first, fall back to any 10-Q
                            acc = acc_map.get(("10-Q", quarter_str)) or form_map.get("10-Q")
                        elif "10-K" in filing:
                            acc = acc_map.get(("10-K", "FY")) or form_map.get("10-K")
                        else:
                            acc = form_map.get("10-K") or form_map.get("10-Q")
                        if acc:
                            url = _build_filing_url(acc, ticker)
                            if url:
                                s["filing_url"] = url
        finally:
            db_pool.putconn(conn)
    except Exception:
        # Non-critical: if DB lookup fails, sources just won't have URLs
        pass


def _sse_event(event_type, data):
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream")
def handle_query_stream(req: QueryRequest, request: Request):
    """SSE endpoint: streams classification + retrieval plan, then the full result."""
    client_ip = _get_client_ip(request)
    allowed, remaining, reason = _check_rate_limit(client_ip)
    if not allowed:
        msg = "Service query limit reached for today. Please try again tomorrow." if reason == "global" else "Daily query limit reached. Please try again tomorrow."
        return JSONResponse(
            status_code=429,
            content={"error": msg, "limit": DAILY_QUERY_LIMIT},
        )
    _increment_rate_limit(client_ip)

    def generate():
        start = time.time()
        tracker = CostTracker()

        try:
            # Fast path: check if full result is cached
            cached_result = get_cached_query_result(req.query)
            if cached_result is not None:
                cached_result["_cache_hit"] = True
                tracker.record_cache_hit("full_query")
                cls = cached_result.get("classification", {})
                route = cached_result.get("route")

                yield _sse_event("classification", {
                    "route": route,
                    "route_name": ROUTE_NAME_MAP.get(route, route),
                    "reasoning": cached_result.get("reasoning"),
                    "elapsed": round(time.time() - start, 2),
                })
                yield _sse_event("retrieval_plan", {
                    "steps": cached_result.get("retrieval_plan", {}).get("steps", []),
                    "elapsed": round(time.time() - start, 2),
                })

                sources = _build_sources(cached_result)
                elapsed = round(time.time() - start, 2)

                yield _sse_event("result", {
                    "answer": cached_result.get("answer"),
                    "route": route,
                    "route_name": ROUTE_NAME_MAP.get(route, route),
                    "reasoning": cached_result.get("reasoning"),
                    "sources": sources,
                    "confidence": cached_result.get("confidence"),
                    "retrieval_plan": cached_result.get("retrieval_plan", {}),
                    "response_time": elapsed,
                    "cache_hit": True,
                    "cost": tracker.summary(),
                })

            else:
                # Step 1: Classify (fast — ~1-2s, cached after first call)
                classification = classify_query(req.query, cost_tracker=tracker)
                route = classification.get("route")
                reasoning = classification.get("reasoning")

                yield _sse_event("classification", {
                    "route": route,
                    "route_name": ROUTE_NAME_MAP.get(route, route),
                    "reasoning": reasoning,
                    "elapsed": round(time.time() - start, 2),
                })

                # Step 2: Build retrieval plan (instant)
                plan = build_retrieval_plan(req.query, classification)
                yield _sse_event("retrieval_plan", {
                    "steps": plan.get("steps", []),
                    "elapsed": round(time.time() - start, 2),
                })

                # Step 3: Execute full query (pass pre-computed classification and cost tracker)
                result = rag_query(req.query, precomputed_classification=classification, cost_tracker=tracker)
                elapsed = round(time.time() - start, 2)

                sources = _build_sources(result)

                yield _sse_event("result", {
                    "answer": result.get("answer"),
                    "route": result.get("route"),
                    "route_name": ROUTE_NAME_MAP.get(result.get("route"), result.get("route")),
                    "reasoning": result.get("reasoning"),
                    "sources": sources,
                    "confidence": result.get("confidence"),
                    "retrieval_plan": result.get("retrieval_plan", {}),
                    "response_time": elapsed,
                    "cache_hit": result.get("_cache_hit", False),
                    "cost": result.get("cost"),
                })

        except Exception as e:
            traceback.print_exc()
            yield _sse_event("error", {
                "error": str(e),
                "response_time": round(time.time() - start, 2),
            })

        yield _sse_event("done", {})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query")
def handle_query(req: QueryRequest, request: Request):
    """Non-streaming endpoint (backwards compatible)."""
    client_ip = _get_client_ip(request)
    allowed, remaining, reason = _check_rate_limit(client_ip)
    if not allowed:
        msg = "Service query limit reached for today. Please try again tomorrow." if reason == "global" else "Daily query limit reached. Please try again tomorrow."
        return JSONResponse(
            status_code=429,
            content={"error": msg, "limit": DAILY_QUERY_LIMIT},
        )
    _increment_rate_limit(client_ip)
    start = time.time()

    try:
        result = rag_query(req.query)
    except Exception as e:
        traceback.print_exc()
        return {
            "answer": None,
            "route": None,
            "route_name": None,
            "reasoning": None,
            "sources": [],
            "confidence": None,
            "response_time": round(time.time() - start, 2),
            "error": str(e),
        }

    elapsed = round(time.time() - start, 2)
    route = result.get("route")
    sources = _build_sources(result)

    return {
        "answer": result.get("answer"),
        "route": route,
        "route_name": ROUTE_NAME_MAP.get(route, route),
        "reasoning": result.get("reasoning"),
        "sources": sources,
        "confidence": result.get("confidence"),
        "retrieval_plan": result.get("retrieval_plan", {}),
        "response_time": elapsed,
        "cache_hit": result.get("_cache_hit", False),
        "cost": result.get("cost"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/rate-limit/status")
def rate_limit_status(request: Request):
    """Check remaining queries for the calling IP."""
    client_ip = _get_client_ip(request)
    allowed, remaining, reason = _check_rate_limit(client_ip)
    return {"limit": DAILY_QUERY_LIMIT, "global_limit": GLOBAL_DAILY_LIMIT, "remaining": remaining, "allowed": allowed}



# --- Cache Management Endpoints ---

@app.get("/cache/stats")
def get_cache_stats():
    """Return Redis cache statistics."""
    return cache_stats()


class CacheClearRequest(BaseModel):
    layer: str | None = None  # "query", "classify", "retrieval", or None for all


@app.post("/cache/clear")
def clear_cache(req: CacheClearRequest = CacheClearRequest()):
    """Clear cache. Optionally specify a layer (query, classify, retrieval)."""
    return cache_clear(req.layer)
