"""
Redis-based caching layer for the SEC RAG pipeline.

Caches three layers:
  1. Full query results (keyed by normalized query text)
  2. Classification results (keyed by normalized query text)
  3. XBRL / retrieval results (keyed by route + ticker + year + quarter + concepts)

Gracefully degrades to no-op if Redis is unavailable.
"""

import hashlib
import json
import os
import time
from functools import wraps

import redis
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # default 1 hour
CACHE_VERSION = "v3"  # bump to invalidate all cached entries after code changes
CACHE_PREFIX = f"sec_rag:{CACHE_VERSION}:"

# Layer-specific TTLs (seconds)
TTL_QUERY_RESULT = int(os.getenv("CACHE_TTL_QUERY", str(CACHE_TTL)))
TTL_CLASSIFICATION = int(os.getenv("CACHE_TTL_CLASSIFY", str(CACHE_TTL * 2)))  # 2h
TTL_RETRIEVAL = int(os.getenv("CACHE_TTL_RETRIEVAL", str(CACHE_TTL)))

# ---------------------------------------------------------------------------
# Redis connection (singleton, lazy)
# ---------------------------------------------------------------------------
_redis_client: redis.Redis | None = None
_redis_available: bool | None = None  # None = not checked yet


def _get_redis() -> redis.Redis | None:
    """Return a Redis client, or None if Redis is unavailable."""
    global _redis_client, _redis_available

    if not CACHE_ENABLED:
        return None

    if _redis_available is False:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        _redis_available = True
        return _redis_client
    except Exception:
        _redis_available = False
        _redis_client = None
        return None


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _normalize_query(query: str) -> str:
    """Normalize query for consistent cache keys."""
    return " ".join(query.lower().strip().split())


def _hash_key(*parts) -> str:
    """Create a deterministic hash from key parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_key(layer: str, *parts) -> str:
    """Build a namespaced Redis key."""
    h = _hash_key(*parts)
    return f"{CACHE_PREFIX}{layer}:{h}"


# ---------------------------------------------------------------------------
# Core cache operations
# ---------------------------------------------------------------------------

def cache_get(key: str) -> dict | None:
    """Get a cached value. Returns None on miss or error."""
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


def cache_set(key: str, value, ttl: int = CACHE_TTL) -> bool:
    """Set a cached value with TTL. Returns True on success."""
    r = _get_redis()
    if r is None:
        return False
    try:
        raw = json.dumps(value, default=str)
        r.setex(key, ttl, raw)
        return True
    except Exception:
        return False


def cache_delete(key: str) -> bool:
    """Delete a specific cache key."""
    r = _get_redis()
    if r is None:
        return False
    try:
        r.delete(key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Layer-specific cache functions
# ---------------------------------------------------------------------------

def get_cached_query_result(query: str) -> dict | None:
    """Check cache for a full query result."""
    key = _make_key("query", _normalize_query(query))
    return cache_get(key)


def set_cached_query_result(query: str, result: dict) -> bool:
    """Cache a full query result."""
    key = _make_key("query", _normalize_query(query))
    return cache_set(key, result, TTL_QUERY_RESULT)


def get_cached_classification(query: str) -> dict | None:
    """Check cache for a classification result."""
    key = _make_key("classify", _normalize_query(query))
    return cache_get(key)


def set_cached_classification(query: str, classification: dict) -> bool:
    """Cache a classification result."""
    key = _make_key("classify", _normalize_query(query))
    return cache_set(key, classification, TTL_CLASSIFICATION)


def get_cached_retrieval(route: str, classification: dict) -> dict | None:
    """Check cache for retrieval results based on route + classification metadata."""
    parts = _retrieval_key_parts(route, classification)
    key = _make_key("retrieval", *parts)
    return cache_get(key)


def set_cached_retrieval(route: str, classification: dict, data) -> bool:
    """Cache retrieval results."""
    parts = _retrieval_key_parts(route, classification)
    key = _make_key("retrieval", *parts)
    return cache_set(key, data, TTL_RETRIEVAL)


def _retrieval_key_parts(route: str, classification: dict) -> list:
    """Extract deterministic key parts from classification for retrieval caching."""
    return [
        route,
        sorted(classification.get("tickers", [])),
        sorted(classification.get("years_involved", [])),
        classification.get("fiscal_quarter"),
        classification.get("year_quarters", {}),
        classification.get("temporal_granularity", "annual"),
        sorted(classification.get("xbrl_concepts", [])),
        sorted(classification.get("concepts", [])),
        sorted(classification.get("statement_types", [])),
    ]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def cache_stats() -> dict:
    """Return cache statistics."""
    r = _get_redis()
    if r is None:
        return {"enabled": CACHE_ENABLED, "connected": False}

    try:
        info = r.info("stats")
        memory = r.info("memory")

        # Count keys by layer
        query_keys = len(r.keys(f"{CACHE_PREFIX}query:*"))
        classify_keys = len(r.keys(f"{CACHE_PREFIX}classify:*"))
        retrieval_keys = len(r.keys(f"{CACHE_PREFIX}retrieval:*"))

        return {
            "enabled": CACHE_ENABLED,
            "connected": True,
            "redis_url": REDIS_URL,
            "total_keys": query_keys + classify_keys + retrieval_keys,
            "layers": {
                "query_results": query_keys,
                "classifications": classify_keys,
                "retrievals": retrieval_keys,
            },
            "ttl": {
                "query": TTL_QUERY_RESULT,
                "classification": TTL_CLASSIFICATION,
                "retrieval": TTL_RETRIEVAL,
            },
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "memory_used_mb": round(memory.get("used_memory", 0) / 1024 / 1024, 2),
        }
    except Exception as e:
        return {"enabled": CACHE_ENABLED, "connected": False, "error": str(e)}


def cache_clear(layer: str | None = None) -> dict:
    """Clear cache keys. If layer is specified, clear only that layer."""
    r = _get_redis()
    if r is None:
        return {"cleared": 0, "error": "Redis not available"}

    try:
        if layer:
            pattern = f"{CACHE_PREFIX}{layer}:*"
        else:
            pattern = f"{CACHE_PREFIX}*"

        keys = r.keys(pattern)
        if keys:
            r.delete(*keys)
        return {"cleared": len(keys), "pattern": pattern}
    except Exception as e:
        return {"cleared": 0, "error": str(e)}
