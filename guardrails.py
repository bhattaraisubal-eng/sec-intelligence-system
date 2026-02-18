"""
SEC RAG Guardrails — Config-driven retrieval filtering, contradiction
detection, and investor-grade confidence scoring.

All behaviour is controlled by guardrails.yaml.  Import the three public
functions into rag_query.py:

    from guardrails import (
        load_guardrails,
        apply_retrieval_guardrails,
        detect_contradictions,
        compute_confidence,
        validate_query,
        validate_scope,
        format_confidence_banner,
        format_contradiction_warnings,
    )
"""

import math
import os
import re
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "guardrails.yaml"
_config: dict | None = None

# Route groupings for 5-route schema
_RELATIONAL_ROUTES = {"metric_lookup", "timeseries", "full_statement", "relational_db"}
_NARRATIVE_ROUTES = {"narrative", "vector_db"}


def load_guardrails(path: str | Path | None = None) -> dict:
    """Load and cache the guardrail config from YAML."""
    global _config
    if _config is not None and path is None:
        return _config
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r") as f:
        _config = yaml.safe_load(f)
    return _config


def _cfg() -> dict:
    """Get the cached config (auto-loads if needed)."""
    if _config is None:
        load_guardrails()
    return _config


# ---------------------------------------------------------------------------
# 1. Query Validation
# ---------------------------------------------------------------------------

def validate_query(query: str) -> tuple[bool, str | None]:
    """Validate the raw query against config rules.

    Returns (is_valid, rejection_reason).
    """
    cfg = _cfg().get("query", {})

    if len(query) < cfg.get("min_length", 5):
        return False, "Query is too short. Please provide more detail."

    if len(query) > cfg.get("max_length", 1000):
        return False, f"Query exceeds {cfg.get('max_length', 1000)} characters."

    for pattern in cfg.get("blocked_patterns", []):
        if re.search(pattern, query):
            return False, "Query contains disallowed content."

    return True, None


# ---------------------------------------------------------------------------
# 1b. Scope Validation (post-classification)
# ---------------------------------------------------------------------------

def validate_scope(classification: dict) -> tuple[bool, str | None]:
    """Check whether the classified query falls within the system's data scope.

    Called AFTER classify_query() so we have extracted tickers, years, and
    target_sections.  Returns (is_in_scope, user_message).
    """
    scope = _cfg().get("scope", {})
    if not scope:
        return True, None

    warnings: list[str] = []

    # --- Ticker scope ---
    supported = {t.upper() for t in scope.get("supported_tickers", [])}
    if supported:
        tickers = [t.upper() for t in classification.get("tickers", [])]
        unsupported = [t for t in tickers if t not in supported]
        if unsupported:
            supported_list = ", ".join(sorted(supported))
            warnings.append(
                f"Ticker(s) **{', '.join(unsupported)}** are outside the scope of this system. "
                f"Supported tickers: {supported_list}."
            )

    # --- Year scope ---
    min_year = scope.get("min_year", 2010)
    max_year = scope.get("max_year", 2027)
    all_years = list(classification.get("years_involved", []))
    if classification.get("fiscal_year"):
        all_years.append(classification["fiscal_year"])
    out_of_range = [y for y in all_years if y < min_year or y > max_year]
    if out_of_range:
        warnings.append(
            f"Year(s) **{', '.join(str(y) for y in sorted(set(out_of_range)))}** "
            f"are outside the data coverage of this system ({min_year}–{max_year}). "
            f"Please query a year within that range."
        )

    # --- Section scope (narrative routes only) ---
    route = classification.get("route", "")
    target_sections = classification.get("target_sections", [])
    if target_sections and route in ("narrative", "hybrid"):
        embedded = {s.lower() for s in scope.get("embedded_sections", [])}
        aliases = {k.lower(): v.lower() for k, v in scope.get("section_aliases", {}).items()}

        unembedded = []
        for section in target_sections:
            sec_lower = section.lower()
            # Check direct match or alias
            normalised = aliases.get(sec_lower, sec_lower)
            if normalised not in embedded:
                unembedded.append(section)

        if unembedded:
            embedded_list = ", ".join(scope.get("embedded_sections", []))
            warnings.append(
                f"Section(s) **{', '.join(unembedded)}** are not available for narrative search. "
                f"Only **{embedded_list}** sections are embedded as vectors. "
                f"Structured financial data (XBRL) is still available for metric queries."
            )

    if warnings:
        return False, "\n\n".join(warnings)

    return True, None


# ---------------------------------------------------------------------------
# 2. Retrieval Quality Gates
# ---------------------------------------------------------------------------

def apply_retrieval_guardrails(
    route: str,
    data,
    classification: dict,
) -> tuple:
    """Filter retrieved data against quality gates defined in config.

    Returns (filtered_data, filter_stats) where filter_stats is a dict
    with counts of kept/dropped items for transparency.
    """
    cfg = _cfg().get("retrieval", {})
    stats = {"kept": 0, "dropped": 0, "warnings": []}

    if route in _NARRATIVE_ROUTES:
        data, stats = _filter_vector_results(data, cfg.get("vector", {}), classification)

    elif route in _RELATIONAL_ROUTES:
        data, stats = _filter_relational_results(data, cfg.get("relational", {}))

    elif route == "hybrid":
        vec_cfg = cfg.get("vector", {})
        rel_cfg = cfg.get("relational", {})

        vec_data = data.get("vector", [])
        rel_data = data.get("relational", {})

        filtered_vec, vec_stats = _filter_vector_results(vec_data, vec_cfg, classification)
        filtered_rel, rel_stats = _filter_relational_results(rel_data, rel_cfg)

        data = {"vector": filtered_vec, "relational": filtered_rel}
        stats = {
            "kept": vec_stats["kept"] + rel_stats["kept"],
            "dropped": vec_stats["dropped"] + rel_stats["dropped"],
            "warnings": vec_stats["warnings"] + rel_stats["warnings"],
        }

    # Check if we should block empty context
    if cfg.get("block_empty_context", True):
        has_data = _has_usable_data(route, data)
        if not has_data:
            stats["warnings"].append("No data passed quality gates.")

    return data, stats


def _filter_vector_results(chunks: list[dict], cfg: dict, classification: dict | None = None) -> tuple[list[dict], dict]:
    """Apply similarity and rerank score thresholds to vector chunks.

    For multi-ticker queries, ensures each ticker gets a fair share of the
    chunk budget so one company's results don't crowd out another's.
    """
    min_sim = cfg.get("min_similarity", 0.0)
    min_rerank = cfg.get("min_rerank_score", -999)
    max_chunks = cfg.get("max_chunks", 50)
    min_warn = cfg.get("min_chunks_warn", 2)

    tickers = (classification or {}).get("tickers", [])
    is_multi_ticker = len(tickers) > 1

    # Multi-ticker: rerank scores from separate sub-query runs are not comparable
    # across tickers (each sub-query reranks against its own query text), so skip
    # the global rerank threshold and rely solely on per-ticker top-N selection.
    if is_multi_ticker:
        from collections import defaultdict
        per_ticker_max = max(max_chunks // len(tickers), 3)

        # Group by ticker, sort each group by its own rerank/similarity score,
        # then take the top N per ticker.
        by_ticker = defaultdict(list)
        for chunk in chunks:
            sim = chunk.get("similarity", 0)
            # Still apply the similarity floor to filter truly irrelevant chunks
            if sim < min_sim:
                continue
            by_ticker[chunk.get("ticker", "")].append(chunk)

        kept = []
        dropped = 0
        for t, t_chunks in by_ticker.items():
            t_chunks.sort(key=lambda c: c.get("rerank_score", c.get("similarity", 0)), reverse=True)
            kept.extend(t_chunks[:per_ticker_max])
            dropped += len(t_chunks) - min(len(t_chunks), per_ticker_max)
        dropped += len(chunks) - sum(len(v) for v in by_ticker.values())
    else:
        kept = []
        dropped = 0

        for chunk in chunks:
            sim = chunk.get("similarity", 0)
            rerank = chunk.get("rerank_score")

            if rerank is not None:
                if rerank < min_rerank:
                    dropped += 1
                    continue
            else:
                if sim < min_sim:
                    dropped += 1
                    continue

            kept.append(chunk)

        kept.sort(key=lambda c: c.get("rerank_score", c.get("similarity", 0)), reverse=True)
        kept = kept[:max_chunks]

    warnings = []
    if len(kept) < min_warn and len(kept) > 0:
        warnings.append(
            f"Only {len(kept)} chunk(s) passed quality filters "
            f"(min_similarity={min_sim}, min_rerank={min_rerank})."
        )

    return kept, {"kept": len(kept), "dropped": dropped, "warnings": warnings}


def _filter_relational_results(data: dict, cfg: dict) -> tuple[dict, dict]:
    """Apply relational data quality gates."""
    max_per_concept = cfg.get("max_facts_per_concept", 5)
    require_min = cfg.get("require_min_facts", True)

    kept_count = 0
    dropped_count = 0
    warnings = []

    # Trim excess facts per concept
    if data.get("xbrl_facts"):
        from collections import defaultdict
        by_concept = defaultdict(list)
        for fact in data["xbrl_facts"]:
            by_concept[fact.get("concept", "unknown")].append(fact)

        trimmed = []
        for concept, facts in by_concept.items():
            trimmed.extend(facts[:max_per_concept])
            dropped_count += max(0, len(facts) - max_per_concept)
        data["xbrl_facts"] = trimmed
        kept_count += len(trimmed)

    if require_min and not data.get("xbrl_facts") and not data.get("statements"):
        warnings.append("No XBRL facts or statements found for this query.")

    # Count other data sources
    kept_count += len(data.get("comparisons", []))
    kept_count += len(data.get("statements", []))
    kept_count += len(data.get("timeseries", []))
    kept_count += len(data.get("earnings", []))

    return data, {"kept": kept_count, "dropped": dropped_count, "warnings": warnings}


def _has_usable_data(route: str, data) -> bool:
    """Check whether filtered data has any usable content."""
    if route in _NARRATIVE_ROUTES:
        return bool(data)
    elif route in _RELATIONAL_ROUTES:
        return bool(
            data.get("xbrl_facts")
            or data.get("statements")
            or data.get("comparisons")
            or data.get("timeseries")
            or data.get("earnings")
        )
    elif route == "hybrid":
        return _has_usable_data("narrative", data.get("vector", [])) or \
               _has_usable_data("metric_lookup", data.get("relational", {}))
    return False


# ---------------------------------------------------------------------------
# 3. Financial Contradiction Detection
# ---------------------------------------------------------------------------

def detect_contradictions(
    vector_chunks: list[dict],
    relational_data: dict,
    classification: dict,
) -> list[dict]:
    """Detect contradictions between narrative claims and XBRL numbers.

    Looks for directional mismatches (narrative says "increased" but numbers
    show a decrease) and magnitude mismatches (narrative says "15%" but
    numbers show 8%).

    Returns a list of contradiction dicts, each with:
        concept, narrative_claim, data_direction, data_value, severity, detail
    """
    cfg = _cfg().get("contradiction_detection", {})
    if not cfg.get("enabled", True):
        return []

    dir_tol = cfg.get("direction_tolerance_pct", 2.0)
    mag_tol = cfg.get("magnitude_tolerance_ppt", 5.0)
    inc_kw = set(cfg.get("increase_keywords", []))
    dec_kw = set(cfg.get("decrease_keywords", []))
    watched = cfg.get("watched_concepts", {})

    # Build a lookup of YoY changes from relational data
    comparisons = relational_data.get("comparisons", [])
    change_map = {}  # concept_pattern -> {pct_change, direction}
    for comp in comparisons:
        concept_name = comp.get("concept", "")
        pct = comp.get("pct_change")
        if pct is not None:
            direction = "increase" if pct > dir_tol else ("decrease" if pct < -dir_tol else "flat")
            change_map[concept_name] = {
                "pct_change": pct,
                "direction": direction,
                "delta": comp.get("delta", 0),
                "from_year": comp.get("from_year"),
                "to_year": comp.get("to_year"),
            }

    if not change_map:
        return []

    # Scan narrative chunks for directional claims
    contradictions = []
    narrative_text = " ".join(c.get("text", "") for c in vector_chunks).lower()

    for human_term, patterns in watched.items():
        # Find matching XBRL data
        matched_data = None
        matched_concept = None
        for concept_name, data in change_map.items():
            for pat in patterns:
                if pat.lower() in concept_name.lower():
                    matched_data = data
                    matched_concept = concept_name
                    break
            if matched_data:
                break

        if not matched_data:
            continue

        # Check for directional claims about this concept
        narrative_direction = _detect_narrative_direction(
            narrative_text, human_term, inc_kw, dec_kw
        )

        if narrative_direction is None:
            continue  # No claim found about this concept

        data_direction = matched_data["direction"]

        # Directional contradiction
        if narrative_direction != data_direction and data_direction != "flat":
            pct_val = float(matched_data["pct_change"])
            severity = "high" if abs(pct_val) > 10 else "medium"
            contradictions.append({
                "concept": human_term,
                "xbrl_concept": matched_concept,
                "narrative_claim": narrative_direction,
                "data_direction": data_direction,
                "data_pct_change": pct_val,
                "severity": severity,
                "type": "direction_mismatch",
                "detail": (
                    f"Narrative suggests {human_term} {narrative_direction}d, "
                    f"but XBRL data shows {pct_val:+.1f}% change "
                    f"({matched_data.get('from_year')} -> {matched_data.get('to_year')})."
                ),
            })
            continue

        # Magnitude contradiction: check if narrative mentions a specific %
        magnitude_claim = _extract_percentage_claim(narrative_text, human_term)
        if magnitude_claim is not None:
            actual_pct = float(abs(matched_data["pct_change"]))
            claimed_pct = float(abs(magnitude_claim))
            gap = abs(actual_pct - claimed_pct)
            if gap > mag_tol:
                contradictions.append({
                    "concept": human_term,
                    "xbrl_concept": matched_concept,
                    "narrative_claim": f"{magnitude_claim:+.1f}%",
                    "data_direction": data_direction,
                    "data_pct_change": actual_pct,
                    "severity": "medium",
                    "type": "magnitude_mismatch",
                    "detail": (
                        f"Narrative claims {human_term} changed ~{magnitude_claim:.0f}%, "
                        f"but XBRL shows {actual_pct:+.1f}% "
                        f"(gap of {gap:.1f} ppt)."
                    ),
                })

    return contradictions


def _detect_narrative_direction(
    text: str, concept: str, inc_kw: set, dec_kw: set
) -> str | None:
    """Detect whether the narrative claims increase or decrease for a concept.

    Looks for sentences containing the concept name near a direction keyword.
    Returns 'increase', 'decrease', or None.
    """
    # Build a regex to find sentences mentioning the concept
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        if concept.lower() not in sentence.lower():
            continue
        words = set(sentence.lower().split())
        has_inc = bool(words & inc_kw)
        has_dec = bool(words & dec_kw)
        if has_inc and not has_dec:
            return "increase"
        if has_dec and not has_inc:
            return "decrease"
    return None


def _extract_percentage_claim(text: str, concept: str) -> float | None:
    """Extract a percentage number mentioned near a concept in narrative text.

    Returns the percentage as a float, or None if not found.
    """
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        if concept.lower() not in sentence.lower():
            continue
        # Look for patterns like "15%", "15.3%", "15 percent"
        match = re.search(r'(\d+\.?\d*)\s*(%|percent)', sentence)
        if match:
            return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# 4. Confidence Scoring
# ---------------------------------------------------------------------------

def _is_effectively_relational(route: str, data) -> bool:
    """Check if the query is effectively relational-only.

    True for pure relational routes, and also for hybrid routes where no
    vector chunks survived filtering but relational data is present.
    """
    if route in _RELATIONAL_ROUTES:
        return True
    if route == "hybrid":
        has_vector = bool((data or {}).get("vector"))
        rel = (data or {}).get("relational", {})
        has_rel = bool(
            rel.get("xbrl_facts") or rel.get("statements")
            or rel.get("comparisons") or rel.get("earnings")
        )
        return has_rel and not has_vector
    return False


def compute_confidence(
    route: str,
    data,
    classification: dict,
    answer: str,
    contradictions: list[dict],
    filter_stats: dict,
) -> dict:
    """Compute a multi-signal confidence score (0-100).

    Uses route-specific weight profiles:
    - Relational (metric_lookup, full_statement): XBRL authority, coverage + recency dominate
    - Timeseries: data-point completeness across year range
    - Narrative: source diversity, retrieval quality, cross-source agreement
    - Hybrid: blended based on which data survived filtering

    Returns dict with:
        overall_score, tier_label, tier_color, tier_description,
        signals: {signal_name: {score, detail}}
    """
    cfg = _cfg().get("confidence", {})
    weights = cfg.get("weights", {})

    # Select the right override profile based on route + data
    rel_overrides = cfg.get("relational_overrides", {})
    ts_overrides = cfg.get("timeseries_overrides", {})
    narr_overrides = cfg.get("narrative_overrides", {})
    effectively_relational = _is_effectively_relational(route, data)

    active_overrides = None  # Track which profile is active for retrieval_quality

    if route == "timeseries" and ts_overrides:
        weights = ts_overrides.get("weights", weights)
        active_overrides = ts_overrides
    elif route in _NARRATIVE_ROUTES and narr_overrides:
        weights = narr_overrides.get("weights", weights)
        active_overrides = narr_overrides
    elif effectively_relational and rel_overrides:
        weights = rel_overrides.get("weights", weights)
        active_overrides = rel_overrides
    elif route == "hybrid":
        # Hybrid: check if it's effectively relational or has vector data
        has_vector = bool((data or {}).get("vector"))
        if has_vector and narr_overrides:
            # Blend toward narrative weights when vector data is present
            weights = narr_overrides.get("weights", weights)
            active_overrides = narr_overrides
        elif rel_overrides:
            weights = rel_overrides.get("weights", weights)
            active_overrides = rel_overrides

    signals = {}

    # Signal 1: Retrieval Quality
    rq_score = _score_retrieval_quality(route, data, cfg.get("retrieval_quality", {}),
                                         active_overrides or rel_overrides)
    signals["retrieval_quality"] = rq_score

    # Signal 2: Source Coverage
    sc_score = _score_source_coverage(route, data, classification)
    signals["source_coverage"] = sc_score

    # Signal 3: Cross-Source Agreement
    cs_score = _score_cross_source_agreement(contradictions, cfg.get("cross_source_agreement", {}))
    signals["cross_source_agreement"] = cs_score

    # Signal 4: Citation Density
    cd_score = _score_citation_density(answer, cfg.get("citation_density", {}))
    signals["citation_density"] = cd_score

    # Signal 5: Data Recency
    dr_score = _score_data_recency(route, data, classification, cfg.get("data_recency", {}))
    signals["data_recency"] = dr_score

    # Weighted overall score
    overall = 0.0
    for signal_name, weight in weights.items():
        if signal_name in signals:
            overall += weight * signals[signal_name]["score"]

    overall = max(0.0, min(100.0, overall))

    # Determine tier
    tier = _get_tier(overall, cfg.get("tiers", []))

    return {
        "overall_score": round(overall, 1),
        "tier_label": tier["label"],
        "tier_color": tier["color"],
        "tier_description": tier["description"],
        "signals": signals,
    }


def _score_retrieval_quality(route: str, data, cfg: dict, overrides: dict | None = None) -> dict:
    """Score based on source authority and retrieval quality.

    - Relational/timeseries: XBRL facts are SEC-filed ground truth (score ~98).
      Falls back to a lower score if only statement text was used.
    - Narrative: average rerank score + bonus for source diversity (multiple
      unique sections/filings corroborating the answer).
    - Hybrid: blend of relational authority and vector retrieval quality.
    """
    low = cfg.get("low", -1.0)
    high = cfg.get("high", 4.0)

    # --- Relational / Timeseries: authority-based scoring ---
    if route in _RELATIONAL_ROUTES or route == "timeseries":
        xbrl_score = (overrides or {}).get("retrieval_quality_score", 90)
        fallback_score = (overrides or {}).get("statement_fallback_score", 72)

        rel = data if isinstance(data, dict) else {}
        has_xbrl = bool(rel.get("xbrl_facts"))
        has_timeseries = bool(rel.get("timeseries"))
        has_statements = bool(rel.get("statements"))

        if has_xbrl or has_timeseries:
            return {
                "score": xbrl_score,
                "detail": f"Direct XBRL data from SEC EDGAR (authoritative, score={xbrl_score}).",
            }
        elif has_statements:
            return {
                "score": fallback_score,
                "detail": f"Financial statement text (reliable but less precise, score={fallback_score}).",
            }
        # No data at all
        return {"score": 30.0, "detail": "No structured data retrieved."}

    # --- Narrative: rerank scores + source diversity ---
    vector_chunks = []
    if route in _NARRATIVE_ROUTES:
        vector_chunks = data or []
    elif route == "hybrid":
        vector_chunks = (data or {}).get("vector", [])
        # Also check relational side for hybrid
        rel = (data or {}).get("relational", {})
        if rel.get("xbrl_facts") or rel.get("timeseries"):
            # Hybrid with XBRL data: boost the floor
            xbrl_boost = (overrides or {}).get("retrieval_quality_score", 90)
            if not vector_chunks:
                return {
                    "score": xbrl_boost,
                    "detail": f"Hybrid route: structured XBRL data (authoritative, score={xbrl_boost}).",
                }

    if not vector_chunks:
        if _is_effectively_relational(route, data):
            score = (overrides or {}).get("retrieval_quality_score", 90)
            return {"score": score, "detail": f"Structured XBRL data (authoritative, score={score})."}
        return {"score": 0.0, "detail": "No vector chunks retrieved."}

    # Compute rerank score component
    scores = []
    for chunk in vector_chunks:
        s = chunk.get("rerank_score", chunk.get("similarity", 0))
        scores.append(s)

    avg = sum(scores) / len(scores)
    if high == low:
        rerank_score = 100.0 if avg >= high else 0.0
    else:
        rerank_score = max(0.0, min(100.0, (avg - low) / (high - low) * 100))

    # Source diversity bonus for narrative: unique sections boost confidence
    unique_sections = set()
    unique_filings = set()
    for chunk in vector_chunks:
        sec = chunk.get("section_type") or chunk.get("section", "")
        if sec:
            unique_sections.add(sec.lower())
        filing_key = (chunk.get("ticker", ""), chunk.get("fiscal_year", ""), chunk.get("filing_type", ""))
        if any(filing_key):
            unique_filings.add(filing_key)

    section_bonus = (overrides or {}).get("section_diversity_bonus", 8)
    diversity_boost = min(20, (len(unique_sections) - 1) * section_bonus) if len(unique_sections) > 1 else 0

    # Chunk count factor: more corroborating chunks = more confidence
    min_for_full = (overrides or {}).get("min_chunks_full_confidence", 5)
    chunk_factor = min(1.0, len(vector_chunks) / max(min_for_full, 1))
    chunk_boost = chunk_factor * 15  # Up to 15 points for having enough chunks

    final_score = min(100.0, rerank_score + diversity_boost + chunk_boost)

    detail_parts = [f"Avg rerank: {avg:.2f} (n={len(scores)}, base={rerank_score:.0f})"]
    if diversity_boost > 0:
        detail_parts.append(f"+{diversity_boost:.0f} diversity ({len(unique_sections)} sections)")
    if chunk_boost > 0:
        detail_parts.append(f"+{chunk_boost:.0f} depth ({len(vector_chunks)} chunks)")

    return {
        "score": round(final_score, 1),
        "detail": "; ".join(detail_parts) + ".",
    }


def _score_source_coverage(route: str, data, classification: dict) -> dict:
    """Score based on fraction of requested dimensions that returned data.

    For timeseries routes, measures data-point completeness across the
    requested year range instead of binary year matching.
    """
    requested_years = set(classification.get("years_involved", []))
    requested_concepts = set(classification.get("concepts", []))
    ticker = classification.get("ticker")

    # Timeseries-specific: count actual data points in the series
    if route == "timeseries":
        ts_data = (data or {}).get("timeseries", [])
        if ts_data:
            data_years = {p.get("fiscal_year") for p in ts_data if p.get("fiscal_year")}
            if requested_years:
                coverage = len(requested_years & data_years) / max(len(requested_years), 1)
            else:
                # No specific years requested — having any timeseries data is good
                coverage = min(1.0, len(data_years) / 3)  # 3+ years = full coverage
            score = coverage * 100
            return {
                "score": round(score, 1),
                "detail": f"{len(data_years)} data points across {min(data_years) if data_years else '?'}-{max(data_years) if data_years else '?'} ({coverage:.0%} of requested range).",
            }
        # Fallback: check xbrl_facts for timeseries-like data
        xbrl_facts = (data or {}).get("xbrl_facts", [])
        if xbrl_facts:
            data_years = {f.get("fiscal_year") for f in xbrl_facts if f.get("fiscal_year")}
            if requested_years:
                coverage = len(requested_years & data_years) / max(len(requested_years), 1)
            else:
                coverage = min(1.0, len(data_years) / 3)
            score = coverage * 100
            return {
                "score": round(score, 1),
                "detail": f"{len(data_years)} XBRL facts spanning {min(data_years) if data_years else '?'}-{max(data_years) if data_years else '?'} ({coverage:.0%}).",
            }

    if not requested_years and not requested_concepts:
        return {"score": 80.0, "detail": "No specific dimensions requested; default coverage."}

    total_slots = 0
    filled_slots = 0

    # Check year coverage
    if requested_years:
        found_years = set()
        if route in _NARRATIVE_ROUTES:
            for chunk in (data or []):
                fy = chunk.get("fiscal_year")
                if fy:
                    found_years.add(fy)
        elif route in _RELATIONAL_ROUTES:
            for fact in (data or {}).get("xbrl_facts", []):
                fy = fact.get("fiscal_year")
                if fy:
                    found_years.add(fy)
            for stmt in (data or {}).get("statements", []):
                fy = stmt.get("fiscal_year")
                if fy:
                    found_years.add(fy)
            for comp in (data or {}).get("comparisons", []):
                for key in ("from_year", "to_year"):
                    fy = comp.get(key)
                    if fy:
                        found_years.add(fy)
        elif route == "hybrid":
            for chunk in (data or {}).get("vector", []):
                fy = chunk.get("fiscal_year")
                if fy:
                    found_years.add(fy)
            rel = (data or {}).get("relational", {})
            for fact in rel.get("xbrl_facts", []):
                fy = fact.get("fiscal_year")
                if fy:
                    found_years.add(fy)
            for stmt in rel.get("statements", []):
                fy = stmt.get("fiscal_year")
                if fy:
                    found_years.add(fy)
            for comp in rel.get("comparisons", []):
                for key in ("from_year", "to_year"):
                    fy = comp.get(key)
                    if fy:
                        found_years.add(fy)

        total_slots += len(requested_years)
        filled_slots += len(requested_years & found_years)

    # Check concept coverage (relational routes)
    # Skip concept-slot counting for event-inferred queries: their narrative concepts
    # (e.g. "acquisition cost") don't map to XBRL names, so penalizing is misleading.
    year_inferred = classification.get("_year_inferred", False)
    if requested_concepts and (route in _RELATIONAL_ROUTES or route == "hybrid") and not year_inferred:
        rel = data if route in _RELATIONAL_ROUTES else (data or {}).get("relational", {})
        found_concepts = set()
        for fact in rel.get("xbrl_facts", []):
            concept = fact.get("concept", "")
            # Match if any requested concept term appears in the XBRL name
            for rc in requested_concepts:
                if rc.lower().replace(" ", "") in concept.lower().replace("-", "").replace(":", ""):
                    found_concepts.add(rc)
        # Also check statements — a fetched statement implicitly covers
        # the concepts it contains (e.g. income_statement covers revenue).
        remaining = requested_concepts - found_concepts
        if remaining:
            for stmt in rel.get("statements", []):
                content = stmt.get("content", "").lower()
                for rc in list(remaining):
                    if rc.lower() in content:
                        found_concepts.add(rc)
                        remaining.discard(rc)
        # Also check earnings data for earnings-related concepts.
        remaining = requested_concepts - found_concepts
        if remaining and rel.get("earnings"):
            for rc in list(remaining):
                for earn in rel.get("earnings", []):
                    text = earn.get("content", "").lower()
                    if rc.lower() in text:
                        found_concepts.add(rc)
                        remaining.discard(rc)
                        break
        total_slots += len(requested_concepts)
        filled_slots += len(found_concepts)

    if total_slots == 0:
        return {"score": 80.0, "detail": "No specific dimensions to check."}

    coverage = filled_slots / total_slots
    score = coverage * 100

    return {
        "score": round(score, 1),
        "detail": f"{filled_slots}/{total_slots} requested dimensions covered ({coverage:.0%}).",
    }


def _score_cross_source_agreement(contradictions: list[dict], cfg: dict) -> dict:
    """Score based on contradictions found (fewer = better)."""
    penalty = cfg.get("penalty_per_contradiction", 25)
    bonus = cfg.get("bonus_per_agreement", 10)
    max_score = cfg.get("max_score", 100)

    if not contradictions:
        # No contradictions = full agreement
        return {"score": min(max_score, 100.0), "detail": "No contradictions detected."}

    high_count = sum(1 for c in contradictions if c.get("severity") == "high")
    med_count = sum(1 for c in contradictions if c.get("severity") == "medium")

    total_penalty = high_count * penalty + med_count * (penalty * 0.6)
    score = max(0.0, max_score - total_penalty)

    return {
        "score": round(score, 1),
        "detail": f"{len(contradictions)} contradiction(s) found ({high_count} high, {med_count} medium).",
    }


def _score_citation_density(answer: str, cfg: dict) -> dict:
    """Score based on how well-cited the answer is."""
    target = cfg.get("target_citations_per_sentence", 0.5)

    # Count sentences (rough)
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    num_sentences = max(len(sentences), 1)

    # Count inline citations: [Source: ...]
    citations = re.findall(r'\[Source:', answer)
    num_citations = len(citations)

    density = num_citations / num_sentences
    score = min(1.0, density / target) * 100 if target > 0 else 100.0

    return {
        "score": round(score, 1),
        "detail": f"{num_citations} citation(s) across ~{num_sentences} sentences (density={density:.2f}).",
    }


def _extract_data_years(route: str, data) -> set:
    """Extract all fiscal years found in retrieved data across all data types."""
    data_years = set()

    def _scan_relational(rel):
        for fact in rel.get("xbrl_facts", []):
            fy = fact.get("fiscal_year")
            if fy:
                data_years.add(fy)
        for stmt in rel.get("statements", []):
            fy = stmt.get("fiscal_year")
            if fy:
                data_years.add(fy)
        for point in rel.get("timeseries", []):
            fy = point.get("fiscal_year")
            if fy:
                data_years.add(fy)
        for comp in rel.get("comparisons", []):
            for key in ("from_year", "to_year"):
                fy = comp.get(key)
                if fy:
                    data_years.add(fy)
        for earn in rel.get("earnings", []):
            fy = earn.get("fiscal_year")
            if fy:
                data_years.add(fy)

    if route in _NARRATIVE_ROUTES:
        for chunk in (data or []):
            fy = chunk.get("fiscal_year")
            if fy:
                data_years.add(fy)
    elif route in _RELATIONAL_ROUTES or route == "timeseries":
        _scan_relational(data or {})
    elif route == "hybrid":
        for chunk in (data or {}).get("vector", []):
            fy = chunk.get("fiscal_year")
            if fy:
                data_years.add(fy)
        _scan_relational((data or {}).get("relational", {}))

    return data_years


def _score_data_recency(route: str, data, classification: dict, cfg: dict) -> dict:
    """Score based on how recent the retrieved data is relative to the query."""
    penalty = cfg.get("penalty_per_year_stale", 20)
    requested_years = classification.get("years_involved", [])

    if not requested_years:
        return {"score": 80.0, "detail": "No specific year requested."}

    max_requested = max(requested_years)
    data_years = _extract_data_years(route, data)

    if not data_years:
        return {"score": 0.0, "detail": "No fiscal year found in retrieved data."}

    max_data_year = max(data_years)
    staleness = max(0, max_requested - max_data_year)
    score = max(0.0, 100.0 - staleness * penalty)

    return {
        "score": round(score, 1),
        "detail": (
            f"Latest data: FY{max_data_year}, requested: FY{max_requested} "
            f"({'current' if staleness == 0 else f'{staleness}yr stale'})."
        ),
    }


def _get_tier(score: float, tiers: list[dict]) -> dict:
    """Return the matching confidence tier for a given score."""
    # Tiers are sorted high-to-low by min_score in the config
    sorted_tiers = sorted(tiers, key=lambda t: t.get("min_score", 0), reverse=True)
    for tier in sorted_tiers:
        if score >= tier.get("min_score", 0):
            return tier
    # Fallback
    return {
        "label": "Unknown",
        "color": "dim",
        "description": "Could not determine confidence.",
    }


# ---------------------------------------------------------------------------
# 5. Formatting Helpers (for terminal / answer output)
# ---------------------------------------------------------------------------

COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def format_confidence_banner(confidence: dict) -> str:
    """Format the confidence score as a terminal-friendly banner."""
    score = confidence["overall_score"]
    tier = confidence["tier_label"]
    color = COLORS.get(confidence.get("tier_color", "dim"), COLORS["dim"])

    lines = [
        f"\n{COLORS['bold']}Confidence Assessment{COLORS['reset']}",
        f"{COLORS['dim']}{'─' * 45}{COLORS['reset']}",
        f"  {color}{COLORS['bold']}{tier}{COLORS['reset']} — {score}/100",
        f"  {COLORS['dim']}{confidence['tier_description']}{COLORS['reset']}",
    ]

    # Signal breakdown
    signals = confidence.get("signals", {})
    if signals:
        lines.append(f"  {COLORS['dim']}Signals:{COLORS['reset']}")
        for name, sig in signals.items():
            label = name.replace("_", " ").title()
            bar = _score_bar(sig["score"])
            lines.append(f"    {label:.<28s} {bar} {sig['score']:5.1f}  {COLORS['dim']}{sig['detail']}{COLORS['reset']}")

    lines.append(f"{COLORS['dim']}{'─' * 45}{COLORS['reset']}")
    return "\n".join(lines)


def _score_bar(score: float, width: int = 10) -> str:
    """Render a small ASCII bar for a 0-100 score."""
    filled = round(score / 100 * width)
    if score >= 75:
        color = COLORS["green"]
    elif score >= 50:
        color = COLORS["yellow"]
    else:
        color = COLORS["red"]
    return f"{color}{'█' * filled}{'░' * (width - filled)}{COLORS['reset']}"


def format_contradiction_warnings(contradictions: list[dict]) -> str:
    """Format contradiction warnings for appending to the answer."""
    if not contradictions:
        return ""

    lines = ["\n---", "**Data Consistency Warnings:**"]
    for c in contradictions:
        icon = "!!" if c.get("severity") == "high" else "!"
        lines.append(f"- [{icon}] {c['detail']}")

    return "\n".join(lines)


def format_confidence_for_answer(confidence: dict) -> str:
    """Format confidence tier for inline answer display (markdown)."""
    score = confidence["overall_score"]
    tier = confidence["tier_label"]
    desc = confidence["tier_description"]
    return f"\n---\n**Confidence: {tier}** ({score}/100) — {desc}"
