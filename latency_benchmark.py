#!/usr/bin/env python3
"""
SEC RAG Pipeline Latency Benchmark

Measures time spent in each phase of query processing:
  1. Query Classification (LLM routing)
  2. Retrieval Plan (metadata extraction)
  3. Data Retrieval (vector search / SQL / hybrid)
     - Sub-phases: embedding, DB query, reranking
  4. Context Formatting
  5. Answer Generation (LLM)

Usage:
  python latency_benchmark.py "What were Apple's risk factors in 2024?"
  python latency_benchmark.py --interactive
  python latency_benchmark.py --benchmark
  python latency_benchmark.py --embedding   # embedding-only latency test
"""

import time
import sys
import os
import statistics
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Tracks latency for each phase of query execution."""

    def __init__(self):
        self.phases: dict[str, float] = {}
        self._phase_start: Optional[float] = None
        self._current_phase: Optional[str] = None

    def start(self, phase_name: str):
        """Start timing a phase (auto-ends the previous one)."""
        if self._current_phase:
            self.stop()
        self._current_phase = phase_name
        self._phase_start = time.perf_counter()

    def stop(self):
        """Stop timing the current phase."""
        if self._current_phase and self._phase_start is not None:
            elapsed_ms = (time.perf_counter() - self._phase_start) * 1000
            self.phases[self._current_phase] = elapsed_ms
            self._current_phase = None

    @property
    def total_ms(self) -> float:
        return sum(self.phases.values())

    def report(self) -> str:
        """Return a formatted latency breakdown."""
        if not self.phases:
            return "No timing data available."

        total = self.total_ms
        width = 50
        lines = [
            "",
            "=" * 78,
            "QUERY LATENCY BREAKDOWN",
            "=" * 78,
            "",
        ]

        for phase, ms in self.phases.items():
            pct = (ms / total * 100) if total > 0 else 0
            filled = int(pct / 100 * width)
            bar = "\u2588" * filled + "\u2591" * (width - filled)
            lines.append(f"  {phase:<38} {ms:>7.1f}ms  ({pct:>5.1f}%)  {bar}")

        lines += [
            "",
            "-" * 78,
            f"  {'TOTAL':<38} {total:>7.1f}ms  (100.0%)",
            "=" * 78,
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Instrumented RAG pipeline
# ---------------------------------------------------------------------------

def run_query(query: str) -> dict:
    """Execute one query through the RAG pipeline with per-phase timing."""

    from rag_query import (
        classify_query,
        build_retrieval_plan,
        retrieve_vector_db,
        retrieve_relational_db,
        retrieve_hybrid,
        format_context,
        generate_answer,
        _build_sources,
    )

    t = LatencyTracker()

    # Phase 1 – classify
    t.start("Phase 1: Query Classification")
    classification = classify_query(query)
    t.stop()
    route = classification["route"]

    # Phase 2 – plan
    t.start("Phase 2: Retrieval Plan")
    retrieval_plan = build_retrieval_plan(query, classification)
    t.stop()

    # Phase 3 – retrieve
    t.start("Phase 3: Data Retrieval")
    if route == "relational_db":
        data = retrieve_relational_db(query, classification)
    elif route == "hybrid":
        data = retrieve_hybrid(query, classification)
    else:
        data = retrieve_vector_db(query, classification)
    t.stop()

    # Phase 4 – format
    t.start("Phase 4: Context Formatting")
    context = format_context(route, data, classification)
    sources = _build_sources(route, data, classification)
    t.stop()

    # Phase 5 – answer
    t.start("Phase 5: Answer Generation")
    if not context.strip():
        answer = (
            "I couldn't find relevant information for your query. "
            "Please check the ticker symbol and try rephrasing your question."
        )
    else:
        answer = generate_answer(query, context, classification)
        if sources:
            source_lines = "\n".join(f"- {s}" for s in sources)
            if "**Sources:**" not in answer:
                answer += f"\n\n---\n**Sources:**\n{source_lines}"
    t.stop()

    return {
        "answer": answer,
        "route": route,
        "reasoning": classification.get("reasoning", ""),
        "classification": classification,
        "sources": sources,
        "retrieval_plan": retrieval_plan,
        "tracker": t,
    }


def print_result(query: str, result: dict):
    """Print classification, timing, and answer."""

    t: LatencyTracker = result["tracker"]
    cls = result["classification"]

    # Timing
    print(t.report())

    # Classification
    print("=" * 78)
    print("CLASSIFICATION")
    print("=" * 78)
    print(f"  Route:        {cls['route'].upper()}")
    print(f"  Reasoning:    {cls.get('reasoning', 'N/A')}")
    print(f"  Query Type:   {cls.get('query_type', 'N/A')}")
    print(f"  Ticker:       {cls.get('ticker', 'N/A')}")
    print(f"  Fiscal Year:  {cls.get('fiscal_year', 'N/A')}")

    # Sources
    if result.get("sources"):
        print()
        print("=" * 78)
        print("SOURCES")
        print("=" * 78)
        for i, src in enumerate(result["sources"], 1):
            print(f"  {i}. {src}")

    # Answer
    print()
    print("=" * 78)
    print("ANSWER")
    print("=" * 78)
    print(result["answer"])
    print()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_single(query: str):
    """Run a single query and print results."""
    print(f"\nQuery: {query}\n")
    result = run_query(query)
    print_result(query, result)


def mode_interactive():
    """Interactive REPL with timing on every query."""

    print("\n" + "=" * 78)
    print("SEC RAG QUERY ENGINE - LATENCY BENCHMARK")
    print("=" * 78)
    print("\nPhases measured:")
    print("  1. Query Classification   2. Retrieval Plan")
    print("  3. Data Retrieval          4. Context Formatting")
    print("  5. Answer Generation")
    print("\nType 'quit' to exit.\n")

    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = run_query(query)
            print_result(query, result)
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


def mode_benchmark():
    """Run several queries and show average latencies per phase."""

    queries = [
        "What were Apple's main risk factors in 2024?",
        "What was Apple's revenue in Q3 2024?",
        "How did Apple's operating expenses change year-over-year?",
        "What is Apple's debt-to-equity ratio?",
        "Describe Apple's competitive advantages",
    ]

    print("\n" + "=" * 78)
    print(f"BENCHMARK  ({len(queries)} queries)")
    print("=" * 78 + "\n")

    phase_totals: dict[str, list[float]] = {}
    run_totals: list[float] = []

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query}")
        try:
            result = run_query(query)
            t: LatencyTracker = result["tracker"]
            run_totals.append(t.total_ms)
            for phase, ms in t.phases.items():
                phase_totals.setdefault(phase, []).append(ms)
            print(f"         -> {t.total_ms:.0f}ms  (route: {result['route']})")
        except Exception as e:
            print(f"         -> ERROR: {e}")

    # Summary
    print("\n" + "=" * 78)
    print("AVERAGE LATENCY PER PHASE")
    print("=" * 78 + "\n")

    phase_order = [
        "Phase 1: Query Classification",
        "Phase 2: Retrieval Plan",
        "Phase 3: Data Retrieval",
        "Phase 4: Context Formatting",
        "Phase 5: Answer Generation",
    ]

    grand_avg = 0.0
    for phase in phase_order:
        vals = phase_totals.get(phase, [])
        if not vals:
            continue
        avg = statistics.mean(vals)
        grand_avg += avg
        lo, hi = min(vals), max(vals)
        print(f"  {phase:<38} avg {avg:>7.1f}ms   [min {lo:>6.1f}, max {hi:>6.1f}]")

    print("-" * 78)
    print(f"  {'TOTAL':<38} avg {grand_avg:>7.1f}ms")
    if run_totals:
        print(f"  {'(end-to-end per query)':<38}     {statistics.mean(run_totals):>7.1f}ms")
    print("=" * 78 + "\n")


def mode_embedding():
    """Benchmark embedding API latency (single & batch)."""

    from openai import OpenAI

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sample_texts = [
        "What was Apple's Q3 2024 revenue?",
        "How did operating expenses change year-over-year?",
        "What are the main risk factors for this company?",
        "Describe the company's competitive advantage.",
        "What is the debt-to-equity ratio?",
        "How has the company's market share evolved?",
        "What is the dividend policy?",
        "Explain the accounting change in revenue recognition.",
        "What were the significant business acquisitions?",
        "How did international sales contribute to revenue?",
    ]

    print("\n" + "=" * 78)
    print(f"EMBEDDING LATENCY BENCHMARK  (model: {model})")
    print("=" * 78 + "\n")

    # --- single-query latencies ---
    print("Single-query latencies:")
    latencies = []
    for i, text in enumerate(sample_texts, 1):
        start = time.perf_counter()
        resp = client.embeddings.create(model=model, input=text)
        ms = (time.perf_counter() - start) * 1000
        latencies.append(ms)
        dim = len(resp.data[0].embedding)
        print(f"  {i:2d}. {ms:>7.2f}ms  {text[:50]}")

    avg = statistics.mean(latencies)
    print(f"\n  Mean: {avg:.2f}ms   Median: {statistics.median(latencies):.2f}ms"
          f"   Min: {min(latencies):.2f}ms   Max: {max(latencies):.2f}ms"
          f"   Dim: {dim}")

    # --- batch latencies ---
    print("\nBatch latencies:")
    for batch_size in (1, 5, 10):
        batches = [sample_texts[i:i+batch_size]
                    for i in range(0, len(sample_texts), batch_size)]
        batch_ms = []
        for batch in batches:
            start = time.perf_counter()
            client.embeddings.create(model=model, input=batch)
            batch_ms.append((time.perf_counter() - start) * 1000)
        print(f"  batch_size={batch_size:<3d}  avg {statistics.mean(batch_ms):>7.2f}ms")

    print("\n" + "=" * 78 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        mode_interactive()
        return

    arg = sys.argv[1]
    if arg in ("--interactive", "-i"):
        mode_interactive()
    elif arg in ("--benchmark", "--bench", "-b"):
        mode_benchmark()
    elif arg in ("--embedding", "-e"):
        mode_embedding()
    elif arg in ("--help", "-h"):
        print(__doc__)
    else:
        query = " ".join(sys.argv[1:])
        mode_single(query)


if __name__ == "__main__":
    main()
