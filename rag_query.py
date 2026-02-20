"""
SEC RAG Query Engine

Natural language query interface that routes questions through vector DB,
relational DB, or hybrid retrieval pipelines based on LLM classification.
"""

import json
import os
import sys
import time
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI

from chunk_and_embed import semantic_search_10k, semantic_search_10q
from xbrl_to_postgres import (
    query_annual,
    query_quarterly,
    search_concepts,
    get_metric_timeseries,
    query_q4,
)
from fetch_financials_to_postgres import get_statement
from cache import (
    get_cached_classification,
    set_cached_classification,
    get_cached_query_result,
    set_cached_query_result,
    get_cached_retrieval,
    set_cached_retrieval,
)
from guardrails import (
    load_guardrails,
    apply_retrieval_guardrails,
    detect_contradictions,
    compute_confidence,
    validate_query,
    validate_scope,
    format_confidence_banner,
    format_contradiction_warnings,
    format_confidence_for_answer,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load guardrail config at startup
_guardrails_cfg = load_guardrails()

MODEL = "gpt-4o-mini"


# --- Cost Tracking ---

# Pricing per 1M tokens (USD)
_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


class CostTracker:
    """Lightweight accumulator for OpenAI API token usage and cost."""

    def __init__(self):
        self._phases = []
        self._start = time.time()

    def record(self, phase: str, model: str, usage, elapsed_ms: float = 0):
        """Record tokens and cost for a completed API call."""
        pricing = _PRICING.get(model, {"input": 0.0, "output": 0.0})
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = prompt_tokens + completion_tokens
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
        throughput = round(total_tokens / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0

        self._phases.append({
            "phase": phase,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "throughput": throughput,
            "cached": False,
        })

    def record_embedding(self, phase: str, model: str, prompt_tokens: int, elapsed_ms: float = 0):
        """Record embedding API usage (no completion tokens)."""
        pricing = _PRICING.get(model, {"input": 0.02, "output": 0.0})
        cost = prompt_tokens * pricing["input"] / 1_000_000
        throughput = round(prompt_tokens / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0

        self._phases.append({
            "phase": phase,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
            "cost": cost,
            "throughput": throughput,
            "cached": False,
        })

    def record_cache_hit(self, phase: str):
        """Record a zero-cost cache hit entry."""
        self._phases.append({
            "phase": phase,
            "model": "cache",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "throughput": 0,
            "cached": True,
        })

    def _efficiency_rating(self, total_cost: float, total_tokens: int) -> dict:
        """Grade cost efficiency: S (cached), A+ (hyper efficient), A, B, C."""
        if total_cost == 0:
            return {"grade": "S", "label": "CACHED", "color": "green"}
        if total_tokens == 0:
            return {"grade": "C", "label": "NO DATA", "color": "red"}
        cost_per_1k = total_cost / (total_tokens / 1000)
        if cost_per_1k < 0.0002:
            return {"grade": "A+", "label": "HYPER EFFICIENT", "color": "green"}
        if cost_per_1k < 0.0005:
            return {"grade": "A", "label": "EFFICIENT", "color": "green"}
        if cost_per_1k < 0.001:
            return {"grade": "B", "label": "NORMAL", "color": "yellow"}
        return {"grade": "C", "label": "HEAVY", "color": "red"}

    def summary(self) -> dict:
        """Return aggregated cost summary."""
        total_cost = sum(p["cost"] for p in self._phases)
        total_tokens = sum(p["total_tokens"] for p in self._phases)
        total_prompt = sum(p["prompt_tokens"] for p in self._phases)
        total_completion = sum(p["completion_tokens"] for p in self._phases)
        wall_time_ms = round((time.time() - self._start) * 1000)
        avg_throughput = round(total_tokens / (wall_time_ms / 1000), 1) if wall_time_ms > 0 else 0
        efficiency = self._efficiency_rating(total_cost, total_tokens)

        return {
            "phases": self._phases,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "wall_time_ms": wall_time_ms,
            "throughput": avg_throughput,
            "efficiency": efficiency,
        }


def _today_str() -> str:
    """Return today's date as YYYY-MM-DD string for LLM context."""
    return date.today().isoformat()


# --- Common Financial Term → XBRL Concept Map ---
# XBRL tags are the ground source of truth for all metric lookups.
# The statements table is only used as fallback when XBRL data is missing
# or when the user explicitly asks for the full financial statement/table.
CONCEPT_ALIASES = {
    # --- Revenue / Sales ---
    "revenue": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:Revenues", "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueGoodsNet"],
    "net revenue": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:Revenues", "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueGoodsNet"],
    "net sales": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueNet", "us-gaap:Revenues", "us-gaap:SalesRevenueGoodsNet"],
    "total revenue": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:Revenues", "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueGoodsNet"],
    "sales": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueNet", "us-gaap:Revenues", "us-gaap:SalesRevenueGoodsNet"],
    "top line": ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:Revenues", "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueGoodsNet"],

    # --- Cost of Revenue / COGS ---
    "cost of revenue": ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfGoodsSold"],
    "cost of sales": ["us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsSold"],
    "cost of goods sold": ["us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsSold"],
    "cogs": ["us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsSold"],

    # --- Gross Profit ---
    "gross profit": ["us-gaap:GrossProfit"],
    "gross margin": ["us-gaap:GrossProfit"],

    # --- Operating Expenses ---
    "operating expenses": ["us-gaap:OperatingExpenses"],
    "opex": ["us-gaap:OperatingExpenses"],
    "total operating expenses": ["us-gaap:OperatingExpenses"],

    # --- R&D ---
    "research and development": ["us-gaap:ResearchAndDevelopmentExpense"],
    "r&d": ["us-gaap:ResearchAndDevelopmentExpense"],
    "r&d expense": ["us-gaap:ResearchAndDevelopmentExpense"],
    "r&d expenses": ["us-gaap:ResearchAndDevelopmentExpense"],
    "r and d": ["us-gaap:ResearchAndDevelopmentExpense"],
    "research and development expense": ["us-gaap:ResearchAndDevelopmentExpense"],
    "research & development": ["us-gaap:ResearchAndDevelopmentExpense"],

    # --- SG&A ---
    "selling general and administrative": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
    "sg&a": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
    "sga": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
    "sg&a expense": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
    "selling and administrative": ["us-gaap:SellingGeneralAndAdministrativeExpense"],
    "general and administrative": ["us-gaap:GeneralAndAdministrativeExpense"],
    "g&a": ["us-gaap:GeneralAndAdministrativeExpense"],

    # --- Operating Income ---
    "operating income": ["us-gaap:OperatingIncomeLoss"],
    "operating profit": ["us-gaap:OperatingIncomeLoss"],
    "operating loss": ["us-gaap:OperatingIncomeLoss"],
    "income from operations": ["us-gaap:OperatingIncomeLoss"],
    "ebit": ["us-gaap:OperatingIncomeLoss"],

    # --- Net Income ---
    "net income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "net profit": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "net loss": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "net earnings": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "bottom line": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "profit": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],

    # --- Income Before Tax ---
    "income before tax": ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "pretax income": ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "pre-tax income": ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "ebt": ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],

    # --- Tax ---
    "income tax": ["us-gaap:IncomeTaxExpenseBenefit"],
    "income tax expense": ["us-gaap:IncomeTaxExpenseBenefit"],
    "tax expense": ["us-gaap:IncomeTaxExpenseBenefit"],
    "provision for income taxes": ["us-gaap:IncomeTaxExpenseBenefit"],

    # --- EPS ---
    "eps": ["us-gaap:EarningsPerShareDiluted", "us-gaap:EarningsPerShareBasic"],
    "eps diluted": ["us-gaap:EarningsPerShareDiluted"],
    "eps basic": ["us-gaap:EarningsPerShareBasic"],
    "earnings per share": ["us-gaap:EarningsPerShareDiluted", "us-gaap:EarningsPerShareBasic"],
    "diluted eps": ["us-gaap:EarningsPerShareDiluted"],
    "basic eps": ["us-gaap:EarningsPerShareBasic"],

    # --- Balance Sheet: Assets ---
    "total assets": ["us-gaap:Assets"],
    "assets": ["us-gaap:Assets"],
    "current assets": ["us-gaap:AssetsCurrent"],
    "non-current assets": ["us-gaap:AssetsNoncurrent"],
    "cash and cash equivalents": ["us-gaap:CashAndCashEquivalentsAtCarryingValue"],
    "cash": ["us-gaap:CashAndCashEquivalentsAtCarryingValue"],
    "cash equivalents": ["us-gaap:CashAndCashEquivalentsAtCarryingValue"],
    "marketable securities": ["us-gaap:MarketableSecuritiesCurrent", "us-gaap:AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "short term investments": ["us-gaap:ShortTermInvestments", "us-gaap:MarketableSecuritiesCurrent"],
    "accounts receivable": ["us-gaap:AccountsReceivableNetCurrent"],
    "receivables": ["us-gaap:AccountsReceivableNetCurrent"],
    "inventory": ["us-gaap:InventoryNet"],
    "inventories": ["us-gaap:InventoryNet"],
    "goodwill": ["us-gaap:Goodwill"],
    "intangible assets": ["us-gaap:IntangibleAssetsNetExcludingGoodwill"],
    "property plant and equipment": ["us-gaap:PropertyPlantAndEquipmentNet"],
    "pp&e": ["us-gaap:PropertyPlantAndEquipmentNet"],
    "ppe": ["us-gaap:PropertyPlantAndEquipmentNet"],
    "fixed assets": ["us-gaap:PropertyPlantAndEquipmentNet"],

    # --- Balance Sheet: Liabilities ---
    "total liabilities": ["us-gaap:Liabilities"],
    "liabilities": ["us-gaap:Liabilities"],
    "current liabilities": ["us-gaap:LiabilitiesCurrent"],
    "non-current liabilities": ["us-gaap:LiabilitiesNoncurrent"],
    "long term debt": ["us-gaap:LongTermDebt", "us-gaap:LongTermDebtNoncurrent"],
    "long-term debt": ["us-gaap:LongTermDebt", "us-gaap:LongTermDebtNoncurrent"],
    "short term debt": ["us-gaap:ShortTermBorrowings"],
    "total debt": ["us-gaap:LongTermDebt", "us-gaap:DebtCurrent"],
    "accounts payable": ["us-gaap:AccountsPayableCurrent"],
    "deferred revenue": ["us-gaap:ContractWithCustomerLiabilityCurrent", "us-gaap:DeferredRevenueCurrent"],
    "commercial paper": ["us-gaap:CommercialPaper"],

    # --- Balance Sheet: Equity ---
    "total equity": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "stockholders equity": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "shareholders equity": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "book value": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "retained earnings": ["us-gaap:RetainedEarningsAccumulatedDeficit"],
    "accumulated deficit": ["us-gaap:RetainedEarningsAccumulatedDeficit"],
    "common stock": ["us-gaap:CommonStockValue"],
    "treasury stock": ["us-gaap:TreasuryStockValue"],

    # --- Shares ---
    "shares outstanding": ["us-gaap:CommonStockSharesOutstanding"],
    "diluted shares": ["us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"],
    "weighted average shares": ["us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", "us-gaap:WeightedAverageNumberOfShareOutstandingBasicAndDiluted"],
    "basic shares": ["us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"],

    # --- Cash Flow Statement ---
    "operating cash flow": ["us-gaap:NetCashProvidedByUsedInOperatingActivities", "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "cash from operations": ["us-gaap:NetCashProvidedByUsedInOperatingActivities", "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "cfo": ["us-gaap:NetCashProvidedByUsedInOperatingActivities", "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "investing cash flow": ["us-gaap:NetCashProvidedByUsedInInvestingActivities", "us-gaap:NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "cash from investing": ["us-gaap:NetCashProvidedByUsedInInvestingActivities", "us-gaap:NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "financing cash flow": ["us-gaap:NetCashProvidedByUsedInFinancingActivities", "us-gaap:NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    "cash from financing": ["us-gaap:NetCashProvidedByUsedInFinancingActivities", "us-gaap:NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    "capital expenditures": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap:PaymentsToAcquireProductiveAssets", "nvda:PurchasesOfPropertyAndEquipmentAndIntangibleAssets"],
    "capex": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap:PaymentsToAcquireProductiveAssets", "nvda:PurchasesOfPropertyAndEquipmentAndIntangibleAssets"],
    "depreciation": ["us-gaap:DepreciationDepletionAndAmortization", "us-gaap:Depreciation", "us-gaap:DepreciationAmortizationAndAccretionNet"],
    "depreciation and amortization": ["us-gaap:DepreciationDepletionAndAmortization", "us-gaap:DepreciationAmortizationAndAccretionNet"],
    "d&a": ["us-gaap:DepreciationDepletionAndAmortization", "us-gaap:DepreciationAmortizationAndAccretionNet"],
    "free cash flow": ["_COMPUTED_FCF"],  # FCF = CFO - CapEx (computed post-retrieval)
    "fcf": ["_COMPUTED_FCF"],
    "dividends paid": ["us-gaap:PaymentsOfDividends"],
    "dividends": ["us-gaap:PaymentsOfDividends"],
    "stock repurchases": ["us-gaap:PaymentsForRepurchaseOfCommonStock"],
    "share buybacks": ["us-gaap:PaymentsForRepurchaseOfCommonStock"],
    "buybacks": ["us-gaap:PaymentsForRepurchaseOfCommonStock"],

    # --- Interest ---
    "interest expense": ["us-gaap:InterestExpense"],
    "interest income": ["us-gaap:InvestmentIncomeInterest", "us-gaap:InterestIncomeExpenseNet"],
    "other income": ["us-gaap:OtherNonoperatingIncomeExpense", "us-gaap:NonoperatingIncomeExpense"],

    # --- EBITDA (derived, but map to components) ---
    "ebitda": ["us-gaap:OperatingIncomeLoss", "us-gaap:DepreciationDepletionAndAmortization"],
}

# Computed metrics: derived from two or more XBRL concepts post-retrieval.
# Each key is a sentinel tag used in CONCEPT_ALIASES; the value describes
# the components, the operation, and a human-friendly label.
COMPUTED_METRICS = {
    "_COMPUTED_FCF": {
        "label": "Free Cash Flow",
        "components": {
            "cfo": ["us-gaap:NetCashProvidedByUsedInOperatingActivities",
                     "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
            "capex": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
                       "us-gaap:PaymentsToAcquireProductiveAssets",
                       "nvda:PurchasesOfPropertyAndEquipmentAndIntangibleAssets"],
        },
        "compute": lambda parts: (parts["cfo"] or 0) - abs(parts.get("capex") or 0),
        "unit": "USD",
    },
}

# --- Colors for terminal output ---
COLORS = {
    "green": "\033[92m",
    "blue": "\033[94m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}

ROUTE_LABELS = {
    "metric_lookup": f"{COLORS['blue']}[METRIC LOOKUP]{COLORS['reset']}",
    "timeseries": f"{COLORS['blue']}[TIMESERIES]{COLORS['reset']}",
    "full_statement": f"{COLORS['blue']}[FULL STATEMENT]{COLORS['reset']}",
    "narrative": f"{COLORS['green']}[NARRATIVE]{COLORS['reset']}",
    "hybrid": f"{COLORS['magenta']}[HYBRID]{COLORS['reset']}",
    # Legacy aliases for backward compat
    "vector_db": f"{COLORS['green']}[NARRATIVE]{COLORS['reset']}",
    "relational_db": f"{COLORS['blue']}[METRIC LOOKUP]{COLORS['reset']}",
}

# --- Classification ---

CLASSIFY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "route_query",
            "description": (
                "Classify a user question about SEC filings and extract metadata.\n\n"
                "ROUTING RULES:\n"
                "- metric_lookup: Specific financial numbers (revenue, EPS, net income for a given year/quarter).\n"
                "- timeseries: Trends over multiple periods ('revenue from 2020 to 2024', 'show revenue trend').\n"
                "- full_statement: Complete financial statements ('show income statement', 'balance sheet for 2024').\n"
                "- narrative: Qualitative/text questions (risk factors, strategy, MD&A discussion, legal proceedings).\n"
                "- hybrid: Questions needing both numbers AND narrative context ('What drove revenue changes?').\n\n"
                "TEMPORAL GRANULARITY RULES:\n"
                "- annual: Query is about full-year figures (default when no quarter is mentioned).\n"
                "- quarterly: Query asks about quarterly data across periods ('quarterly revenue for 2023 and 2024').\n"
                "- specific_quarter: Query asks about one specific quarter ('Q3 2024 revenue')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {
                        "type": "string",
                        "enum": ["metric_lookup", "timeseries", "full_statement", "narrative", "hybrid"],
                        "description": "The retrieval route to use.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this route was chosen.",
                    },
                    "temporal_granularity": {
                        "type": "string",
                        "enum": ["annual", "quarterly", "specific_quarter"],
                        "description": (
                            "Temporal granularity of the query. "
                            "'annual' = full-year figures (default). "
                            "'quarterly' = quarterly breakdown across periods (e.g. 'quarterly revenue for 2023 and 2024'). "
                            "'specific_quarter' = one specific quarter (e.g. 'Q3 2024 revenue')."
                        ),
                    },
                    "retrieval_intent": {
                        "type": "string",
                        "enum": ["specific_metric", "full_statement", "narrative", "comparison", "timeseries"],
                        "description": (
                            "What kind of data the user wants: "
                            "specific_metric (one or a few numbers), "
                            "full_statement (complete financial table), "
                            "narrative (qualitative text from filings), "
                            "comparison (side-by-side metrics across periods or companies), "
                            "timeseries (metric across many periods for trend analysis)."
                        ),
                    },
                    "search_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-4 search queries optimized for semantic search on SEC filing text. "
                            "Include the original query plus expanded variants for better recall. "
                            "For segment queries add 'revenue by business segment breakdown'. "
                            "For causal queries add 'factors contributing to changes and growth drivers'. "
                            "For risk queries add 'key risk factors and uncertainties'. "
                            "For trend queries add 'year over year performance trends'."
                        ),
                    },
                    "xbrl_concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "XBRL concept tags for the requested metrics. Use ONLY well-known tags:\n"
                            "Revenue: us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, us-gaap:Revenues\n"
                            "Net Income: us-gaap:NetIncomeLoss\n"
                            "EPS: us-gaap:EarningsPerShareDiluted, us-gaap:EarningsPerShareBasic\n"
                            "Gross Profit: us-gaap:GrossProfit\n"
                            "Operating Income: us-gaap:OperatingIncomeLoss\n"
                            "Total Assets: us-gaap:Assets\n"
                            "Total Liabilities: us-gaap:Liabilities\n"
                            "Stockholders Equity: us-gaap:StockholdersEquity\n"
                            "Cash: us-gaap:CashAndCashEquivalentsAtCarryingValue\n"
                            "Operating Cash Flow: us-gaap:NetCashProvidedByUsedInOperatingActivities\n"
                            "R&D: us-gaap:ResearchAndDevelopmentExpense\n"
                            "SG&A: us-gaap:SellingGeneralAndAdministrativeExpense\n"
                            "Cost of Revenue: us-gaap:CostOfRevenue\n"
                            "Long-term Debt: us-gaap:LongTermDebt\n"
                            "Shares Outstanding: us-gaap:CommonStockSharesOutstanding\n"
                            "CapEx: us-gaap:PaymentsToAcquirePropertyPlantAndEquipment\n"
                            "Dividends: us-gaap:PaymentsOfDividends\n"
                            "Depreciation: us-gaap:DepreciationDepletionAndAmortization\n"
                            "Income Tax: us-gaap:IncomeTaxExpenseBenefit\n"
                            "Operating Expenses: us-gaap:OperatingExpenses\n"
                            "Leave empty if unsure — the system will resolve from plain-English concepts."
                        ),
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Primary stock ticker symbol mentioned in the query (e.g. 'AAPL'). Null if not mentioned.",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "ALL stock ticker symbols mentioned in the query (e.g. ['AAPL', 'MSFT', 'GOOGL']).",
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": (
                            "Fiscal year explicitly mentioned in the query (e.g. 2024, FY2023). "
                            "MUST be null when the user says 'latest', 'recent', 'current', 'most recent', "
                            "or does not specify any year — the system will auto-resolve to the latest available data."
                        ),
                    },
                    "fiscal_quarter": {
                        "type": "integer",
                        "description": (
                            "Fiscal quarter (1-4) explicitly mentioned in the query. "
                            "Null if not mentioned, if user says 'latest quarter', "
                            "or if DIFFERENT quarters are requested for different years "
                            "(use year_quarters instead)."
                        ),
                    },
                    "year_quarters": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "description": (
                            "Map of fiscal year to specific quarter when different quarters are "
                            "requested for different years, e.g. {\"2023\": 3, \"2016\": 2} for "
                            "'Q3 2023 and Q2 2016'. Leave empty/null when all years use the same quarter "
                            "or no specific quarter is mentioned."
                        ),
                    },
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "causal", "comparative", "ranking", "trend",
                            "geographic", "reconciliation", "risk_analysis",
                            "margin_analysis", "inference", "external_factor",
                            "consistency", "factual",
                        ],
                        "description": (
                            "The type of analysis the question requires: "
                            "causal (what caused X), comparative (X vs Y), ranking (which is largest), "
                            "trend (how did X change over time), geographic (regional comparison), "
                            "reconciliation (quarterly vs annual), risk_analysis (risk factors), "
                            "margin_analysis (cost/profitability), inference (implicit conclusions), "
                            "external_factor (FX/macro), consistency (cross-filing comparison), "
                            "factual (simple lookup)."
                        ),
                    },
                    "years_involved": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "All fiscal years explicitly mentioned or implied in the query (e.g. [2024, 2025]). "
                            "Leave EMPTY when user says 'latest'/'recent' or does not mention any year — "
                            "the system will auto-resolve to the most recent data."
                        ),
                    },
                    "filing_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["10-K", "10-Q"]},
                        "description": "SEC filing types relevant to the query.",
                    },
                    "target_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Filing sections to search, e.g. 'MD&A', 'Risk Factors', "
                            "'Results of Operations', 'Geographic Results'."
                        ),
                    },
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Financial concepts the user is asking about, expressed as plain English terms "
                            "(e.g. 'revenue', 'net income', 'total assets'). Used as fallback if xbrl_concepts is empty."
                        ),
                    },
                    "statement_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "income_statement",
                                "balance_sheet",
                                "cash_flow_statement",
                            ],
                        },
                        "description": "Financial statement types relevant to the query.",
                    },
                },
                "required": ["route", "reasoning", "query_type", "temporal_granularity", "retrieval_intent"],
            },
        },
    }
]

# Backward-compat: map old 3-route names to new 5-route names
_ROUTE_MIGRATION = {
    "relational_db": "metric_lookup",
    "vector_db": "narrative",
    "comparison": "metric_lookup",
}


def _post_classify(query: str, classification: dict) -> dict:
    """Apply post-classification fixes that must run even on cached results.

    This handles corrections added after classifications were already cached
    (e.g. query_type upgrades, route overrides).
    """
    import re as _re
    _RISK_KEYWORDS = r'\brisk(?:s| factors?| factor)\b|\bitem\s*1a\b|\brisk management\b'
    _QUALITATIVE_KEYWORDS = r'\blitigation\b|\bregulatory\b|\bgovernance\b|\bcorporate governance\b'
    if _re.search(_RISK_KEYWORDS + '|' + _QUALITATIVE_KEYWORDS, query, _re.IGNORECASE):
        if classification.get("query_type") == "factual":
            classification["query_type"] = "risk_analysis"
    return classification


def classify_query(query: str, *, cost_tracker: CostTracker | None = None) -> dict:
    """Use LLM function calling to classify the query and extract metadata."""
    # Check cache first
    cached = get_cached_classification(query)
    if cached is not None:
        cached["_cache_hit"] = True
        if cost_tracker:
            cost_tracker.record_cache_hit("classify")
        # Apply post-classification fixes to cached results too (fixes added
        # after the classification was originally cached).
        cached = _post_classify(query, cached)
        return cached

    _t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You classify questions about SEC filings (10-K and 10-Q) and extract structured metadata.\n\n"
                    f"TODAY'S DATE: {_today_str()}. Data coverage: 2010 through 2025 annual filings, "
                    f"plus Q1 2026 quarterly filings where available.\n\n"
                    "COMMON TICKERS: AAPL=Apple, MSFT=Microsoft, NVDA=Nvidia, AMZN=Amazon, "
                    "GOOGL=Google/Alphabet, META=Meta/Facebook, BRK-B=Berkshire Hathaway, "
                    "LLY=Eli Lilly, AVGO=Broadcom, JPM=JPMorgan Chase.\n\n"
                    "LATEST/RECENT HANDLING:\n"
                    "- When the user says 'latest', 'recent', 'current', 'most recent', 'newest', or does NOT "
                    "specify a year, set fiscal_year to null and leave years_involved empty. "
                    "The system will automatically look up the most recent data available in the database.\n"
                    "- Only set fiscal_year when the user explicitly mentions a specific year (e.g. '2024', 'FY2023').\n"
                    "- For 'latest quarterly' or 'most recent quarter', also set fiscal_year to null "
                    "and fiscal_quarter to null — the system will find the most recent quarter.\n\n"
                    "ROUTING (5 routes):\n"
                    "- metric_lookup: User wants specific numbers (revenue, EPS, net income for given year/quarter).\n"
                    "- timeseries: User wants a metric tracked across many periods ('revenue trend 2020-2024').\n"
                    "- full_statement: User wants a complete financial table ('show me the income statement').\n"
                    "- narrative: User wants qualitative text (risk factors, strategy, MD&A, legal proceedings).\n"
                    "- hybrid: User needs BOTH numbers AND narrative ('what drove revenue growth?').\n\n"
                    "TEMPORAL GRANULARITY:\n"
                    "- annual: Full-year figures. Default when no quarter mentioned.\n"
                    "- quarterly: Quarterly breakdown across periods ('quarterly revenue for 2023 and 2024').\n"
                    "  IMPORTANT: 'quarterly revenue for 2023 and 2024' means quarterly granularity, NOT annual.\n"
                    "- specific_quarter: One specific quarter ('Q3 2024 revenue').\n\n"
                    "RETRIEVAL INTENT:\n"
                    "- specific_metric: One or a few numbers.\n"
                    "- full_statement: Complete financial table.\n"
                    "- narrative: Qualitative text from filings.\n"
                    "- comparison: Side-by-side metrics across periods or companies.\n"
                    "- timeseries: Metric across many periods for trend analysis.\n\n"
                    "SEARCH QUERIES: Generate 1-4 semantic search queries for vector retrieval. "
                    "Always include the original query. Add domain-specific expansions for "
                    "segment, causal, risk, or trend queries.\n\n"
                    "XBRL CONCEPTS: Map requested metrics to well-known us-gaap tags. "
                    "Only use tags you are confident about. Leave empty if unsure."
                ),
            },
            {"role": "user", "content": query},
        ],
        tools=CLASSIFY_TOOLS,
        tool_choice={"type": "function", "function": {"name": "route_query"}},
    )

    if cost_tracker and response.usage:
        cost_tracker.record("classify", MODEL, response.usage, (time.time() - _t0) * 1000)

    tool_call = response.choices[0].message.tool_calls[0]
    classification = json.loads(tool_call.function.arguments)

    # Defaults for optional fields – use `or` to coerce JSON null → proper default
    classification["ticker"] = classification.get("ticker") or None
    classification["tickers"] = classification.get("tickers") or []
    classification["fiscal_year"] = classification.get("fiscal_year") or None
    classification["fiscal_quarter"] = classification.get("fiscal_quarter") or None
    classification["concepts"] = classification.get("concepts") or []
    classification["statement_types"] = classification.get("statement_types") or []
    classification["query_type"] = classification.get("query_type") or "factual"
    classification["years_involved"] = classification.get("years_involved") or []
    classification["filing_types"] = classification.get("filing_types") or ["10-K"]
    classification["target_sections"] = classification.get("target_sections") or []

    # New fields (Phase 1)
    classification["temporal_granularity"] = classification.get("temporal_granularity") or "annual"
    classification["retrieval_intent"] = classification.get("retrieval_intent") or "specific_metric"
    classification["search_queries"] = classification.get("search_queries") or []
    classification["xbrl_concepts"] = classification.get("xbrl_concepts") or []

    # Per-year quarter mapping: normalize string keys to int
    raw_yq = classification.get("year_quarters") or {}
    if not isinstance(raw_yq, dict):
        raw_yq = {}
    classification["year_quarters"] = {int(k): v for k, v in raw_yq.items()} if raw_yq else {}

    # Backward-compat route migration: map old 3-route names to new 5-route names
    classification["route"] = _ROUTE_MIGRATION.get(classification["route"], classification["route"])

    # Normalize tickers: ensure ticker ↔ tickers are in sync
    if classification["ticker"] and classification["ticker"] not in classification["tickers"]:
        classification["tickers"].append(classification["ticker"])
    if classification["tickers"] and not classification["ticker"]:
        classification["ticker"] = classification["tickers"][0]
    # If only ticker was set and tickers is empty, populate tickers
    if classification["ticker"] and not classification["tickers"]:
        classification["tickers"] = [classification["ticker"]]

    # Safeguard: scan query text for known tickers AND company names the LLM may have missed
    from config import TICKERS as _KNOWN_TICKERS
    _COMPANY_NAME_TO_TICKER = {
        "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
        "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META",
        "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
        "eli lilly": "LLY", "lilly": "LLY",
        "broadcom": "AVGO", "jpmorgan": "JPM", "jp morgan": "JPM",
        "jpmorgan chase": "JPM", "chase": "JPM",
    }
    _query_upper = query.upper()
    for t in _KNOWN_TICKERS:
        if t in _query_upper and t not in classification["tickers"]:
            classification["tickers"].append(t)
    # Also match company names (case-insensitive)
    _query_lower = query.lower()
    for name, ticker in _COMPANY_NAME_TO_TICKER.items():
        if name in _query_lower and ticker not in classification["tickers"]:
            classification["tickers"].append(ticker)
    # If tickers grew beyond what LLM set, ensure ticker is still the first
    if classification["tickers"] and not classification["ticker"]:
        classification["ticker"] = classification["tickers"][0]
    # Multi-ticker "vs" queries should be comparative
    if len(classification["tickers"]) > 1 and classification["query_type"] == "factual":
        import re as _re
        if _re.search(r'\bvs\.?\b|\bversus\b|\bcompare\b|\bcomparison\b', query, _re.IGNORECASE):
            classification["query_type"] = "comparative"

    # Ensure fiscal_year is populated — fall back to years_involved if LLM left it None
    if not classification["fiscal_year"] and classification["years_involved"]:
        classification["fiscal_year"] = classification["years_involved"][0]

    # Ensure fiscal_year is in years_involved
    if classification["fiscal_year"] and classification["fiscal_year"] not in classification["years_involved"]:
        classification["years_involved"].append(classification["fiscal_year"])

    # Post-classification route override: force risk/qualitative queries to narrative
    # so they retrieve actual Item 1A risk factor text instead of financial numbers.
    import re as _re2
    _RISK_KEYWORDS = r'\brisk(?:s| factors?| factor)\b|\bitem\s*1a\b|\brisk management\b'
    _QUALITATIVE_KEYWORDS = r'\blitigation\b|\bregulatory\b|\bgovernance\b|\bcorporate governance\b'
    _is_risk_query = _re2.search(_RISK_KEYWORDS + '|' + _QUALITATIVE_KEYWORDS, query, _re2.IGNORECASE)
    if (
        classification["route"] in ("metric_lookup", "timeseries", "full_statement")
        and _is_risk_query
        and not classification.get("xbrl_concepts")  # no explicit XBRL metrics requested
    ):
        classification["route"] = "narrative"

    # Ensure risk/qualitative queries get the right query_type so downstream
    # logic fetches more chunks, adds risk sub-queries, and prepends a financial snapshot.
    if _is_risk_query and classification["query_type"] == "factual":
        classification["query_type"] = "risk_analysis"

    # Post-classification route override: correct misroutes when the LLM's own
    # metadata signals a quantitative query (XBRL concepts present, retrieval
    # intent is not narrative) but the route landed on narrative or an unknown value.
    _VALID_ROUTES = {"metric_lookup", "timeseries", "full_statement", "narrative", "hybrid"}
    if (
        classification["route"] not in _VALID_ROUTES
        or (
            classification["route"] == "narrative"
            and classification.get("xbrl_concepts")
            and classification.get("retrieval_intent") in ("comparison", "specific_metric", "timeseries", "full_statement")
        )
    ):
        intent = classification.get("retrieval_intent", "specific_metric")
        if intent in ("comparison", "specific_metric"):
            classification["route"] = "metric_lookup"
        elif intent == "timeseries":
            classification["route"] = "timeseries"
        elif intent == "full_statement":
            classification["route"] = "full_statement"
        else:
            classification["route"] = "metric_lookup"

    # Cache the classification before returning
    set_cached_classification(query, classification)
    return classification


# --- Query Decomposition for Multi-Company Queries ---

DECOMPOSE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "decompose_query",
            "description": (
                "Decompose a complex multi-company financial query into per-company sub-queries. "
                "Each sub-query targets ONE ticker with specific metrics and time range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "A focused sub-query for one company and specific metrics.",
                                },
                                "ticker": {
                                    "type": "string",
                                    "description": "Single ticker symbol for this sub-query.",
                                },
                                "concepts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Plain-English financial concepts for this sub-query "
                                        "(e.g. 'revenue', 'net income', 'operating income')."
                                    ),
                                },
                                "years": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": (
                                        "Fiscal years to fetch. For growth comparisons include 2 consecutive years. "
                                        "Leave empty to auto-resolve to latest."
                                    ),
                                },
                                "year_quarters": {
                                    "type": "object",
                                    "additionalProperties": {"type": "integer"},
                                    "description": (
                                        "Map of fiscal year to specific quarter for that year, "
                                        "e.g. {\"2023\": 3, \"2016\": 2} for Q3 2023 and Q2 2016. "
                                        "Only include entries for years where a specific quarter is requested. "
                                        "Omit years that need annual (full-year) data."
                                    ),
                                },
                                "purpose": {
                                    "type": "string",
                                    "description": "What this sub-query contributes to the overall answer (e.g. 'profitability data for Apple').",
                                },
                            },
                            "required": ["query", "ticker", "concepts", "purpose"],
                        },
                        "description": "List of focused sub-queries, one per ticker per analysis aspect.",
                    },
                },
                "required": ["sub_queries"],
            },
        },
    }
]

_DECOMPOSE_QUERY_TYPES = {"comparative", "ranking", "trend", "causal", "margin_analysis"}


def decompose_query(query: str, classification: dict, *, cost_tracker: CostTracker | None = None) -> list[dict] | None:
    """Decompose a multi-company query into per-ticker sub-queries using LLM.

    Returns a list of sub-query dicts, or None if decomposition is not needed
    (single ticker or non-comparative query type).
    """
    tickers = classification.get("tickers", [])
    query_type = classification.get("query_type", "factual")

    # Decompose multi-ticker queries: comparative/ranking/trend always,
    # plus factual when multiple tickers are present (each needs its own retrieval).
    if len(tickers) <= 1:
        return None
    if query_type not in _DECOMPOSE_QUERY_TYPES and query_type != "factual":
        return None

    concepts = classification.get("concepts", [])
    years_involved = classification.get("years_involved", [])

    _t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You decompose complex multi-company financial queries into focused per-company sub-queries.\n\n"
                    f"TODAY'S DATE: {_today_str()}.\n\n"
                    "RULES:\n"
                    "- Create one sub-query per ticker per analysis aspect.\n"
                    "- Each sub-query must target exactly ONE ticker.\n"
                    "- Include specific financial concepts (plain English: 'revenue', 'net income', etc.).\n"
                    "- For 'growth' or 'growing faster' questions, include 2 consecutive years so YoY change can be computed.\n"
                    "- If no specific years are mentioned, leave years empty (system auto-resolves to latest).\n"
                    "- For profitability questions, include: revenue, net income, operating income, gross profit.\n"
                    "- For growth questions, include: revenue (current and prior year), net income (current and prior year).\n"
                    "- IMPORTANT: When different quarters are requested for different years (e.g. 'Q3 2023 and Q2 2016'), "
                    "use the year_quarters field to map each year to its specific quarter.\n"
                    f"\nTickers in query: {tickers}\n"
                    f"Detected concepts: {concepts}\n"
                    f"Years mentioned: {years_involved if years_involved else 'latest (auto-resolve)'}\n"
                    f"Query type: {query_type}"
                ),
            },
            {"role": "user", "content": query},
        ],
        tools=DECOMPOSE_TOOLS,
        tool_choice={"type": "function", "function": {"name": "decompose_query"}},
    )

    if cost_tracker and response.usage:
        cost_tracker.record("decompose", MODEL, response.usage, (time.time() - _t0) * 1000)

    tool_call = response.choices[0].message.tool_calls[0]
    result = json.loads(tool_call.function.arguments)
    sub_queries = result.get("sub_queries", [])

    if not sub_queries:
        return None

    # Ensure each sub-query has valid defaults
    for sq in sub_queries:
        sq["concepts"] = sq.get("concepts") or concepts
        sq["years"] = sq.get("years") or []
        sq["year_quarters"] = sq.get("year_quarters") or {}

        # Fallback: parse quarter-year pairs from sub-query text when year_quarters is empty
        if not sq["year_quarters"]:
            import re
            text = f"{sq.get('query', '')} {sq.get('purpose', '')}"
            for m in re.finditer(r"Q([1-4])\s*(\d{4})", text, re.IGNORECASE):
                q, y = int(m.group(1)), int(m.group(2))
                sq["year_quarters"][y] = q

    return sub_queries


def _run_subquery_retrieval(sub_queries: list[dict], classification: dict) -> dict:
    """Execute sub-queries from decomposition and merge results.

    Each sub-query gets its own retrieval call with a tailored classification dict.
    For narrative routes, uses retrieve_narrative per ticker; otherwise metric_lookup.
    Sub-queries run in parallel via ThreadPoolExecutor for lower latency.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    original_route = classification.get("route", "metric_lookup")
    is_narrative = original_route in ("narrative", "hybrid")

    def _execute_one(sq):
        year_quarters = sq.get("year_quarters") or {}
        if not isinstance(year_quarters, dict):
            year_quarters = {}
        year_quarters = {int(k): v for k, v in year_quarters.items()} if year_quarters else {}

        temporal = classification.get("temporal_granularity", "annual")
        if year_quarters:
            temporal = "specific_quarter"

        sq_classification = {
            "route": original_route,
            "reasoning": f"Sub-query for {sq['ticker']}: {sq['purpose']}",
            "query_type": classification.get("query_type", "comparative"),
            "temporal_granularity": temporal,
            "retrieval_intent": classification.get("retrieval_intent", "comparison"),
            "ticker": sq["ticker"],
            "tickers": [sq["ticker"]],
            "concepts": sq["concepts"],
            "xbrl_concepts": [],
            "fiscal_year": None,
            "fiscal_quarter": classification.get("fiscal_quarter"),
            "year_quarters": year_quarters,
            "years_involved": sq.get("years", []),
            "filing_types": classification.get("filing_types", ["10-K"]),
            "target_sections": classification.get("target_sections", []),
            "statement_types": classification.get("statement_types", []),
            "search_queries": classification.get("search_queries", []),
        }

        if is_narrative:
            return {"type": "narrative", "data": retrieve_narrative(sq["query"], sq_classification)}
        else:
            return {"type": "relational", "data": retrieve_metric_lookup(sq["query"], sq_classification)}

    merged_relational = {"xbrl_facts": [], "timeseries": [], "statements": [], "comparisons": []}
    merged_vector = []

    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as executor:
        futures = {executor.submit(_execute_one, sq): sq for sq in sub_queries}
        for future in as_completed(futures):
            result = future.result()
            if result["type"] == "narrative":
                merged_vector.extend(result["data"])
            else:
                data = result["data"]
                merged_relational["xbrl_facts"].extend(data.get("xbrl_facts", []))
                merged_relational["timeseries"].extend(data.get("timeseries", []))
                merged_relational["statements"].extend(data.get("statements", []))
                merged_relational["comparisons"].extend(data.get("comparisons", []))

    return {"relational": merged_relational, "vector": merged_vector}


# --- Fiscal Year Default ---

def get_latest_fiscal_year(ticker: str) -> int | None:
    """Query the DB for the most recent fiscal year available for a ticker.

    Checks annual_facts and quarterly_facts to find the absolute latest
    fiscal year — quarterly data may be newer than the most recent annual
    filing (e.g. Q1 2026 before FY2025 10-K is filed).
    """
    from xbrl_to_postgres import get_db_connection
    max_year = None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(fiscal_year) FROM annual_facts WHERE ticker = %s",
                    (ticker,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    max_year = row[0]

                cur.execute(
                    "SELECT MAX(fiscal_year) FROM quarterly_facts WHERE ticker = %s",
                    (ticker,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    max_year = max(max_year or 0, row[0])
    except Exception:
        # Fallback to original approach
        facts = query_annual(ticker=ticker, limit=1)
        if facts:
            return facts[0]["fiscal_year"]
    return max_year


# --- Concept Resolution ---

def resolve_concepts(human_terms: list[str]) -> dict[str, list[str]]:
    """Map human-readable financial terms to XBRL concept names.

    Resolution order:
      1. Exact match in CONCEPT_ALIASES.
      2. Substring match in CONCEPT_ALIASES keys (e.g. "r&d" matches "r&d expense").
      3. Fuzzy ILIKE search against actual DB concepts.
    """
    resolved = {}
    alias_keys = list(CONCEPT_ALIASES.keys())

    for term in human_terms:
        term_lower = term.lower().strip()

        # 1. Exact alias match
        if term_lower in CONCEPT_ALIASES:
            resolved[term] = CONCEPT_ALIASES[term_lower]
            continue

        # 2. Substring match: find alias keys that contain the term or vice-versa
        substring_hits = []
        for alias_key in alias_keys:
            if term_lower in alias_key or alias_key in term_lower:
                substring_hits.extend(CONCEPT_ALIASES[alias_key])
        if substring_hits:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for h in substring_hits:
                if h not in seen:
                    seen.add(h)
                    deduped.append(h)
            resolved[term] = deduped[:3]
            continue

        # 3. Fuzzy DB search
        matches = search_concepts(term, table="annual_facts", limit=5)
        if matches:
            resolved[term] = matches

    return resolved


# --- XBRL Concept Validation (Phase 5) ---

_known_xbrl_concepts: set[str] | None = None


def _load_known_concepts() -> set[str]:
    """Load all distinct XBRL concept names from the DB into a set (once at startup)."""
    global _known_xbrl_concepts
    if _known_xbrl_concepts is not None:
        return _known_xbrl_concepts

    from xbrl_to_postgres import get_db_connection
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT concept FROM annual_facts")
                annual = {row[0] for row in cur.fetchall()}
                cur.execute("SELECT DISTINCT concept FROM quarterly_facts")
                quarterly = {row[0] for row in cur.fetchall()}
                _known_xbrl_concepts = annual | quarterly
    except Exception:
        _known_xbrl_concepts = set()
    return _known_xbrl_concepts


def validate_xbrl_concepts(concepts: list[str]) -> list[str]:
    """Filter LLM-suggested XBRL concepts to only those that exist in the DB."""
    known = _load_known_concepts()
    if not known:
        return []
    return [c for c in concepts if c in known]


# --- Data Availability ---

# IPO / earliest public data year for companies that went public after our
# pipeline start year (2010). Queries for years before IPO should warn the user.
_IPO_YEAR = {
    "META": (2012, "Meta (Facebook) had its IPO in May 2012. No SEC filings or financial data exist before 2012."),
    "AVGO": (2009, "Broadcom (as Avago Technologies) had its IPO in August 2009. Data before FY2010 is limited."),
}

def check_data_availability(classification: dict) -> dict:
    """Check which tickers have filings/data for the requested years.

    Returns dict with:
      - "available": {ticker: [years_with_data]}
      - "missing": {ticker: [years_without_data]}
      - "notes": list of human-readable notes about gaps
    """
    tickers = classification.get("tickers", [])
    if not tickers and classification.get("ticker"):
        tickers = [classification["ticker"]]
    years = sorted(classification.get("years_involved", []))
    if not years and classification.get("fiscal_year"):
        years = [classification["fiscal_year"]]
    if not tickers or not years:
        return {"available": {}, "missing": {}, "notes": []}

    from xbrl_to_postgres import get_db_connection

    available = {}
    missing = {}
    notes = []

    from config import calendar_to_fiscal_year as _cal_to_fy

    # Check for pre-IPO queries first
    for ticker in tickers:
        if ticker in _IPO_YEAR:
            ipo_year, ipo_note = _IPO_YEAR[ticker]
            pre_ipo_years = [y for y in years if y < ipo_year]
            if pre_ipo_years:
                notes.append(ipo_note)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for ticker in tickers:
                    avail_years = []
                    miss_years = []
                    for year in years:
                        fy = _cal_to_fy(ticker, year)
                        cur.execute(
                            "SELECT COUNT(*) FROM filings WHERE ticker = %s AND fiscal_year = %s AND form_type = '10-K'",
                            (ticker, fy),
                        )
                        has_filing = cur.fetchone()[0] > 0
                        if has_filing:
                            avail_years.append(year)
                        else:
                            miss_years.append(year)
                    available[ticker] = avail_years
                    missing[ticker] = miss_years

                    if miss_years:
                        cur.execute(
                            "SELECT MIN(fiscal_year), MAX(fiscal_year) FROM filings WHERE ticker = %s AND form_type = '10-K'",
                            (ticker,),
                        )
                        row = cur.fetchone()
                        min_yr, max_yr = row if row and row[0] else (None, None)
                        year_str = ", ".join(str(y) for y in miss_years)
                        if min_yr:
                            notes.append(
                                f"{ticker} has no 10-K filings for FY{year_str}. "
                                f"Available filing range: FY{min_yr}–FY{max_yr}."
                            )
                        else:
                            notes.append(f"{ticker} has no 10-K filings in the database.")
    except Exception:
        pass

    return {"available": available, "missing": missing, "notes": notes}


# --- Retrieval Plan ---

def build_retrieval_plan(query: str, classification: dict, data_availability: dict | None = None) -> dict:
    """Build a structured retrieval plan that mirrors the actual retrieval execution.

    The plan reflects the real dispatch logic in rag_query():
      - 5-way route dispatch: metric_lookup, timeseries, full_statement, narrative, hybrid
      - XBRL concept resolution (3-layer: validated → aliases → fuzzy DB)
      - Statement fallback when XBRL returns no data
      - Auto-year expansion for comparative/trend queries
      - Multi-ticker top_k adjustment for vector search
      - Post-retrieval: guardrail filtering, contradiction detection, confidence scoring
    """
    route = classification["route"]
    query_type = classification.get("query_type", "factual")
    years = sorted(classification.get("years_involved", []))
    tickers = classification.get("tickers", [])
    ticker = classification.get("ticker")
    concepts = classification.get("concepts", [])
    filing_types = classification.get("filing_types", ["10-K"])
    target_sections = classification.get("target_sections", [])
    fiscal_year = classification.get("fiscal_year")
    fiscal_quarter = classification.get("fiscal_quarter")
    statement_types = classification.get("statement_types", [])
    temporal = classification.get("temporal_granularity", "annual")
    retrieval_intent = classification.get("retrieval_intent", "specific_metric")
    xbrl_concepts = classification.get("xbrl_concepts", [])

    effective_tickers = tickers or ([ticker] if ticker else [])
    is_multi_year = len(years) >= 2
    is_multi_ticker = len(effective_tickers) > 1
    use_quarterly = temporal in ("quarterly", "specific_quarter")

    steps = []

    # Step 1: Intent Detection
    intent = {"type": query_type}
    if effective_tickers:
        intent["tickers"] = ", ".join(effective_tickers)
    if years:
        intent["years"] = " -> ".join(str(y) for y in years) if len(years) > 1 else str(years[0])
    elif fiscal_year:
        intent["years"] = str(fiscal_year)
    else:
        intent["years"] = "latest (auto-resolved)"
    if fiscal_quarter:
        intent["quarter"] = f"Q{fiscal_quarter}"
    if concepts:
        intent["metrics"] = ", ".join(concepts)
    if filing_types and filing_types != ["10-K"]:
        intent["documents"] = ", ".join(filing_types)
    intent["temporal"] = temporal
    intent["retrieval_intent"] = retrieval_intent
    intent["route"] = route
    steps.append({"step": 1, "name": "Intent Detection", "details": intent})

    step_num = 2

    # Step 2: Data Availability Check
    if data_availability and (data_availability.get("notes") or data_availability.get("available")):
        avail_details = {}
        for t in effective_tickers:
            avail_yrs = data_availability.get("available", {}).get(t, [])
            miss_yrs = data_availability.get("missing", {}).get(t, [])
            if avail_yrs and not miss_yrs:
                avail_details[t] = f"10-K available for FY{', FY'.join(str(y) for y in avail_yrs)}"
            elif miss_yrs and not avail_yrs:
                avail_details[t] = f"NO 10-K filings for FY{', FY'.join(str(y) for y in miss_yrs)}"
            elif miss_yrs:
                avail_details[t] = (
                    f"10-K available for FY{', FY'.join(str(y) for y in avail_yrs)}; "
                    f"MISSING FY{', FY'.join(str(y) for y in miss_yrs)}"
                )
        da_step = {
            "step": step_num,
            "name": "Data Availability",
            "details": avail_details,
        }
        if data_availability.get("notes"):
            da_step["warnings"] = data_availability["notes"]
        steps.append(da_step)
        step_num += 1

    # XBRL Concept Resolution (relational and hybrid routes)
    if route in _RELATIONAL_ROUTES or route == "hybrid":
        resolve_actions = []
        if xbrl_concepts:
            resolve_actions.append(f"Validate LLM-suggested tags: {', '.join(xbrl_concepts[:3])}{'...' if len(xbrl_concepts) > 3 else ''}")
        if concepts:
            resolve_actions.append(f"Resolve terms: {', '.join(concepts)}")
            resolve_actions.append("Fallback chain: validated XBRL tags -> CONCEPT_ALIASES -> fuzzy DB search")
        if not resolve_actions:
            resolve_actions.append("No concepts to resolve")
        steps.append({
            "step": step_num, "name": "XBRL Concept Resolution",
            "actions": resolve_actions,
        })
        step_num += 1

    # Step 3: Route-specific retrieval
    if route == "metric_lookup" or (route in ("relational_db",) and retrieval_intent not in ("full_statement", "timeseries")):
        sql_actions = []
        # Show auto-year expansion
        if query_type in ("comparative", "ranking", "trend") and len(years) == 1:
            sql_actions.append(f"Auto-expand years: FY{years[0]-1} -> FY{years[0]} (prior year added for comparison)")
        # XBRL fact queries
        for c in concepts:
            for t in effective_tickers:
                if is_multi_year:
                    year_str = ", ".join(str(y) for y in years)
                    fn = "query_quarterly" if use_quarterly else "query_annual"
                    sql_actions.append(f"{fn}({t}, {c}, [{year_str}])")
                elif fiscal_year:
                    q_str = f", Q{fiscal_quarter}" if fiscal_quarter else ""
                    fn = "query_quarterly" if use_quarterly else "query_annual"
                    sql_actions.append(f"{fn}({t}, {c}, FY{fiscal_year}{q_str})")
                else:
                    sql_actions.append(f"query_annual({t}, {c}, latest)")
        if not sql_actions and concepts:
            sql_actions.append(f"Query XBRL facts for: {', '.join(concepts)}")
        # Statement fallback
        fallback_types = statement_types if statement_types else ["income_statement", "balance_sheet"]
        sql_actions.append(f"Fallback: get_statement() for {', '.join(fallback_types)} if XBRL returns no data")
        # YoY comparisons
        if is_multi_year or query_type in ("comparative", "ranking", "trend"):
            sql_actions.append("Compute YoY deltas (absolute + percentage change)")
        steps.append({
            "step": step_num, "name": "Metric Lookup (XBRL)",
            "tag": "primary" if route != "hybrid" else "supporting",
            "actions": sql_actions,
        })
        step_num += 1

    elif route == "timeseries" or (route == "relational_db" and retrieval_intent == "timeseries"):
        ts_actions = []
        table = "quarterly_facts" if use_quarterly else "annual_facts"
        for c in concepts:
            for t in effective_tickers:
                ts_actions.append(f"get_metric_timeseries({t}, {c}, table={table})")
        if not ts_actions:
            ts_actions.append(f"get_metric_timeseries(table={table})")
        steps.append({
            "step": step_num, "name": "Timeseries Retrieval",
            "tag": "primary",
            "actions": ts_actions,
        })
        step_num += 1

    elif route == "full_statement" or (route == "relational_db" and retrieval_intent == "full_statement"):
        stmt_actions = []
        fetch_types = statement_types if statement_types else ["income_statement", "balance_sheet", "cash_flow_statement"]
        for t in effective_tickers:
            for st in fetch_types:
                year_label = ", ".join(str(y) for y in years) if years else (str(fiscal_year) if fiscal_year else "latest")
                q_str = f", Q{fiscal_quarter}" if fiscal_quarter else ""
                stmt_actions.append(f"get_statement({t}, {st.replace('_', ' ')}, {year_label}{q_str})")
        steps.append({
            "step": step_num, "name": "Full Statement Retrieval",
            "tag": "primary",
            "actions": stmt_actions,
        })
        step_num += 1

    elif route == "hybrid":
        # Hybrid: relational side (dispatched by retrieval_intent)
        hybrid_sql_actions = []
        if retrieval_intent == "timeseries":
            hybrid_sql_actions.append("Dispatch: retrieve_timeseries()")
        elif retrieval_intent == "full_statement":
            hybrid_sql_actions.append("Dispatch: retrieve_full_statement()")
        else:
            hybrid_sql_actions.append("Dispatch: retrieve_metric_lookup()")
        for c in concepts:
            if is_multi_year:
                hybrid_sql_actions.append(f"Pull {c} for {', '.join(str(y) for y in years)}")
            elif fiscal_year:
                q_str = f" Q{fiscal_quarter}" if fiscal_quarter else ""
                hybrid_sql_actions.append(f"Pull {c} for FY{fiscal_year}{q_str}")
            else:
                hybrid_sql_actions.append(f"Pull {c} (latest)")
        if query_type in ("comparative", "ranking", "trend") and len(years) == 1 and years:
            hybrid_sql_actions.append(f"Auto-expand: FY{years[0]-1} -> FY{years[0]}")
        if is_multi_year or query_type in ("comparative", "trend"):
            hybrid_sql_actions.append("Compute YoY deltas")
        hybrid_sql_actions.append("Fallback: get_statement() if XBRL returns no data")
        steps.append({
            "step": step_num, "name": "Structured Data (SQL)",
            "tag": "primary",
            "actions": hybrid_sql_actions,
        })
        step_num += 1

    # Vector search (narrative, vector_db, hybrid)
    if route in ("narrative", "vector_db", "hybrid"):
        vector_actions = []
        filters = []
        if effective_tickers:
            filters.append(f"tickers={', '.join(effective_tickers)}")
        if years:
            filters.append(f"years={', '.join(str(y) for y in years)}")
        elif fiscal_year:
            filters.append(f"year={fiscal_year}")
        else:
            filters.append("year=latest (auto-resolved from DB)")
        if filing_types:
            filters.append(f"filing={', '.join(filing_types)}")
        vector_actions.append(f"Filter: {', '.join(filters)}")

        # Filing type selection based on temporal granularity
        search_10k = "10-K" in filing_types or not filing_types
        search_10q = "10-Q" in filing_types or temporal in ("quarterly", "specific_quarter")
        search_types = []
        if search_10k:
            search_types.append("10-K")
        if search_10q:
            search_types.append("10-Q")
        vector_actions.append(f"Search: {', '.join(search_types)} (based on temporal={temporal})")

        if target_sections:
            vector_actions.append(f"Sections: {', '.join(target_sections)}")

        # Sub-query expansion — show actual queries
        sub_queries = classification.get("search_queries") or [query]
        if len(sub_queries) > 1:
            vector_actions.append(f"Sub-queries ({len(sub_queries)}):")
            for i, sq in enumerate(sub_queries, 1):
                sq_desc = sq[:90] + ("..." if len(sq) > 90 else "")
                vector_actions.append(f"  {i}. \"{sq_desc}\"")
        else:
            search_desc = query[:80] + ("..." if len(query) > 80 else "")
            vector_actions.append(f'Query: "{search_desc}"')

        # Per-year top_k — matches actual retrieve_narrative() logic
        is_deep_narrative_plan = query_type in ("risk_analysis", "causal", "geographic", "comparative")
        if is_multi_ticker:
            per_year_k_10k = (4 if is_deep_narrative_plan else 2) if search_10k else 0
            per_year_k_10q = 1 if search_10q else 0
            vector_actions.append(f"Per-year top_k: 10-K={per_year_k_10k}, 10-Q={per_year_k_10q} (multi-ticker)")
        elif is_deep_narrative_plan:
            per_year_k_10k = 7 if search_10k else 0
            per_year_k_10q = 3 if search_10q else 0
            vector_actions.append(f"Per-year top_k: 10-K={per_year_k_10k}, 10-Q={per_year_k_10q} (deep narrative)")
        else:
            per_year_k_10k = 5 if (search_10k and is_multi_year) else (3 if search_10k else 0)
            per_year_k_10q = 3 if (search_10q and is_multi_year) else (2 if search_10q else 0)
            vector_actions.append(f"Per-year top_k: 10-K={per_year_k_10k}, 10-Q={per_year_k_10q}")

        # Per-ticker search breakdown
        if is_multi_ticker and data_availability:
            for t in effective_tickers:
                avail = data_availability.get("available", {}).get(t, [])
                miss = data_availability.get("missing", {}).get(t, [])
                if avail:
                    yr_str = ", ".join(str(y) for y in avail)
                    total_chunks = len(sub_queries) * len(avail) * per_year_k_10k
                    vector_actions.append(f"  {t}: search FY{yr_str} -> ~{total_chunks} candidate chunks")
                if miss:
                    yr_str = ", ".join(str(y) for y in miss)
                    vector_actions.append(f"  {t}: SKIP FY{yr_str} (no filings)")

        vector_actions.append("Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2")
        vector_actions.append("Deduplication: by chunk ID across sub-queries")

        is_primary = route in ("narrative", "vector_db")
        steps.append({
            "step": step_num, "name": "Semantic Search (Vector)",
            "tag": "primary" if is_primary else "supporting",
            "actions": vector_actions,
        })
        step_num += 1

    # Post-retrieval: Guardrail Filtering
    guardrail_actions = ["Filter low-quality results (similarity threshold, min relevance)"]
    steps.append({
        "step": step_num, "name": "Guardrail Filtering",
        "actions": guardrail_actions,
    })
    step_num += 1

    # Post-retrieval: Contradiction Detection (hybrid only)
    if route == "hybrid":
        steps.append({
            "step": step_num, "name": "Contradiction Detection",
            "actions": ["Cross-check vector narratives against relational data", "Flag mismatches between numbers and commentary"],
        })
        step_num += 1

    # Fusion & Answer Generation
    fusion_actions = []
    if route == "hybrid":
        fusion_actions.append("Align quantitative data with management commentary")
        fusion_actions.append("Reject explanations unsupported by numbers")

    type_specific = {
        "causal": "Map segment/driver deltas to narrative explanations",
        "comparative": "Compare metrics across periods and highlight key differences",
        "trend": "Map explanations to each period's change",
        "ranking": "Lead with numbers; narrative is secondary",
        "geographic": "Ensure narrative matches top region numerically",
        "reconciliation": "Flag timing differences or seasonality mentions",
        "risk_analysis": "Rank risks by potential impact",
        "margin_analysis": "Tie cost changes to margin movement",
        "inference": "Only assert conclusions if data supports them",
        "external_factor": "Separate external vs operational effects",
        "consistency": "Compare explanations across filings; flag contradictions",
        "factual": "Present data directly with source attribution",
    }
    if query_type in type_specific:
        fusion_actions.append(type_specific[query_type])

    if query_type == "risk_analysis" and is_multi_year:
        fusion_actions.append("Highlight new or escalated risks across years")

    fusion_actions.append("Cite all sources with year, document type, ticker, and section")

    steps.append({
        "step": step_num, "name": "Fusion & Answer Generation",
        "actions": fusion_actions,
    })
    step_num += 1

    # Confidence Scoring
    steps.append({
        "step": step_num, "name": "Confidence Scoring",
        "actions": ["Compute confidence from route quality, data coverage, and contradictions", "Assign tier: high / medium / low / insufficient"],
    })

    return {"steps": steps}


def format_retrieval_plan(plan: dict) -> str:
    """Format the retrieval plan for terminal display."""
    lines = []
    lines.append(f"\n{COLORS['bold']}Retrieval Plan{COLORS['reset']}")
    lines.append(f"{COLORS['dim']}{'─' * 45}{COLORS['reset']}")

    for step in plan["steps"]:
        tag = ""
        if step.get("tag"):
            tag = f" ({step['tag']})"
        lines.append(
            f"{COLORS['cyan']}Step {step['step']} — {step['name']}{tag}{COLORS['reset']}"
        )

        if "details" in step:
            for key, value in step["details"].items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")

        if "warnings" in step:
            for warning in step["warnings"]:
                lines.append(f"  {COLORS['yellow']}⚠ {warning}{COLORS['reset']}")

        if "actions" in step:
            for action in step["actions"]:
                lines.append(f"  > {action}")

        lines.append("")  # blank line between steps

    lines.append(f"{COLORS['dim']}{'─' * 45}{COLORS['reset']}")
    return "\n".join(lines)


# --- Retrieval Routes ---

def retrieve_narrative(query: str, classification: dict, *, cost_tracker: CostTracker | None = None) -> list[dict]:
    """Semantic search on 10-K and 10-Q section chunks.

    For multi-year queries, searches each year independently so that
    citations cover all requested periods.
    Supports multi-ticker queries: iterates over all tickers in classification["tickers"].
    """
    tickers = classification.get("tickers", [])
    if not tickers and classification.get("ticker"):
        tickers = [classification["ticker"]]
    if not tickers:
        tickers = [None]  # no ticker filter

    fiscal_year = classification.get("fiscal_year")
    quarter = classification.get("fiscal_quarter")
    years_involved = sorted(classification.get("years_involved", []))
    filing_types = classification.get("filing_types", ["10-K"])

    # Determine which years to search
    if years_involved:
        query_years = years_involved
    elif fiscal_year:
        query_years = [fiscal_year]
    else:
        # Auto-resolve to latest year when user says "latest" or omits year
        latest = None
        for t in tickers:
            if t:
                latest = get_latest_fiscal_year(t)
                break
        query_years = [latest] if latest else [None]  # None = no year filter

    # Determine filing types to search using temporal_granularity
    temporal = classification.get("temporal_granularity", "annual")
    search_10k = "10-K" in filing_types or not filing_types
    search_10q = "10-Q" in filing_types or temporal in ("quarterly", "specific_quarter")

    # Adjust per-year top_k: lower per-ticker when many tickers to avoid huge result sets
    is_multi_year = len(years_involved) >= 2
    is_multi_ticker = len(tickers) > 1
    query_type = classification.get("query_type", "factual")
    # Risk/narrative queries need more chunks to capture quantitative details
    is_deep_narrative = query_type in ("risk_analysis", "causal", "geographic", "comparative")
    if is_multi_ticker:
        per_year_top_k_10k = (4 if is_deep_narrative else 2) if search_10k else 0
        per_year_top_k_10q = 1 if search_10q else 0
    else:
        if is_deep_narrative:
            per_year_top_k_10k = 7 if search_10k else 0
            per_year_top_k_10q = 3 if search_10q else 0
        else:
            per_year_top_k_10k = 5 if (search_10k and is_multi_year) else (3 if search_10k else 0)
            per_year_top_k_10q = 3 if (search_10q and is_multi_year) else (2 if search_10q else 0)

    # Use LLM-generated search queries; fall back to original query
    sub_queries = classification.get("search_queries") or [query]

    # For risk analysis, add targeted sub-queries for quantitative risk details
    if is_deep_narrative and query_type == "risk_analysis":
        risk_sub_queries = [
            "litigation lawsuits regulatory fines penalties enforcement actions settlement",
            "debt obligations credit risk loan losses provisions concentration",
            "competition market share technology disruption cybersecurity",
        ]
        # Add only sub-queries not already covered
        existing_lower = {sq.lower() for sq in sub_queries}
        for rsq in risk_sub_queries:
            if not any(rsq.split()[0] in e for e in existing_lower):
                sub_queries.append(rsq)

    # Batch-embed all unique sub-queries in a single API call
    unique_queries = list(dict.fromkeys(sub_queries))  # dedupe, preserve order
    from chunk_and_embed import generate_embeddings, EMBEDDING_MODEL as _EMB_MODEL
    _t0 = time.time()
    query_embeddings_list, embed_usage = generate_embeddings(unique_queries, return_usage=True)
    if cost_tracker and embed_usage.get("prompt_tokens"):
        cost_tracker.record_embedding("embed", _EMB_MODEL, embed_usage["prompt_tokens"], (time.time() - _t0) * 1000)
    embedding_map = {}
    for sq_text, emb in zip(unique_queries, query_embeddings_list):
        if emb is not None:
            embedding_map[sq_text] = emb

    all_results = []
    seen_ids = set()

    from config import calendar_to_fiscal_year as _cal_to_fy

    for ticker in tickers:
        for year in query_years:
            fy = _cal_to_fy(ticker, year)
            for sq in sub_queries:
                sq_embedding = embedding_map.get(sq)
                if sq_embedding is None:
                    continue

                if search_10k:
                    results_10k = semantic_search_10k(
                        sq, ticker=ticker, fiscal_year=fy,
                        top_k=per_year_top_k_10k, use_reranking=True,
                        query_embedding=sq_embedding,
                    )
                    for r in results_10k:
                        chunk_id = r.get("id")
                        if chunk_id and chunk_id in seen_ids:
                            continue
                        if chunk_id:
                            seen_ids.add(chunk_id)
                        r["source"] = "10-K"
                        all_results.append(r)

                if search_10q:
                    results_10q = semantic_search_10q(
                        sq, ticker=ticker, fiscal_year=fy,
                        fiscal_quarter=quarter, top_k=per_year_top_k_10q,
                        use_reranking=True, query_embedding=sq_embedding,
                    )
                    for r in results_10q:
                        chunk_id = r.get("id")
                        if chunk_id and chunk_id in seen_ids:
                            continue
                        if chunk_id:
                            seen_ids.add(chunk_id)
                        r["source"] = "10-Q"
                        all_results.append(r)

    return all_results


# Backward-compat alias
retrieve_vector_db = retrieve_narrative


def _resolve_xbrl_concepts(classification: dict) -> dict[str, list[str]]:
    """Resolve XBRL concepts from classification using the 3-layer fallback chain.

    Resolution order: xbrl_concepts (validated) → CONCEPT_ALIASES → search_concepts (fuzzy).
    Shared by all relational retrieval functions.
    """
    concepts = classification.get("concepts", [])
    xbrl_concepts = classification.get("xbrl_concepts", [])
    validated = validate_xbrl_concepts(xbrl_concepts) if xbrl_concepts else []

    resolved = {}
    if validated:
        for tag in validated:
            matched = False
            for term in concepts:
                term_lower = term.lower()
                tag_lower = tag.lower().replace("us-gaap:", "").replace("-", "")
                if term_lower.replace(" ", "") in tag_lower:
                    resolved.setdefault(term, []).append(tag)
                    matched = True
                    break
            if not matched:
                resolved.setdefault(tag, []).append(tag)

    # Expand resolved terms with full CONCEPT_ALIASES variants.
    # The classifier may return only one XBRL concept (e.g. us-gaap:Revenues)
    # but different companies use different concept names for the same metric.
    # Ensure all known aliases are available so each ticker can match its variant.
    for term in list(resolved.keys()):
        term_lower = term.lower()
        if term_lower in CONCEPT_ALIASES:
            existing = set(resolved[term])
            for alias in CONCEPT_ALIASES[term_lower]:
                if alias not in existing:
                    resolved[term].append(alias)

    resolved_terms = set(resolved.keys())
    unresolved = [c for c in concepts if c not in resolved_terms]
    if unresolved:
        fallback = resolve_concepts(unresolved)
        resolved.update(fallback)
    elif not resolved and concepts:
        resolved = resolve_concepts(concepts)

    # Last resort: if concepts and xbrl_concepts were both empty, try to
    # extract known financial terms directly from the query text.
    if not resolved:
        query_text = classification.get("_original_query", "")
        if query_text:
            query_lower = query_text.lower()
            # Sort by key length descending so "free cash flow" matches before "cash flow"
            for alias_key in sorted(CONCEPT_ALIASES, key=len, reverse=True):
                if alias_key in query_lower:
                    resolved[alias_key] = CONCEPT_ALIASES[alias_key]
                    break

    return resolved


def _get_tickers_and_years(classification: dict) -> tuple[list[str], list[int]]:
    """Extract tickers and query years from classification. Shared helper."""
    tickers = classification.get("tickers", [])
    if not tickers and classification.get("ticker"):
        tickers = [classification["ticker"]]
    fiscal_year = classification.get("fiscal_year")
    years_involved = sorted(classification.get("years_involved", []))

    effective_years = []
    for ticker in (tickers or [None]):
        ticker_fiscal_year = fiscal_year
        if ticker and ticker_fiscal_year is None:
            ticker_fiscal_year = get_latest_fiscal_year(ticker)
        if len(years_involved) >= 2:
            effective_years = years_involved
        elif ticker_fiscal_year:
            effective_years = [ticker_fiscal_year]
        break  # years are the same for all tickers

    return tickers, effective_years


def retrieve_metric_lookup(query: str, classification: dict) -> dict:
    """Targeted XBRL fact query for specific numbers with YoY comparisons."""
    tickers, query_years = _get_tickers_and_years(classification)
    fiscal_year = classification.get("fiscal_year")
    fiscal_quarter = classification.get("fiscal_quarter")
    year_quarters = classification.get("year_quarters") or {}  # {year: quarter}
    years_involved = sorted(classification.get("years_involved", []))

    data = {"xbrl_facts": [], "timeseries": [], "statements": [], "comparisons": []}

    resolved = _resolve_xbrl_concepts(classification)

    temporal = classification.get("temporal_granularity", "annual")
    use_quarterly = temporal in ("quarterly", "specific_quarter")

    query_type = classification.get("query_type", "factual")
    all_years_used = set()

    # Detect segment/breakdown intent from query text
    _q_lower = query.lower()
    want_segments = any(kw in _q_lower for kw in (
        "by segment", "by product", "by region", "by geography", "by country",
        "segment breakdown", "revenue breakdown", "breakdown by",
        "by business segment", "by operating segment", "by category",
        "product mix", "segment mix", "geographic breakdown",
        "revenue mix", "per segment", "each segment",
    ))

    for ticker in tickers:
        ticker_fiscal_year = fiscal_year
        if ticker and ticker_fiscal_year is None:
            ticker_fiscal_year = get_latest_fiscal_year(ticker)

        # Per-ticker year resolution: prefer explicit years_involved, then
        # per-ticker fiscal year, then shared query_years as last resort.
        if years_involved:
            ticker_years = years_involved
        elif ticker_fiscal_year:
            ticker_years = [ticker_fiscal_year]
        else:
            ticker_years = query_years

        # Auto-expand for growth/comparison: if only 1 year resolved and the
        # query needs YoY data, prepend the previous year.
        if query_type in ("comparative", "ranking", "trend") and len(ticker_years) == 1:
            ticker_years = [ticker_years[0] - 1, ticker_years[0]]

        all_years_used.update(ticker_years)

        # XBRL fact retrieval
        ticker_got_facts = False
        # Detect mixed annual+quarterly: query asks for both FY total and a specific quarter
        mixed_annual_quarterly = (
            not use_quarterly
            and fiscal_quarter is not None
            and fiscal_quarter in (1, 2, 3, 4)
        )
        for term, xbrl_names in resolved.items():
            # Handle computed metrics (e.g. FCF = CFO - CapEx)
            if xbrl_names and xbrl_names[0] in COMPUTED_METRICS:
                cm = COMPUTED_METRICS[xbrl_names[0]]
                for year in ticker_years:
                    year_quarter = year_quarters.get(year, fiscal_quarter)
                    part_values = {}
                    for part_name, part_concepts in cm["components"].items():
                        for pc in part_concepts:
                            if use_quarterly and year_quarter:
                                facts = query_quarterly(
                                    ticker=ticker, concept=pc,
                                    fiscal_year=year, fiscal_quarter=year_quarter,
                                    limit=1,
                                )
                            else:
                                facts = query_annual(
                                    ticker=ticker, concept=pc,
                                    fiscal_year=year, limit=1,
                                )
                            if facts and facts[0].get("value") is not None:
                                part_values[part_name] = facts[0]["value"]
                                break
                    if "cfo" in part_values:
                        computed_val = cm["compute"](part_values)
                        data["xbrl_facts"].append({
                            "ticker": ticker,
                            "concept": cm["label"],
                            "fiscal_year": year,
                            "fiscal_quarter": year_quarter if use_quarterly else None,
                            "value": computed_val,
                            "unit": cm["unit"],
                            "resolved_from": term,
                            "computed": True,
                        })
                        ticker_got_facts = True
                continue

            for year in ticker_years:
                # Resolve the quarter for this specific year:
                # 1. Per-year mapping (year_quarters) takes priority
                # 2. Then fall back to the global fiscal_quarter
                year_quarter = year_quarters.get(year, fiscal_quarter)

                for concept_name in xbrl_names:
                    if use_quarterly and year_quarter == 4:
                        # Q4 not in DB; derive via edgartools
                        q4_result = query_q4(ticker, concept_name, year)
                        facts = [q4_result] if q4_result else []
                    elif use_quarterly and year_quarter:
                        facts = query_quarterly(
                            ticker=ticker, concept=concept_name,
                            fiscal_year=year, fiscal_quarter=year_quarter,
                            limit=10 if not want_segments else 50,
                            include_segments=want_segments,
                        )
                    elif use_quarterly:
                        # Quarterly granularity but no specific quarter — fetch all quarters
                        facts = query_quarterly(
                            ticker=ticker, concept=concept_name,
                            fiscal_year=year, fiscal_quarter=None,
                            limit=10 if not want_segments else 50,
                            include_segments=want_segments,
                        )
                    else:
                        facts = query_annual(
                            ticker=ticker, concept=concept_name,
                            fiscal_year=year,
                            limit=10 if not want_segments else 50,
                            include_segments=want_segments,
                        )
                    if facts:
                        for f in facts:
                            f["resolved_from"] = term
                        data["xbrl_facts"].extend(facts)
                        ticker_got_facts = True
                        if not want_segments:
                            break

                # Mixed mode: also fetch the specific quarter alongside annual
                if mixed_annual_quarterly and ticker_got_facts:
                    for concept_name in xbrl_names:
                        if year_quarter == 4:
                            q4_result = query_q4(ticker, concept_name, year)
                            q_facts = [q4_result] if q4_result else []
                        else:
                            q_facts = query_quarterly(
                                ticker=ticker, concept=concept_name,
                                fiscal_year=year, fiscal_quarter=year_quarter,
                                limit=10,
                            )
                        if q_facts:
                            for f in q_facts:
                                f["resolved_from"] = term
                            data["xbrl_facts"].extend(q_facts)
                            break

        # Fallback to statements if XBRL returned nothing for THIS ticker
        if not ticker_got_facts and ticker and ticker_years:
            statement_types = classification.get("statement_types", [])
            fetch_types = statement_types if statement_types else ["income_statement", "balance_sheet"]
            for year in ticker_years:
                year_quarter = year_quarters.get(year, fiscal_quarter)
                for st in fetch_types:
                    md = get_statement(ticker, st, year, year_quarter)
                    if md:
                        data["statements"].append({"type": st, "fiscal_year": year, "ticker": ticker, "content": md})

    # YoY comparisons — use the actual years found across all tickers
    comparison_years = sorted(all_years_used)
    if len(comparison_years) >= 2 and data["xbrl_facts"]:
        data["comparisons"] = _compute_yoy_comparisons(data["xbrl_facts"], comparison_years)

    return data


def retrieve_timeseries(query: str, classification: dict) -> dict:
    """Fetch metric timeseries across periods using get_metric_timeseries()."""
    tickers, query_years = _get_tickers_and_years(classification)

    data = {"xbrl_facts": [], "timeseries": [], "statements": [], "comparisons": []}

    resolved = _resolve_xbrl_concepts(classification)

    temporal = classification.get("temporal_granularity", "annual")
    use_quarterly = temporal in ("quarterly", "specific_quarter")
    table = "quarterly_facts" if use_quarterly else "annual_facts"

    for ticker in tickers:
        if ticker and resolved:
            for term, xbrl_names in resolved.items():
                ts = get_metric_timeseries(ticker, xbrl_names[0], table=table,
                                           concepts=xbrl_names)
                # Derive missing Q4 data via edgartools for quarterly timeseries
                if use_quarterly and ts:
                    existing_quarters = {
                        (p["fiscal_year"], p.get("fiscal_quarter"))
                        for p in ts
                    }
                    years_in_ts = sorted({p["fiscal_year"] for p in ts})
                    for year in years_in_ts:
                        if (year, 4) not in existing_quarters:
                            for concept_name in xbrl_names:
                                q4 = query_q4(ticker, concept_name, year)
                                if q4 and q4.get("value") is not None:
                                    ts.append({
                                        "fiscal_year": year,
                                        "fiscal_quarter": 4,
                                        "value": q4["value"],
                                        "unit": q4.get("unit", "USD"),
                                        "end_date": q4.get("period_end"),
                                    })
                                    break
                    # Re-sort so Q4 appears in correct order
                    ts.sort(key=lambda p: (
                        p.get("fiscal_year", 0),
                        p.get("fiscal_quarter", 0),
                    ))
                data["timeseries"].extend(ts)

    return data


def retrieve_full_statement(query: str, classification: dict) -> dict:
    """Fetch complete financial statements via get_statement()."""
    tickers, query_years = _get_tickers_and_years(classification)
    fiscal_quarter = classification.get("fiscal_quarter")
    statement_types = classification.get("statement_types", [])

    data = {"xbrl_facts": [], "timeseries": [], "statements": [], "comparisons": []}

    fetch_types = statement_types if statement_types else ["income_statement", "balance_sheet", "cash_flow_statement"]

    for ticker in tickers:
        if ticker and query_years:
            for year in query_years:
                for st in fetch_types:
                    md = get_statement(ticker, st, year, fiscal_quarter)
                    if md:
                        data["statements"].append({"type": st, "fiscal_year": year, "ticker": ticker, "content": md})

    return data


def retrieve_relational_db(query: str, classification: dict) -> dict:
    """Legacy wrapper: dispatches to the appropriate relational retrieval function."""
    intent = classification.get("retrieval_intent", "specific_metric")
    if intent == "full_statement":
        return retrieve_full_statement(query, classification)
    elif intent == "timeseries":
        return retrieve_timeseries(query, classification)
    else:
        return retrieve_metric_lookup(query, classification)


def _compute_yoy_comparisons(xbrl_facts: list[dict], years: list[int]) -> list[dict]:
    """Compute year-over-year deltas from fetched XBRL facts.

    Groups facts by (ticker, concept), then computes absolute and percentage
    change between consecutive years.
    """
    from collections import defaultdict

    # Group: (ticker, concept) -> {year: value}
    by_key = defaultdict(dict)
    for fact in xbrl_facts:
        ticker = fact.get("ticker", "N/A")
        concept = fact.get("concept")
        year = fact.get("fiscal_year")
        value = fact.get("value")
        if concept and year and value is not None:
            by_key[(ticker, concept)][year] = value

    comparisons = []
    sorted_years = sorted(years)
    for (ticker, concept), year_vals in by_key.items():
        for i in range(1, len(sorted_years)):
            prev_year = sorted_years[i - 1]
            curr_year = sorted_years[i]
            if prev_year in year_vals and curr_year in year_vals:
                prev_val = year_vals[prev_year]
                curr_val = year_vals[curr_year]
                delta = curr_val - prev_val
                pct_change = (delta / abs(prev_val) * 100) if prev_val != 0 else None
                comparisons.append({
                    "ticker": ticker,
                    "concept": concept,
                    "from_year": prev_year,
                    "to_year": curr_year,
                    "from_value": prev_val,
                    "to_value": curr_val,
                    "delta": delta,
                    "pct_change": pct_change,
                })

    return comparisons


def retrieve_hybrid(query: str, classification: dict, *, cost_tracker: CostTracker | None = None) -> dict:
    """Combine narrative and relational retrieval based on retrieval_intent."""
    intent = classification.get("retrieval_intent", "specific_metric")
    if intent == "timeseries":
        relational_data = retrieve_timeseries(query, classification)
    elif intent == "full_statement":
        relational_data = retrieve_full_statement(query, classification)
    else:
        relational_data = retrieve_metric_lookup(query, classification)
    vector_results = retrieve_narrative(query, classification, cost_tracker=cost_tracker)
    return {"relational": relational_data, "vector": vector_results}


# --- Context Formatting ---

_RELATIONAL_ROUTES = {"metric_lookup", "timeseries", "full_statement", "relational_db"}


def _format_narrative_context(data: list[dict], classification: dict) -> list[str]:
    """Format narrative (vector) chunks into context parts."""
    parts = []
    years_involved = sorted(classification.get("years_involved", []))
    is_multi_year = len(years_involved) >= 2

    if is_multi_year:
        from collections import defaultdict
        chunks_by_year = defaultdict(list)
        for chunk in data:
            fy = chunk.get("fiscal_year", "N/A")
            chunks_by_year[fy].append(chunk)

        parts.append("## Retrieved Document Sections\n")
        chunk_counter = 0
        for fy in sorted(chunks_by_year.keys(), key=lambda y: (isinstance(y, str), y)):
            parts.append(f"### FY{fy} Narrative\n")
            for chunk in chunks_by_year[fy]:
                chunk_counter += 1
                source = chunk.get("source", "10-K")
                ticker = chunk.get("ticker", "N/A")
                quarter = chunk.get("fiscal_quarter")
                section = chunk.get("section_title") or chunk.get("section_id", "N/A")
                score = chunk.get("rerank_score", chunk.get("similarity", 0))
                q_str = f" Q{quarter}" if quarter else ""
                parts.append(
                    f"#### Chunk {chunk_counter} [{source}] {ticker} FY{fy}{q_str} - {section} (score: {score:.3f})\n"
                    f"[Source: {source}, Ticker: {ticker}, Year: {fy}{q_str}, Section: {section}]\n"
                    f"{chunk.get('expanded_text', chunk['text'])}\n"
                )
    else:
        parts.append("## Retrieved Document Sections\n")
        for i, chunk in enumerate(data, 1):
            source = chunk.get("source", "10-K")
            ticker = chunk.get("ticker", "N/A")
            year = chunk.get("fiscal_year", "N/A")
            quarter = chunk.get("fiscal_quarter")
            section = chunk.get("section_title") or chunk.get("section_id", "N/A")
            score = chunk.get("rerank_score", chunk.get("similarity", 0))
            q_str = f" Q{quarter}" if quarter else ""
            parts.append(
                f"### Chunk {i} [{source}] {ticker} FY{year}{q_str} - {section} (score: {score:.3f})\n"
                f"[Source: {source}, Ticker: {ticker}, Year: {year}{q_str}, Section: {section}]\n"
                f"{chunk.get('expanded_text', chunk['text'])}\n"
            )
    return parts


def _fmt_val(value, signed: bool = False) -> str:
    """Format a numeric value, converting to billions if >= 1 billion."""
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        v = value / 1_000_000_000
        fmt = f"{v:+,.2f}" if signed else f"{v:,.2f}"
        return f"{fmt} billion"
    fmt = f"{value:+,.2f}" if signed else f"{value:,.2f}"
    return fmt


def _format_fact_line(fact: dict, fallback_ticker: str) -> str:
    """Format a single XBRL fact into a context line."""
    ticker_val = fact.get("ticker", fallback_ticker)
    value = fact.get("value")
    unit = fact.get("unit", "")
    concept = fact.get("concept", "")
    year = fact.get("fiscal_year", "")
    has_quarter = "fiscal_quarter" in fact and fact.get("fiscal_quarter") is not None
    quarter_str = f" Q{fact['fiscal_quarter']}" if has_quarter else ""
    doc_type = "10-Q" if has_quarter else "10-K"
    line = (
        f"- **{concept}** ({year}{quarter_str}): {_fmt_val(value)} {unit}" if value else
        f"- **{concept}** ({year}{quarter_str}): N/A"
    )
    dim = fact.get("dimension")
    mem = fact.get("member")
    if dim and mem:
        # Clean up member name: "aapl:IPhoneMember" -> "iPhone"
        mem_label = mem.split(":")[-1] if ":" in mem else mem
        mem_label = mem_label.replace("Member", "").replace("Segment", "")
        # Convert CamelCase to spaced: "RestOfAsiaPacific" -> "Rest Of Asia Pacific"
        import re as _re
        mem_label = _re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', mem_label)
        # Fix common run-together words
        mem_label = mem_label.replace("Homeand", "Home and").replace("Ofand", "Of and")
        line += f" [Segment: {mem_label}]"
    elif dim:
        line += f" [Segment: {dim}]"
    line += f" [XBRL: {concept} | {ticker_val}, FY {year}{quarter_str}]"
    return line


def _format_relational_context(data: dict, classification: dict) -> list[str]:
    """Format relational (XBRL/statements) data into context parts.

    When facts come from multiple tickers, groups them by ticker with section
    headers for clearer multi-company comparison context.
    """
    parts = []
    fallback_ticker = classification.get("ticker", "N/A")

    if data.get("xbrl_facts"):
        # Detect multi-ticker facts
        fact_tickers = {f.get("ticker", fallback_ticker) for f in data["xbrl_facts"]}
        is_multi_ticker = len(fact_tickers) > 1

        if is_multi_ticker:
            from collections import defaultdict
            facts_by_ticker = defaultdict(list)
            for fact in data["xbrl_facts"]:
                t = fact.get("ticker", fallback_ticker)
                facts_by_ticker[t].append(fact)

            parts.append("## XBRL Financial Facts (by Company)\n")
            for ticker in sorted(facts_by_ticker.keys()):
                parts.append(f"### {ticker}\n")
                for fact in facts_by_ticker[ticker]:
                    parts.append(_format_fact_line(fact, fallback_ticker))
                parts.append("")
        else:
            # Separate consolidated totals from segment breakdowns
            consolidated = [f for f in data["xbrl_facts"] if not f.get("dimension")]
            segments = [f for f in data["xbrl_facts"] if f.get("dimension")]

            if segments:
                # Filter out aggregate/parent members that are subtotals
                _AGGREGATE_MEMBERS = {
                    "us-gaap:ProductMember",
                    "us-gaap:OperatingSegmentsMember",
                }
                leaf_segments = [f for f in segments if f.get("member") not in _AGGREGATE_MEMBERS]
                # If filtering removed everything, keep originals
                if not leaf_segments:
                    leaf_segments = segments

                if consolidated:
                    parts.append("## XBRL Financial Facts (Consolidated Total)\n")
                    for fact in consolidated:
                        parts.append(_format_fact_line(fact, fallback_ticker))
                    parts.append("")

                parts.append("## XBRL Financial Facts (Segment Breakdown)\n")
                # Group segments by dimension axis for clarity
                from collections import defaultdict
                by_dim = defaultdict(list)
                for fact in leaf_segments:
                    dim = fact.get("dimension", "Other")
                    by_dim[dim].append(fact)

                # Friendly axis labels
                _AXIS_LABELS = {
                    "srt:ProductOrServiceAxis": "By Product / Service",
                    "srt:StatementGeographicalAxis": "By Geography",
                    "srt:ConsolidationItemsAxis": "By Consolidation",
                    "us-gaap:StatementBusinessSegmentsAxis": "By Business Segment (Geographic)",
                }
                for dim_name in sorted(by_dim.keys()):
                    axis_label = _AXIS_LABELS.get(dim_name)
                    if not axis_label:
                        axis_label = dim_name.split(":")[-1] if ":" in dim_name else dim_name
                        axis_label = axis_label.replace("Axis", "").replace("Statement", "")
                    parts.append(f"### {axis_label}\n")
                    for fact in sorted(by_dim[dim_name], key=lambda f: -(f.get("value") or 0)):
                        parts.append(_format_fact_line(fact, fallback_ticker))
                    parts.append("")
            else:
                parts.append("## XBRL Financial Facts\n")
                for fact in data["xbrl_facts"]:
                    parts.append(_format_fact_line(fact, fallback_ticker))
                parts.append("")

    if data.get("comparisons"):
        # Detect multi-ticker comparisons
        comp_tickers = {c.get("ticker", fallback_ticker) for c in data["comparisons"]}
        is_multi_ticker_comp = len(comp_tickers) > 1

        if is_multi_ticker_comp:
            from collections import defaultdict
            comps_by_ticker = defaultdict(list)
            for comp in data["comparisons"]:
                t = comp.get("ticker", fallback_ticker)
                comps_by_ticker[t].append(comp)

            parts.append("## Year-over-Year Comparisons (by Company)\n")
            for ticker in sorted(comps_by_ticker.keys()):
                parts.append(f"### {ticker}\n")
                for comp in comps_by_ticker[ticker]:
                    concept = comp["concept"]
                    pct_str = f" ({comp['pct_change']:+.1f}%)" if comp["pct_change"] is not None else ""
                    parts.append(
                        f"- **{concept}**: FY{comp['from_year']} {_fmt_val(comp['from_value'])} → "
                        f"FY{comp['to_year']} {_fmt_val(comp['to_value'])} | "
                        f"Change: {_fmt_val(comp['delta'], signed=True)}{pct_str}\n"
                        f"  [XBRL: {concept} | {ticker}, FY {comp['from_year']}]\n"
                        f"  [XBRL: {concept} | {ticker}, FY {comp['to_year']}]"
                    )
                parts.append("")

            # Cross-company comparison summary table
            parts.append("## Cross-Company Comparison Summary\n")
            # Group by concept across tickers
            concept_data = defaultdict(dict)
            for comp in data["comparisons"]:
                t = comp.get("ticker", fallback_ticker)
                concept_data[comp["concept"]][t] = comp

            for concept, ticker_comps in concept_data.items():
                parts.append(f"### {concept}\n")
                parts.append("| Company | FY (prior) | FY (latest) | Change | % Change |")
                parts.append("|---------|-----------|------------|--------|----------|")
                for t in sorted(ticker_comps.keys()):
                    c = ticker_comps[t]
                    pct = f"{c['pct_change']:+.1f}%" if c["pct_change"] is not None else "N/A"
                    parts.append(
                        f"| {t} | {_fmt_val(c['from_value'])} ({c['from_year']}) | "
                        f"{_fmt_val(c['to_value'])} ({c['to_year']}) | "
                        f"{_fmt_val(c['delta'], signed=True)} | {pct} |"
                    )
                parts.append("")
                for t in sorted(ticker_comps.keys()):
                    c = ticker_comps[t]
                    parts.append(
                        f"[XBRL: {concept} | {t}, FY {c['from_year']}] "
                        f"[XBRL: {concept} | {t}, FY {c['to_year']}]"
                    )
                parts.append("")
        else:
            parts.append("## Year-over-Year Comparisons\n")
            for comp in data["comparisons"]:
                ticker_val = comp.get("ticker", fallback_ticker)
                concept = comp["concept"]
                pct_str = f" ({comp['pct_change']:+.1f}%)" if comp["pct_change"] is not None else ""
                parts.append(
                    f"- **{concept}** ({ticker_val}): FY{comp['from_year']} {_fmt_val(comp['from_value'])} → "
                    f"FY{comp['to_year']} {_fmt_val(comp['to_value'])} | "
                    f"Change: {_fmt_val(comp['delta'], signed=True)}{pct_str}\n"
                    f"  [XBRL: {concept} | {ticker_val}, FY {comp['from_year']}]\n"
                    f"  [XBRL: {concept} | {ticker_val}, FY {comp['to_year']}]"
                )
            parts.append("")

    if data.get("timeseries"):
        parts.append("## Historical Timeseries\n")
        for point in data["timeseries"]:
            ticker_val = point.get("ticker") or fallback_ticker
            year = point.get("fiscal_year", "")
            has_quarter = "fiscal_quarter" in point and point["fiscal_quarter"] is not None
            quarter_num = point.get("fiscal_quarter")
            q = f" Q{quarter_num}" if has_quarter else ""
            val = point.get("value")
            unit = point.get("unit", "")
            concept = point.get("concept", "Timeseries Data")
            # Q1-Q3 come from 10-Q, Q4 (derived from annual) and annual come from 10-K
            is_q4 = has_quarter and quarter_num == 4
            doc_type = "10-K" if (not has_quarter or is_q4) else "10-Q"
            val_str = f"{_fmt_val(val)} {unit}" if val else "N/A"
            parts.append(
                f"- {year}{q}: {val_str} "
                f"[XBRL: {concept} | {ticker_val}, FY {year}{q}]"
            )
        parts.append("")

    if data.get("statements"):
        parts.append("## Financial Statements\n")
        fiscal_quarter_val = classification.get("fiscal_quarter")
        for stmt in data["statements"]:
            ticker_val = stmt.get("ticker") or fallback_ticker
            stmt_year = stmt.get("fiscal_year") or classification.get("fiscal_year", "N/A")
            q_str = f" Q{fiscal_quarter_val}" if fiscal_quarter_val else ""
            doc_type = "10-Q" if fiscal_quarter_val else "10-K"
            stmt_name = stmt["type"].replace("_", " ").title()
            parts.append(
                f"### {stmt_name} — {ticker_val} (FY{stmt_year})\n"
                f"[Source: {doc_type}, Ticker: {ticker_val}, Year: {stmt_year}{q_str}, "
                f"Section: {stmt_name}]\n"
            )
            parts.append(stmt["content"])
            parts.append("")

    return parts


def _build_financial_snapshot(classification: dict) -> str:
    """Build a concise financial snapshot for each ticker/year to enrich risk analysis context."""
    tickers = classification.get("tickers", [])
    if not tickers and classification.get("ticker"):
        tickers = [classification["ticker"]]
    years_involved = sorted(classification.get("years_involved", []))
    fiscal_year = classification.get("fiscal_year")
    if not years_involved and fiscal_year:
        years_involved = [fiscal_year]
    # Auto-resolve to latest fiscal year when none specified (e.g. "latest 10-K")
    if not years_involved and tickers:
        latest = get_latest_fiscal_year(tickers[0])
        if latest:
            years_involved = [latest]
    if not tickers or not years_involved:
        return ""

    # Key US-GAAP and common concepts to look for
    _SNAPSHOT_CONCEPTS = [
        ("us-gaap:Revenues", "Revenue"),
        ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue"),
        ("us-gaap:SalesRevenueServicesNet", "Revenue (Services)"),
        ("us-gaap:RevenuesNetOfInterestExpense", "Net Revenue"),
        ("us-gaap:NetIncomeLoss", "Net Income"),
        ("us-gaap:OperatingIncomeLoss", "Operating Income"),
        ("us-gaap:Assets", "Total Assets"),
        ("us-gaap:Liabilities", "Total Liabilities"),
        ("us-gaap:StockholdersEquity", "Stockholders' Equity"),
        ("us-gaap:LongTermDebt", "Long-Term Debt"),
        ("us-gaap:LongTermDebtNoncurrent", "Long-Term Debt"),
        ("us-gaap:CashAndCashEquivalentsAtCarryingValue", "Cash & Equivalents"),
        ("dei:EntityNumberOfEmployees", "Employees"),
    ]

    from xbrl_to_postgres import get_metric_timeseries

    snapshot_parts = []
    for ticker in tickers:
        for year in years_involved:
            metrics_found = {}
            for concept, label in _SNAPSHOT_CONCEPTS:
                if label in metrics_found:
                    continue  # Skip duplicate labels (e.g. multiple revenue concepts)
                try:
                    ts = get_metric_timeseries(ticker, concept, table="annual_facts")
                    vals = [t for t in ts if t.get("fiscal_year") == year]
                    if vals and vals[0].get("value"):
                        val = float(vals[0]["value"])
                        if label == "Employees":
                            metrics_found[label] = f"{val:,.0f}"
                        elif abs(val) >= 1e9:
                            metrics_found[label] = f"${val/1e9:,.2f}B"
                        elif abs(val) >= 1e6:
                            metrics_found[label] = f"${val/1e6:,.0f}M"
                        else:
                            metrics_found[label] = f"${val:,.0f}"
                except Exception:
                    continue

            if metrics_found:
                lines = [f"## {ticker} FY{year} — Financial Snapshot"]
                for label, formatted in metrics_found.items():
                    lines.append(f"- {label}: {formatted}")
                snapshot_parts.append("\n".join(lines))

    if snapshot_parts:
        return "# Financial Context (from XBRL filings)\n\n" + "\n\n".join(snapshot_parts) + "\n"
    return ""


def format_context(route: str, data, classification: dict, data_availability: dict | None = None) -> str:
    """Assemble retrieved data into a structured context string for the LLM."""
    parts = []

    # Prepend data availability warnings so the LLM knows about gaps
    if data_availability and data_availability.get("notes"):
        avail_lines = ["## Data Availability Notices"]
        for note in data_availability["notes"]:
            avail_lines.append(f"- ⚠ {note}")
        avail_lines.append("")
        parts.append("\n".join(avail_lines))

    # Prepend financial snapshot for risk/comparative queries to give quantitative grounding
    query_type = classification.get("query_type", "factual")
    tickers_for_snapshot = classification.get("tickers", [])
    _needs_snapshot = (
        query_type in ("risk_analysis", "comparative", "causal")
        or (len(tickers_for_snapshot) > 1 and route in ("narrative", "hybrid"))
    )
    if _needs_snapshot:
        snapshot = _build_financial_snapshot(classification)
        if snapshot:
            parts.append(snapshot)

    if route in ("narrative", "vector_db"):
        parts.extend(_format_narrative_context(data, classification))

    elif route in _RELATIONAL_ROUTES:
        parts.extend(_format_relational_context(data, classification))

    elif route == "hybrid":
        rel = data.get("relational", {})
        if rel:
            rel_parts = _format_relational_context(rel, classification)
            if any(p.strip() for p in rel_parts):
                parts.append("# Structured Financial Data\n")
                parts.extend(rel_parts)

        vec = data.get("vector", [])
        if vec:
            vec_parts = _format_narrative_context(vec, classification)
            if any(p.strip() for p in vec_parts):
                parts.append("\n# Narrative Sections\n")
                parts.extend(vec_parts)

    return "\n".join(parts)


# --- Answer Generation ---

def generate_answer(query: str, context: str, classification: dict, *, cost_tracker: CostTracker | None = None) -> str:
    """Generate a grounded answer using the retrieved context."""
    route = classification["route"]

    system_prompt = (
        f"You are a financial analyst assistant answering questions about SEC filings. "
        f"Today's date is {_today_str()}. Data covers 2010 through the latest available filings (up to 2025/Q1 2026). "
        "Base your answers strictly on the provided context. "
        "If the context doesn't contain enough information, say so clearly. "
        "Be concise and factual.\n\n"
        "CITATION RULES (mandatory):\n"
        "- Every factual claim must have an inline citation.\n"
        "- Use this exact format: [Source: <Document>, Ticker: <TICKER>, Year: <YEAR>, Section: <SECTION>]\n"
        "  Example: [Source: 10-K, Ticker: AAPL, Year: 2025, Section: MD&A]\n"
        "- Extract the citation details from the [Source: ...] tags in the context.\n"
        "- Do NOT omit any of the four fields (document type, ticker, year, section).\n"
        "- At the end, include a **Sources:** list summarising all unique sources used."
    )

    if route in _RELATIONAL_ROUTES:
        system_prompt += (
            " You have structured financial data. Present numbers clearly with proper formatting. "
            "Note the fiscal year and quarter for all figures."
        )
        if route == "timeseries":
            system_prompt += " Present the data as a chronological trend with period-over-period changes."
        elif route == "full_statement":
            system_prompt += " Present the full financial statement in a clear tabular format."
    elif route == "hybrid":
        system_prompt += (
            "\n\nHYBRID ANALYSIS INSTRUCTIONS:\n"
            "You have both structured XBRL data and narrative MD&A sections.\n"
            "- Use the XBRL data for top-line metrics (total revenue, net income, etc.).\n"
            "- Use the narrative sections to provide SPECIFIC, QUANTITATIVE breakdowns.\n"
            "- When narrative text mentions segment growth (e.g. 'Azure grew 28%', "
            "'Office 365 revenue increased $4.4 billion or 13%'), cite these EXACT figures.\n"
            "- Break down the growth attribution by segment or product with dollar amounts "
            "and percentages from the narrative, not just vague summaries.\n"
            "- Present the breakdown in a structured format (bullet points or table).\n"
            "- Always cite the source for each specific figure."
        )

    # Multi-company comparison instructions
    tickers = classification.get("tickers", [])
    if len(tickers) > 1:
        system_prompt += (
            "\n\nMULTI-COMPANY COMPARISON INSTRUCTIONS:\n"
            "- Present each company's metrics in its own section.\n"
            "- Use a comparison table for side-by-side figures.\n"
            "- State which company leads on each metric.\n"
            "- When answering 'which is most profitable' or 'growing faster', give a definitive answer with numbers.\n"
            "- Note if companies have different fiscal year ends (e.g. Apple=Sep, Microsoft=Jun, JPMorgan=Dec).\n"
            "- Always include absolute values AND percentages when comparing growth."
        )

    # Multi-year, query-type-aware prompt extensions
    query_type = classification.get("query_type", "factual")
    years_involved = classification.get("years_involved", [])
    is_multi_year = len(years_involved) >= 2

    if is_multi_year:
        if query_type == "comparative":
            system_prompt += (
                "\n\nCOMPARATIVE ANALYSIS INSTRUCTIONS:\n"
                "- Present each year's data separately before comparing.\n"
                "- List segment-level figures for each year.\n"
                "- State year-over-year changes (absolute and percentage).\n"
                "- Identify the top contributing segments to the change.\n"
                "- You MUST cite sources from EACH year involved in the comparison."
            )
        elif query_type == "causal":
            system_prompt += (
                "\n\nCAUSAL ANALYSIS INSTRUCTIONS:\n"
                "- State the observed change between years.\n"
                "- Cite specific drivers for each year separately.\n"
                "- Distinguish between segment-level drivers and macro/external drivers.\n"
                "- You MUST cite sources from EACH year involved."
            )
        elif query_type == "trend":
            system_prompt += (
                "\n\nTREND ANALYSIS INSTRUCTIONS:\n"
                "- Present data chronologically across all years.\n"
                "- Identify acceleration or deceleration in the trend.\n"
                "- You MUST cite sources from EACH year involved."
            )
        elif query_type == "risk_analysis":
            system_prompt += (
                "\n\nMULTI-YEAR RISK ANALYSIS INSTRUCTIONS:\n"
                "- Compare risk disclosures across years.\n"
                "- Highlight new, escalated, or removed risks.\n"
                "- You MUST cite sources from EACH year involved."
            )

    # Detect if query is risk-related even if query_type is "comparative"
    import re as _re_risk
    _is_risk_query = (
        query_type == "risk_analysis"
        or _re_risk.search(r'\brisk(?:s| factor)', query, _re_risk.IGNORECASE)
    )

    # Single-year or general risk analysis instructions
    if _is_risk_query:
        system_prompt += (
            "\n\nRISK ANALYSIS — QUANTITATIVE DETAIL INSTRUCTIONS:\n"
            "- IMPORTANT: The context includes a 'Financial Snapshot' section with key metrics (revenue, total assets,\n"
            "  debt, net income, equity, etc.) for each company. You MUST reference these metrics when discussing risks.\n"
            "  For EACH company, open with a brief financial profile using the snapshot data before listing risk factors.\n"
            "  Example: 'JPMorgan Chase, with $2.4T in total assets, $268B in long-term debt, and $25.1B in revenue,\n"
            "  faces the following key risks:'\n"
            "- For EVERY risk factor, extract and cite ALL specific quantitative figures from the context:\n"
            "  dollar amounts ($1.5 billion), percentages (75% of revenue), customer concentration (11% from one customer),\n"
            "  debt figures, litigation amounts, geographic revenue splits, regulatory fines, employee counts, etc.\n"
            "- Use the financial snapshot to SIZE risks — e.g. relate debt risk to total debt/equity ratio,\n"
            "  relate revenue risk to total revenue, relate legal reserves to net income.\n"
            "- Present each risk with its quantitative impact — do NOT give vague qualitative-only summaries.\n"
            "- If a risk factor has NO quantitative figure in the context, still describe it with maximum specificity\n"
            "  (name the regulation, the counterparty, the product, the jurisdiction — whatever detail exists).\n"
            "- Organize risks by category (Regulatory/Legal, Market/Credit, Operational, Technology, Competition, etc.).\n"
            "- If a risk mentions specific counterparties, regulators, lawsuits, or entities, name them.\n"
            "- If a risk cites specific time periods, deadlines, or events, include those dates.\n"
            "- Quote distinctive language directly when the filing uses strong or unusual phrasing."
        )
        if len(tickers) > 1:
            system_prompt += (
                "\n\nMULTI-COMPANY RISK COMPARISON INSTRUCTIONS:\n"
                "- After presenting each company's risks separately, include a COMPARATIVE SUMMARY section.\n"
                "- Use a comparison table with risk categories as rows and companies as columns.\n"
                "- For each category, note the SEVERITY and SPECIFICITY of each company's exposure.\n"
                "- Highlight risks UNIQUE to each company (e.g. banking-specific vs. tech-platform-specific).\n"
                "- Highlight risks SHARED by both companies (e.g. regulatory, cybersecurity, macro).\n"
                "- State which company has greater exposure in each category, with supporting figures where available."
            )

    # General narrative quantitative extraction for non-risk narrative queries
    if route in ("narrative", "vector_db") and query_type != "risk_analysis":
        system_prompt += (
            "\n\nNARRATIVE QUANTITATIVE INSTRUCTIONS:\n"
            "- When the context contains specific dollar amounts, percentages, or other quantitative figures,\n"
            "  you MUST include them in your answer — do not summarize them away.\n"
            "- Cite the exact figures as they appear in the source text."
        )

    # Add data availability instruction if context contains availability notices
    if "Data Availability Notices" in context:
        system_prompt += (
            "\n\nDATA AVAILABILITY INSTRUCTIONS:\n"
            "- The context contains 'Data Availability Notices' about missing data or pre-IPO limitations.\n"
            "- You MUST explicitly inform the user about these limitations in your answer.\n"
            "- For pre-IPO years, clearly state the company's IPO date and that no SEC data exists before it.\n"
            "- Do NOT silently omit a company from the comparison — explain WHY data is unavailable."
        )

    user_message = f"Question: {query}\n\n---\n\nContext:\n{context}"

    # Wire max_tokens and temperature from guardrails.yaml
    answer_cfg = _guardrails_cfg.get("answer", {})
    temperature = answer_cfg.get("temperature") or 0.1
    max_tokens = answer_cfg.get("max_tokens")

    api_kwargs = {
        "model": MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if max_tokens:
        api_kwargs["max_tokens"] = max_tokens

    _t0 = time.time()
    response = client.chat.completions.create(**api_kwargs)
    if cost_tracker and response.usage:
        cost_tracker.record("generate", MODEL, response.usage, (time.time() - _t0) * 1000)
    return response.choices[0].message.content


# --- Accession Number Lookup ---

def _lookup_accession(ticker: str, fiscal_year, fiscal_quarter=None) -> str | None:
    """Look up accession number from the filings table.

    For quarterly data (fiscal_quarter 1-3), looks up the 10-Q for that quarter.
    For Q4 or annual data, looks up the 10-K for that fiscal year.
    """
    if not ticker or not fiscal_year:
        return None
    try:
        from xbrl_to_postgres import get_db_connection
        is_quarterly = fiscal_quarter is not None and fiscal_quarter in (1, 2, 3)
        if is_quarterly:
            fiscal_period = f"Q{fiscal_quarter}"
            form_type = "10-Q"
        else:
            fiscal_period = "FY"
            form_type = "10-K"

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT accession_number FROM filings "
                    "WHERE ticker = %s AND fiscal_year = %s AND form_type = %s AND fiscal_period = %s "
                    "LIMIT 1",
                    (ticker, fiscal_year, form_type, fiscal_period),
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception:
        return None


# --- Source Citation Builder ---

def _build_sources(route: str, data, classification: dict) -> list[dict]:
    """Build a deduplicated list of source citation dicts from retrieved data.

    Each dict has keys: filing (display text), accession_number, ticker, filing_type.
    Uses chunk-level metadata (from DB) rather than classification for accuracy.
    """
    sources = []
    seen = set()

    def _add(filing_text, accession_number=None, ticker=None, filing_type="10-K"):
        if filing_text not in seen:
            seen.add(filing_text)
            sources.append({
                "filing": filing_text,
                "accession_number": accession_number,
                "ticker": ticker,
                "filing_type": filing_type,
            })

    fallback_ticker = classification.get("ticker", "N/A")

    if route in ("narrative", "vector_db"):
        for chunk in (data or []):
            doc_type = chunk.get("source", "10-K")
            ticker = chunk.get("ticker") or fallback_ticker
            year = chunk.get("fiscal_year", "N/A")
            quarter = chunk.get("fiscal_quarter")
            section = chunk.get("section_title") or chunk.get("section_id", "N/A")
            accession = chunk.get("accession_number")
            q_str = f" Q{quarter}" if quarter else ""
            _add(
                f"[{doc_type}] {ticker}, FY {year}{q_str}, Section: {section}",
                accession_number=accession, ticker=ticker, filing_type=doc_type,
            )

    elif route in _RELATIONAL_ROUTES:
        for fact in (data or {}).get("xbrl_facts", []):
            ticker = fact.get("ticker") or fallback_ticker
            year = fact.get("fiscal_year", "N/A")
            concept = fact.get("concept", "N/A")
            has_quarter = "fiscal_quarter" in fact and fact["fiscal_quarter"] is not None
            quarter_num = fact.get("fiscal_quarter")
            # Q4 data is derived from the 10-K annual filing (Annual - Q1 - Q2 - Q3)
            is_q4 = has_quarter and quarter_num == 4
            doc_type = "10-K" if (not has_quarter or is_q4) else "10-Q"
            if is_q4:
                q_str = " Q4 (derived from 10-K)"
            elif has_quarter:
                q_str = f" Q{quarter_num}"
            else:
                q_str = ""
            # Look up accession number for this specific filing
            accession = _lookup_accession(ticker, year, quarter_num)
            _add(
                f"[{doc_type} XBRL] {ticker}, FY {year}{q_str}, Concept: {concept}",
                accession_number=accession, ticker=ticker, filing_type=doc_type,
            )
        for comp in (data or {}).get("comparisons", []):
            ticker = comp.get("ticker") or fallback_ticker
            concept = comp.get("concept", "N/A")
            for year_key in ("from_year", "to_year"):
                year_val = comp.get(year_key)
                if year_val:
                    accession = _lookup_accession(ticker, year_val)
                    _add(
                        f"[10-K XBRL] {ticker}, FY {year_val}, Concept: {concept}",
                        accession_number=accession, ticker=ticker, filing_type="10-K",
                    )
        for stmt in (data or {}).get("statements", []):
            ticker = stmt.get("ticker") or fallback_ticker
            fy = stmt.get("fiscal_year") or classification.get("fiscal_year", "N/A")
            fq = classification.get("fiscal_quarter")
            doc_type = "10-Q" if fq else "10-K"
            q_str = f" Q{fq}" if fq else ""
            stmt_name = stmt["type"].replace("_", " ").title()
            accession = _lookup_accession(ticker, fy, fq)
            _add(
                f"[{doc_type}] {ticker}, FY {fy}{q_str}, Statement: {stmt_name}",
                accession_number=accession, ticker=ticker, filing_type=doc_type,
            )
        # Only cite timeseries points for the requested year(s), not the full history
        years_involved = set(classification.get("years_involved", []))
        if not years_involved and classification.get("fiscal_year"):
            years_involved = {classification["fiscal_year"]}
        for point in (data or {}).get("timeseries", []):
            year = point.get("fiscal_year")
            if years_involved and year not in years_involved:
                continue
            ticker = point.get("ticker") or fallback_ticker
            has_quarter = "fiscal_quarter" in point and point["fiscal_quarter"] is not None
            quarter_num = point.get("fiscal_quarter")
            is_q4 = has_quarter and quarter_num == 4
            doc_type = "10-K" if (not has_quarter or is_q4) else "10-Q"
            concept = point.get("concept", "Timeseries Data")
            if is_q4:
                q_str = " Q4 (derived from 10-K)"
            elif has_quarter:
                q_str = f" Q{quarter_num}"
            else:
                q_str = ""
            accession = _lookup_accession(ticker, year, quarter_num)
            _add(
                f"[{doc_type}] {ticker}, FY {year}{q_str}, Concept: {concept}",
                accession_number=accession, ticker=ticker, filing_type=doc_type,
            )

    elif route == "hybrid":
        rel = (data or {}).get("relational", {})
        vec = (data or {}).get("vector", [])
        for s in _build_sources("metric_lookup", rel, classification):
            if s["filing"] not in seen:
                seen.add(s["filing"])
                sources.append(s)
        for s in _build_sources("narrative", vec, classification):
            if s["filing"] not in seen:
                seen.add(s["filing"])
                sources.append(s)

    return sources


# --- Main Orchestrator ---

def rag_query(query: str, *, skip_result_cache: bool = False,
              precomputed_classification: dict | None = None,
              cost_tracker: CostTracker | None = None) -> dict:
    """
    Main entry point. Classifies the query, retrieves context, generates an answer,
    applies guardrails, detects contradictions, and computes confidence.

    Args:
        query: The user query.
        skip_result_cache: If True, bypass the result cache.
        precomputed_classification: If provided, skip the classify_query() LLM call.
        cost_tracker: Optional CostTracker to accumulate API usage.

    Returns dict with keys: answer, route, reasoning, classification, sources,
        retrieval_plan, confidence, contradictions, filter_stats, query_rejected, cost
    """
    if cost_tracker is None:
        cost_tracker = CostTracker()

    # Step 0: Query validation (guardrail)
    is_valid, rejection_reason = validate_query(query)
    if not is_valid:
        return {
            "answer": rejection_reason,
            "route": None,
            "reasoning": None,
            "classification": {},
            "sources": [],
            "retrieval_plan": {},
            "confidence": None,
            "contradictions": [],
            "filter_stats": {},
            "query_rejected": True,
            "cost": cost_tracker.summary(),
        }

    # Step 0b: Check full result cache
    if not skip_result_cache:
        cached_result = get_cached_query_result(query)
        if cached_result is not None:
            cached_result["_cache_hit"] = True
            cost_tracker.record_cache_hit("full_query")
            cached_result["cost"] = cost_tracker.summary()
            return cached_result

    # Step 1: Classify (skip if pre-computed)
    if precomputed_classification:
        classification = precomputed_classification
    else:
        classification = classify_query(query, cost_tracker=cost_tracker)
    classification["_original_query"] = query  # for fallback concept extraction
    route = classification["route"]
    reasoning = classification["reasoning"]

    # Step 1b: Scope validation (ticker, year, section boundaries)
    in_scope, scope_message = validate_scope(classification)
    if not in_scope:
        return {
            "answer": scope_message,
            "route": route,
            "reasoning": reasoning,
            "classification": classification,
            "sources": [],
            "retrieval_plan": {},
            "confidence": None,
            "contradictions": [],
            "filter_stats": {},
            "query_rejected": True,
            "cost": cost_tracker.summary(),
        }

    # Step 1c: Query decomposition for multi-company queries
    sub_queries = decompose_query(query, classification, cost_tracker=cost_tracker)
    if sub_queries:
        print(
            f"{COLORS['dim']}Decomposed into {len(sub_queries)} sub-queries: "
            f"{', '.join(sq['ticker'] + ' (' + sq['purpose'] + ')' for sq in sub_queries)}{COLORS['reset']}"
        )

    # Step 1d: Data availability check
    data_availability = check_data_availability(classification)
    if data_availability.get("notes"):
        for note in data_availability["notes"]:
            print(f"{COLORS['yellow']}⚠ {note}{COLORS['reset']}")

    # Step 2: Build retrieval plan
    retrieval_plan = build_retrieval_plan(query, classification, data_availability=data_availability)

    # Step 3: Retrieve (with retrieval-level cache)
    cached_retrieval = get_cached_retrieval(route, classification)
    if cached_retrieval is not None:
        data = cached_retrieval
        cost_tracker.record_cache_hit("retrieve")
    elif sub_queries:
        # Multi-company decomposed retrieval
        data = _run_subquery_retrieval(sub_queries, classification)
        route = "hybrid"
        classification["route"] = route
        set_cached_retrieval(route, classification, data)
    else:
        # Standard single-path retrieval (5-way dispatch)
        _route_dispatch = {
            "metric_lookup": retrieve_metric_lookup,
            "timeseries": retrieve_timeseries,
            "full_statement": retrieve_full_statement,
            "narrative": retrieve_narrative,
            "hybrid": retrieve_hybrid,
            # Legacy aliases
            "vector_db": retrieve_narrative,
            "relational_db": retrieve_relational_db,
        }
        retriever = _route_dispatch.get(route, retrieve_narrative)
        # Pass cost_tracker to retrievers that support embedding calls
        if route in ("narrative", "hybrid", "vector_db"):
            data = retriever(query, classification, cost_tracker=cost_tracker)
        else:
            data = retriever(query, classification)
        if route not in _route_dispatch:
            route = "narrative"
            classification["route"] = route
        set_cached_retrieval(route, classification, data)

    # Step 4: Apply retrieval guardrails (filter low-quality results)
    data, filter_stats = apply_retrieval_guardrails(route, data, classification)

    # Step 5: Detect contradictions (hybrid route)
    contradictions = []
    if route == "hybrid":
        contradictions = detect_contradictions(
            vector_chunks=data.get("vector", []),
            relational_data=data.get("relational", {}),
            classification=classification,
        )

    # Step 6: Format context
    context = format_context(route, data, classification, data_availability=data_availability)

    # Step 7: Build sources list
    sources = _build_sources(route, data, classification)

    # Step 8: Generate answer
    if not context.strip():
        answer = (
            "I couldn't find relevant information for your query. "
            "Please check the ticker symbol and try rephrasing your question."
        )
    else:
        answer = generate_answer(query, context, classification, cost_tracker=cost_tracker)
        # Always append the authoritative source list
        if sources:
            source_lines = "\n".join(f"- {s['filing']}" for s in sources)
            if "**Sources:**" not in answer:
                answer += f"\n\n---\n**Sources:**\n{source_lines}"

    # Step 9: Append contradiction warnings to answer (if config says to)
    answer_cfg = _guardrails_cfg.get("answer", {})
    if answer_cfg.get("surface_contradictions", True) and contradictions:
        answer += format_contradiction_warnings(contradictions)

    # Step 10: Compute confidence score
    confidence = compute_confidence(
        route=route,
        data=data,
        classification=classification,
        answer=answer,
        contradictions=contradictions,
        filter_stats=filter_stats,
    )

    # Step 11: Append confidence tier to answer
    if answer_cfg.get("surface_confidence", True) and confidence:
        answer += format_confidence_for_answer(confidence)

    result = {
        "answer": answer,
        "route": route,
        "reasoning": reasoning,
        "classification": classification,
        "sources": sources,
        "retrieval_plan": retrieval_plan,
        "data_availability": data_availability,
        "confidence": confidence,
        "contradictions": contradictions,
        "filter_stats": filter_stats,
        "query_rejected": False,
        "cost": cost_tracker.summary(),
    }

    # Cache the full result
    set_cached_query_result(query, result)
    return result


# --- Interactive CLI ---

def print_header():
    print(f"\n{COLORS['bold']}{'='*60}")
    print("  SEC Filing RAG Query Engine")
    print(f"{'='*60}{COLORS['reset']}")
    print(f"{COLORS['dim']}Routes: metric_lookup | timeseries | full_statement | narrative | hybrid")
    print(f"Type 'quit' or 'exit' to stop.{COLORS['reset']}\n")


def main():
    print_header()

    while True:
        try:
            query = input(f"{COLORS['cyan']}Question: {COLORS['reset']}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{COLORS['dim']}Goodbye.{COLORS['reset']}")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print(f"{COLORS['dim']}Goodbye.{COLORS['reset']}")
            break

        print(f"\n{COLORS['dim']}Classifying query...{COLORS['reset']}")

        try:
            result = rag_query(query)
        except Exception as e:
            print(f"\n{COLORS['yellow']}Error: {e}{COLORS['reset']}\n")
            continue

        # Handle rejected queries
        if result.get("query_rejected"):
            print(f"\n{COLORS['yellow']}{result['answer']}{COLORS['reset']}\n")
            continue

        route = result["route"]
        label = ROUTE_LABELS.get(route, route)

        print(f"\n{COLORS['bold']}Route:{COLORS['reset']} {label}")
        print(f"{COLORS['dim']}Reason: {result['reasoning']}{COLORS['reset']}")

        cls = result["classification"]
        meta_parts = []
        if cls.get("tickers") and len(cls["tickers"]) > 1:
            meta_parts.append(f"tickers={cls['tickers']}")
        elif cls.get("ticker"):
            meta_parts.append(f"ticker={cls['ticker']}")
        if cls.get("fiscal_year"):
            meta_parts.append(f"year={cls['fiscal_year']}")
        if cls.get("fiscal_quarter"):
            meta_parts.append(f"Q{cls['fiscal_quarter']}")
        if cls.get("concepts"):
            meta_parts.append(f"concepts={cls['concepts']}")
        if cls.get("query_type"):
            meta_parts.append(f"type={cls['query_type']}")
        if cls.get("temporal_granularity") and cls["temporal_granularity"] != "annual":
            meta_parts.append(f"temporal={cls['temporal_granularity']}")
        if cls.get("retrieval_intent"):
            meta_parts.append(f"intent={cls['retrieval_intent']}")
        if cls.get("xbrl_concepts"):
            meta_parts.append(f"xbrl={cls['xbrl_concepts']}")
        if meta_parts:
            print(f"{COLORS['dim']}Extracted: {', '.join(meta_parts)}{COLORS['reset']}")

        # Display retrieval plan
        if result.get("retrieval_plan"):
            print(format_retrieval_plan(result["retrieval_plan"]))

        # Display filter stats
        fs = result.get("filter_stats", {})
        if fs.get("dropped"):
            print(f"{COLORS['dim']}Guardrails: kept {fs['kept']}, dropped {fs['dropped']} low-quality results{COLORS['reset']}")
        for warn in fs.get("warnings", []):
            print(f"{COLORS['yellow']}  Warning: {warn}{COLORS['reset']}")

        print(f"\n{COLORS['dim']}Retrieving data...{COLORS['reset']}")
        print(f"\n{COLORS['bold']}Answer:{COLORS['reset']}")
        print(result["answer"])

        # Display confidence banner
        if result.get("confidence"):
            print(format_confidence_banner(result["confidence"]))

        # Display contradiction details
        if result.get("contradictions"):
            print(f"\n{COLORS['yellow']}{COLORS['bold']}Contradictions Detected:{COLORS['reset']}")
            for c in result["contradictions"]:
                sev_color = COLORS['yellow'] if c['severity'] == 'high' else COLORS['dim']
                print(f"  {sev_color}[{c['severity'].upper()}] {c['detail']}{COLORS['reset']}")

        print(f"\n{'—'*60}\n")


if __name__ == "__main__":
    main()
