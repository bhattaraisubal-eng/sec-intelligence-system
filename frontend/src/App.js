import { useState, useEffect, useCallback, useRef } from "react";
import "./index.css";

/* ------------------------------------------------------------------ */
/*  Typing animation hook                                              */
/* ------------------------------------------------------------------ */
function useTypewriter(text, speed = 80) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);
  useEffect(() => {
    setDisplayed("");
    setDone(false);
    let i = 0;
    const timer = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) { clearInterval(timer); setDone(true); }
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed]);
  return { displayed, done };
}

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

const TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM"];

const ROUTE_LABELS = {
  metric_lookup: "Metric Lookup",
  timeseries: "Timeseries",
  full_statement: "Full Statement",
  narrative: "Narrative Search",
  hybrid: "Hybrid",
  comparison: "Comparison",
  multi_company: "Multi-Company",
};

const EXAMPLE_QUERIES = [
  "What was total revenue in 2023?",
  "Show revenue trend from 2018 to 2023",
  "Compare net income AAPL vs MSFT 2023",
  "What did management say about AI in latest 10-K?",
  "Show full income statement for 2023",
];

const STEP_ICONS = {
  "Intent Detection": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  ),
  "Data Availability": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2 3.6 4 8 4s8-2 8-4V7M4 7c0-2 3.6-4 8-4s8 2 8 4M12 11v4m-2-2h4" />
  ),
  "XBRL Concept Resolution": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
  ),
  "Metric Lookup (XBRL)": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
  ),
  "Timeseries Retrieval": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  ),
  "Full Statement Retrieval": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  ),
  "Structured Data (SQL)": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
  ),
  "Semantic Search (Vector)": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
  ),
  "Guardrail Filtering": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
  ),
  "Contradiction Detection": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  ),
  "Fusion & Answer Generation": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  ),
  "Confidence Scoring": (
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
  ),
};

/* ------------------------------------------------------------------ */
/*  XBRL concept → human-friendly label map                            */
/* ------------------------------------------------------------------ */
const XBRL_LABELS = {
  // Revenue
  "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
  "us-gaap:Revenues": "Revenue",
  "us-gaap:SalesRevenueNet": "Net Sales",
  "us-gaap:SalesRevenueGoodsNet": "Net Sales (Goods)",
  // Cost of Revenue
  "us-gaap:CostOfRevenue": "Cost of Revenue",
  "us-gaap:CostOfGoodsAndServicesSold": "COGS",
  "us-gaap:CostOfGoodsSold": "COGS",
  // Gross Profit
  "us-gaap:GrossProfit": "Gross Profit",
  // Operating Expenses
  "us-gaap:OperatingExpenses": "Operating Expenses",
  "us-gaap:ResearchAndDevelopmentExpense": "R&D Expense",
  "us-gaap:SellingGeneralAndAdministrativeExpense": "SG&A Expense",
  "us-gaap:GeneralAndAdministrativeExpense": "G&A Expense",
  // Operating Income
  "us-gaap:OperatingIncomeLoss": "Operating Income",
  // Net Income
  "us-gaap:NetIncomeLoss": "Net Income",
  "us-gaap:ProfitLoss": "Net Income",
  // Pre-tax Income
  "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "Pre-Tax Income",
  "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments": "Pre-Tax Income",
  // Tax
  "us-gaap:IncomeTaxExpenseBenefit": "Income Tax Expense",
  // EPS
  "us-gaap:EarningsPerShareDiluted": "EPS (Diluted)",
  "us-gaap:EarningsPerShareBasic": "EPS (Basic)",
  // Assets
  "us-gaap:Assets": "Total Assets",
  "us-gaap:AssetsCurrent": "Current Assets",
  "us-gaap:AssetsNoncurrent": "Non-Current Assets",
  "us-gaap:CashAndCashEquivalentsAtCarryingValue": "Cash & Equivalents",
  "us-gaap:MarketableSecuritiesCurrent": "Marketable Securities",
  "us-gaap:AvailableForSaleSecuritiesDebtSecuritiesCurrent": "Marketable Securities",
  "us-gaap:ShortTermInvestments": "Short-Term Investments",
  "us-gaap:AccountsReceivableNetCurrent": "Accounts Receivable",
  "us-gaap:InventoryNet": "Inventory",
  "us-gaap:Goodwill": "Goodwill",
  "us-gaap:IntangibleAssetsNetExcludingGoodwill": "Intangible Assets",
  "us-gaap:PropertyPlantAndEquipmentNet": "PP&E",
  // Liabilities
  "us-gaap:Liabilities": "Total Liabilities",
  "us-gaap:LiabilitiesCurrent": "Current Liabilities",
  "us-gaap:LiabilitiesNoncurrent": "Non-Current Liabilities",
  "us-gaap:LongTermDebt": "Long-Term Debt",
  "us-gaap:LongTermDebtNoncurrent": "Long-Term Debt",
  "us-gaap:ShortTermBorrowings": "Short-Term Debt",
  "us-gaap:DebtCurrent": "Current Debt",
  "us-gaap:AccountsPayableCurrent": "Accounts Payable",
  "us-gaap:ContractWithCustomerLiabilityCurrent": "Deferred Revenue",
  "us-gaap:DeferredRevenueCurrent": "Deferred Revenue",
  "us-gaap:CommercialPaper": "Commercial Paper",
  // Equity
  "us-gaap:StockholdersEquity": "Stockholders' Equity",
  "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "Total Equity",
  "us-gaap:RetainedEarningsAccumulatedDeficit": "Retained Earnings",
  "us-gaap:CommonStockValue": "Common Stock",
  "us-gaap:TreasuryStockValue": "Treasury Stock",
  // Shares
  "us-gaap:CommonStockSharesOutstanding": "Shares Outstanding",
  "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": "Diluted Shares",
  "us-gaap:WeightedAverageNumberOfShareOutstandingBasicAndDiluted": "Shares Outstanding",
  "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic": "Basic Shares",
  // Cash Flow
  "us-gaap:NetCashProvidedByUsedInOperatingActivities": "Operating Cash Flow",
  "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "Operating Cash Flow",
  "us-gaap:NetCashProvidedByUsedInInvestingActivities": "Investing Cash Flow",
  "us-gaap:NetCashProvidedByUsedInInvestingActivitiesContinuingOperations": "Investing Cash Flow",
  "us-gaap:NetCashProvidedByUsedInFinancingActivities": "Financing Cash Flow",
  "us-gaap:NetCashProvidedByUsedInFinancingActivitiesContinuingOperations": "Financing Cash Flow",
  "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": "Capital Expenditures",
  "us-gaap:PaymentsToAcquireProductiveAssets": "Capital Expenditures",
  "us-gaap:DepreciationDepletionAndAmortization": "D&A",
  "us-gaap:Depreciation": "Depreciation",
  "us-gaap:DepreciationAmortizationAndAccretionNet": "D&A",
  "us-gaap:PaymentsOfDividends": "Dividends Paid",
  "us-gaap:PaymentsForRepurchaseOfCommonStock": "Stock Repurchases",
  // Interest & Other
  "us-gaap:InterestExpense": "Interest Expense",
  "us-gaap:InvestmentIncomeInterest": "Interest Income",
  "us-gaap:InterestIncomeExpenseNet": "Net Interest Income",
  "us-gaap:OtherNonoperatingIncomeExpense": "Other Income",
  "us-gaap:NonoperatingIncomeExpense": "Non-Operating Income",
};

/** Convert raw XBRL concept tag to human-friendly label */
function xbrlLabel(concept) {
  if (!concept) return concept;
  if (XBRL_LABELS[concept]) return XBRL_LABELS[concept];
  // Fallback: strip namespace and convert CamelCase to words
  const local = concept.includes(":") ? concept.split(":")[1] : concept;
  return local.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2");
}

/* ------------------------------------------------------------------ */
/*  Inline markdown parser                                             */
/* ------------------------------------------------------------------ */
function findSourceUrl(sourceText, sources) {
  if (!sources || sources.length === 0) return null;

  // Handle XBRL format: "us-gaap:NetIncomeLoss | AAPL, FY 2023"
  const xbrlMatch = sourceText.match(/^(.+?)\s*\|\s*(\w+),\s*FY\s*(\d{4})/);
  if (xbrlMatch) {
    const concept = xbrlMatch[1].trim().toLowerCase();
    const ticker = xbrlMatch[2].toLowerCase();
    const year = xbrlMatch[3];
    for (const src of sources) {
      const filing = (src.filing || "").toLowerCase();
      if (filing.includes(ticker) && filing.includes(`fy ${year}`) && filing.includes(concept) && src.filing_url) {
        return src.filing_url;
      }
    }
  }

  // Fallback: fuzzy word-overlap matching
  const lower = sourceText.toLowerCase();
  for (const src of sources) {
    const filing = (src.filing || "").toLowerCase();
    if (!filing) continue;
    const words = lower.split(/\s+/).filter((w) => w.length > 2);
    const matchCount = words.filter((w) => filing.includes(w)).length;
    if (matchCount >= Math.max(1, words.length * 0.5) && src.filing_url) {
      return src.filing_url;
    }
  }
  return null;
}

const UP_ARROW = (
  <svg className="inline h-3.5 w-3.5 -mt-0.5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M10 17a.75.75 0 01-.75-.75V5.612L5.29 9.77a.75.75 0 01-1.08-1.04l5.25-5.5a.75.75 0 011.08 0l5.25 5.5a.75.75 0 11-1.08 1.04l-3.96-4.158V16.25A.75.75 0 0110 17z" clipRule="evenodd" />
  </svg>
);
const DOWN_ARROW = (
  <svg className="inline h-3.5 w-3.5 -mt-0.5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M10 3a.75.75 0 01.75.75v10.638l3.96-4.158a.75.75 0 111.08 1.04l-5.25 5.5a.75.75 0 01-1.08 0l-5.25-5.5a.75.75 0 111.08-1.04l3.96 4.158V3.75A.75.75 0 0110 3z" clipRule="evenodd" />
  </svg>
);

// Patterns that indicate an increase (green arrow up)
const INCREASE_RE = /\(Increase\b[^)]*\)|↑|\bgrew\b|\bgrowth\b|\bincreas(?:e|ed|ing)\b|\brose\b|\bup\b(?:\s+by\b)/i;
// Patterns that indicate a decrease (red arrow down)
const DECREASE_RE = /\(Decrease\b[^)]*\)|↓|\bdeclin(?:e|ed|ing)\b|\bdecreas(?:e|ed|ing)\b|\bdrop(?:ped)?\b|\bfell\b|\bdown\b(?:\s+by\b)/i;
// Signed percentage or dollar: +6.9%, -2.8%, +$13.64, -$11.04
// Also match unicode arrow patterns: (↑ 26.5%), (↓ 1.1%)
// Match: ↓ -$75.14 billion | ↑ +6.9% | ↓ 1.1% | +6.9% | -2.8% | -$75.14 billion
const SIGNED_CHANGE_RE = /([\u2191\u2193])\s*([+-]?\$?[\d,.]+(?:\s*(?:billion|million|thousand))?\s*(?:USD\s*)?(?:%)?)|([+-])(\$?[\d,.]+(?:\s*(?:billion|million|thousand))?\s*(?:USD\s*)?(?:%)?)/ ;

function renderInline(text, sources) {
  if (!text) return null;
  const parts = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    const sourceMatch = remaining.match(/\[(?:Source|XBRL):\s*([^\]]+)\]/);
    const changeMatch = remaining.match(SIGNED_CHANGE_RE);
    const inlineMathMatch = remaining.match(/\\\((.+?)\\\)/);

    const matches = [
      boldMatch && { type: "bold", match: boldMatch },
      sourceMatch && { type: "source", match: sourceMatch },
      changeMatch && { type: "change", match: changeMatch },
      inlineMathMatch && { type: "inlineMath", match: inlineMathMatch },
    ].filter(Boolean);

    if (matches.length === 0) {
      // No special tokens left — check for word-based increase/decrease in plain text
      parts.push(<span key={key++}>{remaining}</span>);
      break;
    }

    matches.sort((a, b) => a.match.index - b.match.index);
    const earliest = matches[0];
    const m = earliest.match;

    if (m.index > 0) {
      parts.push(<span key={key++}>{remaining.slice(0, m.index)}</span>);
    }

    if (earliest.type === "bold") {
      // Check if bold content is a signed change value like **+6.9%**, **-2.8%**, **↑ 26.5%**, **↓ -$75.14 billion**
      const innerArrow = m[1].match(/^([\u2191\u2193])\s*(.+)$/);
      const innerChange = m[1].match(/^([+-])(.+)$/);
      if (innerArrow) {
        const isUp = innerArrow[1] === "\u2191";
        parts.push(
          <strong key={key++} className={`font-semibold inline-flex items-center gap-0.5 ${isUp ? "text-term-green" : "text-bb_red"}`}>
            {isUp ? UP_ARROW : DOWN_ARROW}{innerArrow[2]}
          </strong>
        );
      } else if (innerChange) {
        const isUp = innerChange[1] === "+";
        parts.push(
          <strong key={key++} className={`font-semibold inline-flex items-center gap-0.5 ${isUp ? "text-term-green" : "text-bb_red"}`}>
            {isUp ? UP_ARROW : DOWN_ARROW}{innerChange[2]}
          </strong>
        );
      } else {
        parts.push(
          <strong key={key++} className="font-semibold text-bb-gray-100">{m[1]}</strong>
        );
      }
    } else if (earliest.type === "change") {
      // m[1]=arrow m[2]=value for ↑/↓ patterns; m[3]=sign m[4]=value for +/- patterns
      const isUp = m[1] === "\u2191" || m[3] === "+";
      const displayText = m[1] ? m[2] : m[0];
      parts.push(
        <span key={key++} className={`inline-flex items-center gap-0.5 font-semibold ${isUp ? "text-term-green" : "text-bb_red"}`}>
          {isUp ? UP_ARROW : DOWN_ARROW}{displayText}
        </span>
      );
    } else if (earliest.type === "source") {
      const rawLabel = m[1];
      // Format label for display
      const shortLabel = (() => {
        // New XBRL format: "us-gaap:NetIncomeLoss | AAPL, FY 2023"
        const xbrlMatch = rawLabel.match(/^(.+?)\s*\|\s*(\w+),\s*FY\s*(\d{4}.*)/);
        if (xbrlMatch) {
          const concept = xbrlMatch[1].trim();
          const ticker = xbrlMatch[2];
          const period = xbrlMatch[3].trim();
          const friendly = xbrlLabel(concept);
          return `${friendly} · ${ticker} FY${period}`;
        }
        // Legacy Source format: "10-K XBRL, Ticker: AAPL, Year: 2023, Concept: us-gaap:Foo"
        // Also handles LLM-generated: "XBRL, Ticker: AAPL, Year: 2023, Section: us-gaap:Foo"
        const filing = (rawLabel.match(/\b(10-[KQ]|8-K)\b/i) || [""])[0].toUpperCase();
        const ticker = (rawLabel.match(/Ticker:\s*(\w+)/i) || [, ""])[1];
        const year = (rawLabel.match(/Year:\s*(\d{4})/i) || [, ""])[1];
        const concept = (rawLabel.match(/(?:Concept|Section):\s*([\w\-.:]+)/i) || [, ""])[1];
        const friendly = concept ? xbrlLabel(concept) : "";
        if (friendly && ticker && year) return `${friendly} · ${ticker} FY${year}`;
        if (filing && ticker && year) return `${filing} · ${ticker} FY${year}`;
        if (filing && ticker) return `${filing} · ${ticker}`;
        // Fallback: show full text
        return rawLabel;
      })();
      const url = findSourceUrl(rawLabel, sources);
      if (url) {
        parts.push(
          <a key={key++} href={url} target="_blank" rel="noopener noreferrer"
            title={rawLabel}
            className="ml-1 inline-flex items-center gap-0.5 rounded bg-bb_blue-bg px-1.5 py-0.5 text-xxs font-mono text-bb_blue-bright hover:text-bb_blue border border-bb_blue/20 hover:border-bb_blue/40 transition-colors no-underline">
            <svg className="h-2.5 w-2.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            {shortLabel}
          </a>
        );
      } else {
        parts.push(
          <span key={key++} title={rawLabel} className="ml-1 inline-flex items-center rounded bg-bb_blue-bg px-1.5 py-0.5 text-xxs font-mono text-bb_blue-bright">
            {shortLabel}
          </span>
        );
      }
    } else if (earliest.type === "inlineMath") {
      parts.push(
        <span key={key++} className="inline rounded bg-bb-surface border border-bb-border px-1.5 py-0.5 font-mono text-xs text-bb-gray-200">
          {latexToReadable(m[1])}
        </span>
      );
    }
    remaining = remaining.slice(m.index + m[0].length);
  }
  return parts;
}

/* Render table cells with change arrows */
function renderTableCell(cell, sources) {
  if (!cell) return null;
  const trimmed = cell.trim();
  // Handle: ↓ -$75.14 billion, ↑ +6.9%, ↓ 1.1%
  const arrowMatch = trimmed.match(/^([\u2191\u2193])\s*(.+)$/);
  if (arrowMatch) {
    const isUp = arrowMatch[1] === "\u2191";
    return (
      <span className={`inline-flex items-center gap-0.5 font-semibold ${isUp ? "text-term-green" : "text-bb_red"}`}>
        {isUp ? UP_ARROW : DOWN_ARROW}{arrowMatch[2]}
      </span>
    );
  }
  const signedMatch = trimmed.match(/^([+-])(.+)$/);
  if (signedMatch) {
    const isUp = signedMatch[1] === "+";
    return (
      <span className={`inline-flex items-center gap-0.5 font-semibold ${isUp ? "text-term-green" : "text-bb_red"}`}>
        {isUp ? UP_ARROW : DOWN_ARROW}{trimmed}
      </span>
    );
  }
  // Check for negative values like -2.8% or -11.04 billion
  const negMatch = trimmed.match(/^-[\d,.]+\s*(?:billion|million|thousand)?\s*(?:USD\s*)?%?$/i);
  if (negMatch) {
    return (
      <span className="inline-flex items-center gap-0.5 font-semibold text-bb_red">
        {DOWN_ARROW}{trimmed}
      </span>
    );
  }
  return renderInline(cell, sources);
}

/* ------------------------------------------------------------------ */
/*  Retrieval Plan — terminal stepper                                  */
/* ------------------------------------------------------------------ */
function RetrievalPlan({ steps, activeStep, completed }) {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="mt-4 bb-panel-inset rounded bg-bb-panel p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xxs font-mono font-semibold uppercase tracking-widest text-term-green">
          Retrieval Plan
        </span>
        {!completed && (
          <span className="ml-auto text-xxs font-mono text-amber animate-pulse">EXECUTING...</span>
        )}
        {completed && (
          <span className="ml-auto text-xxs font-mono text-term-green">COMPLETE</span>
        )}
      </div>

      <div className="space-y-0.5">
        {steps.map((step, i) => {
          const isActive = !completed && i === activeStep;
          const isDone = completed || i < activeStep;
          const isPending = !completed && i > activeStep;
          const icon = STEP_ICONS[step.name];

          return (
            <div
              key={i}
              className={`flex items-start gap-2 py-1.5 px-2 rounded transition-all duration-300 ${
                isActive ? "bg-term-bg" : isDone ? "bg-transparent" : "opacity-30"
              }`}
            >
              {/* Status indicator */}
              <div className="flex-shrink-0 mt-0.5">
                {isDone ? (
                  <span className="text-term-green font-mono text-xs">&#10003;</span>
                ) : isActive ? (
                  <span className="inline-block h-2 w-2 rounded-full bg-term-green animate-pulse-dot mt-1" />
                ) : (
                  <span className="text-bb-gray-500 font-mono text-xs">{step.step}</span>
                )}
              </div>

              {/* Step content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  {icon && (
                    <svg
                      className={`h-3.5 w-3.5 flex-shrink-0 ${
                        isDone ? "text-term-dim" : isActive ? "text-term-green" : "text-bb-gray-500"
                      }`}
                      fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}
                    >
                      {icon}
                    </svg>
                  )}
                  <span className={`text-xs font-mono ${
                    isDone ? "text-bb-gray-300" : isActive ? "text-term-green" : "text-bb-gray-500"
                  }`}>
                    {step.name}
                  </span>
                  {step.tag && (
                    <span className={`rounded px-1 py-0.5 text-xxs font-mono uppercase ${
                      step.tag === "primary"
                        ? "bg-bb_blue-bg text-bb_blue-bright"
                        : "bg-bb-surface text-bb-gray-400"
                    }`}>
                      {step.tag}
                    </span>
                  )}
                </div>

                {(isDone || isActive) && step.actions && step.actions.length > 0 && (
                  <div className="mt-0.5 space-y-0">
                    {step.actions.map((action, j) => (
                      <p key={j} className="text-xxs font-mono text-bb-gray-400 pl-5">
                        &rsaquo; {action}
                      </p>
                    ))}
                  </div>
                )}

                {(isDone || isActive) && step.details && (
                  <div className="mt-1 flex flex-wrap gap-1 pl-5">
                    {Object.entries(step.details)
                      .filter(([k]) => !["route"].includes(k))
                      .map(([k, v]) => {
                        const isMissing = typeof v === "string" && (v.startsWith("NO ") || v.includes("MISSING"));
                        return (
                          <span key={k} className={`inline-flex items-center rounded px-1.5 py-0.5 text-xxs font-mono border ${
                            isMissing
                              ? "bg-amber/10 border-amber/30"
                              : "bg-bb-surface border-bb-border"
                          }`}>
                            <span className="mr-1 text-bb-gray-400">{k}:</span>
                            <span className={isMissing ? "text-amber" : "text-bb-gray-200"}>
                              {typeof v === "object" ? JSON.stringify(v) : String(v)}
                            </span>
                          </span>
                        );
                      })}
                  </div>
                )}

                {(isDone || isActive) && step.warnings && step.warnings.length > 0 && (
                  <div className="mt-1 space-y-0 pl-5">
                    {step.warnings.map((warning, j) => (
                      <p key={j} className="text-xxs font-mono text-amber">
                        &#x26A0; {warning}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Terminal Spinner                                                    */
/* ------------------------------------------------------------------ */
function TerminalSpinner() {
  return (
    <div className="flex items-center gap-3 py-8 px-4">
      <span className="inline-block h-2 w-2 rounded-full bg-term-green animate-pulse-dot" />
      <span className="text-sm font-mono text-term-green">Classifying query...</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Skeleton loader — mimics the answer layout while loading            */
/* ------------------------------------------------------------------ */
function AnswerSkeleton() {
  const shimmerClass = "relative overflow-hidden bg-bb-surface rounded after:absolute after:inset-0 after:bg-gradient-to-r after:from-transparent after:via-bb-hover after:to-transparent after:animate-[shimmer_1.5s_infinite]";
  return (
    <div className="space-y-4 animate-fade-up" style={{ animationDelay: "0s" }}>
      {/* Metadata strip skeleton */}
      <div className="flex items-center gap-2">
        <div className={`h-6 w-28 rounded ${shimmerClass}`} />
        <div className={`h-6 w-16 rounded ${shimmerClass}`} />
        <div className={`h-6 w-14 rounded ${shimmerClass}`} />
      </div>

      {/* Answer card skeleton */}
      <div className="bb-panel-inset rounded bg-bb-panel p-4">
        <div className="flex items-center gap-2 mb-3 pb-2 border-b border-bb-border">
          <div className="w-1 h-4 bg-term-green/30 rounded-full" />
          <div className={`h-4 w-20 rounded ${shimmerClass}`} />
        </div>

        <div className="space-y-3">
          {/* Section heading */}
          <div className={`h-5 w-48 rounded ${shimmerClass}`} />

          {/* Bullet points */}
          <div className="flex gap-2 pl-2">
            <span className="mt-2 h-1 w-1 flex-shrink-0 rounded-full bg-bb-border" />
            <div className={`h-4 w-full rounded ${shimmerClass}`} />
          </div>
          <div className="flex gap-2 pl-2">
            <span className="mt-2 h-1 w-1 flex-shrink-0 rounded-full bg-bb-border" />
            <div className={`h-4 w-5/6 rounded ${shimmerClass}`} />
          </div>

          {/* Section heading */}
          <div className={`h-5 w-40 rounded mt-2 ${shimmerClass}`} />

          {/* Bullet points */}
          <div className="flex gap-2 pl-2">
            <span className="mt-2 h-1 w-1 flex-shrink-0 rounded-full bg-bb-border" />
            <div className={`h-4 w-full rounded ${shimmerClass}`} />
          </div>
          <div className="flex gap-2 pl-2">
            <span className="mt-2 h-1 w-1 flex-shrink-0 rounded-full bg-bb-border" />
            <div className={`h-4 w-4/6 rounded ${shimmerClass}`} />
          </div>

          {/* Table skeleton */}
          <div className="mt-3 rounded border border-bb-border overflow-hidden">
            <div className="flex border-b border-bb-border bg-bb-surface">
              <div className={`h-7 w-1/4 m-1 rounded ${shimmerClass}`} />
              <div className={`h-7 w-1/4 m-1 rounded ${shimmerClass}`} />
              <div className={`h-7 w-1/4 m-1 rounded ${shimmerClass}`} />
            </div>
            {[0, 1].map((r) => (
              <div key={r} className="flex border-b border-bb-border/50">
                <div className={`h-6 w-1/4 m-1.5 rounded ${shimmerClass}`} />
                <div className={`h-6 w-1/4 m-1.5 rounded ${shimmerClass}`} />
                <div className={`h-6 w-1/4 m-1.5 rounded ${shimmerClass}`} />
              </div>
            ))}
          </div>

          {/* Summary line */}
          <div className={`h-4 w-3/4 rounded mt-2 ${shimmerClass}`} />
          <div className={`h-4 w-2/3 rounded ${shimmerClass}`} />
        </div>
      </div>

      {/* Confidence skeleton */}
      <div className={`h-12 w-full rounded ${shimmerClass}`} />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  LaTeX math to readable text (fallback for inline)                  */
/* ------------------------------------------------------------------ */
function latexToReadable(tex) {
  let s = tex;
  s = s.replace(/\\text\{([^}]*)\}/g, "$1");
  s = s.replace(/\\frac\{([^}]*)}\{([^}]*)}/g, "$1 / $2");
  s = s.replace(/\\left\s*\(/g, "(").replace(/\\right\s*\)/g, ")");
  s = s.replace(/\^\{\\frac\{([^}]*)}\{([^}]*)}\}/g, "^($1/$2)");
  s = s.replace(/\^\{([^}]*)}/g, "^($1)");
  s = s.replace(/\\approx/g, " ≈ ");
  s = s.replace(/\\times/g, " × ");
  s = s.replace(/\\cdot/g, "·");
  s = s.replace(/\\%/g, "%");
  s = s.replace(/\\[a-zA-Z]+/g, "");
  s = s.replace(/[{}]/g, "").replace(/\s{2,}/g, " ").trim();
  return s;
}

/* ------------------------------------------------------------------ */
/*  LaTeX math to React elements (rich rendering for display blocks)   */
/* ------------------------------------------------------------------ */
function renderMath(tex) {
  let s = tex;
  // Normalize: strip \text{}, \left, \right
  s = s.replace(/\\text\{([^}]*)\}/g, "$1");
  s = s.replace(/\\left\s*/g, "").replace(/\\right\s*/g, "");
  s = s.replace(/\\%/g, "%");
  s = s.replace(/\\approx/g, "≈");
  s = s.replace(/\\times/g, "×");
  s = s.replace(/\\cdot/g, "·");
  // Remove remaining backslash commands (but not braces yet)
  s = s.replace(/\\[a-zA-Z]+/g, "");

  // Tokenize into React elements
  const elements = [];
  let key = 0;
  let remaining = s;

  while (remaining.length > 0) {
    // Match \frac{...}{...} pattern (already stripped \frac, look for consecutive {a}{b})
    // Since we stripped \frac, look for the pattern in pre-processed form
    // Actually let's re-approach: process from the original tex
    break;
  }

  // Simpler approach: process the full string with regex replacements to build JSX
  // Split on key math constructs and rebuild
  const parts = [];
  let rest = tex;
  let k = 0;

  // Pre-clean
  rest = rest.replace(/\\text\{([^}]*)\}/g, "$1");
  rest = rest.replace(/\\left\s*/g, "").replace(/\\right\s*/g, "");
  rest = rest.replace(/\\%/g, "%");

  while (rest.length > 0) {
    // Look for \frac{...}{...} possibly followed by ^{...}
    const fracExp = rest.match(/\\frac\{([^}]*)}\{([^}]*)}(\s*\^\{([^}]*)})?\s*/);
    // Look for standalone ^{...}
    const supMatch = !fracExp ? rest.match(/\^\{([^}]*)}/) : null;
    // Look for ≈ or \approx
    const approxMatch = !fracExp && !supMatch ? rest.match(/\\approx/) : null;

    const candidates = [
      fracExp && { type: "frac", match: fracExp },
      supMatch && { type: "sup", match: supMatch },
      approxMatch && { type: "approx", match: approxMatch },
    ].filter(Boolean);

    if (candidates.length === 0) {
      // Clean remaining
      let clean = rest.replace(/\\[a-zA-Z]+/g, "").replace(/[{}]/g, "").replace(/\s{2,}/g, " ").trim();
      if (clean) parts.push(<span key={k++}>{clean}</span>);
      break;
    }

    candidates.sort((a, b) => a.match.index - b.match.index);
    const first = candidates[0];
    const fm = first.match;

    // Text before match
    if (fm.index > 0) {
      let before = rest.slice(0, fm.index).replace(/\\[a-zA-Z]+/g, "").replace(/[{}]/g, "").replace(/\s{2,}/g, " ");
      if (before.trim()) parts.push(<span key={k++}>{before}</span>);
    }

    if (first.type === "frac") {
      const numer = fm[1].replace(/\\[a-zA-Z]+/g, "").replace(/[{}]/g, "").trim();
      const denom = fm[2].replace(/\\[a-zA-Z]+/g, "").replace(/[{}]/g, "").trim();
      const exponent = fm[4] ? fm[4].replace(/\\frac\{([^}]*)}\{([^}]*)}/g, "$1/$2").replace(/\\[a-zA-Z]+/g, "").replace(/[{}]/g, "").trim() : null;

      parts.push(
        <span key={k++} className="inline-flex items-center">
          <span className="inline-flex flex-col items-center mx-0.5 relative" style={{ verticalAlign: "middle" }}>
            <span className="text-xs leading-tight px-1">{numer}</span>
            <span className="w-full border-t border-bb-gray-400" />
            <span className="text-xs leading-tight px-1">{denom}</span>
          </span>
          {exponent && (
            <sup className="text-xxs text-amber -ml-0.5 -translate-y-3 inline-block">{exponent}</sup>
          )}
        </span>
      );
    } else if (first.type === "sup") {
      const exp = fm[1].replace(/\\frac\{([^}]*)}\{([^}]*)}/g, "$1/$2").replace(/\\[a-zA-Z]+/g, "").replace(/[{}]/g, "").trim();
      parts.push(<sup key={k++} className="text-xxs text-amber">{exp}</sup>);
    } else if (first.type === "approx") {
      parts.push(<span key={k++}> ≈ </span>);
    }

    rest = rest.slice(fm.index + fm[0].length);
  }

  return parts.length > 0 ? parts : latexToReadable(tex);
}

/* ------------------------------------------------------------------ */
/*  Answer block with markdown parsing                                 */
/* ------------------------------------------------------------------ */
function AnswerBlock({ answer, sources }) {
  if (!answer) return null;
  const lines = answer.split("\n");
  const isSeparator = (line) => /^\|[\s-:|]+\|$/.test(line.trim());

  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (/^-{3,}$/.test(line.trim())) {
      blocks.push({ type: "hr", key: i });
      i++;
      continue;
    }

    if (line.trimStart().startsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].trimStart().startsWith("|")) {
        if (!isSeparator(lines[i])) tableLines.push(lines[i]);
        i++;
      }
      if (tableLines.length > 0) {
        const parseCells = (l) => l.split("|").slice(1, -1).map((c) => c.trim());
        blocks.push({
          type: "table", key: blocks.length,
          header: parseCells(tableLines[0]),
          rows: tableLines.slice(1).map(parseCells),
        });
      }
      continue;
    }

    if (line.startsWith("#### ")) { blocks.push({ type: "h4", key: i, text: line.slice(5) }); i++; continue; }
    if (line.startsWith("### ")) { blocks.push({ type: "h3", key: i, text: line.slice(4) }); i++; continue; }
    if (line.startsWith("## ")) { blocks.push({ type: "h2", key: i, text: line.slice(3) }); i++; continue; }
    if (line.startsWith("# ")) { blocks.push({ type: "h1", key: i, text: line.slice(2) }); i++; continue; }
    if (line.trimStart().startsWith("- ")) {
      const indent = line.length - line.trimStart().length;
      const depth = Math.floor(indent / 2);
      blocks.push({ type: "bullet", key: i, text: line.replace(/^\s*-\s/, ""), depth });
      i++; continue;
    }
    if (/^\d+\.\s/.test(line.trimStart())) {
      const indent = line.length - line.trimStart().length;
      const depth = Math.floor(indent / 2);
      blocks.push({ type: "bullet", key: i, text: line.replace(/^\s*\d+\.\s/, ""), depth });
      i++; continue;
    }
    if (line.trim() === "") { blocks.push({ type: "space", key: i }); i++; continue; }

    // LaTeX display math block: \[ ... \] (may span multiple lines)
    if (line.trim().startsWith("\\[")) {
      let mathLines = [line];
      while (i + 1 < lines.length && !mathLines.join("\n").includes("\\]")) {
        i++;
        mathLines.push(lines[i]);
      }
      blocks.push({ type: "math", key: i, text: mathLines.join(" ").replace(/\\\[|\\\]/g, "").trim() });
      i++;
      continue;
    }

    blocks.push({ type: "p", key: i, text: line });
    i++;
  }

  return (
    <div className="space-y-1.5 text-sm leading-relaxed text-bb-gray-200 font-sans">
      {blocks.map((block) => {
        switch (block.type) {
          case "hr": return <hr key={block.key} className="my-3 border-bb-border" />;
          case "h1": return <h2 key={block.key} className="mt-4 mb-2 text-base font-bold text-amber-bright font-mono">{renderInline(block.text, sources)}</h2>;
          case "h2": return <h3 key={block.key} className="mt-3 mb-1 text-sm font-semibold text-amber font-mono">{renderInline(block.text, sources)}</h3>;
          case "h3": return <h4 key={block.key} className="mt-2 mb-1 text-sm font-medium text-amber-dim font-mono">{renderInline(block.text, sources)}</h4>;
          case "h4": return <h5 key={block.key} className="mt-2 mb-1 text-sm font-normal text-bb-gray-200 font-mono">{renderInline(block.text, sources)}</h5>;
          case "bullet": {
            const isUp = INCREASE_RE.test(block.text) && !DECREASE_RE.test(block.text);
            const isDown = DECREASE_RE.test(block.text) && !INCREASE_RE.test(block.text);
            const depth = block.depth || 0;
            const dotColor = isUp ? "bg-term-green" : isDown ? "bg-bb_red" : "bg-term-green";
            const dotSize = depth === 0 ? "h-1 w-1 rounded-full" : "h-1 w-1 rounded-sm";
            const textOpacity = depth === 0 ? "" : "opacity-85";
            return (
              <div key={block.key} className="flex gap-2" style={{ paddingLeft: `${8 + depth * 16}px` }}>
                <span className={`mt-2 flex-shrink-0 ${dotSize} ${dotColor}`} />
                <span className={textOpacity}>{renderInline(block.text, sources)}</span>
              </div>
            );
          }
          case "table": {
            // Detect which columns are "change" columns for arrow rendering
            const changeColIdx = new Set();
            block.header.forEach((h, idx) => {
              const lc = h.toLowerCase().trim();
              if (lc.includes("change") || lc.includes("growth") || lc.includes("delta") || lc.includes("difference")) changeColIdx.add(idx);
            });
            return (
              <div key={block.key} className="my-3 overflow-x-auto rounded border border-bb-border">
                <table className="w-full text-left text-sm font-mono">
                  <thead>
                    <tr className="border-b border-bb-border bg-bb-surface">
                      {block.header.map((cell, j) => (
                        <th key={j} className="px-3 py-2 text-xxs font-semibold uppercase tracking-wider text-amber">{renderInline(cell, sources)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, ri) => (
                      <tr key={ri} className="border-b border-bb-border/50 bb-row-hover">
                        {row.map((cell, ci) => (
                          <td key={ci} className="px-3 py-1.5 text-bb-gray-200">
                            {changeColIdx.has(ci) ? renderTableCell(cell, sources) : renderInline(cell, sources)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }
          case "math":
            return (
              <div key={block.key} className="my-2 rounded bg-bb-surface border border-bb-border px-4 py-3 font-mono text-sm text-bb-gray-200 overflow-x-auto flex items-center flex-wrap gap-1">
                {renderMath(block.text)}
              </div>
            );
          case "space": return <div key={block.key} className="h-1" />;
          case "p": default: return <p key={block.key}>{renderInline(block.text, sources)}</p>;
        }
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Confidence Breakdown — horizontal bars                             */
/* ------------------------------------------------------------------ */
const SIGNAL_META = {
  retrieval_quality: { label: "SRC AUTH", color: "bg-bb_blue" },
  source_coverage: { label: "COVERAGE", color: "bg-term-green" },
  cross_source_agreement: { label: "AGREEMENT", color: "bg-term-green" },
  citation_density: { label: "CITATION", color: "bg-amber" },
  data_recency: { label: "RECENCY", color: "bg-bb_blue" },
};

function ConfidenceBreakdown({ confidence }) {
  if (!confidence || !confidence.signals) return null;

  const { overall_score, tier_label, tier_color, tier_description, signals } = confidence;
  const scoreColor =
    tier_color === "green" ? "text-term-green" : tier_color === "yellow" ? "text-amber" : "text-bb_red";
  const scoreBg =
    tier_color === "green" ? "bg-term-green" : tier_color === "yellow" ? "bg-amber" : "bg-bb_red";

  return (
    <div className="bb-panel-inset rounded bg-bb-panel p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono font-semibold uppercase tracking-widest text-bb-gray-300">
          Confidence
        </span>
        <span className={`rounded px-2 py-0.5 text-xs font-mono font-bold border ${
          tier_color === "green" ? "bg-term-bg text-term-green border-term-green/30" :
          tier_color === "yellow" ? "bg-amber-bg text-amber border-amber/30" :
          "bg-bb_red-bg text-bb_red border-bb_red/30"
        }`}>
          {tier_label}
        </span>
      </div>

      {/* Big score number */}
      <div className="mb-1">
        <span className={`text-2xl font-mono font-bold tabular-nums ${scoreColor}`}>
          {Math.round(overall_score)}
        </span>
        <span className="text-xs font-mono text-bb-gray-400 ml-1">/ 100</span>
      </div>

      {tier_description && (
        <p className="text-xs font-mono text-bb-gray-400 mb-3">{tier_description}</p>
      )}

      <div className="space-y-2">
        {Object.entries(signals).map(([key, signal]) => {
          const meta = SIGNAL_META[key] || { label: key.toUpperCase(), color: "bg-bb-gray-400" };
          const score = signal.score;
          const barColor = score >= 80 ? "bg-term-green" : score >= 50 ? "bg-amber" : "bg-bb_red";

          return (
            <div key={key} className="flex items-center gap-2.5">
              <span className="text-xs font-mono text-bb-gray-200 font-medium w-24 flex-shrink-0">{meta.label}</span>
              <div className="relative h-2 w-28 flex-shrink-0 overflow-hidden rounded-sm bg-bb-surface">
                <div
                  className={`absolute inset-y-0 left-0 rounded-sm ${barColor} transition-all duration-1000 ease-out`}
                  style={{ width: `${Math.max(score, 2)}%` }}
                />
              </div>
              <span className={`text-xs font-mono font-semibold tabular-nums w-7 text-right flex-shrink-0 ${
                score >= 80 ? "text-term-green" : score >= 50 ? "text-amber" : "text-bb_red"
              }`}>
                {Math.round(score)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Cost Breakdown — "Trade Ticket" style execution cost panel          */
/* ------------------------------------------------------------------ */
const PHASE_COLORS = {
  classify:  { bar: "bg-bb_blue", dot: "bg-bb_blue-bright", label: "Classify" },
  decompose: { bar: "bg-violet-500", dot: "bg-violet-400", label: "Decompose" },
  embed:     { bar: "bg-cyan-500", dot: "bg-cyan-400", label: "Embed" },
  generate:  { bar: "bg-amber", dot: "bg-amber", label: "Generate" },
  full_query:{ bar: "bg-term-green", dot: "bg-term-green", label: "Full Query" },
  retrieve:  { bar: "bg-term-green", dot: "bg-term-green", label: "Retrieve" },
};

function formatCost(cost) {
  if (cost === 0) return "0.00¢";
  if (cost < 0.01) return `${(cost * 100).toFixed(3)}¢`;
  if (cost < 1) return `${(cost * 100).toFixed(2)}¢`;
  return `$${cost.toFixed(2)}`;
}

/* costEquivalent removed — no longer displayed */

function CostBreakdown({ cost }) {
  if (!cost || !cost.phases || cost.phases.length === 0) return null;

  const { phases, total_cost, total_tokens, wall_time_ms, efficiency } = cost;
  const grade = efficiency?.grade || "B";
  const allCached = phases.every(p => p.cached);

  // For the stacked pipeline bar
  const costPhases = phases.filter(p => !p.cached && p.cost > 0);
  const totalPhaseCost = costPhases.reduce((s, p) => s + p.cost, 0) || 1;

  const gradeColor = grade === "S" || grade === "A+" || grade === "A"
    ? "text-term-green" : grade === "B" ? "text-amber" : "text-bb_red";
  const gradeBg = grade === "S" || grade === "A+" || grade === "A"
    ? "bg-term-bg border-term-green/30" : grade === "B"
    ? "bg-amber-bg border-amber/30" : "bg-bb_red-bg border-bb_red/30";

  return (
    <div className="bb-panel-inset rounded bg-bb-panel p-4">
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono font-semibold uppercase tracking-widest text-bb-gray-300">
          Execution Cost
        </span>
        <span className={`rounded px-2 py-0.5 text-xs font-mono font-bold border ${gradeBg} ${gradeColor}`}>
          {grade}
        </span>
      </div>

      {/* Big cost number */}
      <div className="mb-3">
        <span className="text-2xl font-mono font-bold text-white tabular-nums">
          {formatCost(total_cost)}
        </span>
      </div>

      {/* Models used */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 mb-3">
        {[
          { label: "LLM", value: "gpt-4o-mini" },
          { label: "Embed", value: "text-embedding-3-small" },
          { label: "Reranker", value: "ms-marco-MiniLM-L-6" },
        ].map(m => (
          <span key={m.label} className="text-[10px] font-mono text-bb-gray-400">
            <span className="text-bb-gray-500 uppercase">{m.label}</span>{" "}
            <span className="text-bb-gray-300">{m.value}</span>
          </span>
        ))}
      </div>

      {/* Individual phase bars */}
      {!allCached && costPhases.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] font-mono text-bb-gray-500 uppercase tracking-widest mb-2">
            Cost Breakdown
          </div>
          <div className="space-y-2">
            {phases.map((p, i) => {
              const phaseStyle = PHASE_COLORS[p.phase] || PHASE_COLORS.generate;
              const pct = !p.cached && totalPhaseCost > 0 ? (p.cost / totalPhaseCost) * 100 : 0;
              return (
                <div key={i}>
                  <div className="flex items-center justify-between mb-0.5">
                    <div className="flex items-center gap-1.5">
                      <span className={`inline-block h-2 w-2 rounded-full ${p.cached ? "bg-term-green" : phaseStyle.dot}`} />
                      <span className="text-xs font-mono text-bb-gray-200 font-medium">
                        {phaseStyle.label}
                      </span>
                    </div>
                    {p.cached ? (
                      <span className="text-[10px] font-mono font-bold text-term-green tracking-wider">CACHED</span>
                    ) : (
                      <span className="text-xs font-mono tabular-nums">
                        <span className="text-white font-semibold">{formatCost(p.cost)}</span>
                      </span>
                    )}
                  </div>
                  {!p.cached && (
                    <div className="h-1.5 rounded-full overflow-hidden bg-bb-surface">
                      <div
                        className={`${phaseStyle.bar} h-full rounded-full transition-all duration-700 ease-out`}
                        style={{ width: `${Math.max(pct, 3)}%` }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Cached state */}
      {allCached && (
        <div className="flex items-center gap-2 mb-3 rounded bg-term-bg/50 border border-term-green/20 px-3 py-2.5">
          <svg className="h-4 w-4 text-term-green flex-shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
          </svg>
          <span className="text-xs font-mono text-term-green font-medium">
            Instant — served from cache
          </span>
        </div>
      )}

      {/* Stats row */}
      <div className="flex items-center gap-3 text-xs font-mono pt-2 border-t border-bb-border">
        <span className="text-bb-gray-400">
          <span className="text-bb-gray-200 font-semibold tabular-nums">{total_tokens.toLocaleString()}</span> tokens
        </span>
        <span className="text-bb-gray-700">|</span>
        <span className="text-bb-gray-400">
          <span className="text-bb-gray-200 font-semibold tabular-nums">{(wall_time_ms / 1000).toFixed(1)}s</span> wall
        </span>
        {!allCached && total_tokens > 0 && wall_time_ms > 0 && (
          <>
            <span className="text-bb-gray-700">|</span>
            <span className="text-bb-gray-400">
              <span className="text-bb-gray-200 font-semibold tabular-nums">{Math.round(total_tokens / (wall_time_ms / 1000)).toLocaleString()}</span> tok/s
            </span>
          </>
        )}
      </div>
    </div>
  );
}


/* ------------------------------------------------------------------ */
/*  Source parser (shared between timeline and main)                    */
/* ------------------------------------------------------------------ */
function parseSource(src) {
  const filing = src.filing || "";
  // Check 10-Q first (more specific), then 10-K as default for annual filings
  // A source like "[10-K XBRL]" should be typed as 10-K, not XBRL
  let type = "10-K";
  if (filing.includes("10-Q")) type = "10-Q";
  else if (filing.includes("10-K")) type = "10-K";
  else if (filing.toLowerCase().includes("comparison") || filing.toLowerCase().includes("yoy")) type = "YoY";
  // Prefer explicit ticker field from API; fall back to regex after the "[...]" prefix
  const tickerMatch = filing.match(/\]\s*([A-Z]{1,5})\b/);
  const yearMatch = filing.match(/(?:FY\s?)(\d{4})/);
  return {
    type, ticker: src.ticker || tickerMatch?.[1] || "", year: yearMatch?.[1] || "",
    filing, reference: src.reference, date: src.filing_date,
    url: src.filing_url || null,
  };
}

const FILING_TYPE_COLORS = {
  "10-K": { dot: "bg-bb_blue", badge: "bg-bb_blue-bg text-bb_blue-bright border-bb_blue/30" },
  "10-Q": { dot: "bg-term-green", badge: "bg-term-bg text-term-green border-term-green/30" },
};

/* ------------------------------------------------------------------ */
/*  Document Timeline (right sidebar)                                  */
/* ------------------------------------------------------------------ */
function DocumentTimeline({ sources }) {
  if (!sources || sources.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-4 py-8">
        <span className="text-bb-gray-500 font-mono text-xs">No filings loaded</span>
      </div>
    );
  }

  // Parse all sources, keep only 10-K and 10-Q, deduplicate by ticker+type+year+quarter
  const parsed = sources.map(parseSource);
  const seen = new Set();
  const deduped = [];
  for (const src of parsed) {
    if (src.type !== "10-K" && src.type !== "10-Q") continue;
    const qMatch = src.filing.match(/Q([1-4])/);
    const quarter = qMatch ? qMatch[1] : "";
    const key = `${src.ticker}|${src.type}|${src.year}|${quarter}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push({ ...src, quarter });
  }

  if (deduped.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-4 py-8">
        <span className="text-bb-gray-500 font-mono text-xs">No filings loaded</span>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {deduped.map((src, i) => {
        const colors = FILING_TYPE_COLORS[src.type] || FILING_TYPE_COLORS["10-K"];
        let periodLabel = "";
        if (src.year) {
          periodLabel = `FY${src.year}`;
          if (src.quarter) periodLabel += ` Q${src.quarter}`;
        }

        const content = (
          <div className="flex items-center gap-1.5 w-full">
            <span className={`inline-block rounded px-1 py-0.5 text-xxs font-mono font-semibold border ${colors.badge}`}>
              {src.type}
            </span>
            {src.ticker && (
              <span className="text-xxs font-mono font-semibold text-bb-gray-100">{src.ticker}</span>
            )}
            {periodLabel && (
              <span className="text-xxs font-mono text-bb-gray-400">{periodLabel}</span>
            )}
            {src.url && (
              <svg className="ml-auto h-3 w-3 flex-shrink-0 text-bb_blue-bright" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            )}
          </div>
        );

        if (src.url) {
          return (
            <a
              key={i}
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block py-1.5 px-2 rounded bb-row-hover no-underline hover:bg-bb-hover transition-colors"
            >
              {content}
            </a>
          );
        }
        return (
          <div key={i} className="py-1.5 px-2 rounded">
            {content}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Result Metadata Strip                                              */
/* ------------------------------------------------------------------ */
function ResultMetadataStrip() {
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className="font-semibold text-term-green tracking-wide">SEC Filing Intelligence System</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Animated counter component                                         */
/* ------------------------------------------------------------------ */
function AnimatedNumber({ value, duration = 1500 }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = value;
    const startTime = performance.now();
    const tick = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(Math.round(eased * end));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value, duration]);
  return <>{display.toLocaleString()}</>;
}

/* ------------------------------------------------------------------ */
/*  Landing page                                                       */
/* ------------------------------------------------------------------ */
/* Scroll-reveal hook — triggers once when element enters viewport */
function useScrollReveal(threshold = 0.15) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); obs.unobserve(el); } },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return [ref, visible];
}

/* Reusable scroll-reveal wrapper */
function Reveal({ children, delay = 0, className = "" }) {
  const [ref, visible] = useScrollReveal(0.12);
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

function LandingPage() {
  const { displayed, done } = useTypewriter("SEC Filing Intelligence Engine", 65);

  const STATS = [
    { value: 70, suffix: "+", label: "XBRL Metrics" },
    { value: 1, suffix: "M+", label: "Data Points" },
    { value: 10, suffix: "", label: "S&P 500 Companies" },
    { value: 16, suffix: "", label: "Years of Coverage" },
  ];

  const FEATURES = [
    {
      label: "Metric Lookup",
      desc: "70+ XBRL metrics with 3-layer concept resolution (validated tags, alias map, fuzzy DB search). Automatic statement fallback and Q4 derivation from annual filings.",
      icon: <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />,
    },
    {
      label: "Trend Analysis",
      desc: "Dedicated timeseries route tracks metrics across years with auto-year expansion for YoY comparison, quarterly + annual hybrid support, and recency-aware data selection.",
      icon: <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />,
    },
    {
      label: "Narrative Search",
      desc: "LLM-generated sub-queries searched via batch embeddings, cosine similarity (pgvector), and cross-encoder reranking over Risk Factors and MD&A from 10-K and 10-Q filings.",
      icon: <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />,
    },
    {
      label: "Multi-Company",
      desc: "LLM decomposes queries into per-ticker sub-queries, executed in parallel with fair-share chunk budgeting, data availability pre-checks, and IPO year validation.",
      icon: <path strokeLinecap="round" strokeLinejoin="round" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />,
    },
    {
      label: "Smart Routing",
      desc: "5-way route dispatch (metric lookup, timeseries, narrative, hybrid, full statement) with 12 query types, fiscal year auto-resolution, and dynamic retrieval depth.",
      icon: <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />,
    },
    {
      label: "Source Verification",
      desc: "Every answer links to the SEC filing on sec.gov, with 0\u2013100 confidence scoring across 5 signals, contradiction detection warnings, and data availability notices.",
      icon: <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />,
    },
  ];

  return (
    <div className="flex flex-col items-center justify-center py-10 text-center max-w-3xl mx-auto">
      {/* Animated title */}
      <h1 className="font-mono font-bold text-3xl text-term-green animate-glow-pulse tracking-wide h-10">
        {displayed}
        {!done && <span className="animate-blink">_</span>}
      </h1>

      {/* Subtitle */}
      <p className="mt-5 text-base font-mono text-bb-gray-300 leading-relaxed animate-fade-up" style={{ animationDelay: "2s" }}>
        AI-powered retrieval over official SEC EDGAR filings.
        Structured XBRL data and vector search across 10-K and 10-Q documents &mdash;
        answered in seconds with full source attribution.
      </p>

      {/* Animated stats bar */}
      <div className="mt-8 w-full grid grid-cols-4 gap-3 animate-fade-up" style={{ animationDelay: "2.5s" }}>
        {STATS.map((s) => (
          <div key={s.label} className="rounded border border-bb-border bg-bb-surface px-3 py-4 animate-count-border" style={{ animationDelay: "3s" }}>
            <div className="text-xl font-mono font-bold text-term-green text-glow-green tabular-nums">
              <AnimatedNumber value={s.value} duration={2000} />{s.suffix}
            </div>
            <div className="mt-1 text-sm font-mono text-bb-gray-400">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Feature cards */}
      <div className="mt-8 grid grid-cols-3 gap-2.5 w-full text-left animate-fade-up" style={{ animationDelay: "3s" }}>
        {FEATURES.map((c, i) => (
          <div
            key={c.label}
            className="group rounded border border-bb-border bg-bb-surface px-3 py-2.5 transition-all duration-300 hover:border-term-green/30 hover:bg-bb-hover"
            style={{ animationDelay: `${3.2 + i * 0.1}s` }}
          >
            <div className="flex items-center gap-1.5">
              <svg className="h-3.5 w-3.5 text-term-green opacity-70 group-hover:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                {c.icon}
              </svg>
              <span className="text-xs font-mono font-semibold text-term-green">{c.label}</span>
            </div>
            <p className="mt-1 text-xs font-mono text-bb-gray-400 leading-relaxed">{c.desc}</p>
          </div>
        ))}
      </div>

      {/* Fiscal year note */}
      <div className="mt-8 w-full flex items-center gap-2.5 rounded border border-amber/20 bg-amber-bg/30 px-4 py-3 animate-fade-up" style={{ animationDelay: "3.5s" }}>
        <svg className="h-4 w-4 flex-shrink-0 text-amber" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span className="text-sm font-mono text-amber/80 leading-relaxed">
          All dates reference fiscal years, which may differ from calendar years.
          For example, NVIDIA's FY2023 runs Feb 2022 &ndash; Jan 2023.
        </span>
      </div>

      {/* ── Architecture Deep Dive ── */}
      <div className="mt-16 w-full text-left">

        {/* Section divider */}
        <Reveal>
          <div className="flex items-center gap-3 mb-8">
            <div className="h-px flex-1 bg-bb-border" />
            <span className="text-xs font-mono font-semibold uppercase tracking-widest text-bb-gray-400">System Architecture</span>
            <div className="h-px flex-1 bg-bb-border" />
          </div>
        </Reveal>

        {/* Data ingestion pipeline */}
        <Reveal delay={100}>
          <div className="rounded border border-bb-border bg-bb-surface p-5 mb-5">
            <h3 className="text-base font-mono font-semibold text-term-green mb-3">Data Ingestion Pipeline</h3>
            <p className="text-sm font-mono text-bb-gray-300 leading-relaxed mb-4">
              SEC EDGAR filings are fetched, parsed, and stored across multiple specialized tables.
              A single EDGAR fetch per filing feeds all downstream processors to avoid duplicate requests.
            </p>

            {/* Ingestion flow: Fetch → fan-out → Chunk & Embed */}
            {(() => {
              const StepBox = ({ s, className: cx }) => (
                <div className={`rounded border border-term-green/30 bg-bb-panel px-3 py-2 text-center ${cx || ""}`}>
                  <div className={`text-xs font-mono font-bold ${s.color}`}>{s.step}</div>
                  <div className="text-[10px] font-mono text-bb-gray-500 mt-0.5">{s.detail}</div>
                </div>
              );
              const HArrow = ({ direction }) => (
                <svg className="h-3 w-4 text-term-green/50 flex-shrink-0" fill="none" viewBox="0 0 16 12" stroke="currentColor" strokeWidth={1.5}>
                  {direction === "left" ? (
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 6H3m0 0l3-3M3 6l3 3" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" d="M1 6h12m0 0l-3-3m3 3l-3 3" />
                  )}
                </svg>
              );
              const VArrow = () => (
                <svg className="h-4 w-3 text-term-green/50" fill="none" viewBox="0 0 12 16" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 1v12m0 0l-3-3m3 3l3-3" />
                </svg>
              );
              return (
                <div className="py-2">
                  {/* Row 1: Fetch → fan-out */}
                  <div className="flex items-center gap-0">
                    <div className="flex-1 min-w-0">
                      <StepBox s={{ step: "Fetch", detail: "SEC EDGAR filings", color: "text-term-green" }} className="flex-1" />
                    </div>
                    <HArrow direction="right" />
                    <div className="flex-[3] min-w-0 rounded border border-bb-border/30 bg-bb-panel/30 px-2 py-1.5">
                      <div className="text-[10px] font-mono text-bb-gray-500 text-center mb-1.5">parallel per filing</div>
                      <div className="grid grid-cols-3 gap-1.5">
                        {/* XBRL — goes straight to DB */}
                        <div className="rounded border border-term-green/30 bg-bb-panel px-2 py-1.5 text-center">
                          <div className="text-[10px] font-mono font-bold text-bb_blue-bright">Parse XBRL</div>
                          <div className="text-[9px] font-mono text-bb-gray-500 mt-0.5">Facts → PostgreSQL</div>
                        </div>
                        {/* Sections — feeds into Chunk & Embed */}
                        <div className="rounded border border-amber/30 bg-bb-panel px-2 py-1.5 text-center">
                          <div className="text-[10px] font-mono font-bold text-amber">Extract Sections</div>
                          <div className="text-[9px] font-mono text-bb-gray-500 mt-0.5">Risk Factors, MD&A</div>
                        </div>
                        {/* Statements — goes straight to DB */}
                        <div className="rounded border border-term-green/30 bg-bb-panel px-2 py-1.5 text-center">
                          <div className="text-[10px] font-mono font-bold text-bb_blue-bright">Statements</div>
                          <div className="text-[9px] font-mono text-bb-gray-500 mt-0.5">Markdown → PostgreSQL</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  {/* Down arrow from Extract Sections only */}
                  <div className="flex">
                    <div className="flex-1" />
                    <div className="flex-[3] flex justify-center">
                      <div className="flex flex-col items-center">
                        <VArrow />
                        <span className="text-[9px] font-mono text-bb-gray-500 -mt-1">only sections</span>
                      </div>
                    </div>
                  </div>
                  {/* Row 2: Chunk & Embed — only section text */}
                  <div className="flex">
                    <div className="flex-1" />
                    <div className="flex-[3] flex justify-center">
                      <StepBox s={{ step: "Chunk & Embed", detail: "Sliding window → OpenAI embeddings → pgvector", color: "text-amber" }} className="min-w-[220px]" />
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* Ingestion details: two-column */}
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="rounded border border-bb-border/50 bg-bb-panel px-4 py-3">
                <div className="text-xs font-mono font-semibold text-term-green mb-2 uppercase tracking-wider">Structured Data</div>
                <ul className="space-y-1.5">
                  {[
                    "XBRL facts parsed into annual_facts and quarterly_facts tables",
                    "Facts classified by period duration (70\u2013120d quarterly, 330\u2013420d annual)",
                    "Financial statements converted to markdown (Q4 derived automatically)",
                  ].map((text, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-bb_blue-bright/70 flex-shrink-0" />
                      <span className="text-xs font-mono text-bb-gray-400 leading-relaxed">{text}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="rounded border border-bb-border/50 bg-bb-panel px-4 py-3">
                <div className="text-xs font-mono font-semibold text-amber mb-2 uppercase tracking-wider">Vector Data</div>
                <ul className="space-y-1.5">
                  {[
                    "Section text chunked with sliding window (sentence-boundary snapping)",
                    "Subsection headings prepended to each chunk for context",
                    "Batch embedded via OpenAI text-embedding-3-small (1536 dims)",
                    "Stored in pgvector for cosine similarity search",
                  ].map((text, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber/70 flex-shrink-0" />
                      <span className="text-xs font-mono text-bb-gray-400 leading-relaxed">{text}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </Reveal>

        {/* RAG Pipeline */}
        <Reveal delay={150}>
          <div className="rounded border border-bb-border bg-bb-surface p-5 mb-5">
            <h3 className="text-base font-mono font-semibold text-term-green mb-3">RAG Pipeline</h3>
            <p className="text-sm font-mono text-bb-gray-300 leading-relaxed mb-4">
              Each query is classified, decomposed into sub-queries, routed through up to 5 retrieval
              strategies (XBRL lookup, timeseries, vector search, hybrid fusion, full statements), then
              validated with guardrails and contradiction detection before generating a grounded,
              citation-backed answer.
            </p>

            {/* Pipeline flow: 2 rows of 4, connected horizontally + vertically */}
            {(() => {
              const ROW1 = [
                { step: "Input", detail: "Query + validation", color: "text-term-green" },
                { step: "Classify", detail: "Route, tickers, years", color: "text-term-green" },
                { step: "Availability", detail: "FY mapping, IPO checks", color: "text-amber" },
                { step: "Decompose", detail: "Sub-query split", color: "text-term-green" },
              ];
              /* Row 2 is reversed: Retrieve is on the right (under Decompose), flows left to Output */
              const ROW2 = [
                { step: "Output", detail: "Citations, confidence", color: "text-term-green" },
                { step: "Generate", detail: "LLM + grounded context", color: "text-term-green" },
                { step: "Guardrails", detail: "Filter, contradiction", color: "text-amber" },
                { step: "Retrieve", detail: "5-route data fetch + rerank", color: "text-bb_blue-bright" },
              ];
              const HArrow = ({ direction }) => (
                <svg className="h-3 w-4 text-term-green/50 flex-shrink-0" fill="none" viewBox="0 0 16 12" stroke="currentColor" strokeWidth={1.5}>
                  {direction === "left" ? (
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 6H3m0 0l3-3M3 6l3 3" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" d="M1 6h12m0 0l-3-3m3 3l-3 3" />
                  )}
                </svg>
              );
              const VArrow = () => (
                <svg className="h-4 w-3 text-term-green/50" fill="none" viewBox="0 0 12 16" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 1v12m0 0l-3-3m3 3l3-3" />
                </svg>
              );
              const StepBox = ({ s }) => (
                <div className="rounded border border-term-green/30 bg-bb-panel px-3 py-2 text-center flex-1 min-w-0">
                  <div className={`text-xs font-mono font-bold ${s.color}`}>{s.step}</div>
                  <div className="text-[10px] font-mono text-bb-gray-500 mt-0.5">{s.detail}</div>
                </div>
              );
              return (
                <div className="py-2">
                  {/* Row 1: left to right */}
                  <div className="flex items-center gap-0">
                    {ROW1.map((s, i) => (
                      <div key={s.step} className="flex items-center flex-1 min-w-0">
                        <StepBox s={s} />
                        {i < ROW1.length - 1 && <HArrow direction="right" />}
                      </div>
                    ))}
                  </div>
                  {/* Down arrow: right-aligned under Decompose, pointing to Retrieve */}
                  <div className="flex justify-end pr-[calc(12.5%-6px)]">
                    <VArrow />
                  </div>
                  {/* Row 2: right to left (Retrieve → Guardrails → Generate → Output) */}
                  <div className="flex items-center gap-0">
                    {ROW2.map((s, i) => (
                      <div key={s.step} className="flex items-center flex-1 min-w-0">
                        {i > 0 && <HArrow direction="left" />}
                        <StepBox s={s} />
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Tier 2: Expanded sub-steps panel */}
            <div className="mt-4 grid grid-cols-2 gap-4">
              {/* Left column — Classification & Planning */}
              <div className="rounded border border-bb-border/50 bg-bb-panel px-4 py-3">
                <div className="text-xs font-mono font-semibold text-term-green mb-2 uppercase tracking-wider">Classification &amp; Planning</div>
                <ul className="space-y-1.5">
                  {[
                    "LLM intent detection (route, query type, tickers, years)",
                    "Sub-query generation (1\u20134 semantic search queries per question)",
                    "Fiscal year mapping (calendar year \u2192 fiscal year)",
                    "Data availability check (filing existence, IPO year validation)",
                    "Multi-company decomposition (per-ticker sub-query split)",
                  ].map((text, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-term-green/70 flex-shrink-0" />
                      <span className="text-xs font-mono text-bb-gray-400 leading-relaxed">{text}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Right column — Retrieval Pipeline */}
              <div className="rounded border border-bb-border/50 bg-bb-panel px-4 py-3">
                <div className="text-xs font-mono font-semibold text-bb_blue-bright mb-2 uppercase tracking-wider">Retrieval Pipeline</div>
                <ul className="space-y-1.5">
                  {[
                    "XBRL concept resolution (validated tags \u2192 alias map \u2192 fuzzy DB search)",
                    "5-way route dispatch (metric_lookup, timeseries, narrative, hybrid, full_statement)",
                    "Batch embedding of sub-queries (single OpenAI API call)",
                    "Cosine similarity search (pgvector) per ticker \u00d7 year \u00d7 sub-query",
                    "Cross-encoder reranking (MiniLM-L-6-v2) for precision",
                  ].map((text, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-bb_blue-bright/70 flex-shrink-0" />
                      <span className="text-xs font-mono text-bb-gray-400 leading-relaxed">{text}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </Reveal>

        {/* Data layer */}
        <Reveal delay={150}>
          <div className="rounded border border-bb-border bg-bb-surface p-5 mb-5">
            <h3 className="text-base font-mono font-semibold text-term-green mb-4">Data Layer</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="rounded border border-bb-border/50 bg-bb-panel px-4 py-3">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="h-4 w-4 text-bb_blue-bright" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                  <span className="text-sm font-mono font-semibold text-bb_blue-bright">XBRL Structured Data</span>
                </div>
                <p className="text-sm font-mono text-bb-gray-400 leading-relaxed">
                  Machine-readable financial facts filed directly with the SEC.
                  Revenue, net income, EPS, total assets, and 70+ concepts parsed
                  from 10-K and 10-Q filings into <span className="text-bb-gray-300">annual_facts</span> and <span className="text-bb-gray-300">quarterly_facts</span> tables.
                </p>
              </div>
              <div className="rounded border border-bb-border/50 bg-bb-panel px-4 py-3">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="h-4 w-4 text-amber" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                  </svg>
                  <span className="text-sm font-mono font-semibold text-amber">Vector Embeddings</span>
                </div>
                <p className="text-sm font-mono text-bb-gray-400 leading-relaxed">
                  Narrative sections (Risk Factors, MD&A) chunked and embedded
                  using OpenAI <span className="text-bb-gray-300">text-embedding-3-small</span> (1536 dims).
                  Stored in PostgreSQL with <span className="text-bb-gray-300">pgvector</span>, then reranked with a cross-encoder.
                </p>
              </div>
              <div className="rounded border border-bb-border/50 bg-bb-panel px-4 py-3">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="h-4 w-4 text-term-green" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 01-2.25 2.25M16.5 7.5V18a2.25 2.25 0 002.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 002.25 2.25h13.5" />
                  </svg>
                  <span className="text-sm font-mono font-semibold text-term-green">Financial Statements</span>
                </div>
                <p className="text-sm font-mono text-bb-gray-400 leading-relaxed">
                  Full income statements, balance sheets, and cash flow statements
                  from SEC EDGAR. Q4 values are auto-calculated by
                  subtracting Q1-Q3 from annual totals when not directly reported.
                </p>
              </div>
            </div>
          </div>
        </Reveal>

        {/* Retrieval routes */}
        <Reveal delay={150}>
          <div className="rounded border border-bb-border bg-bb-surface p-5 mb-5">
            <h3 className="text-base font-mono font-semibold text-term-green mb-3">Retrieval Routes</h3>
            <p className="text-sm font-mono text-bb-gray-300 leading-relaxed mb-4">
              The query classifier analyzes each question and selects the optimal retrieval strategy.
              Five routes cover the full spectrum from precise metric lookups to open-ended narrative analysis.
            </p>
            <div className="space-y-2.5">
              {[
                { route: "metric_lookup", color: "text-bb_blue-bright", border: "border-bb_blue/20",
                  label: "Metric Lookup", desc: "Exact XBRL fact retrieval for specific financial metrics. Queries annual_facts and quarterly_facts tables directly for highest precision." },
                { route: "timeseries", color: "text-term-green", border: "border-term-green/20",
                  label: "Timeseries", desc: "Multi-year trend retrieval with automatic YoY growth calculations, period-over-period changes, CAGR computation, and trend direction." },
                { route: "narrative", color: "text-amber", border: "border-amber/20",
                  label: "Narrative Search", desc: "Semantic vector search over embedded filing sections (MD&A, Risk Factors). Cosine similarity finds candidates, cross-encoder reranks for relevance." },
                { route: "hybrid", color: "text-bb-gray-200", border: "border-bb-gray-400/20",
                  label: "Hybrid Fusion", desc: "Combines XBRL structured data with narrative vector search. Numbers provide precision while narrative sections add context and qualitative insights." },
                { route: "full_statement", color: "text-bb-gray-300", border: "border-bb-gray-400/20",
                  label: "Full Statement", desc: "Retrieves complete financial statements (income statement, balance sheet, cash flow) for broad financial overview queries." },
              ].map((r, i) => (
                <Reveal key={r.route} delay={i * 80}>
                  <div className={`flex gap-4 items-start rounded border ${r.border} bg-bb-panel px-4 py-3`}>
                    <span className={`text-xs font-mono font-bold ${r.color} mt-0.5 w-32 flex-shrink-0`}>{r.label}</span>
                    <span className="text-sm font-mono text-bb-gray-400 leading-relaxed">{r.desc}</span>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </Reveal>

        {/* Quality assurance */}
        <Reveal delay={150}>
          <div className="rounded border border-bb-border bg-bb-surface p-5 mb-5">
            <h3 className="text-base font-mono font-semibold text-term-green mb-4">Quality Assurance</h3>
            <p className="text-sm font-mono text-bb-gray-300 leading-relaxed mb-4">
              Config-driven guardrails ensure every answer is grounded, validated, and scored
              before reaching the user. All thresholds and weights are tunable
              via <span className="text-bb-gray-200">guardrails.yaml</span>.
            </p>
            <div className="space-y-4">
              <Reveal delay={0}>
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="h-2 w-2 rounded-full bg-term-green" />
                    <span className="text-sm font-mono font-semibold text-bb-gray-200">Retrieval Guardrails</span>
                  </div>
                  <p className="text-sm font-mono text-bb-gray-400 leading-relaxed pl-4">
                    Every chunk passes similarity and rerank score thresholds before reaching the LLM.
                    Relational results are capped per-concept to prevent context overload, and
                    multi-ticker queries use fair-share budgeting so no single company dominates the context.
                    Zero-result queries return a clear "insufficient data" response instead of hallucinating.
                  </p>
                </div>
              </Reveal>
              <Reveal delay={80}>
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="h-2 w-2 rounded-full bg-amber" />
                    <span className="text-sm font-mono font-semibold text-bb-gray-200">Contradiction Detection</span>
                  </div>
                  <p className="text-sm font-mono text-bb-gray-400 leading-relaxed pl-4">
                    On hybrid queries, narrative claims are cross-checked against XBRL numbers across
                    7 watched financial concepts. Both directional mismatches (e.g. "revenue increased"
                    vs. actual decline) and magnitude mismatches (e.g. "grew 15%" vs. actual 8%) are
                    flagged with severity levels and surfaced alongside the answer.
                  </p>
                </div>
              </Reveal>
              <Reveal delay={160}>
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="h-2 w-2 rounded-full bg-bb_blue-bright" />
                    <span className="text-sm font-mono font-semibold text-bb-gray-200">Multi-Signal Confidence Scoring</span>
                  </div>
                  <p className="text-sm font-mono text-bb-gray-400 leading-relaxed pl-4">
                    Each answer receives a 0&ndash;100 confidence score from five weighted signals:
                    retrieval quality, source coverage, cross-source agreement, citation density,
                    and data recency. Route-specific weight profiles (relational, timeseries, narrative, hybrid)
                    ensure fair scoring across query types, mapped to four investor-facing tiers.
                  </p>
                </div>
              </Reveal>
              <Reveal delay={240}>
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="h-2 w-2 rounded-full bg-bb_red" />
                    <span className="text-sm font-mono font-semibold text-bb-gray-200">Scope &amp; Input Validation</span>
                  </div>
                  <p className="text-sm font-mono text-bb-gray-400 leading-relaxed pl-4">
                    Queries are validated for length and blocked prompt-injection patterns before classification.
                    Post-classification, scope checks catch unsupported tickers, out-of-range years, and
                    unembedded filing sections &mdash; returning actionable rejection messages with coverage boundaries.
                  </p>
                </div>
              </Reveal>
            </div>
          </div>
        </Reveal>

        {/* Tech stack */}
        <Reveal delay={150}>
          <div className="rounded border border-bb-border bg-bb-surface p-5">
            <h3 className="text-base font-mono font-semibold text-term-green mb-4">Tech Stack</h3>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "LLM", value: "GPT-4o" },
                { label: "Embeddings", value: "text-embedding-3-small" },
                { label: "Reranker", value: "MiniLM-L-6-v2" },
                { label: "Database", value: "PostgreSQL + pgvector" },
                { label: "Cache", value: "Redis (LRU)" },
                { label: "Data Source", value: "SEC EDGAR / XBRL" },
                { label: "Backend", value: "Python / FastAPI" },
                { label: "Frontend", value: "React" },
                { label: "Streaming", value: "Server-Sent Events" },
              ].map((t) => (
                <div key={t.label} className="rounded border border-bb-border/50 bg-bb-panel px-3 py-2.5 text-center">
                  <div className="text-xs font-mono text-bb-gray-500">{t.label}</div>
                  <div className="text-sm font-mono font-semibold text-bb-gray-200 mt-0.5">{t.value}</div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>

      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main App component                                                 */
/* ------------------------------------------------------------------ */
function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Streaming state
  const [classification, setClassification] = useState(null);
  const [planSteps, setPlanSteps] = useState([]);
  const [activeStep, setActiveStep] = useState(0);
  const [planComplete, setPlanComplete] = useState(false);
  const stepTimerRef = useRef(null);
  const [sessionCost, setSessionCost] = useState(0);

  // Clock
  const [currentTime, setCurrentTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Submit query using SSE streaming
  const handleSearch = useCallback(async (overrideQuery) => {
    const q = (typeof overrideQuery === "string" ? overrideQuery : query).trim();
    if (!q) return;

    setQuery(q);
    setLoading(true);
    setError(null);
    setResult(null);
    setClassification(null);
    setPlanSteps([]);
    setActiveStep(0);
    setPlanComplete(false);
    if (stepTimerRef.current) clearInterval(stepTimerRef.current);

    try {
      const res = await fetch(`${BACKEND_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}: ${res.statusText}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let stepsReceived = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const eventMatch = part.match(/^event:\s*(.+)/m);
          const dataMatch = part.match(/^data:\s*(.+)/m);
          if (!eventMatch || !dataMatch) continue;

          const eventType = eventMatch[1].trim();
          let eventData;
          try {
            eventData = JSON.parse(dataMatch[1]);
          } catch {
            continue;
          }

          if (eventType === "classification") {
            setClassification(eventData);
          }

          if (eventType === "retrieval_plan") {
            stepsReceived = eventData.steps || [];
            setPlanSteps(stepsReceived);
            setActiveStep(0);

            let step = 0;
            const intervalMs = Math.max(800, Math.min(2000, 8000 / Math.max(stepsReceived.length, 1)));
            stepTimerRef.current = setInterval(() => {
              step++;
              if (step < stepsReceived.length) {
                setActiveStep(step);
              } else {
                clearInterval(stepTimerRef.current);
              }
            }, intervalMs);
          }

          if (eventType === "result") {
            if (stepTimerRef.current) clearInterval(stepTimerRef.current);
            setPlanComplete(true);
            setActiveStep(stepsReceived.length);
            setResult(eventData);
            if (eventData.cost?.total_cost) {
              setSessionCost(prev => prev + eventData.cost.total_cost);
            }
            setLoading(false);
          }

          if (eventType === "error") {
            if (stepTimerRef.current) clearInterval(stepTimerRef.current);
            setError(eventData.error || "An unknown error occurred");
            setLoading(false);
          }
        }
      }
    } catch (err) {
      if (stepTimerRef.current) clearInterval(stepTimerRef.current);
      const isNetworkError = err.message === "Failed to fetch" || err.message === "Load failed";
      setError(
        isNetworkError
          ? "Unable to connect to backend. Ensure FastAPI server is running on " + BACKEND_URL
          : err.message
      );
      setLoading(false);
    }
  }, [query]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    if (q) handleSearch(q);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    return () => { if (stepTimerRef.current) clearInterval(stepTimerRef.current); };
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !loading) {
      e.preventDefault();
      handleSearch();
    }
  };

  const handleExample = (q) => {
    setQuery(q);
    handleSearch(q);
  };

  const cleanAnswer = (raw) => {
    if (!raw) return null;
    return raw
      .replace(/---\s*\n\*\*Sources:\*\*.*$/s, "")
      .replace(/\*\*Sources:\*\*.*$/s, "")
      .replace(/---\s*\n\*\*Confidence:.*$/s, "")
      .replace(/\*\*Confidence:.*$/s, "")
      .trimEnd();
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-bb-black text-bb-gray-200 font-sans">
      {/* ==================== HEADER BAR ==================== */}
      <header className="flex-shrink-0 h-10 flex items-center justify-between px-4 bg-bb-panel border-b border-bb-border">
        <div className="flex items-center gap-3">
          {classification && loading && (
            <span className="text-xxs font-mono text-amber animate-pulse">
              PROCESSING: {classification.route_name || classification.route}
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xxs font-mono text-bb-gray-400">
            {currentTime.toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" })}
          </span>
          <span className="text-xxs font-mono text-term-green tabular-nums">
            {currentTime.toLocaleTimeString("en-US", { hour12: false })}
          </span>
        </div>
      </header>

      {/* ==================== MAIN BODY ==================== */}
      <div className="flex flex-1 overflow-hidden">

        {/* ==================== LEFT SIDEBAR ==================== */}
        <aside className="hidden lg:flex flex-col w-64 flex-shrink-0 border-r border-bb-border bg-bb-panel overflow-y-auto">
          {/* Scope — Tickers */}
          <div className="p-3 border-b border-bb-border">
            <label className="text-xxs font-mono font-semibold uppercase tracking-widest text-bb-gray-400 mb-2 block">
              Coverage &mdash; Tickers
            </label>
            <div className="flex flex-wrap gap-1">
              {TICKERS.map((t) => (
                <span
                  key={t}
                  className="rounded bg-bb-surface border border-bb-border px-2 py-1 text-xs font-mono font-semibold text-term-green"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          {/* Scope — Years */}
          <div className="p-3 border-b border-bb-border">
            <label className="text-xxs font-mono font-semibold uppercase tracking-widest text-bb-gray-400 mb-1.5 block">
              Coverage &mdash; Years
            </label>
            <span className="text-xs font-mono text-bb-gray-200">2010 &ndash; 2026</span>
          </div>

          {/* Scope — Filing Types */}
          <div className="p-3 border-b border-bb-border">
            <label className="text-xxs font-mono font-semibold uppercase tracking-widest text-bb-gray-400 mb-2 block">
              Filing Types
            </label>
            <div className="flex gap-1.5">
              <span className="rounded bg-bb_blue-bg border border-bb_blue/30 px-2 py-1 text-xs font-mono font-semibold text-bb_blue-bright">
                10-K
              </span>
              <span className="rounded bg-term-bg border border-term-green/30 px-2 py-1 text-xs font-mono font-semibold text-term-green">
                10-Q
              </span>
            </div>
          </div>

          {/* Query Input */}
          <div className="p-3 border-b border-bb-border flex-shrink-0">
            <label className="text-xxs font-mono font-semibold uppercase tracking-widest text-bb-gray-400 mb-2 block">
              Query
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter query..."
              disabled={loading}
              rows={8}
              className="w-full rounded bg-bb-surface border border-bb-border px-2 py-1.5 text-xs font-mono text-bb-gray-100 placeholder-bb-gray-500 outline-none focus:border-term-green/50 resize-none"
            />
            <button
              onClick={() => handleSearch()}
              disabled={loading || !query.trim()}
              className="mt-2 w-full rounded bg-term-green/10 border border-term-green/30 px-4 py-3 text-sm font-mono font-bold text-term-green transition hover:bg-term-green/20 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {loading ? "EXECUTING..." : "EXECUTE QUERY"}
            </button>
          </div>

          {/* Example Queries */}
          <div className="p-3 flex-1 overflow-y-auto">
            <label className="text-xxs font-mono font-semibold uppercase tracking-widest text-bb-gray-400 mb-2 block">
              Examples
            </label>
            <div className="space-y-1">
              {EXAMPLE_QUERIES.map((eq) => (
                <button
                  key={eq}
                  onClick={() => handleExample(eq)}
                  disabled={loading}
                  className="w-full text-left rounded px-2 py-1.5 text-xxs font-mono text-bb-gray-300 bb-row-hover disabled:opacity-30"
                >
                  <span className="text-term-dim mr-1">&gt;</span>{eq}
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* ==================== MAIN CONTENT ==================== */}
        <main className="flex-1 overflow-y-auto">
          {/* Mobile query bar (when sidebars hidden) */}
          <div className="lg:hidden p-3 border-b border-bb-border bg-bb-panel">
            <div className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter query..."
                disabled={loading}
                className="flex-1 rounded bg-bb-surface border border-bb-border px-3 py-2 text-xs font-mono text-bb-gray-100 placeholder-bb-gray-500 outline-none focus:border-term-green/50"
              />
              <button
                onClick={() => handleSearch()}
                disabled={loading || !query.trim()}
                className="rounded bg-term-green/10 border border-term-green/30 px-4 py-2 text-xs font-mono font-bold text-term-green disabled:opacity-30"
              >
                {loading ? "..." : "GO"}
              </button>
            </div>
          </div>

          <div className="p-4 space-y-4">
            {/* Initial loading spinner */}
            {loading && !classification && <TerminalSpinner />}

            {/* Classification badge */}
            {classification && !result && (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded bg-bb_blue-bg px-2 py-1 text-xxs font-mono font-semibold text-bb_blue-bright border border-bb_blue/20">
                  {classification.route_name}
                </span>
                <span className="text-xxs font-mono text-bb-gray-400">{classification.reasoning}</span>
              </div>
            )}

            {/* Retrieval plan — during loading */}
            {planSteps.length > 0 && !result && (
              <RetrievalPlan steps={planSteps} activeStep={activeStep} completed={false} />
            )}

            {/* Skeleton loader — shown while waiting for answer */}
            {loading && classification && !result && <AnswerSkeleton />}

            {/* Error */}
            {error && (
              <div className="rounded border border-bb_red/30 bg-bb_red-bg px-4 py-3 text-xs font-mono text-bb_red">
                <span className="font-bold mr-2">ERROR:</span>{error}
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="space-y-4">
                {/* Metadata strip */}
                <ResultMetadataStrip result={result} />

                {/* Retrieval plan — collapsed */}
                {planSteps.length > 0 && (
                  <details className="group">
                    <summary className="flex cursor-pointer items-center gap-2 text-xxs font-mono text-bb-gray-400 transition hover:text-bb-gray-200">
                      <svg className="h-3 w-3 transition group-open:rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                      </svg>
                      RETRIEVAL PLAN ({planSteps.length} steps)
                    </summary>
                    <RetrievalPlan steps={planSteps} activeStep={planSteps.length} completed={true} />
                  </details>
                )}

                {/* Answer card */}
                <div className="bb-panel-inset rounded bg-bb-panel p-4">
                  <div className="flex items-center gap-2 mb-3 pb-2 border-b border-bb-border">
                    <div className="w-1 h-4 bg-term-green rounded-full" />
                    <h2 className="text-xs font-mono font-semibold uppercase tracking-widest text-amber">
                      Analysis
                    </h2>
                  </div>
                  <AnswerBlock answer={cleanAnswer(result.answer || (typeof result.data === "string" ? result.data : null))} sources={result.sources} />

                  {Array.isArray(result.data) && result.data.length > 0 && (
                    <div className="mt-4 overflow-x-auto rounded border border-bb-border">
                      <table className="w-full text-left text-xs font-mono">
                        <thead>
                          <tr className="border-b border-bb-border bg-bb-surface">
                            {Object.keys(result.data[0]).map((key) => (
                              <th key={key} className="px-3 py-2 text-xxs font-semibold uppercase tracking-wider text-amber">{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {result.data.map((row, i) => (
                            <tr key={i} className="border-b border-bb-border/50 bb-row-hover">
                              {Object.values(row).map((val, j) => (
                                <td key={j} className="px-3 py-1.5 text-bb-gray-200 tabular-nums">{val != null ? String(val) : "\u2014"}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Fiscal year note */}
                {result.sources && result.sources.length > 0 && (
                  <div className="flex items-start gap-2 rounded border border-bb-border bg-bb-surface px-3 py-2">
                    <svg className="h-3.5 w-3.5 flex-shrink-0 mt-0.5 text-bb-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-xxs font-mono text-bb-gray-500 leading-relaxed">
                      Sources link to official SEC EDGAR filings. Dates shown are <span className="text-bb-gray-300">fiscal years</span>, which
                      may differ from calendar years depending on the company's fiscal year-end
                      (e.g., NVIDIA FY ends January, Apple FY ends September).
                    </span>
                  </div>
                )}

                {/* Confidence + Cost breakdown — side by side */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <ConfidenceBreakdown confidence={result.confidence} />
                  <CostBreakdown cost={result.cost} />
                </div>

              </div>
            )}

            {/* Empty state — RAG application overview */}
            {!loading && !result && !error && !classification && (
              <LandingPage />
            )}
          </div>
        </main>

        {/* ==================== RIGHT SIDEBAR ==================== */}
        <aside className="hidden lg:flex flex-col w-72 flex-shrink-0 border-l border-bb-border bg-bb-panel overflow-y-auto">
          <div className="p-3 border-b border-bb-border">
            <div className="flex items-center justify-between">
              <span className="text-xxs font-mono font-semibold uppercase tracking-widest text-bb-gray-400">
                Sources{result?.sources?.length > 0 ? ` (${result.sources.length})` : ""}
              </span>
              <span className="flex items-center gap-1 rounded bg-bb_blue-bg/50 border border-bb_blue/20 px-1.5 py-0.5">
                <svg className="h-2.5 w-2.5 text-bb_blue-bright" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
                <span className="text-xxs font-mono font-semibold text-bb_blue-bright">SEC Official</span>
              </span>
            </div>
          </div>
          <div className="flex-1 p-2 overflow-y-auto">
            <DocumentTimeline sources={result?.sources} />
          </div>

        </aside>
      </div>

      {/* ==================== FOOTER STATUS BAR ==================== */}
      <footer className="flex-shrink-0 h-7 flex items-center justify-between px-4 bg-bb-panel border-t border-bb-border text-xxs font-mono">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-term-green">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-term-green" />
            Connected
          </span>
          {result?.route && (
            <span className="text-bb-gray-400">
              Route: <span className="text-bb-gray-200">{result.route_name || ROUTE_LABELS[result.route] || result.route}</span>
            </span>
          )}
          {result?.response_time != null && (
            <span className="text-bb-gray-400">
              Latency: <span className="text-bb-gray-200 tabular-nums">{result.response_time.toFixed(2)}s</span>
            </span>
          )}
          {result?.cache_hit && (
            <span className="text-term-green font-semibold flex items-center gap-1">
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
              </svg>
              Cached
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {sessionCost > 0 && (
            <span className="text-bb-gray-400">
              Session:{" "}
              <span className="text-amber tabular-nums font-semibold">
                {sessionCost < 1
                  ? `${(sessionCost * 100).toFixed(3)}¢`
                  : `$${sessionCost.toFixed(2)}`}
              </span>
            </span>
          )}
        </div>
      </footer>
    </div>
  );
}

export default App;
