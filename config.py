"""
Shared configuration for SEC RAG System.

Central location for tickers, years, fiscal year mappings, and rate limiting.
"""

# Top 10 S&P 500 companies by market cap
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM"]

# Years to cover: 2010 through present (inclusive of 2027 for Q1 2026+ filings)
YEARS = range(2010, 2028)

# SEC EDGAR rate limit delay (seconds between API calls)
SEC_RATE_LIMIT_DELAY = 0.15

# Fiscal year-end month for each ticker
# Most companies end in December (12), but some have non-calendar fiscal years
FISCAL_YEAR_END_MONTH = {
    "AAPL": 9,    # September (e.g., FY2024 ends Sep 2024)
    "MSFT": 6,    # June (e.g., FY2024 ends Jun 2024)
    "NVDA": 1,    # January (e.g., FY2025 ends Jan 2025)
    "AMZN": 12,   # December
    "GOOGL": 12,  # December
    "META": 12,   # December
    "BRK-B": 12,  # December
    "LLY": 12,    # December
    "AVGO": 10,   # October (e.g., FY2024 ends Oct/Nov 2024)
    "JPM": 12,    # December
}


def _parse_date(value):
    """Convert a date value to a date object if it's a string."""
    from datetime import date, datetime
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def infer_fiscal_year(ticker: str, filing) -> int:
    """Infer the fiscal year from a filing.

    Uses period_of_report (preferred) which reflects the period the filing
    covers, falling back to filing_date.year.

    For companies with non-calendar fiscal years (e.g., AAPL ends Sep,
    MSFT ends Jun), the period_of_report is the most reliable indicator.
    """
    # Prefer period_of_report — it reflects the actual reporting period
    period = _parse_date(getattr(filing, 'period_of_report', None))
    if period is not None:
        fy_end_month = FISCAL_YEAR_END_MONTH.get(ticker, 12)
        if fy_end_month == 12:
            return period.year
        # For non-calendar FY: if the period month is after the FY-end month,
        # it belongs to the NEXT fiscal year
        if period.month > fy_end_month:
            return period.year + 1
        return period.year

    # Fallback to filing_date
    filing_date = _parse_date(getattr(filing, 'filing_date', None))
    if filing_date:
        return filing_date.year
    return filing.filing_date.year


def infer_fiscal_quarter(ticker: str, filing) -> int | None:
    """Infer the fiscal quarter from a filing.

    Derives the quarter from period_of_report relative to the company's
    fiscal year-end month. Returns None for annual (10-K) filings.
    """
    form = getattr(filing, 'form', '')
    if form == '10-K':
        return None

    period = _parse_date(getattr(filing, 'period_of_report', None))
    if period is None:
        # Fallback: calendar quarter from filing date
        filing_date = _parse_date(getattr(filing, 'filing_date', None))
        if filing_date:
            return (filing_date.month - 1) // 3 + 1
        return (filing.filing_date.month - 1) // 3 + 1

    fy_end_month = FISCAL_YEAR_END_MONTH.get(ticker, 12)

    # Calculate months since fiscal year start
    # FY starts the month after fy_end_month
    fy_start_month = (fy_end_month % 12) + 1
    month = period.month

    # Handle 52/53-week fiscal calendars (e.g., AAPL, AVGO) where period
    # end dates can spill 1-7 days into the next month.
    # If the day is in the first week of the month, treat it as the prior month.
    if period.day <= 7:
        # Roll back to previous month
        month = 12 if month == 1 else month - 1

    # Calculate offset from fiscal year start
    if month >= fy_start_month:
        months_into_fy = month - fy_start_month
    else:
        months_into_fy = (12 - fy_start_month) + month

    # Quarter = which 3-month bucket (0-based // 3 + 1)
    quarter = (months_into_fy // 3) + 1

    # Clamp to 1-4
    return max(1, min(4, quarter))


def calendar_to_fiscal_year(ticker: str, calendar_year: int) -> int:
    """Map a calendar year to the fiscal year used in the database.

    For companies whose fiscal year ends before December, the 10-K covering
    calendar year activity is stored under fiscal_year = calendar_year + 1.
    E.g. AVGO (FY ends Oct): calendar 2013 → FY2014 in DB.
         NVDA (FY ends Jan): calendar 2013 → FY2014 in DB.
         MSFT (FY ends Jun): calendar 2013 → FY2014 in DB.
         AAPL (FY ends Sep): calendar 2013 → FY2014 in DB.
    For calendar-year companies (Dec): calendar 2013 → FY2013.
    """
    fy_end_month = FISCAL_YEAR_END_MONTH.get(ticker, 12)
    if fy_end_month == 12:
        return calendar_year
    return calendar_year + 1
