"""
Fetch Apple's 10-K and 10-Q financial statements (2023-2025)
and save them as markdown files.
"""

from edgar import Company, set_identity
from pathlib import Path
from datetime import datetime

# Set identity for SEC Edgar API
set_identity("User user@example.com")

# Base directory for output
BASE_DIR = Path("/Users/subalbhattarai/Downloads/sec_rag_system/apple_financials")


def save_statement_as_markdown(statement, filepath, form_type, year, quarter=None):
    """Save a financial statement as a markdown file using built-in formatting."""
    try:
        if statement is None:
            print(f"  Warning: No statement data")
            return False

        # Use the built-in render().to_markdown() method
        rendered = statement.render()
        md_content = rendered.to_markdown()

        # Add metadata footer
        period_str = f"{year}" + (f" Q{quarter}" if quarter else "")
        md_content += f"\n\n---\n*Source: {form_type} {period_str} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        # Save to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(md_content)

        print(f"  Saved: {filepath}")
        return True
    except Exception as e:
        print(f"  Error saving: {e}")
        return False


def process_filing(filing, form_type, year, quarter=None):
    """Process a single filing and extract financial statements."""
    try:
        print(f"\nProcessing {form_type} for {year}" + (f" Q{quarter}" if quarter else ""))

        xbrl = filing.xbrl()
        if xbrl is None:
            print(f"  No XBRL data available")
            return

        financials = xbrl.statements

        # Determine folder and filename
        folder = BASE_DIR / form_type
        prefix = f"{year}" + (f"_Q{quarter}" if quarter else "")

        # Extract and save each statement
        statements = [
            ("balance_sheet", financials.balance_sheet),
            ("income_statement", financials.income_statement),
            ("cash_flow", financials.cashflow_statement),
        ]

        for stmt_folder, stmt_func in statements:
            try:
                statement = stmt_func()
                filepath = folder / stmt_folder / f"{prefix}.md"
                save_statement_as_markdown(statement, filepath, form_type, year, quarter)
            except Exception as e:
                print(f"  Error extracting {stmt_folder}: {e}")

    except Exception as e:
        print(f"  Error processing filing: {e}")


def main():
    print("Fetching Apple (AAPL) financial data from SEC Edgar...")

    company = Company("AAPL")
    print(f"Company: {company.name}")

    # Get 10-K filings (annual reports)
    print("\n" + "=" * 60)
    print("Fetching 10-K (Annual) filings...")
    print("=" * 60)

    for filing in company.get_filings(form="10-K", amendments=False):
        year = filing.filing_date.year
        if 2023 <= year <= 2025:
            process_filing(filing, "10-K", year)

    # Get 10-Q filings (quarterly reports)
    print("\n" + "=" * 60)
    print("Fetching 10-Q (Quarterly) filings...")
    print("=" * 60)

    for filing in company.get_filings(form="10-Q", amendments=False):
        year = filing.filing_date.year
        if 2023 <= year <= 2025:
            month = filing.filing_date.month
            # Apple's fiscal quarters: Q1 (Jan-Feb), Q2 (Apr-May), Q3 (Jul-Aug)
            quarter_map = {1: 1, 2: 1, 4: 2, 5: 2, 7: 3, 8: 3}
            quarter = quarter_map.get(month)
            if quarter is None:
                print(f"  Warning: Unexpected filing month {month}, skipping")
                continue
            process_filing(filing, "10-Q", year, quarter)

    print("\n" + "=" * 60)
    print(f"Completed! Files saved to: {BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
