# ----------------------------------------------------------------------
#  world_bank_fetcher.py
#  (Pink Sheet Commodity API implementation - Fixed)
# ----------------------------------------------------------------------
import time
import logging
from datetime import datetime
import pandas as pd
import wbgapi as wb
import requests
from requests.exceptions import RequestException
import io

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1️⃣  Obtain a `requests.Session` that wbgapi normally uses
# ----------------------------------------------------------------------
def _get_wbgapi_session() -> requests.Session:
    """
    Return the Session object that wbgapi uses (or create our own if the
    library does not expose one).
    """
    if hasattr(wb, "session"):
        return wb.session
    if hasattr(wb, "api") and hasattr(wb.api, "session"):
        return wb.api.session
    if hasattr(wb, "session_obj"):
        return wb.session_obj

    log.debug("wbgapi did not expose a Session; creating a fresh one.")
    sess = requests.Session()
    try:
        wb.options.session = sess
    except Exception:
        pass
    return sess


_SESSION = _get_wbgapi_session()


# ----------------------------------------------------------------------
# 2️⃣  Global configuration – runs once at import time
# ----------------------------------------------------------------------
def _configure_wbgapi() -> None:
    """Add a friendly User‑Agent and a urllib3‑Retry adapter."""
    _SESSION.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; WorldBankDataFetcher/1.0; "
                "+https://github.com/yourorg/yourrepo)"
            )
        }
    )

    try:
        from urllib3.util import Retry
        from requests.adapters import HTTPAdapter

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
            raise_on_redirect=False,
        )
        _SESSION.mount("https://", HTTPAdapter(max_retries=retry))
    except Exception:
        log.debug("urllib3 not installed – retry adapter not added.")


_configure_wbgapi()


# ----------------------------------------------------------------------
# 3️⃣  Public fetcher – Pink Sheet Commodity Price Data
# ----------------------------------------------------------------------
def fetch_worldbank_commodities(pink_sheet: bool = True) -> pd.DataFrame:
    """
    Pull commodity price series from the World Bank Pink Sheet (monthly historical data).

    Downloads the official CMO-Historical-Data-Monthly.xlsx file and extracts
    copper, aluminum, nickel, zinc, lead, and tin prices.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``material``, ``price``, ``source``.
        Empty DataFrame if fetch fails.
    """
    if not pink_sheet:
        log.warning(
            "pink_sheet=False – World Bank commodity fetch disabled; returning empty DataFrame."
        )
        return pd.DataFrame()

    # The official World Bank Pink Sheet historical data URL
    url = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"

    log.info("Fetching World Bank Pink Sheet data (monthly historical Excel file)...")

    try:
        resp = _SESSION.get(url, timeout=30)
        log.info(f"World Bank Pink Sheet GET → status={resp.status_code}")

        if resp.status_code != 200:
            log.error(f"Failed to fetch Pink Sheet: status={resp.status_code}")
            return pd.DataFrame()

        # Read the Excel file from memory
        excel_file = io.BytesIO(resp.content)

        # Read the "Monthly Prices" sheet with proper header detection
        sheet_name = 'Monthly Prices'

        # First, let's examine the structure by reading without headers
        df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        log.info(f"📊 Raw data shape: {df_raw.shape}")

        # Find the actual header row - it contains the commodity names
        header_row_idx = None
        for idx in range(min(10, len(df_raw))):
            row_vals = df_raw.iloc[idx].dropna()
            if len(row_vals) > 5:  # Header row has many non-null values
                # Check if this row contains commodity names
                if any('copper' in str(val).lower() or 'crude' in str(val).lower() or 'aluminum' in str(val).lower() for val in row_vals):
                    header_row_idx = idx
                    log.info(f"📊 Found header row at index {idx}: {list(row_vals[:5])}")
                    break

        if header_row_idx is None:
            # If we can't find by content, use row 4 (0-indexed) as it's typically the header
            header_row_idx = 4
            log.info(f"📊 Using default header row at index {header_row_idx}")

        # Read with the proper header row
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row_idx)

        # Clean up column names - the first column should be the date
        df.columns = [str(col).strip() for col in df.columns]

        log.info(f"📊 After setting header at row {header_row_idx}, shape: {df.shape}")
        log.info(f"📊 Column names (first 10): {df.columns[:10].tolist()}")

        # Find the date column - it should be the first column or contain 'date'/'month'
        date_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'date' in col_lower or 'month' in col_lower or col_lower == 'nan' or '1960' in col_lower:
                date_col = col
                break

        if date_col is None:
            date_col = df.columns[0]  # Use first column as default

        log.info(f"📊 Using date column: '{date_col}'")

        # Map Pink Sheet column names to our standardized material names
        # Based on the actual column names from your file
        commodity_column_map = {
            'Copper': 'copper',
            'COPPER': 'copper',
            'Copper ': 'copper',  # with space
            'ALUMINUM': 'aluminum',
            'Aluminum': 'aluminum',
            'Aluminium': 'aluminum',
            'NICKEL': 'nickel',
            'Nickel': 'nickel',
            'ZINC': 'zinc',
            'Zinc': 'zinc',
            'LEAD': 'lead',
            'Lead': 'lead',
            'Tin': 'tin',
            'TIN': 'tin',
            'Tin ': 'tin',  # with space
        }

        # Also check for partial matches
        commodity_keywords = {
            'copper': 'copper',
            'aluminum': 'aluminum',
            'aluminium': 'aluminum',
            'nickel': 'nickel',
            'zinc': 'zinc',
            'lead': 'lead',
            'tin': 'tin'
        }

        rows = []

        # Filter to last 5 years
        current_year = datetime.now().year
        start_date = pd.Timestamp(f"{current_year - 5}-01-01")

        # Process each column to find commodity data
        for col in df.columns:
            if col == date_col:
                continue

            col_str = str(col).strip()
            material = None

            # Try exact match first
            if col_str in commodity_column_map:
                material = commodity_column_map[col_str]
            else:
                # Try partial match (case-insensitive)
                col_lower = col_str.lower()
                for key, value in commodity_keywords.items():
                    if key in col_lower:
                        material = value
                        break

            if material is None:
                continue

            log.info(f"📊 Found commodity column: '{col}' → {material}")

            # Extract data for this commodity
            for idx, row in df.iterrows():
                try:
                    date_val = row[date_col]
                    price_val = row[col]

                    # Skip if price is None/NaN or contains placeholder
                    if pd.isna(price_val) or str(price_val).strip() in ['', '…', '...']:
                        continue

                    # Parse date - handle different date formats
                    if isinstance(date_val, pd.Timestamp):
                        dt = date_val
                    elif isinstance(date_val, str):
                        # Handle formats like "1960M01", "1960-01", etc.
                        if 'M' in date_val:
                            # Format: "1960M01"
                            year = date_val.split('M')[0]
                            month = date_val.split('M')[1]
                            dt = pd.to_datetime(f"{year}-{month}-01")
                        else:
                            dt = pd.to_datetime(date_val)
                    elif pd.isna(date_val):
                        continue
                    else:
                        # Try to convert whatever it is
                        dt = pd.to_datetime(date_val)

                    # Filter to last 5 years
                    if dt < start_date:
                        continue

                    # Convert price to float, handling any string formatting
                    try:
                        price_float = float(price_val)
                        rows.append({
                            'date': dt,
                            'material': material,
                            'price': price_float,
                            'source': 'worldbank-pinksheet'
                        })
                    except (ValueError, TypeError):
                        continue

                except Exception as e:
                    # Skip problematic rows
                    continue

        if not rows:
            log.warning("⚠️ No commodity data found in Pink Sheet")
            # Let's debug what columns we actually have
            all_cols = [str(col).strip() for col in df.columns]
            log.info(f"📊 All available columns: {all_cols}")
            return pd.DataFrame()

        result_df = pd.DataFrame(rows)
        log.info(
            f"✅ World Bank Pink Sheet provided {len(result_df)} rows across "
            f"{result_df['material'].nunique()} materials: {sorted(result_df['material'].unique())}"
        )

        # Show sample of data
        if len(result_df) > 0:
            log.info(f"📊 Sample data:\n{result_df.head(10)}")

        return result_df

    except Exception as e:
        log.error(f"Failed to fetch/parse Pink Sheet: {type(e).__name__}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return pd.DataFrame()