# ----------------------------------------------------------------------
#  world_bank_fetcher.py
#  (compatible with every released version of wbgapi)
# ----------------------------------------------------------------------
import time
import logging
from datetime import datetime
import pandas as pd
import wbgapi as wb
from requests.exceptions import RequestException
import requests

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1️⃣  Obtain a `requests.Session` that wbgapi actually uses
# ----------------------------------------------------------------------
def _get_wbgapi_session() -> requests.Session:
    """Return the Session object that wbgapi uses for HTTP calls."""
    if hasattr(wb, "session"):
        return wb.session
    if hasattr(wb, "api") and hasattr(wb.api, "session"):
        return wb.api.session
    if hasattr(wb, "session_obj"):
        return wb.session_obj

    # fallback – create our own Session
    log.debug("wbgapi did not expose a Session; creating a fresh one.")
    my_session = requests.Session()
    try:
        wb.options.session = my_session      # official public API
    except Exception:                       # pragma: no cover
        pass
    return my_session


_SESSION = _get_wbgapi_session()


# ----------------------------------------------------------------------
# 2️⃣  Global configuration – runs only once
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
    except Exception:                         # pragma: no cover
        log.debug("urllib3 not installed – retry adapter not added.")


_configure_wbgapi()


# ----------------------------------------------------------------------
# 3️⃣  Public fetcher – now uses the *Commodity Markets* database (db = 14)
# ----------------------------------------------------------------------
def fetch_worldbank_commodities(pink_sheet: bool = True) -> pd.DataFrame:
    """
    Pull the six commodity price series (Copper, Aluminum, …) from the
    **World Bank – Commodity Markets** database (source = 14).

    The historic name “Pink Sheet” is misleading – the price series you need
    are stored in source = 14, not source = 2.  Setting ``pink_sheet=True``
    therefore points to db = 14.

    Parameters
    ----------
    pink_sheet : bool, default=True
        When True we query **database 14** (Commodity Markets).  When False
        the default database (no explicit db) is used – this will very likely
        return no data for the six series.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``material``, ``price``, ``source``.
        Empty DataFrame if every request fails.
    """
    # --------------------------------------------------------------
    # 1️⃣  Indicator → material mapping (exact series IDs from WB)
    # --------------------------------------------------------------
    indicators = {
        "PCOPP": "copper",
        "PALUM": "aluminum",
        "PNICK": "nickel",
        "PZINC": "zinc",
        "PLEAD": "lead",
        "PTIN": "tin",
    }

    # --------------------------------------------------------------
    # 2️⃣  Build the list of years (plain numbers)
    # --------------------------------------------------------------
    cur_year = datetime.now().year
    years = list(range(cur_year - 5, cur_year + 1))   # e.g. 2020‑2025
    years_str = ";".join(str(y) for y in years)      # "2020;2021;…;2025"

    # --------------------------------------------------------------
    # 3️⃣  Choose the correct database id
    # --------------------------------------------------------------
    # Commodity Markets (the series we need) == source 14
    db_id = 14 if pink_sheet else None

    # --------------------------------------------------------------
    # 4️⃣  Loop over each indicator separately
    # --------------------------------------------------------------
    all_rows = []

    for indicator, material in indicators.items():
        log.info(f"Fetching {material} ({indicator}) …")

        for attempt in range(3):
            try:
                raw = wb.data.fetch(
                    indicator,
                    "WLD",
                    db=db_id,          # <- 14 for the price series
                    time=years_str,
                )
                break
            except RequestException as exc:
                log.warning(
                    f"World Bank attempt {attempt + 1}/3 for {material} failed "
                    f"({type(exc).__name__}): {exc}"
                )
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    raw = []
                    log.error(
                        f"❌ All retries exhausted for {material} ({indicator})"
                    )
                    break

        for rec in raw:
            if rec.get("value") is None:
                continue
            try:
                dt = pd.Timestamp(f"{rec['time']}-01-01")
            except Exception:                     # pragma: no cover
                continue

            all_rows.append(
                {
                    "date": dt,
                    "material": material,
                    "price": float(rec["value"]),
                    "source": "worldbank-commodity-markets",
                }
            )

    # --------------------------------------------------------------
    # 5️⃣  Return result (or empty DF)
    # --------------------------------------------------------------
    if not all_rows:
        log.info(
            "⚠️ World Bank returned no usable records for the commodity series."
        )
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    log.info(
        f"✅ World Bank provided {len(df)} rows across "
        f"{df['material'].nunique()} materials."
    )
    return df