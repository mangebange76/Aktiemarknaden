import os
import time
from typing import List, Dict, Optional, Tuple

import requests

FMP_BASE = "https://financialmodelingprep.com/api/v3"

# --- Hämta API-nyckeln säkert: st.secrets först, annars env ---
try:
    import streamlit as st  # kan saknas i vissa sammanhang → därför try/except
    _ST_FMP = st.secrets.get("FMP_API_KEY", None)
except Exception:
    _ST_FMP = None

API_KEY = _ST_FMP or os.environ.get("FMP_API_KEY")


def _get(path: str, params: Dict = None, timeout: int = 25):
    """
    Bas-GEt mot FMP med tydlig felhantering och maskering av apikey i feltext.
    """
    if params is None:
        params = {}

    key = API_KEY or params.get("apikey")
    if not key:
        raise RuntimeError(
            "FMP_API_KEY saknas. Lägg till i Streamlit Secrets som FMP_API_KEY (utanför [GOOGLE_CREDENTIALS])."
        )

    params["apikey"] = key
    url = f"{FMP_BASE}/{path}"
    r = requests.get(url, params=params, timeout=timeout)

    # Enkel backoff vid 429 (rate limit)
    if r.status_code == 429:
        time.sleep(1.2)
        r = requests.get(url, params=params, timeout=timeout)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Lägg med en kort snippet men maska bort nyckeln
        snippet = (r.text or "")[:300].replace(key, "***")
        raise requests.HTTPError(f"FMP HTTP {r.status_code} på {path}. Svar: {snippet}") from e

    # FMP kan svara med text även vid 200 i vissa edgefall; försök json först
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}


# ----------------------------- Universe/Screener -----------------------------
def list_universe(exchange: Optional[str] = None,
                  country: Optional[str] = None,
                  sector: Optional[str] = None,
                  industry: Optional[str] = None,
                  limit: int = 100000) -> List[Dict]:
    """
    Global screener. Använd exchange ('NASDAQ','NYSE','STO','OSE','CPH','HEL','FWB','LSE','TSX','TSXV'…)
    eller country ('United States','Sweden','Norway','Denmark','Finland','Germany','United Kingdom','Canada', 'Europe').
    """
    params = {"limit": limit}
    if exchange: params["exchange"] = exchange
    if country:  params["country"]  = country
    if sector:   params["sector"]   = sector
    if industry: params["industry"] = industry
    data = _get("stock-screener", params=params)
    # FMP returnerar ibland {"Error Message": "..."} – fånga det
    if isinstance(data, dict) and "Error Message" in data:
        raise requests.HTTPError(f"FMP fel: {data['Error Message']}")
    return [d for d in data if isinstance(d, dict) and d.get("symbol")]


def sectors_in_universe(exchange: Optional[str] = None, country: Optional[str] = None) -> List[str]:
    rows = list_universe(exchange=exchange, country=country, limit=50000)
    return sorted({r.get("sector") for r in rows if r.get("sector")})


def industries_in_universe(exchange: Optional[str] = None, country: Optional[str] = None, sector: Optional[str] = None) -> List[str]:
    rows = list_universe(exchange=exchange, country=country, sector=sector, limit=50000)
    return sorted({r.get("industry") for r in rows if r.get("industry")})


# ----------------------------- Quotes/Financials -----------------------------
def quote_batch(symbols: List[str]) -> List[Dict]:
    if not symbols:
        return []
    out = []
    CHUNK = 150
    for i in range(0, len(symbols), CHUNK):
        chunk = ",".join(symbols[i:i+CHUNK])
        data = _get(f"quote/{chunk}")
        if isinstance(data, dict) and "Error Message" in data:
            raise requests.HTTPError(f"FMP fel: {data['Error Message']}")
        out.extend(data if isinstance(data, list) else [])
        time.sleep(0.35)
    return out


def financial_ratios_ttm(symbol: str) -> Dict:
    fr = _get(f"ratios-ttm/{symbol}")
    return fr[0] if isinstance(fr, list) and fr else {}


def key_metrics_ttm(symbol: str) -> Dict:
    km = _get(f"key-metrics-ttm/{symbol}")
    return km[0] if isinstance(km, list) and km else {}


def historical_key_metrics(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict]:
    return _get(f"historical-key-metrics/{symbol}", params={"period": period, "limit": limit}) or []


def income_statement(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict]:
    return _get(f"income-statement/{symbol}", params={"period": period, "limit": limit}) or []


def cashflow_statement(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict]:
    return _get(f"cash-flow-statement/{symbol}", params={"period": period, "limit": limit}) or []


def batch_safe(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i+chunk_size]


# ----------------------------- Index presets -----------------------------
def _index_api_symbols(path: Optional[str]) -> List[str]:
    if not path:
        return []
    try:
        data = _get(path)
        syms = []
        for r in data:
            s = r.get("symbol") or r.get("Symbol") or r.get("ticker")
            if s:
                syms.append(s)
        return syms
    except Exception:
        return []


def index_symbols(index_code: str) -> List[str]:
    """
    API-först. Fallback: använd exchange/land som proxy om listan saknas.
    Täcker USA, Sverige, Norge, Danmark, Finland, Kanada, Tyskland, UK.
    """
    code = index_code.upper()
    path_map = {
        # USA
        "SP500": "sp500_constituent",
        "NDX":   "nasdaq_constituent",     # Nasdaq-100
        "DOW":   "dowjones_constituent",
        # Sverige/Europa
        "OMXS30":   "omxs30_constituent",
        "OMXSB":    "omx_stockholm_benchmark_constituent",
        "OMXS30GI": "omxs30_gi_constituent",
        "STOXX600": "stoxx600_constituent",
        "STOXX50E": "stoxx50e_constituent",
        # Övriga (fallback via exchange)
        "DAX": None, "MDAX": None, "SDAX": None, "TECDAX": None,
        "FTSE100": None, "FTSE250": None,
        "TSXCOMP": None, "TSXV": None,
        "OBX": None, "OSEBX": None,
        "OMXC25": None, "OMXH25": None,
    }
    syms = _index_api_symbols(path_map.get(code))
    if syms:
        return syms

    # Fallback – proxy via exchange
    rows: List[Dict] = []
    if code in ("DAX","MDAX","SDAX","TECDAX"):
        rows = list_universe(exchange="FWB", limit=50000)
    elif code in ("FTSE100","FTSE250"):
        rows = list_universe(exchange="LSE", limit=50000)
    elif code == "TSXCOMP":
        rows = list_universe(exchange="TSX", limit=50000)
    elif code == "TSXV":
        rows = list_universe(exchange="TSXV", limit=50000)
    elif code in ("OBX","OSEBX"):
        rows = list_universe(exchange="OSE", limit=50000)
    elif code == "OMXC25":
        rows = list_universe(exchange="CPH", limit=50000)
    elif code == "OMXH25":
        rows = list_universe(exchange="HEL", limit=50000)
    elif code in ("OMXS30","OMXSB","OMXS30GI"):
        rows = list_universe(exchange="STO", limit=50000)
    elif code in ("SP500","NDX","DOW"):
        rows = list_universe(country="United States", limit=50000)
    elif code in ("STOXX600","STOXX50E"):
        rows = list_universe(country="Europe", limit=50000)
    return [r["symbol"] for r in rows if r.get("symbol")]


# ----------------------------- Hjälp: testa nyckeln -----------------------------
def test_fmp(symbol: str = "AAPL") -> Tuple[bool, str]:
    """
    Enkel hälsokoll: hämtar profile för en symbol.
    Returnerar (ok, meddelande).
    """
    try:
        data = _get(f"profile/{symbol}")
        if isinstance(data, list) and data:
            return True, "FMP-anslutning OK."
        return False, "Oväntat svar från FMP (ingen data)."
    except Exception as e:
        return False, str(e)
