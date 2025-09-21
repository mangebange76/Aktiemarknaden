from __future__ import annotations
import os, time
from typing import Iterable, List, Optional, Dict, Any, Union, Tuple
import requests

# ---- Bas-url: nya "stable" istället för legacy /api/v3 ----
BASE_URL = "https://financialmodelingprep.com/stable"

# ---- Läs API-nyckel: Streamlit secrets först, annars env ----
def _get_api_key() -> Optional[str]:
    try:
        import streamlit as st
        key = st.secrets.get("FMP_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("FMP_API_KEY")

def _symbols_param(symbols: Union[str, Iterable[str]]) -> str:
    if isinstance(symbols, str):
        return symbols.strip().upper()
    return ",".join(sorted({s.strip().upper() for s in symbols if s and s.strip()}))

def _get(endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    params = dict(params or {})
    apikey = _get_api_key()
    if not apikey:
        raise RuntimeError("FMP_API_KEY saknas. Lägg in i Streamlit Secrets (utanför [GOOGLE_CREDENTIALS]).")
    params["apikey"] = apikey

    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, params=params, timeout=timeout)

    # enkel backoff vid 429
    if r.status_code == 429:
        time.sleep(1.2)
        r = requests.get(url, params=params, timeout=timeout)

    # FMP returnerar 403 med 'Legacy' för gamla vägar → ge tydligt fel
    if r.status_code == 403 and "Legacy" in (r.text or ""):
        raise requests.HTTPError(f"FMP 403 Legacy Endpoint på {endpoint}. (Byter nu till /stable i klienten.)")

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        snippet = (r.text or "")[:300].replace(apikey, "***")
        raise requests.HTTPError(f"FMP HTTP {r.status_code} på {endpoint}. Svar: {snippet}") from e

    # vissa svar kan vara text; försök json först
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}


# =========================
#   Nya "stable" wrappers
# =========================

# --- Directory / Screener ---
def company_screener(**params) -> List[Dict[str, Any]]:
    return _get("company-screener", params=params)

def stock_list() -> List[Dict[str, Any]]:
    return _get("stock-list")

def available_sectors() -> List[str]:
    out = _get("available-sectors")
    if out and isinstance(out, list) and isinstance(out[0], dict) and "sector" in out[0]:
        return [x.get("sector") for x in out if "sector" in x]
    return out

def available_industries() -> List[str]:
    out = _get("available-industries")
    if out and isinstance(out, list) and isinstance(out[0], dict) and "industry" in out[0]:
        return [x.get("industry") for x in out if "industry" in x]
    return out

# --- Constituents ---
def sp500_constituent() -> List[Dict[str, Any]]:
    return _get("sp500-constituent")

def nasdaq_constituent() -> List[Dict[str, Any]]:
    # OBS: detta är Nasdaq-100 hos FMP
    return _get("nasdaq-constituent")

def dowjones_constituent() -> List[Dict[str, Any]]:
    try:
        return _get("dowjones-constituent")
    except Exception:
        return []

# --- Quotes / Profile ---
def get_quote(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("quote", params={"symbol": _symbols_param(symbols)})

def get_profile(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("profile", params={"symbol": _symbols_param(symbols)})

# --- Key metrics / Ratios ---
def get_ratios_ttm(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("ratios-ttm", params={"symbol": _symbols_param(symbols)})

def get_key_metrics_ttm(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("key-metrics-ttm", params={"symbol": _symbols_param(symbols)})

def get_key_metrics_history(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _get("key-metrics", params={"symbol": symbol.upper(), "period": period, "limit": limit})

# --- Financial statements ---
def get_income_statement(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _get("income-statement", params={"symbol": symbol.upper(), "period": period, "limit": limit})

def get_cashflow_statement(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _get("cashflow-statement", params={"symbol": symbol.upper(), "period": period, "limit": limit})


# =========================
#  Backwards compatibility
#  (behåller appens imports)
# =========================

def list_universe(exchange: Optional[str] = None,
                  country: Optional[str] = None,
                  sector: Optional[str] = None,
                  industry: Optional[str] = None,
                  limit: int = 100000) -> List[Dict[str, Any]]:
    """
    Ersätter gamla 'stock-screener' med 'company-screener'.
    """
    params: Dict[str, Any] = {"limit": limit}
    if exchange: params["exchange"] = exchange
    if country:  params["country"]  = country
    if sector:   params["sector"]   = sector
    if industry: params["industry"] = industry
    data = company_screener(**params)
    # Normalisera: vissa fält kan saknas beroende på filter; vi returnerar dicts med 'symbol'
    return [d for d in data if isinstance(d, dict) and d.get("symbol")]

def sectors_in_universe(exchange: Optional[str] = None, country: Optional[str] = None) -> List[str]:
    rows = list_universe(exchange=exchange, country=country, limit=50000)
    return sorted({r.get("sector") for r in rows if r.get("sector")})

def industries_in_universe(exchange: Optional[str] = None, country: Optional[str] = None, sector: Optional[str] = None) -> List[str]:
    rows = list_universe(exchange=exchange, country=country, sector=sector, limit=50000)
    return sorted({r.get("industry") for r in rows if r.get("industry")})

def quote_batch(symbols: List[str]) -> List[Dict[str, Any]]:
    if not symbols:
        return []
    out: List[Dict[str, Any]] = []
    CHUNK = 150
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i+CHUNK]
        data = get_quote(chunk)
        if isinstance(data, list):
            out.extend(data)
        time.sleep(0.35)
    return out

def financial_ratios_ttm(symbol: str) -> Dict[str, Any]:
    data = get_ratios_ttm(symbol)
    return data[0] if isinstance(data, list) and data else {}

def key_metrics_ttm(symbol: str) -> Dict[str, Any]:
    data = get_key_metrics_ttm(symbol)
    return data[0] if isinstance(data, list) and data else {}

def historical_key_metrics(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict[str, Any]]:
    return get_key_metrics_history(symbol, period=period, limit=limit) or []

def income_statement(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict[str, Any]]:
    return get_income_statement(symbol, period=period, limit=limit) or []

def cashflow_statement(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict[str, Any]]:
    return get_cashflow_statement(symbol, period=period, limit=limit) or []

def batch_safe(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i+chunk_size]

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
    Försök API-konstituenter först (där FMP har endpoints), annars fallback:
    använd exchange/land som proxy för ett brett universum.
    """
    code = index_code.upper()

    # Kända "stable"-endpoints (konstituenter)
    path_map = {
        "SP500": "sp500-constituent",
        "NDX":   "nasdaq-constituent",    # Nasdaq-100
        "DOW":   "dowjones-constituent",
        # (om europeiska index saknas i API → fallback längre ned)
    }
    syms = _index_api_symbols(path_map.get(code))
    if syms:
        return syms

    # Fallback via börs/land som proxy
    def _lu(ex=None, ct=None):
        return [r["symbol"] for r in list_universe(exchange=ex, country=ct, limit=50000) if r.get("symbol")]

    if code in ("DAX","MDAX","SDAX","TECDAX"):
        return _lu(ex="FWB")
    if code in ("FTSE100","FTSE250"):
        return _lu(ex="LSE")
    if code == "TSXCOMP":
        return _lu(ex="TSX")
    if code == "TSXV":
        return _lu(ex="TSXV")
    if code in ("OBX","OSEBX"):
        return _lu(ex="OSE")
    if code == "OMXC25":
        return _lu(ex="CPH")
    if code == "OMXH25":
        return _lu(ex="HEL")
    if code in ("OMXS30","OMXSB","OMXS30GI"):
        return _lu(ex="STO")
    if code in ("SP500","NDX","DOW"):
        return _lu(ct="United States")
    if code in ("STOXX600","STOXX50E"):
        return _lu(ct="Europe")

    return []

def test_fmp(symbol: str = "AAPL") -> Tuple[bool, str]:
    try:
        q = get_quote(symbol)
        if isinstance(q, list) and q:
            return True, "FMP-anslutning OK."
        return False, "Oväntat svar från FMP."
    except Exception as e:
        return False, str(e)
