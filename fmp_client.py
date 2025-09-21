from __future__ import annotations
import os, time
from typing import Iterable, List, Optional, Dict, Any, Union, Tuple
import requests

BASE_URL = "https://financialmodelingprep.com/stable"

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

    if r.status_code == 429:
        time.sleep(1.2)
        r = requests.get(url, params=params, timeout=timeout)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        snippet = (r.text or "")[:300].replace(apikey, "***")
        raise requests.HTTPError(f"FMP HTTP {r.status_code} på {endpoint}. Svar: {snippet}") from e

    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

# -------------------------- gratis helpers --------------------------
def stock_list() -> List[Dict[str, Any]]:
    return _get("stock-list")

_COUNTRY_TO_EXCH = {
    "United States": ["NASDAQ","NYSE","AMEX"],
    "Sweden": ["STO"],
    "Norway": ["OSE"],
    "Denmark": ["CPH"],
    "Finland": ["HEL"],
    "Germany": ["FWB"],
    "United Kingdom": ["LSE"],
    "Canada": ["TSX","TSXV"],
    "Europe": ["FWB","LSE","STO","OSE","CPH","HEL"],  # grov proxy
}

def _filter_stock_list(rows: List[Dict[str, Any]],
                       exchange: Optional[str],
                       country: Optional[str]) -> List[Dict[str, Any]]:
    ex_field = "exchangeShortName"  # fältet i stock-list
    wanted = set()
    if exchange:
        wanted.add(exchange.upper())
    if country and country in _COUNTRY_TO_EXCH:
        wanted.update(_COUNTRY_TO_EXCH[country])
    if not wanted:
        return rows
    out = []
    for r in rows:
        ex = (r.get(ex_field) or r.get("exchange") or "").upper()
        if ex in wanted:
            out.append({"symbol": r.get("symbol"), "exchange": ex})
    return out

# -------------------------- screener (med fallback) --------------------------
def company_screener(**params) -> List[Dict[str, Any]]:
    return _get("company-screener", params=params)

def list_universe(exchange: Optional[str] = None,
                  country: Optional[str] = None,
                  sector: Optional[str] = None,
                  industry: Optional[str] = None,
                  limit: int = 100000) -> List[Dict[str, Any]]:
    """
    Försök premium 'company-screener'. Vid 402 faller vi tillbaka till gratis 'stock-list'
    och filtrerar lokalt på exchange/country.
    """
    params: Dict[str, Any] = {"limit": limit}
    if exchange: params["exchange"] = exchange
    if country:  params["country"]  = country
    if sector:   params["sector"]   = sector
    if industry: params["industry"] = industry
    try:
        data = company_screener(**params)
        # Normalisera minimalt
        return [d for d in data if isinstance(d, dict) and d.get("symbol")]
    except requests.HTTPError as e:
        msg = str(e)
        if "402" in msg or "Premium Endpoint" in msg:
            # Fallback: gratis stock-list + lokal filtrering
            raw = stock_list()
            filt = _filter_stock_list(raw, exchange, country)
            return filt[:limit]
        raise

def sectors_in_universe(exchange: Optional[str] = None, country: Optional[str] = None) -> List[str]:
    # Kräver premium för att bli bra; vi returnerar tom lista i gratisläge.
    try:
        rows = list_universe(exchange=exchange, country=country, limit=20000)
        ss = sorted({r.get("sector") for r in rows if r.get("sector")})
        return ss
    except Exception:
        return []

def industries_in_universe(exchange: Optional[str] = None, country: Optional[str] = None, sector: Optional[str] = None) -> List[str]:
    try:
        rows = list_universe(exchange=exchange, country=country, limit=20000)
        ii = sorted({r.get("industry") for r in rows if r.get("industry")})
        return ii
    except Exception:
        return []

# -------------------------- index-konstituenter --------------------------
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
    code = index_code.upper()
    path_map = {
        "SP500": "sp500-constituent",
        "NDX":   "nasdaq-constituent",
        "DOW":   "dowjones-constituent",
    }
    syms = _index_api_symbols(path_map.get(code))
    if syms:
        return syms

    # Fallback via börs/land
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

# -------------------------- quotes/profile/nyckeltal --------------------------
def get_quote(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("quote", params={"symbol": _symbols_param(symbols)})

def get_profile(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("profile", params={"symbol": _symbols_param(symbols)})

def get_ratios_ttm(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("ratios-ttm", params={"symbol": _symbols_param(symbols)})

def get_key_metrics_ttm(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("key-metrics-ttm", params={"symbol": _symbols_param(symbols)})

def get_key_metrics_history(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _get("key-metrics", params={"symbol": symbol.upper(), "period": period, "limit": limit})

def get_income_statement(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _get("income-statement", params={"symbol": symbol.upper(), "period": period, "limit": limit})

def get_cashflow_statement(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _get("cashflow-statement", params={"symbol": symbol.upper(), "period": period, "limit": limit})

# -------------------------- batch utilities + kompatibla wrappers --------------------------
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

def test_fmp(symbol: str = "AAPL") -> Tuple[bool, str]:
    try:
        q = get_quote(symbol)
        if isinstance(q, list) and q:
            return True, "FMP-anslutning OK."
        return False, "Oväntat svar från FMP."
    except Exception as e:
        return False, str(e)
