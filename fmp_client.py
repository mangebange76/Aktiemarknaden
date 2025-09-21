# aktiemarknaden/fmp_client.py
from __future__ import annotations
import os
from typing import Iterable, List, Optional, Dict, Any, Union
import requests

BASE_URL = "https://financialmodelingprep.com/stable"

def _get_api_key() -> Optional[str]:
    # Miljövariabel först
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    # Streamlit secrets om det finns
    try:
        import streamlit as st  # type: ignore
        key = st.secrets.get("FMP_API_KEY")  # pyright: ignore[reportOptionalMemberAccess]
        if key:
            return str(key)
    except Exception:
        pass
    return None

def _symbols_param(symbols: Union[str, Iterable[str]]) -> str:
    if isinstance(symbols, str):
        return symbols.strip().upper()
    return ",".join(sorted({s.strip().upper() for s in symbols if s and s.strip()}))

def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    params = dict(params or {})
    apikey = _get_api_key()
    if apikey:
        params["apikey"] = apikey
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, params=params, timeout=30)
    # FMP skickar ett tydligt meddelande vid legacy-route
    if r.status_code == 403 and "Legacy" in (r.text or ""):
        raise RuntimeError(
            f"FMP 403 Legacy Endpoint på {url}. Byt till /stable-varianten av endpointen."
        )
    r.raise_for_status()
    data = r.json()
    return data

# ---------- DIRECTORY / UNIVERSE ----------
def list_universe(
    exchange: Optional[str] = None,
    country: Optional[str] = None,
    page: int = 0,
    limit: int = 10000,
    **filters: Any,
) -> List[Dict[str, Any]]:
    """
    Hämtar tickers via nya company-screener (ersätter stock-screener).
    Vanliga filter: exchange, sector, industry, priceMoreThan, marketCapMoreThan, isEtf, isActivelyTrading etc.
    """
    params: Dict[str, Any] = {"page": page, "limit": limit}
    if exchange:
        params["exchange"] = exchange
    if country:
        params["country"] = country
    params.update({k: v for k, v in filters.items() if v is not None})
    return _get("company-screener", params=params)

def list_symbols_all() -> List[Dict[str, Any]]:
    """Full symbol-lista (alla marknader)"""
    return _get("stock-list")

def available_sectors() -> List[str]:
    out = _get("available-sectors")
    # FMP returnerar ibland som list of dicts; normalisera till str-lista
    if out and isinstance(out, list) and isinstance(out[0], dict) and "sector" in out[0]:
        return [x.get("sector") for x in out if "sector" in x]
    return out

def available_industries() -> List[str]:
    out = _get("available-industries")
    if out and isinstance(out, list) and isinstance(out[0], dict) and "industry" in out[0]:
        return [x.get("industry") for x in out if "industry" in x]
    return out

# ---------- INDEX KONSTITUENTER ----------
def sp500_constituents() -> List[Dict[str, Any]]:
    return _get("sp500-constituent")

def nasdaq_constituents() -> List[Dict[str, Any]]:
    return _get("nasdaq-constituent")

# ---------- QUOTES & PROFIL ----------
def get_quote(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("quote", params={"symbol": _symbols_param(symbols)})

def get_profile(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    # NYTT: profile kräver ?symbol=... (inte /profile/{sym})
    return _get("profile", params={"symbol": _symbols_param(symbols)})

# ---------- NYCKELTAL ----------
def get_ratios_ttm(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("ratios-ttm", params={"symbol": _symbols_param(symbols)})

def get_key_metrics_ttm(symbols: Union[str, Iterable[str]]) -> List[Dict[str, Any]]:
    return _get("key-metrics-ttm", params={"symbol": _symbols_param(symbols)})

def get_key_metrics_history(
    symbol: str, period: str = "annual", limit: int = 20
) -> List[Dict[str, Any]]:
    return _get("key-metrics", params={"symbol": symbol.upper(), "period": period, "limit": limit})

# ---------- FINANSIELLA RAPPORTER ----------
def get_income_statement(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    return _get("income-statement", params={"symbol": symbol.upper(), "period": period, "limit": limit})

def get_cashflow_statement(symbol: str, period: str = "annual", limit: int = 8) -> List[Dict[str, Any]]:
    # OBS: cashflow-statement i stable
    return _get("cashflow-statement", params={"symbol": symbol.upper(), "period": period, "limit": limit})
