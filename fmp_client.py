import os
import time
from typing import List, Dict, Optional

import requests

FMP_BASE = "https://financialmodelingprep.com/api/v3"
API_KEY = os.environ.get("FMP_API_KEY") or os.getenv("FMP_API_KEY")


# ----------------------------- Low-level -----------------------------
def _get(path: str, params: Dict = None, timeout: int = 25):
    if params is None:
        params = {}
    if "apikey" not in params:
        params["apikey"] = API_KEY
    r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


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
    return [d for d in data if d.get("symbol")]


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
        out.extend(data)
        time.sleep(0.35)
    return out


def financial_ratios_ttm(symbol: str) -> Dict:
    fr = _get(f"ratios-ttm/{symbol}")
    return fr[0] if isinstance(fr, list) and fr else {}


def key_metrics_ttm(symbol: str) -> Dict:
    km = _get(f"key-metrics-ttm/{symbol}")
    return km[0] if isinstance(km, list) and km else {}


def historical_key_metrics(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict]:
    return _get(f"historical-key-metrics/{symbol}", params={"period": period, "limit": limit})


def income_statement(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict]:
    return _get(f"income-statement/{symbol}", params={"period": period, "limit": limit})


def cashflow_statement(symbol: str, period: str = "annual", limit: int = 6) -> List[Dict]:
    return _get(f"cash-flow-statement/{symbol}", params={"period": period, "limit": limit})


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
    if code in ("DAX","MDAX","SDAX","TECDAX"):
        rows = list_universe(exchange="FWB", limit=50000); return [r["symbol"] for r in rows if r.get("symbol")]
    if code in ("FTSE100","FTSE250"):
        rows = list_universe(exchange="LSE", limit=50000); return [r["symbol"] for r in rows if r.get("symbol")]
    if code == "TSXCOMP":
        rows = list_universe(exchange="TSX", limit=50000);  return [r["symbol"] for r in rows if r.get("symbol")]
    if code == "TSXV":
        rows = list_universe(exchange="TSXV", limit=50000); return [r["symbol"] for r in rows if r.get("symbol")]
    if code in ("OBX","OSEBX"):
        rows = list_universe(exchange="OSE", limit=50000);  return [r["symbol"] for r in rows if r.get("symbol")]
    if code == "OMXC25":
        rows = list_universe(exchange="CPH", limit=50000);  return [r["symbol"] for r in rows if r.get("symbol")]
    if code == "OMXH25":
        rows = list_universe(exchange="HEL", limit=50000);  return [r["symbol"] for r in rows if r.get("symbol")]
    if code in ("OMXS30","OMXSB","OMXS30GI"):
        rows = list_universe(exchange="STO", limit=50000);  return [r["symbol"] for r in rows if r.get("symbol")]
    if code in ("SP500","NDX","DOW"):
        rows = list_universe(country="United States", limit=50000); return [r["symbol"] for r in rows if r.get("symbol")]
    if code in ("STOXX600","STOXX50E"):
        rows = list_universe(country="Europe", limit=50000); return [r["symbol"] for r in rows if r.get("symbol")]

    return []
