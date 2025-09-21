from functools import lru_cache
from typing import Optional

import yfinance as yf

SUPPORTED = {"SEK","USD","EUR","NOK","CAD","GBP"}


def _pair(base: str, target: str) -> str:
    return f"{base}{target}=X"


@lru_cache(maxsize=64)
def fx_rate(base: str, target: str) -> Optional[float]:
    base = base.upper()
    target = target.upper()
    if base == target:
        return 1.0
    if base not in SUPPORTED or target not in SUPPORTED:
        return None
    t = yf.Ticker(_pair(base, target))
    # fast_info
    try:
        p = getattr(t, "fast_info", None)
        if p and p.get("last_price"):
            return float(p["last_price"])
    except Exception:
        pass
    # history fallback
    try:
        hist = t.history(period="5d", interval="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def convert(amount: Optional[float], base: str, target: str) -> Optional[float]:
    if amount is None:
        return None
    r = fx_rate(base, target)
    return amount * r if r else None
