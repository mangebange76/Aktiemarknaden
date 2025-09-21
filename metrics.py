from typing import Dict, List, Tuple, Optional
import math
import statistics as stats


def safe_float(x) -> Optional[float]:
    try:
        if x in (None, ""):
            return None
        return float(x)
    except Exception:
        return None


def ps_pe_from_ttm(ttm_ratios: Dict) -> Tuple[Optional[float], Optional[float]]:
    ps = safe_float(ttm_ratios.get("priceToSalesRatioTTM"))
    pe = safe_float(ttm_ratios.get("peTTM") or ttm_ratios.get("priceEarningsRatioTTM"))
    return ps, pe


def hist_avg_ps_pe(hist_km: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
    ps_vals, pe_vals = [], []
    for row in hist_km:
        ps_vals.append(safe_float(row.get("priceToSalesRatio")))
        pe_vals.append(safe_float(row.get("priceEarningsRatio")))
    ps_vals = [v for v in ps_vals if v and math.isfinite(v)]
    pe_vals = [v for v in pe_vals if v and math.isfinite(v)]
    ps_avg = stats.fmean(ps_vals) if ps_vals else None
    pe_avg = stats.fmean(pe_vals) if pe_vals else None
    return ps_avg, pe_avg


def cagr(values: List[Optional[float]]) -> Optional[float]:
    series = [v for v in values if v is not None and math.isfinite(v)]
    if len(series) < 2:
        return None
    first, last = series[0], series[-1]
    years = len(series) - 1
    if first <= 0 or last <= 0:
        return None
    return (last/first) ** (1/years) - 1


def intrinsic_from_hist_multiples(price: Optional[float],
                                  sales_per_share_ttm: Optional[float],
                                  eps_ttm: Optional[float],
                                  hist_ps_avg_5y: Optional[float],
                                  hist_pe_avg_5y: Optional[float]) -> Dict:
    target_ps = sales_per_share_ttm * hist_ps_avg_5y if (sales_per_share_ttm and hist_ps_avg_5y) else None
    target_pe = eps_ttm * hist_pe_avg_5y if (eps_ttm and hist_pe_avg_5y) else None
    candidates = [v for v in [target_ps, target_pe] if v and math.isfinite(v)]
    conservative = min(candidates) if candidates else None

    def underval(p, t):
        if p and t and t > 0:
            return (t - p) / t * 100.0
        return None

    return {
        "Target via P/S": target_ps,
        "Target via P/E": target_pe,
        "Target konservativ": conservative,
        "Undervärdering via P/S (%)": underval(price, target_ps),
        "Undervärdering via P/E (%)": underval(price, target_pe),
        "Undervärdering konservativ (%)": underval(price, conservative),
    }


def debt_safety_score(debt_to_equity: float = None,
                      interest_coverage: float = None,
                      current_ratio: float = None) -> float:
    score = 0.0
    if debt_to_equity is not None:
        if debt_to_equity < 0.3: score += 40
        elif debt_to_equity < 0.6: score += 30
        elif debt_to_equity < 1.0: score += 20
        elif debt_to_equity < 1.5: score += 10
    if interest_coverage is not None:
        if interest_coverage > 12: score += 40
        elif interest_coverage > 6:  score += 30
        elif interest_coverage > 3:  score += 20
        elif interest_coverage > 1.5:score += 10
    if current_ratio is not None:
        if current_ratio > 2.0: score += 20
        elif current_ratio > 1.5: score += 15
        elif current_ratio > 1.0: score += 10
        elif current_ratio > 0.8:  score += 5
    return min(100.0, score)


def composite_rank(underv_konserv: float = None,
                   growth_cagr_rev_5y: float = None,
                   growth_cagr_eps_5y: float = None,
                   debt_score: float = None,
                   weight_value: float = 0.45,
                   weight_growth: float = 0.35,
                   weight_debt: float = 0.20) -> float:
    uv = max(-100.0, min(100.0, underv_konserv)) if underv_konserv is not None else 0.0
    gr = 0.5 * ((growth_cagr_rev_5y or 0.0) + (growth_cagr_eps_5y or 0.0))
    gr = max(-50.0, min(100.0, gr))
    ds = debt_score or 0.0
    return weight_value * uv + weight_growth * gr + weight_debt * ds


def top_k_per_industry(df, k_per_industry: int, cap_total: int) -> list:
    """
    Tar topp K per industri enligt df:s aktuella sortering.
    Returnerar ordnad unik tickerlista (max cap_total).
    """
    import pandas as pd  # lazy import för att undvika hårt beroende när modulen repurposas
    if "Industri" not in df.columns or "Ticker" not in df.columns:
        return []
    chosen = []
    for ind, sub in df.groupby("Industri", sort=False):
        picks = sub.head(int(k_per_industry))["Ticker"].dropna().astype(str).tolist()
        chosen.extend(picks)
    ordered = [t for t in df["Ticker"].dropna().astype(str).tolist() if t in set(chosen)]
    ordered = list(dict.fromkeys(ordered))
    return ordered[:int(cap_total)]
