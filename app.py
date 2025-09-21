# app.py — Pro Screener: multi-presets, FX, peer-z, EV/EBIT(DA), FCF-yield, SQLite-cache, backtest & Sheets
import os
import time
import json
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from fmp_client import (
    list_universe, sectors_in_universe, industries_in_universe,
    quote_batch, financial_ratios_ttm, key_metrics_ttm,
    historical_key_metrics, income_statement, cashflow_statement,
    batch_safe, index_symbols
)
from fx_utils import convert as fx_convert, fx_rate
from metrics import (
    safe_float, ps_pe_from_ttm, hist_avg_ps_pe, cagr,
    intrinsic_from_hist_multiples, debt_safety_score, composite_rank,
    top_k_per_industry
)
from sheets_utils import read_df, write_df

st.set_page_config(page_title="Global Screener – Pro", layout="wide")
st.title("🌐 Pro Screener – multi-marknad, peer-z, EV/EBIT(DA), FCF-yield, cache, backtest & Sheets")

FMP_API_KEY = os.environ.get("FMP_API_KEY") or st.secrets.get("FMP_API_KEY")
SHEET_URL   = st.secrets.get("SHEET_URL", "")
SHEET_NAME  = st.secrets.get("SHEET_NAME", "Bolag")
CACHE_DB    = "cache.db"

if not FMP_API_KEY:
    st.warning("⚠️ Sätt FMP_API_KEY i Secrets för att hämta data från FMP.")


# ----------------------------- Benchmarks (backtest) -----------------------------
BENCH_ETF = {
    "USA": "SPY",
    "Sverige": "EWD",
    "Norge": "NORW",
    "Danmark": "EDEN",
    "Finland": "EFNL",
    "Kanada": "EWC",
    "Tyskland": "EWG",
    "UK": "EWU",
}
def preset_to_region(preset_name: str) -> str:
    for key in ["USA","Sverige","Norge","Danmark","Finland","Kanada","Tyskland","UK"]:
        if preset_name.startswith(key):
            return key
    return "USA"


# ----------------------------- SQLite cache (raw data) -----------------------------
def cache_init():
    con = sqlite3.connect(CACHE_DB)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            symbol TEXT PRIMARY KEY,
            updated_at TEXT,
            ratios_ttm TEXT,
            key_ttm TEXT,
            hist_km TEXT,
            income TEXT,
            cashflow TEXT
        )
    """)
    con.commit(); con.close()

def cache_get(symbol: str, max_age_hours: int):
    con = sqlite3.connect(CACHE_DB); cur = con.cursor()
    cur.execute("SELECT updated_at, ratios_ttm, key_ttm, hist_km, income, cashflow FROM cache WHERE symbol=?", (symbol,))
    row = cur.fetchone(); con.close()
    if not row: return None
    updated_at = datetime.fromisoformat(row[0])
    if datetime.utcnow() - updated_at > timedelta(hours=max_age_hours):
        return None
    loads = lambda s: json.loads(s) if s else None
    return {
        "ratios_ttm": loads(row[1]),
        "key_ttm": loads(row[2]),
        "hist_km": loads(row[3]),
        "income": loads(row[4]),
        "cashflow": loads(row[5]),
    }

def cache_put(symbol: str, ratios_ttm, key_ttm, hist_km, income, cashflow):
    con = sqlite3.connect(CACHE_DB); cur = con.cursor()
    cur.execute("""
        INSERT INTO cache(symbol, updated_at, ratios_ttm, key_ttm, hist_km, income, cashflow)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(symbol) DO UPDATE SET
          updated_at=excluded.updated_at,
          ratios_ttm=excluded.ratios_ttm,
          key_ttm=excluded.key_ttm,
          hist_km=excluded.hist_km,
          income=excluded.income,
          cashflow=excluded.cashflow
    """, (
        symbol,
        datetime.utcnow().isoformat(timespec="seconds"),
        json.dumps(ratios_ttm or {}),
        json.dumps(key_ttm or {}),
        json.dumps(hist_km or []),
        json.dumps(income or []),
        json.dumps(cashflow or [])
    ))
    con.commit(); con.close()

cache_init()


# ----------------------------- Picks (backtest snapshots) -----------------------------
def picks_init():
    con = sqlite3.connect(CACHE_DB); cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_utc TEXT,
            preset_name TEXT,
            base_ccy TEXT,
            tickers TEXT,     -- json-list
            weights TEXT      -- json-list (lika vikt)
        )
    """)
    con.commit(); con.close()

def picks_add(preset_name: str, base_ccy: str, tickers: list, weights: list):
    con = sqlite3.connect(CACHE_DB); cur = con.cursor()
    cur.execute(
        "INSERT INTO picks(created_utc,preset_name,base_ccy,tickers,weights) VALUES(?,?,?,?,?)",
        (datetime.utcnow().isoformat(timespec="seconds"), preset_name, base_ccy, json.dumps(tickers), json.dumps(weights))
    )
    con.commit(); con.close()

def picks_fetch() -> pd.DataFrame:
    con = sqlite3.connect(CACHE_DB)
    df = pd.read_sql_query("SELECT * FROM picks ORDER BY created_utc DESC, id DESC", con)
    con.close()
    return df

picks_init()


# ----------------------------- Sidopanel -----------------------------
with st.sidebar:
    st.header("Google Sheets & valuta")
    sheet_url  = st.text_input("Sheet URL", value=SHEET_URL)
    sheet_name_base = st.text_input("Basflik", value=SHEET_NAME)
    base_ccy   = st.selectbox("Visningsvaluta", ["SEK","USD","EUR","NOK","CAD","GBP"], index=0)
    only_new   = st.checkbox("Skriv endast nya fält (bevara övrigt)", value=False)

    st.header("Cache & uppdatering")
    cache_hours = st.number_input("Cache max ålder (timmar)", 0, 72, 6, 1)  # 0 = alltid färskt
    delay       = st.number_input("Fördröjning per symbol (sek)", 0.0, 3.0, 0.8, 0.1)
    hist_years  = st.slider("Historik (år) för P/S & P/E-snitt", 3, 10, 5)

    st.header("Quality gate (före rank)")
    min_roic = st.number_input("Min ROIC TTM (%)", 0.0, 100.0, 0.0, 0.5)
    min_icr  = st.number_input("Min Interest Coverage", 0.0, 100.0, 2.0, 0.5)
    require_pos_fcf = st.checkbox("Kräv positivt FCF (TTM)", value=True)

    st.header("Likviditet")
    min_price = st.number_input(f"Min pris ({base_ccy})", 0.0, 1000.0, 1.0, 0.5)
    min_avgvol = st.number_input("Min snittvolym (10d)", 0, 50_000_000, 100_000, 10_000)

    st.header("Rank-vikter")
    w_val = st.slider("Vikt: Värde",    0.0, 1.0, 0.45, 0.05)
    w_gro = st.slider("Vikt: Tillväxt", 0.0, 1.0, 0.35, 0.05)
    w_deb = st.slider("Vikt: Skuld",    0.0, 1.0, 0.20, 0.05)
    de_cap = st.number_input("Max D/E för full skuldbetygsvikt", 0.1, 10.0, 1.0, 0.1)


# ----------------------------- 1) Välj presets (multi) -----------------------------
st.subheader("1) Välj marknader/index (multi-val)")

PRESETS = {
    # USA
    "USA – Index (S&P500/NDX/DOW)": ["SP500","NDX","DOW"],
    "USA – Alla listor": ["NASDAQ","NYSE","AMEX"],
    # Sverige
    "Sverige – Index (OMXS30/OMXSB/OMXS30GI)": ["OMXS30","OMXSB","OMXS30GI"],
    "Sverige – Alla listor (STO)": ["STO"],
    # Norge
    "Norge – Index (OBX/OSEBX)": ["OBX","OSEBX"],
    "Norge – Alla listor (OSE)": ["OSE"],
    # Danmark/Finland
    "Danmark – Index (OMXC25)": ["OMXC25"],
    "Danmark – Alla listor (CPH)": ["CPH"],
    "Finland – Index (OMXH25)": ["OMXH25"],
    "Finland – Alla listor (HEL)": ["HEL"],
    # Kanada
    "Kanada – Index (TSX Composite/TSXV)": ["TSXCOMP","TSXV"],
    "Kanada – Alla listor (TSX+TSXV)": ["TSX","TSXV"],
    # Tyskland
    "Tyskland – Index (DAX/MDAX/SDAX/TecDAX)": ["DAX","MDAX","SDAX","TECDAX"],
    "Tyskland – Alla listor (FWB/Xetra)": ["FWB"],
    # UK
    "UK – Index (FTSE100/FTSE250)": ["FTSE100","FTSE250"],
    "UK – Alla listor (LSE)": ["LSE"],
}

chosen = st.multiselect("Välj en eller flera presets", list(PRESETS.keys()))

def gather_symbols_from_preset(key: str) -> list:
    items = PRESETS[key]
    syms: list = []
    for it in items:
        if it.isalpha() and it not in ("NASDAQ","NYSE","AMEX","TSX","TSXV","LSE","FWB","SIX","STO","OSE","CPH","HEL"):
            syms += index_symbols(it)
        else:
            rows = list_universe(exchange=it, limit=20000)
            syms += [r["symbol"] for r in rows if r.get("symbol")]
        time.sleep(0.1)
    return list(dict.fromkeys(syms))


preset_universes = {}
if st.button("Hämta universum för valda presets"):
    if not chosen:
        st.warning("Välj minst ett preset.")
    else:
        with st.spinner("Hämtar tickers…"):
            for k in chosen:
                preset_universes[k] = gather_symbols_from_preset(k)
        total = sum(len(v) for v in preset_universes.values())
        st.success(f"Klar. {len(preset_universes)} presets, totalt {total} tickers.")


# ----------------------------- 2) Massuppdatera & beräkna -----------------------------
st.subheader("2) Massuppdatera & beräkna (inkl. cache, quality gate, peer-z, EV/EBIT(DA), FCF-yield)")
if st.button("Uppdatera alla valda presets"):
    if not preset_universes:
        st.error("Hämta först universum för dina presets.")
        st.stop()

    all_tabs = []  # (fliknamn, df)

    for preset_name, symbols in preset_universes.items():
        st.write(f"### {preset_name} — {len(symbols)} tickers")
        if not symbols:
            st.info("Inga tickers i preset."); continue

        # Prisbatch (pris, valuta, volym, namn, börs)
        qrows = quote_batch(symbols)
        qmap = {r["symbol"]: r for r in qrows if r.get("symbol")}

        base_df = pd.DataFrame({"Ticker": symbols})
        base_df["Aktuell kurs (orig)"] = base_df["Ticker"].map(lambda s: qmap.get(s, {}).get("price"))
        base_df["Valuta (pris)"]       = base_df["Ticker"].map(lambda s: qmap.get(s, {}).get("currency"))
        base_df["AvgVol10d"]           = base_df["Ticker"].map(lambda s: qmap.get(s, {}).get("avgVolume10Day") or qmap.get(s, {}).get("avgVolume"))
        base_df["Bolag"]               = base_df["Ticker"].map(lambda s: qmap.get(s, {}).get("name"))
        base_df["Börs"]                = base_df["Ticker"].map(lambda s: qmap.get(s, {}).get("exchange"))

        # Pris i basvaluta
        def _to_base(row):
            p = safe_float(row.get("Aktuell kurs (orig)"))
            ccy = row.get("Valuta (pris)")
            return fx_convert(p, ccy, base_ccy) if ccy else p
        base_df[f"Aktuell kurs ({base_ccy})"] = base_df.apply(_to_base, axis=1)

        progress = st.progress(0.0); status = st.empty()
        rows = []; total = len(symbols); done = 0

        for sym in symbols:
            done += 1
            status.write(f"Bearbetar {done}/{total}: {sym}")
            blob = cache_get(sym, max_age_hours=cache_hours) if cache_hours > 0 else None
            try:
                if blob:
                    ttm = blob["ratios_ttm"] or {}
                    km  = blob["key_ttm"] or {}
                    hist= blob["hist_km"] or []
                    inc = blob["income"] or []
                    cfs = blob["cashflow"] or []
                else:
                    ttm = financial_ratios_ttm(sym) or {}
                    km  = key_metrics_ttm(sym) or {}
                    hist= historical_key_metrics(sym, period="annual", limit=hist_years+1) or []
                    inc = income_statement(sym, period="annual", limit=hist_years+1) or []
                    cfs = cashflow_statement(sym, period="annual", limit=hist_years+1) or []
                    cache_put(sym, ttm, km, hist, inc, cfs)

                ps_ttm, pe_ttm = ps_pe_from_ttm(ttm)
                sps = safe_float(km.get("salesPerShareTTM"))
                eps = safe_float(km.get("epsTTM"))

                # EV/EBITDA & EV/Sales
                ev_to_ebitda = safe_float(km.get("evToEbitdaTTM"))
                ev_to_sales  = safe_float(km.get("evToSalesTTM") or ttm.get("enterpriseValueToSalesRatioTTM"))
                # EV/EBIT proxy från income
                ebit_annual = None
                if inc:
                    last = inc[0]  # nyaste först
                    ebit_annual = safe_float(last.get("ebit") or last.get("operatingIncome"))
                ev_over_ebit = None
                if ebit_annual and ebit_annual > 0 and "enterpriseValueTTM" in km:
                    ev_over_ebit = safe_float(km.get("enterpriseValueTTM")) / ebit_annual

                # 5y historik-snitt
                ps_avg, pe_avg = hist_avg_ps_pe(hist)

                # CAGR (revenue/eps)
                rev_series = [safe_float(r.get("revenue")) for r in reversed(inc)]
                eps_series = [safe_float(r.get("eps")) for r in reversed(inc)]
                rev_cagr = cagr(rev_series); eps_cagr = cagr(eps_series)

                # FCF & yield
                fcf_ttm = safe_float(km.get("freeCashFlowTTM"))
                mcap = safe_float(km.get("marketCapTTM")) or safe_float(qmap.get(sym, {}).get("marketCap"))
                fcf_yield = (fcf_ttm/mcap*100.0) if (fcf_ttm is not None and mcap and mcap > 0) else None

                # ROIC/Skuld
                roic = safe_float(km.get("roicTTM") or ttm.get("returnOnCapitalEmployedTTM"))
                dte  = safe_float(ttm.get("debtEquityRatioTTM"))
                icr  = safe_float(ttm.get("interestCoverageTTM"))
                cr   = safe_float(ttm.get("currentRatioTTM"))
                dscore = debt_safety_score(dte, icr, cr)

                # Historiska targets → basvaluta
                price_base = base_df.loc[base_df["Ticker"]==sym, f"Aktuell kurs ({base_ccy})"].values
                price_base = safe_float(price_base[0]) if len(price_base) else None
                list_ccy = base_df.loc[base_df["Ticker"]==sym, "Valuta (pris)"].values
                list_ccy = list_ccy[0] if len(list_ccy) and list_ccy[0] else None

                uv_native = intrinsic_from_hist_multiples(None, sps, eps, ps_avg, pe_avg)
                def _conv(v): return fx_convert(v, list_ccy, base_ccy) if (v is not None and list_ccy) else v
                tgt_ps_b = _conv(uv_native["Target via P/S"])
                tgt_pe_b = _conv(uv_native["Target via P/E"])
                tgt_cons = min([x for x in [tgt_ps_b, tgt_pe_b] if x is not None], default=None)
                def uv(p,t): return ((t - p)/t*100.0) if (p and t and t>0) else None

                rows.append({
                    "Ticker": sym,
                    "P/S TTM": ps_ttm, "P/E TTM": pe_ttm,
                    "EV/Sales TTM": ev_to_sales,
                    "EV/EBITDA TTM": ev_to_ebitda,
                    "EV/EBIT (proxy)": ev_over_ebit,
                    "Sales/Share TTM": sps, "EPS TTM": eps,
                    "P/S 5y snitt": ps_avg, "P/E 5y snitt": pe_avg,
                    "Revenue CAGR 5y (%)": round(rev_cagr*100,2) if rev_cagr is not None else None,
                    "EPS CAGR 5y (%)": round(eps_cagr*100,2) if eps_cagr is not None else None,
                    "FCF TTM": fcf_ttm, "FCF yield (%)": round(fcf_yield,2) if fcf_yield is not None else None,
                    "ROIC TTM (%)": roic,
                    "Debt/Equity TTM": dte, "Interest coverage TTM": icr, "Current ratio TTM": cr,
                    "Skuld-säkerhet (0-100)": dscore,
                    f"Target via P/S ({base_ccy})": tgt_ps_b,
                    f"Target via P/E ({base_ccy})": tgt_pe_b,
                    f"Target konservativ ({base_ccy})": tgt_cons,
                    "Undervärdering konservativ (%)": uv(price_base, tgt_cons),
                })
            except Exception as e:
                rows.append({"Ticker": sym, "Fel": str(e)})

            progress.progress(done/total)
            time.sleep(delay)

        feats = pd.DataFrame(rows)
        out = base_df.merge(feats, on="Ticker", how="left")

        # Placeholder-kolumner om saknas
        for col in ["Sektor","Industri","Land","Valuta"]:
            if col not in out.columns:
                out[col] = None

        # Peer stats och z-scores per industri
        grp = out.groupby("Industri", dropna=False)
        def peer_stats(col):
            g = grp[col].agg(["median","mean","std"]).rename(
                columns={"median":f"Peer {col} med","mean":f"Peer {col} mean","std":f"Peer {col} std"}
            )
            return g
        for metric in ["P/E TTM","EV/EBITDA TTM","EV/Sales TTM","P/S TTM"]:
            out = out.merge(peer_stats(metric), left_on="Industri", right_index=True, how="left")
            out[f"Peer z {metric}"] = (out[metric] - out[f"Peer {metric} mean"]) / out[f"Peer {metric} std"]

        # Peer-multiplar → peer-targets och peer-undervärdering
        out = out.merge(grp["P/S TTM"].median().rename("Peer PS TTM median"), left_on="Industri", right_index=True, how="left")
        out = out.merge(grp["P/E TTM"].median().rename("Peer PE TTM median"), left_on="Industri", right_index=True, how="left")
        def _peer_target(row):
            sps = safe_float(row.get("Sales/Share TTM")); eps = safe_float(row.get("EPS TTM"))
            ps_m = safe_float(row.get("Peer PS TTM median")); pe_m = safe_float(row.get("Peer PE TTM median"))
            list_ccy = row.get("Valuta (pris)") or row.get("Valuta")
            t_ps = (sps*ps_m) if (sps and ps_m) else None
            t_pe = (eps*pe_m) if (eps and pe_m) else None
            t_ps_b = fx_convert(t_ps, list_ccy, base_ccy) if (t_ps is not None and list_ccy) else t_ps
            t_pe_b = fx_convert(t_pe, list_ccy, base_ccy) if (t_pe is not None and list_ccy) else t_pe
            t_cons = min([x for x in [t_ps_b, t_pe_b] if x is not None], default=None)
            p_b = safe_float(row.get(f"Aktuell kurs ({base_ccy})"))
            uv = ((t_cons - p_b)/t_cons*100.0) if (p_b and t_cons and t_cons>0) else None
            return pd.Series({f"Peer Target konservativ ({base_ccy})": t_cons, "Peer Undervärdering konservativ (%)": uv})
        out = pd.concat([out, out.apply(_peer_target, axis=1)], axis=1)

        # Likviditet
        out = out[
            (out[f"Aktuell kurs ({base_ccy})"].fillna(0) >= min_price) &
            (out["AvgVol10d"].fillna(0) >= min_avgvol)
        ]

        # Quality gate
        if require_pos_fcf:
            out = out[out["FCF TTM"].fillna(-1) > 0]
        if min_roic > 0:
            out = out[out["ROIC TTM (%)"].fillna(0) >= min_roic]
        if min_icr > 0:
            out = out[out["Interest coverage TTM"].fillna(0) >= min_icr]

        # Rank (skala skuldbetygsvikt vid hög D/E)
        def scaled_rank(r):
            uv = r.get("Undervärdering konservativ (%)")
            gr1= r.get("Revenue CAGR 5y (%)"); gr2 = r.get("EPS CAGR 5y (%)")
            ds = r.get("Skuld-säkerhet (0-100)") or 0.0
            de = r.get("Debt/Equity TTM")
            debt_weight = w_deb
            if de is not None and de > de_cap:
                factor = max(0.0, 1.0 - (de - de_cap)/(2*de_cap))
                debt_weight = w_deb * factor
            total_w = w_val + w_gro + debt_weight
            wv = w_val/total_w; wg = w_gro/total_w; wd = debt_weight/total_w
            return composite_rank(uv, gr1, gr2, ds, weight_value=wv, weight_growth=wg, weight_debt=wd)
        out["Rank (0-100+)"] = out.apply(scaled_rank, axis=1)

        # ≥10% uppsida (historisk ELLER peer)
        cond_hist = out["Undervärdering konservativ (%)"].fillna(-999) >= 10
        cond_peer = out["Peer Undervärdering konservativ (%)"].fillna(-999) >= 10
        out = out[cond_hist | cond_peer]

        # Industri-rank
        out["Industri-rank (peer underv)"] = out.groupby("Industri")["Peer Undervärdering konservativ (%)"] \
                                                 .rank(ascending=False, method="min")

        out_sorted = out.sort_values(
            by=["Peer Undervärdering konservativ (%)","Rank (0-100+)"],
            ascending=[False, False]
        ).reset_index(drop=True)

        st.dataframe(
            out_sorted[
                ["Ticker","Bolag","Börs","Sektor","Industri",
                 f"Aktuell kurs ({base_ccy})","AvgVol10d",
                 "P/E TTM","P/S TTM","EV/Sales TTM","EV/EBITDA TTM","EV/EBIT (proxy)",
                 "P/S 5y snitt","P/E 5y snitt",
                 "FCF TTM","FCF yield (%)","ROIC TTM (%)",
                 "Debt/Equity TTM","Interest coverage TTM","Current ratio TTM","Skuld-säkerhet (0-100)",
                 f"Target konservativ ({base_ccy})","Undervärdering konservativ (%)",
                 f"Peer Target konservativ ({base_ccy})","Peer Undervärdering konservativ (%)",
                 "Peer z P/E TTM","Peer z EV/EBITDA TTM","Peer z EV/Sales TTM","Peer z P/S TTM",
                 "Revenue CAGR 5y (%)","EPS CAGR 5y (%)",
                 "Rank (0-100+)","Industri-rank (peer underv)"]
            ],
            use_container_width=True, height=600
        )

        # Spara picks (Top-N eller Top-K/industri cap N)
        st.markdown("**Spara urval som backtest-picks**")
        cA, cB, cC, cD = st.columns([1,1,1,2])
        mode_key = f"save_mode_{preset_name}"
        with cA:
            save_mode = st.radio("Urvalsmetod", ["Top-N totalt", "Top-K/industri (cap N)"], index=0, key=mode_key)
        if save_mode == "Top-N totalt":
            with cB:
                topn = st.number_input(f"Top-N ({preset_name})", 1, 500, 25, 1, key=f"topn_{preset_name}")
            with cD:
                if st.button(f"Spara Top-{int(topn)} ({preset_name})"):
                    tops = out_sorted.head(int(topn))["Ticker"].dropna().astype(str).tolist()
                    if not tops:
                        st.warning("Inga tickers att spara.")
                    else:
                        weights = [1.0/len(tops)]*len(tops)
                        picks_add(preset_name, base_ccy, tops, weights)
                        st.success(f"Sparade {len(tops)} picks (Top-{int(topn)}).")
        else:
            with cB:
                k_per = st.number_input("K per industri", 1, 20, 2, 1, key=f"kper_{preset_name}")
            with cC:
                capN = st.number_input("Total cap N", 1, 500, 30, 1, key=f"cap_{preset_name}")
            with cD:
                if st.button(f"Spara Top-{int(k_per)}/industri (cap {int(capN)}) — {preset_name}"):
                    picks = top_k_per_industry(out_sorted, int(k_per), int(capN))
                    if not picks:
                        st.warning("Inga tickers att spara (saknas Industri/Ticker eller tomt urval).")
                    else:
                        weights = [1.0/len(picks)]*len(picks)
                        picks_add(preset_name, base_ccy, picks, weights)
                        st.success(f"Sparade {len(picks)} picks (Top-{int(k_per)}/industri, cap {int(capN)}).")

        # CSV per preset
        st.download_button(f"Ladda ner CSV – {preset_name}",
                           out_sorted.to_csv(index=False).encode("utf-8"),
                           file_name=f"screen_{preset_name.replace(' ','_')}.csv",
                           mime="text/csv")

        # Lägg för Sheets-skrivning
        all_tabs.append((preset_name, out_sorted))

        progress.empty(); status.empty()

    # Skriv till Google Sheets – en flik per preset
    if sheet_url and all_tabs:
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            gc = gspread.authorize(Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope))
            sh = gc.open_by_url(sheet_url)
            preserve = ["Bolag","Sektor","Industri","Börs","Land","Valuta"]
            for (preset_name, df_tab) in all_tabs:
                tab_name = preset_name.split("–")[0].strip()[:25]
                titles = [w.title for w in sh.worksheets()]
                ws = sh.worksheet(tab_name) if tab_name in titles else sh.add_worksheet(tab_name, rows="2000", cols="50")
                existing = ws.get_all_records()
                if existing:
                    ex_df = pd.DataFrame(existing)
                    all_cols = list(dict.fromkeys(list(ex_df.columns) + list(df_tab.columns)))
                    ex_df = ex_df.reindex(columns=all_cols)
                    df_out= df_tab.reindex(columns=all_cols)
                    for c in preserve:
                        if c in ex_df.columns and c in df_out.columns:
                            df_out[c] = df_out[c].combine_first(ex_df[c])
                    if only_new:
                        df_out = df_out.where(df_out.notna() & (df_out != ""), ex_df)
                else:
                    df_out = df_tab
                ws.clear()
                ws.update([df_out.columns.tolist()] + df_out.astype(str).fillna("").values.tolist())
            st.success("Sparat till Google Sheets (en flik per preset).")
        except Exception as e:
            st.warning(f"Google Sheets-fel: {e}")


st.divider()

# ----------------------------- 3) Backtest -----------------------------
st.subheader("3) Backtest – utvärdera sparade Top-N/Top-K-picks (utan grafer)")
bt_df = picks_fetch()
if bt_df.empty:
    st.caption("Inga sparade picks ännu. Spara Top-N eller Top-K/industri i panelen ovan efter en körning.")
else:
    st.caption("Välj snapshots att utvärdera. Avkastning approx. med justerade stängningar (utan utdelningar).")
    show_df = bt_df.head(200).copy()
    show_df["tickers_count"] = show_df["tickers"].apply(lambda s: len(json.loads(s)) if s else 0)
    st.dataframe(show_df[["id","created_utc","preset_name","base_ccy","tickers_count"]], use_container_width=True, height=240)

    ids_str = st.text_input("Ange id:n (kommaseparerade) att backtesta", value=str(show_df["id"].head(5).tolist())[1:-1])
    horizons = st.multiselect("Horisont (mån)", [1,3,6,12], default=[1,3,6,12])
    rebalance_monthly = st.checkbox("Månadsvis ombalansering (lika vikt)", value=False)

    def _load_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            px = data["Close"].copy()
        else:
            px = data.copy()
        if isinstance(px, pd.Series):
            px = px.to_frame()
        return px

    def _portfolio_return(px: pd.DataFrame, date0, date1, weights: list):
        try:
            p0 = px.loc[px.index.get_loc(date0, method="nearest")]
            p1 = px.loc[px.index.get_loc(date1, method="nearest")]
        except Exception:
            try:
                p0 = px.iloc[0]; p1 = px.iloc[-1]
            except Exception:
                return None
        common = p0.dropna().index.intersection(p1.dropna().index)
        if len(common) == 0:
            return None
        w = pd.Series(weights, index=px.columns).reindex(common).fillna(0.0)
        if w.sum() == 0:
            w = pd.Series([1.0/len(common)]*len(common), index=common)
        r = (p1[common].values / p0[common].values) - 1.0
        return float((pd.Series(r, index=common) * w).sum())

    def _portfolio_return_rebalanced_monthly(px: pd.DataFrame, date0, months: int):
        try:
            end_date = pd.to_datetime(date0) + pd.DateOffset(months=int(months))
            sub = px.loc[(px.index >= (pd.to_datetime(date0) - pd.Timedelta(days=3))) & (px.index <= (end_date + pd.Timedelta(days=3)))]
            if sub.empty:
                return None
            first_row = sub.iloc[[0]]
            mclose = sub.resample("M").last()
            mclose = pd.concat([first_row, mclose])
            mclose = mclose[mclose.index <= end_date + pd.Timedelta(days=3)]
            rets = []
            for i in range(1, len(mclose)):
                p0 = mclose.iloc[i-1]; p1 = mclose.iloc[i]
                common = p0.dropna().index.intersection(p1.dropna().index)
                if len(common) == 0:
                    continue
                w = pd.Series([1.0/len(common)]*len(common), index=common)
                r = (p1[common].values / p0[common].values) - 1.0
                rets.append(float((pd.Series(r, index=common) * w).sum()))
            if not rets:
                return None
            cum = 1.0
            for r in rets:
                cum *= (1.0 + r)
            return cum - 1.0
        except Exception:
            return None

    if st.button("Kör backtest"):
        if not ids_str.strip():
            st.warning("Ange minst ett id.")
        else:
            ids = []
            for x in ids_str.replace(" ", "").split(","):
                try:
                    ids.append(int(x))
                except Exception:
                    pass
            sel = bt_df[bt_df["id"].isin(ids)].copy()
            if sel.empty:
                st.warning("Inga matchande id:n.")
            else:
                rows = []
                for _, snap in sel.iterrows():
                    created = pd.to_datetime(snap["created_utc"])
                    region  = preset_to_region(snap["preset_name"])
                    bench   = BENCH_ETF.get(region, "SPY")
                    tickers = json.loads(snap["tickers"])
                    weights = json.loads(snap["weights"])
                    start = (created - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
                    end   = (created + pd.Timedelta(days=380)).strftime("%Y-%m-%d")
                    try:
                        px = _load_prices(tickers, start, end)
                        bx = _load_prices([bench], start, end)
                    except Exception as e:
                        st.warning(f"Prisfel för snapshot {snap['id']}: {e}")
                        continue
                    for m in horizons:
                        d1 = created + pd.DateOffset(months=m)
                        if rebalance_monthly:
                            pr = _portfolio_return_rebalanced_monthly(px[tickers], created, m)
                        else:
                            pr = _portfolio_return(px[tickers], created, d1, weights)
                        br = _portfolio_return(bx[[bench]], created, d1, [1.0])
                        rows.append({
                            "id": snap["id"],
                            "Datum": created.date().isoformat(),
                            "Preset": snap["preset_name"],
                            "Region": region,
                            "Benchmark": bench,
                            "Horisont (mån)": m,
                            "Portfölj (%)": None if pr is None else round(pr*100, 2),
                            "Benchmark (%)": None if br is None else round(br*100, 2),
                            "Alfa (p.p.)": None if (pr is None or br is None) else round((pr - br)*100, 2),
                            "Antal innehav": len(tickers)
                        })
                resbt = pd.DataFrame(rows).sort_values(["Datum","Preset","Horisont (mån)"])
                if resbt.empty:
                    st.info("Ingen backtest-resultat kunde beräknas (saknade priser?).")
                else:
                    st.dataframe(resbt, use_container_width=True, height=450)
                    st.download_button("Ladda ner backtest-resultat (CSV)",
                                       resbt.to_csv(index=False).encode("utf-8"),
                                       file_name="backtest_resultat.csv",
                                       mime="text/csv")
