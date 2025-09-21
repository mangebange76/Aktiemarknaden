from typing import List
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


def _client():
    import streamlit as st
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(credentials)


def open_ws(sheet_url: str, sheet_name: str):
    sh = _client().open_by_url(sheet_url)
    titles = [w.title for w in sh.worksheets()]
    return sh.worksheet(sheet_name) if sheet_name in titles else sh.add_worksheet(sheet_name, rows="2000", cols="50")


def read_df(sheet_url: str, sheet_name: str) -> pd.DataFrame:
    ws = open_ws(sheet_url, sheet_name)
    rows = ws.get_all_records()
    return pd.DataFrame(rows)


def write_df(df: pd.DataFrame, sheet_url: str, sheet_name: str,
             preserve_cols: List[str] = None,
             only_new_fields: bool = False):
    ws = open_ws(sheet_url, sheet_name)
    existing = ws.get_all_records()
    if existing:
        ex_df = pd.DataFrame(existing)
        # Align kolumner
        all_cols = list(dict.fromkeys(list(ex_df.columns) + list(df.columns)))
        ex_df = ex_df.reindex(columns=all_cols)
        df    = df.reindex(columns=all_cols)
        # Bevara manuellt valda kolumner där df saknar värde
        if preserve_cols:
            for col in preserve_cols:
                if col in ex_df.columns and col in df.columns:
                    df[col] = df[col].combine_first(ex_df[col])
        if only_new_fields:
            # Behåll alla befintliga värden där df är NaN/"" (per cell)
            df = df.where(df.notna() & (df != ""), ex_df)
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(str).fillna("").values.tolist())
