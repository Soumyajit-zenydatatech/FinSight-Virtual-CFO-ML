"""
Format-safe date parsing shared by prediction, heatmap and ext-factor code.

Two date formats exist in this project:
  * external_factors.csv (model_store)   -> ISO  "YYYY-MM-DD"
  * SUBLEDGER CSVs from Supabase Storage -> "DD-MM-YYYY" (also seen with "/")

pandas' `dayfirst=True` silently mis-reads ISO strings ("2021-01-09" -> 1 Sep 2021),
so each value is routed to an explicit format based on its shape.
"""

from __future__ import annotations

import pandas as pd

_ISO      = r"^\d{4}-\d{1,2}-\d{1,2}"
_DMY_DASH = r"^\d{1,2}-\d{1,2}-\d{4}"
_DMY_SLSH = r"^\d{1,2}/\d{1,2}/\d{4}"


def parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse a Series of date strings where ISO and day-first formats may be mixed.
    Unparseable values become NaT.
    """
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    iso = s.str.match(_ISO)
    if iso.any():
        out[iso] = pd.to_datetime(s[iso].str.slice(0, 10), format="%Y-%m-%d", errors="coerce")

    dmy = s.str.match(_DMY_DASH)
    if dmy.any():
        out[dmy] = pd.to_datetime(s[dmy].str.slice(0, 10), format="%d-%m-%Y", errors="coerce")

    dmy2 = s.str.match(_DMY_SLSH)
    if dmy2.any():
        out[dmy2] = pd.to_datetime(s[dmy2].str.slice(0, 10), format="%d/%m/%Y", errors="coerce")

    # Anything else (e.g. "Jan 2024", full timestamps): best effort, day-first
    rest = out.isna() & ~s.isin(["", "nan", "None", "NaT"])
    if rest.any():
        try:
            out[rest] = pd.to_datetime(s[rest], format="mixed", dayfirst=True, errors="coerce")
        except (TypeError, ValueError):
            out[rest] = pd.to_datetime(s[rest], dayfirst=True, errors="coerce")

    return out
