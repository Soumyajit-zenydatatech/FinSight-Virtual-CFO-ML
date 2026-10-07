"""
POST /predict/{model_id}
  – Accepts a date range (start_date → end_date) + one snapshot of external
    factor values.  Internally expands the range into monthly rows, applies
    the same external factors to every month, runs the model, and returns
    the full monthly breakdown so the frontend can render a forecast chart /
    map without any extra processing.
"""

import io
import os
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests as http_requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from auth import get_current_user
from config import settings
from db.main_session import get_main_db
from db.models import FileUpload, IngestionBatch, TrainedModel
from db.session import get_db
from ml.dates import parse_dates
from ml.ext_factors import load_ext_factors, merge_ext_factors
from ml.ext_forecaster import get_ext_forecaster
from ml.predictor import predict
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictRow,
    BatchPredictSummary,
    BatchPredictRowEnhanced,
    BatchPredictSummaryEnhanced,
    BatchPredictResponseEnhanced,
    ForecastRequest,
    ForecastResponse,
    ForecastSummary,
    MonthlyForecast,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _monthly_dates(start: date, end: date) -> list[date]:
    """Return the 1st of every month from start's month to end's month inclusive."""
    dates = []
    current = start.replace(day=1)
    end_anchor = end.replace(day=1)
    while current <= end_anchor:
        dates.append(current)
        month = current.month + 1
        year = current.year
        if month > 12:
            month = 1
            year += 1
        current = current.replace(year=year, month=month)
    return dates


@router.post(
    "/{model_id}",
    response_model=ForecastResponse,
    summary="Date-range forecast: monthly predictions from start to end date",
    description=(
        "Provide a start/end date range plus optional cost-component defaults "
        "and a single external-factor snapshot (CCI, CPI, Oil, GDP, "
        "Unemployment, ROI).  The API generates one input row per month, "
        "applies the same external factors to every month as a constant "
        "baseline, runs the model, and returns the full monthly breakdown "
        "ready for frontend charting or a forecast map."
    ),
)
def run_prediction(
    model_id: int,
    payload: ForecastRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    # ── Parse dates ───────────────────────────────────────────────────────────
    try:
        start = datetime.strptime(payload.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(payload.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="Dates must be in YYYY-MM-DD format.")

    if end < start:
        raise HTTPException(status_code=422, detail="end_date must be >= start_date.")

    months = _monthly_dates(start, end)
    if not months:
        raise HTTPException(status_code=422, detail="Date range produced no months.")

    # ── Build one input row per month ─────────────────────────────────────────
    # The external factors are constant (latest snapshot) across all months.
    # Cost components default to whatever the frontend supplied (or None → NaN).
    base_row = {
        "Region":        payload.Region,
        "Geo":           payload.Geo,
        "Country":       payload.Country,
        "Item type":     payload.Item_type,
        "Customer":      payload.Customer,
        "Raw Material":  payload.Raw_Material,
        "Direct Labor":  payload.Direct_Labor,
        "Freight":       payload.Freight,
        "Storage":       payload.Storage,
        "Packaging":     payload.Packaging,
        "Indirect Labor": payload.Indirect_Labor,
        "Rent & Utility": payload.Rent_Utility,
        "Overhead":      payload.Overhead,
        "CCI":           payload.CCI,
        "CPI":           payload.CPI,
        "Oil":           payload.Oil,
        "GDP":           payload.GDP,
        "Unemployment":  payload.Unemployment,
        "ROI":           payload.ROI,
    }

    input_rows = [
        {**base_row, "order_date": m.isoformat()}
        for m in months
    ]

    # ── Run model ─────────────────────────────────────────────────────────────
    try:
        results = predict(tm.model_file_path, input_rows)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model file not found on disk. It may have been deleted manually.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # ── Build monthly forecast list ───────────────────────────────────────────
    monthly_predictions: list[MonthlyForecast] = []
    total_revenue = total_cogs = total_sga = 0.0

    for m, row in zip(months, results):
        rev  = row.get("predicted_total_revenue") or 0.0
        cogs = row.get("predicted_COGS") or 0.0
        sga  = row.get("predicted_SGA") or 0.0

        total_revenue += rev
        total_cogs    += cogs
        total_sga     += sga

        monthly_predictions.append(
            MonthlyForecast(
                month=m.strftime("%Y-%m"),
                date=m.isoformat(),
                predicted_total_revenue=row.get("predicted_total_revenue"),
                predicted_COGS=row.get("predicted_COGS"),
                predicted_SGA=row.get("predicted_SGA"),
                model_used_revenue=row.get("model_used_revenue"),
                model_used_COGS=row.get("model_used_COGS"),
                model_used_SGA=row.get("model_used_SGA"),
            )
        )

    summary = ForecastSummary(
        total_revenue=round(total_revenue, 4),
        total_COGS=round(total_cogs, 4),
        total_SGA=round(total_sga, 4),
        months_count=len(months),
    )

    return ForecastResponse(
        model_id=tm.id,
        model_name=tm.model_name,
        start_date=payload.start_date,
        end_date=payload.end_date,
        monthly_predictions=monthly_predictions,
        summary=summary,
    )


# ── Column mapping: SL CSV column names → model input names ──────────────────
_SL_RENAME = {
    "Order Date": "order_date",
    "order_date": "order_date",
    "Item type": "Item_type",
    "Item Type": "Item_type",
    "Raw Material": "Raw_Material",
    "Direct Labor": "Direct_Labor",
    "Indirect Labor": "Indirect_Labor",
    "Rent & Utility": "Rent_Utility",
}


def _resolve_file_url(file_path: str) -> str:
    """
    Convert a Supabase Storage relative path to a full public URL.
    If it's already a full URL, return as-is.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return file_path
    supabase_url = settings.SUPABASE_URL.rstrip("/")
    bucket = "File%20Storage"
    return f"{supabase_url}/storage/v1/object/public/{bucket}/{file_path}"


def _clean_sl_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning as load_financial_csv() in preprocessor.py:
    - Strip $, commas, whitespace from all columns
    - Convert numeric columns to float
    - Keep categorical and date columns as-is
    """
    KEEP_AS_STR = {"Region", "Geo", "Country", "Item type", "Customer",
                   "Order Date", "order_date"}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col in KEEP_AS_STR:
            continue
        df[col] = (
            df[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_sl_csv(file_path: str) -> pd.DataFrame:
    """
    Load the SUBLEDGER CSV from a Supabase Storage path or full URL.
    Returns a DataFrame with original column names intact.
    """
    url = _resolve_file_url(file_path)
    resp = http_requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content))


@router.post(
    "/from-batch/{model_id}",
    response_model=BatchPredictResponseEnhanced,
    summary="Predict from batch SUBLEDGER — actual vs predicted, with optional filters and future forecast",
    description=(
        "Reads the SUBLEDGER CSV for the given batch, merges external factors by month, "
        "and runs predictions on every row. "
        "Optional filters: country, region, geo (single values, case-insensitive). "
        "Optional date filters: prediction_start (show CSV rows from this date onward), "
        "prediction_end (also generate synthetic future monthly rows beyond the CSV). "
        "Each row in the response shows the actual values from the CSV alongside the "
        "predicted values. Future rows (beyond the CSV date range) show predicted values only. "
        "Summary includes total actual revenue/gross profit vs total predicted."
    ),
)
def predict_from_batch(
    model_id: int,
    payload: BatchPredictRequest,
    # ── Optional filters as query params ──────────────────────────────────────
    prediction_start: Optional[str] = None,
    prediction_end:   Optional[str] = None,
    show_all:         bool = False,
    db: Session = Depends(get_db),
    main_db: Session = Depends(get_main_db),
    current_user=Depends(get_current_user),
):
    try:
     return _predict_from_batch_inner(model_id, payload, prediction_start, prediction_end, show_all, db, main_db, current_user)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


def _predict_from_batch_inner(model_id, payload, prediction_start, prediction_end, show_all, db, main_db, current_user):
    # ── Validate model ────────────────────────────────────────────────────────
    tm = db.query(TrainedModel).get(model_id)
    if not tm:
        raise HTTPException(status_code=404, detail="Model not found.")

    # ── Validate date params ──────────────────────────────────────────────────
    pred_start_dt = pred_end_dt = None

    # ── Look up batch + SUBLEDGER file (Supabase) ────────────────────────────
    batch = main_db.query(IngestionBatch).filter(
        IngestionBatch.id == payload.batch_id,
        IngestionBatch.company_id == current_user.company_id,
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {payload.batch_id} not found.")

    sl_upload = main_db.query(FileUpload).filter(
        FileUpload.batch_id == payload.batch_id,
        FileUpload.file_type == "SUBLEDGER",
    ).first()
    if not sl_upload or not sl_upload.file_path:
        raise HTTPException(status_code=404, detail="No SUBLEDGER file found for this batch.")

    # ── Load + clean CSV ──────────────────────────────────────────────────────
    try:
        df = _load_sl_csv(sl_upload.file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read SL file: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="SUBLEDGER CSV is empty.")

    df = _clean_sl_df(df)
    df = df.rename(columns=_SL_RENAME)

    if "order_date" not in df.columns:
        raise HTTPException(status_code=422, detail="SUBLEDGER CSV must have an 'Order Date' column.")

    # ── Parse order_date ──────────────────────────────────────────────────────
    df["order_date"] = parse_dates(df["order_date"])
    df = df.dropna(subset=["order_date"])

    # Capture the full-CSV last date BEFORE any filters so future generation
    # is only triggered when date_to is truly beyond the entire dataset.
    full_csv_last_date = df["order_date"].max().date()

    # ── Apply filters (case-insensitive) ──────────────────────────────────────
    if payload.region:
        df = df[df["Region"].astype(str).str.strip().str.lower() == payload.region.strip().lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail="No rows found after applying filters: Region")
    if payload.geo:
        df = df[df["Geo"].astype(str).str.strip().str.lower() == payload.geo.strip().lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail="No rows found after applying filters: Geo")
    if payload.country:
        df = df[df["Country"].astype(str).str.strip().str.lower() == payload.country.strip().lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail="No rows found after applying filters: Country.")

    if df.empty:
        raise HTTPException(status_code=404, detail="No rows found after applying filters.")

    # Keep full filtered df (before date range) for combo/frequency detection
    df_full_for_combos = df.copy()

    # ── Date range filter (only when show_all is False) ───────────────────────
    if not show_all:
        if prediction_start:
            try:
                pred_start_dt = datetime.strptime(prediction_start, "%Y-%m-%d")
                df = df[df["order_date"] >= pred_start_dt]
            except ValueError:
                raise HTTPException(status_code=422, detail="prediction_start must be YYYY-MM-DD.")
        if prediction_end:
            try:
                df_to_dt = datetime.strptime(prediction_end, "%Y-%m-%d")
                df = df[df["order_date"] <= df_to_dt]
            except ValueError:
                raise HTTPException(status_code=422, detail="prediction_end must be YYYY-MM-DD.")
        # If date range yields no historical rows it's okay — future rows may still be generated

    # ── prediction_end also drives future generation if beyond full CSV range ──
    if prediction_end:
        try:
            pred_end_dt = datetime.strptime(prediction_end, "%Y-%m-%d").date()
        except ValueError:
            pass


    # ── Stash actual values before merging external factors ───────────────────
    ACTUAL_COLS = {
        "Total Revenue": "actual_total_revenue",
        "COGS":          "actual_COGS",
        "SG&A":          "actual_SGA",
        "Gross Profit":  "actual_gross_profit",
    }
    for src, dst in ACTUAL_COLS.items():
        df[dst] = pd.to_numeric(df.get(src, pd.Series(dtype=float)), errors="coerce") if src in df.columns else float("nan")

    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")

    # ── Merge external factors (historical) ───────────────────────────────────
    ext_df = load_ext_factors(settings.MODEL_STORE_DIR)
    if ext_df is not None:
        df, ext_msg = merge_ext_factors(df, ext_df)
        ext_msg += " Future months beyond ext factor range use linear trend projection."
    else:
        ext_msg = "No external_factors.csv in model_store — model used training medians."

    # ── Payload ext factors used for future rows only (historical keeps real values) ──
    EXT_COLS = ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"]
    payload_overrides = {c: getattr(payload, c) for c in EXT_COLS if getattr(payload, c, None) not in (None, 0)}

    # ── Predict historical rows (skip if date range yields no historical rows) ──
    hist_input = df.to_dict(orient="records") if not df.empty else []
    hist_results = []
    if hist_input:
        try:
            hist_results = predict(tm.model_file_path, hist_input)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Model .pkl file not found on disk.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # ── Build historical prediction rows ──────────────────────────────────────
    predictions: list[BatchPredictRowEnhanced] = []
    total_actual_revenue = total_actual_gp = 0.0
    total_pred_revenue = total_pred_gp = 0.0

    for raw_row, res in zip(hist_input, hist_results):
        pred_rev  = res.get("predicted_total_revenue") or 0.0
        pred_cogs = res.get("predicted_COGS") or 0.0
        pred_gp   = pred_rev - pred_cogs

        act_rev  = raw_row.get("actual_total_revenue")
        act_cogs = raw_row.get("actual_COGS")
        act_sga  = raw_row.get("actual_SGA")
        act_gp   = raw_row.get("actual_gross_profit")

        total_actual_revenue += act_rev if act_rev and not pd.isna(act_rev) else 0.0
        total_actual_gp      += act_gp  if act_gp  and not pd.isna(act_gp)  else 0.0
        total_pred_revenue   += pred_rev
        total_pred_gp        += pred_gp

        predictions.append(BatchPredictRowEnhanced(
            order_date=str(raw_row.get("order_date", "")),
            row_type="historical",
            region=str(raw_row.get("Region") or ""),
            geo=str(raw_row.get("Geo") or ""),
            country=str(raw_row.get("Country") or ""),
            item_type=str(raw_row.get("Item_type") or ""),
            customer=str(raw_row.get("Customer") or ""),
            actual_total_revenue=None if (act_rev is None or (isinstance(act_rev, float) and pd.isna(act_rev))) else round(act_rev, 4),
            actual_COGS=None if (act_cogs is None or (isinstance(act_cogs, float) and pd.isna(act_cogs))) else round(act_cogs, 4),
            actual_SGA=None if (act_sga is None or (isinstance(act_sga, float) and pd.isna(act_sga))) else round(act_sga, 4),
            actual_gross_profit=None if (act_gp is None or (isinstance(act_gp, float) and pd.isna(act_gp))) else round(act_gp, 4),
            predicted_total_revenue=round(pred_rev, 4),
            predicted_COGS=round(pred_cogs, 4),
            predicted_SGA=round(res.get("predicted_SGA") or 0.0, 4),
            predicted_gross_profit=round(pred_gp, 4),
            model_used_revenue=res.get("model_used_revenue"),
            model_used_COGS=res.get("model_used_COGS"),
            model_used_SGA=res.get("model_used_SGA"),
        ))

    hist_count = len(predictions)

    # ── Generate future rows if prediction_end is beyond last CSV date ─────────
    future_count = 0
    ext_forecast_out: list = []   # macro path assumed per future month (for the response)
    if pred_end_dt:
        last_csv_date = full_csv_last_date

        # Future start = prediction_start if given and it's beyond CSV, else next month after CSV
        if pred_start_dt and pred_start_dt.date() > last_csv_date:
            future_gen_start = pred_start_dt.date().replace(day=1)
        else:
            # Next month after last CSV date
            m = last_csv_date.month + 1
            y = last_csv_date.year + (1 if m > 12 else 0)
            m = 1 if m > 12 else m
            future_gen_start = date(y, m, 1)

        if pred_end_dt >= future_gen_start:
            future_months = _monthly_dates(future_gen_start, pred_end_dt)
        else:
            future_months = []

        if future_months:
                COST_COLS = ["Raw_Material", "Direct_Labor", "Freight", "Storage",
                             "Packaging", "Indirect_Labor", "Rent_Utility", "Overhead"]
                combo_cols = ["Region", "Geo", "Country", "Customer", "Item_type"]
                # Use full filtered df (ignoring date range) for combos/frequency/cost medians
                df_num = df_full_for_combos.copy()
                for c in COST_COLS:
                    if c in df_num.columns:
                        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

                # Real ext factor values by month (from file); payload overrides win
                ext_lookup: dict = {}
                if ext_df is not None:
                    for _, er in ext_df.sort_values("year_month").iterrows():
                        ext_lookup[str(er["year_month"])] = {c: er[c] for c in EXT_COLS if c in er}

                # ── Stage 1: forecast ext factors for months not in the file ──────
                # Fits one time-series model per factor on the file's history
                # (cached by file mtime) and predicts each requested future month.
                future_periods = [pd.Period(d, "M") for d in future_months]
                missing_periods = [p for p in future_periods if str(p) not in ext_lookup]
                ext_forecast_lookup: dict = {}
                if missing_periods and ext_df is not None:
                    forecaster = get_ext_forecaster(settings.MODEL_STORE_DIR)
                    if forecaster is not None:
                        fc_df = forecaster.predict(missing_periods)
                        for _, fr in fc_df.iterrows():
                            ext_forecast_lookup[str(fr["year_month"])] = fr.to_dict()
                        ext_msg = ext_msg.replace(
                            "use linear trend projection.",
                            f"forecast with {forecaster.describe()}.",
                        )

                # Per-combo: record which calendar months had data + avg order count per month
                _od = parse_dates(df_num["order_date"])
                df_num["_order_month"] = _od.dt.month
                df_num["_order_year"]  = _od.dt.year
                combo_month_map: dict = {}   # combo_key → set of calendar months
                combo_freq_map:  dict = {}   # (combo_key, month) → avg rows per year-month
                for _, row in df_num.iterrows():
                    key = tuple(str(row.get(c, "")) for c in combo_cols if c in df_num.columns)
                    m = int(row["_order_month"]) if not pd.isna(row["_order_month"]) else 0
                    combo_month_map.setdefault(key, set()).add(m)
                # Compute average order count per (combo, calendar-month) across years
                for combo_key in combo_month_map:
                    mask0 = pd.Series([True] * len(df_num))
                    for ci, c in enumerate(combo_cols):
                        if c in df_num.columns:
                            mask0 = mask0 & (df_num[c].astype(str) == combo_key[ci])
                    combo_rows = df_num[mask0]
                    for m in combo_month_map[combo_key]:
                        month_rows = combo_rows[combo_rows["_order_month"] == m]
                        # Group by year to get count per year-month occurrence, then average
                        counts_per_ym = month_rows.groupby("_order_year").size()
                        avg_count = round(counts_per_ym.mean()) if len(counts_per_ym) > 0 else 1
                        combo_freq_map[(combo_key, m)] = max(1, int(avg_count))

                future_input_rows = []
                _ext_used_by_month: dict = {}   # month → macro values actually fed to the model

                for future_date in future_months:
                    period_key = str(pd.Period(future_date, "M"))
                    if period_key in ext_lookup:
                        # Real data exists for this month
                        base_ext = dict(ext_lookup[period_key])
                        ext_source = "file"
                    elif period_key in ext_forecast_lookup:
                        # Stage 1 output: forecasted macro values for this month
                        fr = ext_forecast_lookup[period_key]
                        base_ext = {c: fr[c] for c in EXT_COLS if c in fr}
                        ext_source = "forecast"
                    else:
                        base_ext = {}
                        ext_source = "none"
                    # Payload overrides always win (user-supplied values take priority)
                    base_ext.update(payload_overrides)

                    # Record the macro path assumed for this month (once per month)
                    if period_key not in _ext_used_by_month:
                        entry = {"month": period_key, "source": ext_source}
                        for c in EXT_COLS:
                            entry[c] = float(base_ext[c]) if c in base_ext and pd.notna(base_ext[c]) else None
                            if ext_source == "forecast":
                                fr = ext_forecast_lookup[period_key]
                                entry[f"{c}_low"]  = float(fr[f"{c}_low"])  if f"{c}_low"  in fr and pd.notna(fr[f"{c}_low"])  else None
                                entry[f"{c}_high"] = float(fr[f"{c}_high"]) if f"{c}_high" in fr and pd.notna(fr[f"{c}_high"]) else None
                        _ext_used_by_month[period_key] = entry

                    for _, combo in df_num[[c for c in combo_cols if c in df_num.columns]].drop_duplicates().iterrows():
                        combo_key = tuple(str(combo.get(c, "")) for c in combo_cols if c in df_num.columns)
                        # Only generate a row if this combo historically had data in this calendar month
                        if future_date.month not in combo_month_map.get(combo_key, set()):
                            continue

                        # How many orders did this combo typically place in this calendar month?
                        repeat = combo_freq_map.get((combo_key, future_date.month), 1)

                        mask = pd.Series([True] * len(df_num))
                        for c in combo_cols:
                            if c in df_num.columns and c in combo.index:
                                mask = mask & (df_num[c].astype(str) == str(combo[c]))

                        combo_df = df_num[mask].copy()
                        combo_df["_order_date_dt"] = parse_dates(combo_df["order_date"])
                        combo_df_sorted = combo_df.sort_values("_order_date_dt")
                        cost_features = {}
                        for c in COST_COLS:
                            if c in combo_df_sorted.columns:
                                last_val = combo_df_sorted[c].dropna().iloc[-1] if combo_df_sorted[c].dropna().shape[0] > 0 else None
                                cost_features[c] = float(last_val) if last_val is not None else None

                        for _ in range(repeat):
                         future_row = {
                             "order_date": future_date.isoformat(),
                             "Region": combo.get("Region", ""),
                             "Geo": combo.get("Geo", ""),
                             "Country": combo.get("Country", ""),
                             "Item_type": combo.get("Item_type", ""),
                             "Customer": combo.get("Customer", ""),
                             **cost_features,
                             **base_ext,
                         }
                         future_input_rows.append(future_row)

                ext_forecast_out = list(_ext_used_by_month.values())

                if future_input_rows:
                    try:
                        future_results = predict(tm.model_file_path, future_input_rows)
                    except Exception as exc:
                        raise HTTPException(status_code=500, detail=f"Future prediction error: {exc}")

                    for f_raw, f_res in zip(future_input_rows, future_results):
                        pred_rev  = f_res.get("predicted_total_revenue") or 0.0
                        pred_cogs = f_res.get("predicted_COGS") or 0.0
                        pred_gp   = pred_rev - pred_cogs

                        total_pred_revenue += pred_rev
                        total_pred_gp      += pred_gp

                        predictions.append(BatchPredictRowEnhanced(
                            order_date=str(f_raw.get("order_date", "")),
                            row_type="future",
                            region=str(f_raw.get("Region") or ""),
                            geo=str(f_raw.get("Geo") or ""),
                            country=str(f_raw.get("Country") or ""),
                            item_type=str(f_raw.get("Item_type") or ""),
                            customer=str(f_raw.get("Customer") or ""),
                            actual_total_revenue=None,
                            actual_COGS=None,
                            actual_gross_profit=None,
                            predicted_total_revenue=round(pred_rev, 4),
                            predicted_COGS=round(pred_cogs, 4),
                            predicted_SGA=round(f_res.get("predicted_SGA") or 0.0, 4),
                            predicted_gross_profit=round(pred_gp, 4),
                            model_used_revenue=f_res.get("model_used_revenue"),
                            model_used_COGS=f_res.get("model_used_COGS"),
                            model_used_SGA=f_res.get("model_used_SGA"),
                        ))
                        future_count += 1

    return BatchPredictResponseEnhanced(
        model_id=tm.id,
        model_name=tm.model_name,
        batch_id=payload.batch_id,
        sl_file_path=sl_upload.file_path,
        filters_applied={
            "country": payload.country,
            "region": payload.region,
            "geo": payload.geo,
            "prediction_start": prediction_start,
            "prediction_end": prediction_end,

            "show_all": str(show_all),
        },
        predictions=predictions,
        summary=BatchPredictSummaryEnhanced(
            historical_row_count=hist_count,
            future_row_count=future_count,
            total_row_count=hist_count + future_count,
            gross_actual_revenue=round(total_actual_revenue, 4),
            total_actual_gross_profit=round(total_actual_gp, 4),
            gross_predicted_revenue=round(total_pred_revenue, 4),
            total_predicted_gross_profit=round(total_pred_gp, 4),
        ),
        external_factors_info=ext_msg,
        external_factors_forecast=ext_forecast_out,
    )
