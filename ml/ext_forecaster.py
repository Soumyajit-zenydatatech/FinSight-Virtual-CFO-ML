"""
External Factor Forecaster
==========================
Stage 1 of future prediction: forecast the macro inputs (CCI, CPI, Oil, GDP,
Unemployment, ROI) for months that are not yet in model_store/external_factors.csv.
Stage 2 (the Ridge revenue model) then consumes those values as features.

One univariate time-series model is fitted per factor on the monthly history:

  * SARIMA  – statsmodels SARIMAX with a small order search, seasonal period 12.
              Used when the series has >= 36 months.
  * Holt-Winters – additive trend (damped) + additive 12-month seasonality.
              Used when the series has 24..35 months, or if SARIMA fails.
  * Linear  – last-resort straight line when fewer than 24 months exist.

Results are cached per (file path, file mtime) so a request never refits unless
the external factors file changed (e.g. via the heatmap append / replace APIs).
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.ext_factors import EXT_FACTOR_COLS, EXT_FACTORS_FILENAME, load_ext_factors

SEASONAL_PERIOD = 12
MIN_MONTHS_SARIMA = 36
MIN_MONTHS_HW = 24
INTERVAL_ALPHA = 0.20   # 80% prediction interval
SANITY_HORIZON = 36     # months ahead checked for divergence when picking a model

# Small SARIMA grid; enough for smooth macro series, cheap to search.
# Seasonal differencing (D=1) is deliberately avoided: with ~5 years of data it
# leaves too few effective observations and diverges easily.
_SARIMA_ORDERS: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = [
    ((1, 1, 0), (1, 0, 0, SEASONAL_PERIOD)),
    ((0, 1, 1), (1, 0, 0, SEASONAL_PERIOD)),
    ((1, 0, 0), (1, 0, 0, SEASONAL_PERIOD)),
    ((0, 1, 1), (0, 0, 0, 0)),
]

# Silence optimiser chatter from statsmodels; a non-converged fit is handled by
# the plausibility check below, not by the user reading a warning.
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=ValueWarning)
except Exception:  # statsmodels missing → forecaster degrades to linear
    pass


def _plausible(forecast: np.ndarray, history: np.ndarray) -> bool:
    """
    A forecast is accepted only if it stays within a generous band around the
    observed history: [min - 2*range, max + 2*range]. Rejects the exploding fits
    that SARIMAX can produce on short series.
    """
    if forecast.size == 0 or not np.all(np.isfinite(forecast)):
        return False
    lo, hi = float(np.nanmin(history)), float(np.nanmax(history))
    rng = max(hi - lo, abs(hi) * 0.05, 1e-9)
    return bool(np.all(forecast >= lo - 2 * rng) and np.all(forecast <= hi + 2 * rng))


@dataclass
class _FactorModel:
    kind: str                       # "sarima" | "holt_winters" | "linear" | "constant"
    last_period: pd.Period
    model: object = None            # fitted statsmodels results, or (slope, intercept)
    fallback_value: float = float("nan")


@dataclass
class ExtFactorForecaster:
    """Fit once on the ext factors history; predict any list of future periods."""

    models: Dict[str, _FactorModel] = field(default_factory=dict)
    history_months: int = 0
    first_period: Optional[pd.Period] = None
    last_period: Optional[pd.Period] = None

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, ext_df: pd.DataFrame) -> "ExtFactorForecaster":
        """
        ext_df : from load_ext_factors() – columns year_month + factor columns,
                 one row per month (gaps allowed; they are interpolated).
        """
        df = ext_df.dropna(subset=["year_month"]).sort_values("year_month")
        df = df.drop_duplicates("year_month").set_index("year_month")

        # Regular monthly index; interpolate interior gaps so the models see a clean series
        full_idx = pd.period_range(df.index.min(), df.index.max(), freq="M")
        df = df.reindex(full_idx)

        self.first_period = full_idx[0]
        self.last_period = full_idx[-1]
        self.history_months = len(full_idx)

        for col in EXT_FACTOR_COLS:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both")
            series = series.astype(float)
            if series.notna().sum() == 0:
                continue
            self.models[col] = self._fit_one(col, series)

        return self

    def _fit_one(self, name: str, series: pd.Series) -> _FactorModel:
        n = int(series.notna().sum())
        last_val = float(series.dropna().iloc[-1])
        last_period = series.index[-1]

        # Constant series → nothing to model
        if float(series.std(ddof=0)) == 0.0:
            return _FactorModel("constant", last_period, None, last_val)

        # Statsmodels wants a DatetimeIndex with a freq for forecasting
        ts = series.copy()
        ts.index = ts.index.to_timestamp(how="start")
        ts = ts.asfreq("MS")

        history = ts.values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if n >= MIN_MONTHS_SARIMA:
                fitted = self._fit_sarima(ts, history)
                if fitted is not None:
                    return _FactorModel("sarima", last_period, fitted, last_val)

            if n >= MIN_MONTHS_HW:
                fitted = self._fit_holt_winters(ts, history)
                if fitted is not None:
                    return _FactorModel("holt_winters", last_period, fitted, last_val)

        # Linear fallback
        x = np.arange(len(series), dtype=float)
        y = series.values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            a, b = np.polyfit(x[mask], y[mask], 1)
            return _FactorModel("linear", last_period, (float(a), float(b)), last_val)
        return _FactorModel("constant", last_period, None, last_val)

    @staticmethod
    def _fit_sarima(ts: pd.Series, history: np.ndarray):
        """Best-AIC SARIMA whose 36-month forecast stays within a sane band."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        best, best_aic = None, np.inf
        for order, sorder in _SARIMA_ORDERS:
            try:
                res = SARIMAX(ts, order=order, seasonal_order=sorder).fit(
                    disp=False, maxiter=100, method="lbfgs",
                )
                if not np.isfinite(res.aic) or res.aic >= best_aic:
                    continue
                fc = np.asarray(res.get_forecast(steps=SANITY_HORIZON).predicted_mean, dtype=float)
                if _plausible(fc, history):
                    best, best_aic = res, res.aic
            except Exception:
                continue
        return best

    @staticmethod
    def _fit_holt_winters(ts: pd.Series, history: np.ndarray):
        """Damped additive Holt-Winters; seasonal first, non-seasonal fallback."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        for seasonal in ("add", None):
            try:
                res = ExponentialSmoothing(
                    ts, trend="add", damped_trend=True,
                    seasonal=seasonal,
                    seasonal_periods=SEASONAL_PERIOD if seasonal else None,
                    initialization_method="estimated",
                ).fit(optimized=True)
                fc = np.asarray(res.forecast(SANITY_HORIZON), dtype=float)
                if _plausible(fc, history):
                    return res
            except Exception:
                continue
        return None

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, periods: List[pd.Period]) -> pd.DataFrame:
        """
        Returns one row per requested period with columns:
          year_month, <factor>, <factor>_low, <factor>_high   (80% interval)
        Periods at or before the last history month return the fitted/actual value
        with a zero-width interval.
        """
        if not periods:
            return pd.DataFrame(columns=["year_month"])

        periods = sorted(set(pd.Period(p, "M") for p in periods))
        horizon = max(int((p - self.last_period).n) for p in periods)
        horizon = max(horizon, 1)

        out = pd.DataFrame({"year_month": [str(p) for p in periods]})

        for col, fm in self.models.items():
            mean, lo, hi = self._forecast_path(fm, horizon)
            vals, lows, highs = [], [], []
            for p in periods:
                step = int((p - self.last_period).n)   # 1 = first future month
                if step <= 0:
                    vals.append(fm.fallback_value); lows.append(fm.fallback_value); highs.append(fm.fallback_value)
                else:
                    vals.append(float(mean[step - 1])); lows.append(float(lo[step - 1])); highs.append(float(hi[step - 1]))
            out[col] = vals
            out[f"{col}_low"] = lows
            out[f"{col}_high"] = highs

        return out

    def _forecast_path(self, fm: _FactorModel, horizon: int):
        """Return (mean, low, high) arrays of length `horizon`."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if fm.kind == "sarima":
                try:
                    fc = fm.model.get_forecast(steps=horizon)
                    mean = np.asarray(fc.predicted_mean, dtype=float)
                    ci = fc.conf_int(alpha=INTERVAL_ALPHA)
                    lo = np.asarray(ci.iloc[:, 0], dtype=float)
                    hi = np.asarray(ci.iloc[:, 1], dtype=float)
                    if np.all(np.isfinite(mean)):
                        return mean, lo, hi
                except Exception:
                    pass

            if fm.kind == "holt_winters":
                try:
                    mean = np.asarray(fm.model.forecast(horizon), dtype=float)
                    # HW has no analytic interval in statsmodels' simple API;
                    # use residual std scaled by sqrt(horizon) as an 80% band.
                    resid = np.asarray(fm.model.resid, dtype=float)
                    sigma = float(np.nanstd(resid)) if resid.size else 0.0
                    z = 1.2816
                    spread = z * sigma * np.sqrt(np.arange(1, horizon + 1))
                    if np.all(np.isfinite(mean)):
                        return mean, mean - spread, mean + spread
                except Exception:
                    pass

            if fm.kind == "linear":
                a, b = fm.model
                n_hist = self.history_months
                x = np.arange(n_hist, n_hist + horizon, dtype=float)
                mean = a * x + b
                return mean, mean, mean

        # constant / failure
        mean = np.full(horizon, fm.fallback_value, dtype=float)
        return mean, mean, mean

    # ── Introspection ─────────────────────────────────────────────────────────
    def describe(self) -> str:
        kinds = sorted({fm.kind for fm in self.models.values()})
        label = {"sarima": "SARIMA", "holt_winters": "Holt-Winters",
                 "linear": "linear trend", "constant": "constant"}
        parts = [label.get(k, k) for k in kinds]
        return (f"{'/'.join(parts)} fitted on {self.history_months} months "
                f"({self.first_period}..{self.last_period})")

    def model_kinds(self) -> Dict[str, str]:
        return {c: fm.kind for c, fm in self.models.items()}


# ── Cache: one fitted forecaster per external_factors.csv version ────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Tuple[float, ExtFactorForecaster]] = {}


def get_ext_forecaster(model_store_dir: str) -> Optional[ExtFactorForecaster]:
    """
    Return a fitted forecaster for the ext factors file in `model_store_dir`,
    refitting only when the file's mtime changes. None if the file is missing.
    """
    path = os.path.join(model_store_dir, EXT_FACTORS_FILENAME)
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)

    with _cache_lock:
        hit = _cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]

    ext_df = load_ext_factors(model_store_dir)
    if ext_df is None or ext_df.empty:
        return None

    forecaster = ExtFactorForecaster().fit(ext_df)

    with _cache_lock:
        _cache[path] = (mtime, forecaster)
    return forecaster
