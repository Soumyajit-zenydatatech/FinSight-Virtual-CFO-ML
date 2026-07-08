# SM Revenue Forecasting API — Full Documentation

Complete reference for all endpoints. Every request requires a JWT token issued by the main platform.

```
Authorization: Bearer <your_token>
```

Use the **Authorize** button in Swagger UI at `/docs` to set it once for all endpoints.

---

## Table of Contents

1. [Training](#1-training)
2. [Datasets](#2-datasets)
3. [Models](#3-models)
4. [Prediction](#4-prediction)
5. [Forecast Map](#5-forecast-map)
6. [Heatmap — Base Endpoints](#6-heatmap--base-endpoints)
7. [Heatmap — What-If Endpoints](#7-heatmap--what-if-endpoints)

---

## 1. Training

### `POST /train`

Upload a CSV and train new Revenue, COGS, and SG&A models. Two algorithms are trained per target (LinearRegression and RidgeCV) and the better one is automatically selected.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | Main financial CSV |
| `model_name` | string | ✅ | Unique name for this model |
| `description` | string | ❌ | Optional notes |
| `test_size` | float | ❌ | Train/test split ratio (default `0.25`) |
| `random_state` | int | ❌ | Reproducibility seed (default `42`) |
| `external_factors_json` | string | ❌ | JSON string of monthly ext factor values |

**Response**
```json
{
  "model_id": 1,
  "dataset_id": 2,
  "model_name": "revenue_model_v1",
  "is_new_dataset": true,
  "training_duration_seconds": 4.21,
  "metrics": {
    "Total Revenue": {
      "baseline": { "MAE": 12000, "RMSE": 18000, "R2": 0.87, "MAPE": 4.2 },
      "ridge":    { "MAE": 9500,  "RMSE": 14000, "R2": 0.91, "MAPE": 3.1 }
    },
    "COGS": { "..." },
    "SG&A": { "..." }
  },
  "best_model_per_target": {
    "Total Revenue": "ridge",
    "COGS": "baseline",
    "SG&A": "ridge"
  },
  "message": "Model trained successfully. New dataset registered."
}
```

---

## 2. Datasets

### `GET /datasets`

List all CSV datasets uploaded by the authenticated user.

**Response**
```json
[
  {
    "id": 2,
    "user_id": "uuid-here",
    "original_filename": "subledger.csv",
    "file_hash": "sha256...",
    "file_size_bytes": 204800,
    "row_count": 800,
    "column_names": ["Order Date", "Region", "Total Revenue", "..."],
    "date_range_start": "2021-01-01",
    "date_range_end": "2025-12-31",
    "target_columns": ["Total Revenue", "COGS", "SG&A"],
    "uploaded_at": "2025-01-15T10:30:00",
    "notes": null,
    "model_count": 2
  }
]
```

---

### `GET /datasets/{dataset_id}`

Get metadata for a single dataset.

**Path param:** `dataset_id` (integer)

**Response:** Same structure as single item above.

---

### `GET /datasets/{dataset_id}/models`

List all trained models associated with a dataset.

**Path param:** `dataset_id` (integer)

**Response**
```json
[
  {
    "id": 1,
    "dataset_id": 2,
    "user_id": "uuid-here",
    "model_name": "revenue_model_v1",
    "description": null,
    "targets": ["Total Revenue", "COGS", "SG&A"],
    "feature_columns": ["Raw Material", "Direct Labor", "Freight", "..."],
    "external_factors_used": { "CCI": true, "CPI": true, "Oil": true },
    "test_size": 0.25,
    "random_state": 42,
    "metrics": { "..." },
    "trained_at": "2025-01-15T10:30:00",
    "training_duration_seconds": 4.21
  }
]
```

---

## 3. Models

### `GET /models`

List all trained models for the authenticated user.

**Query param:** `dataset_id` (optional) — filter by dataset

**Response:** Array of model objects (same structure as above).

---

### `GET /models/{model_id}`

Get a single trained model by ID.

**Path param:** `model_id` (integer)

**Response:** Single model object.

---

### `DELETE /models/{model_id}`

Delete a trained model and its `.pkl` file from disk.

**Path param:** `model_id` (integer)

**Response**
```json
{ "detail": "Model 'revenue_model_v1' deleted." }
```

---

## 4. Prediction

### `POST /predict/{model_id}`

Date-range forecast. Give a start and end date — the backend generates one row per month and predicts Revenue, COGS, and SG&A for each.

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "Region": "Australia & Oceania",
  "Country": "Australia",
  "Item_type": "Beverages",
  "Customer": "MNO Coffee",
  "Raw_Material": 120000,
  "Direct_Labor": 45000,
  "Freight": 8000,
  "Storage": 5000,
  "Packaging": 3000,
  "Indirect_Labor": 12000,
  "Rent_Utility": 9000,
  "Overhead": 6000,
  "CCI": 105.0,
  "CPI": 3.2,
  "Oil": 78.5,
  "GDP": 650.0,
  "Unemployment": 4.1,
  "ROI": 5.2
}
```

All fields except `start_date` and `end_date` are optional.

**Response**
```json
{
  "model_id": 1,
  "model_name": "revenue_model_v1",
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "monthly_predictions": [
    {
      "month": "2026-01",
      "date": "2026-01-01",
      "predicted_total_revenue": 468200.0,
      "predicted_COGS": 352100.0,
      "predicted_SGA": 43800.0,
      "model_used_revenue": "ridge",
      "model_used_COGS": "baseline",
      "model_used_SGA": "ridge"
    }
  ],
  "summary": {
    "total_revenue": 5618400.0,
    "total_COGS": 4225200.0,
    "total_SGA": 525600.0,
    "months_count": 12
  }
}
```

---

### `POST /predict/from-batch/{model_id}` ⭐ Main production endpoint

Loads the SUBLEDGER CSV from Supabase Storage for a batch, merges external factors by month, and predicts every row. Shows actual vs predicted for historical rows. Generates future rows if `prediction_end` is beyond the CSV date range.

**Path param:** `model_id` (integer)

**Query params (all optional)**

| Param | Description |
|-------|-------------|
| `country` | Filter to this country (case-insensitive) |
| `region` | Filter to this region |
| `geo` | Filter to this geo |
| `prediction_start` | Start of future forecast window (YYYY-MM-DD) |
| `prediction_end` | End of future forecast window — generates future rows if beyond CSV |

**Request body**
```json
{ "batch_id": 3 }
```

**Response**
```json
{
  "model_id": 1,
  "model_name": "alice_check",
  "batch_id": 3,
  "sl_file_path": "2/3/SUBLEDGER/file.csv",
  "filters_applied": {
    "country": null,
    "region": null,
    "geo": null,
    "prediction_start": null,
    "prediction_end": null
  },
  "predictions": [
    {
      "order_date": "2021-09-01",
      "row_type": "historical",
      "region": "Australia & Oceania",
      "geo": "APAC",
      "country": "Australia",
      "item_type": "Office Supplies",
      "customer": "ABC Infra",
      "actual_total_revenue": 330115.3,
      "actual_COGS": 248732.2,
      "actual_SGA": 83133.36,
      "actual_gross_profit": 81383.1,
      "predicted_total_revenue": 364134.04,
      "predicted_COGS": 248732.2,
      "predicted_SGA": 83133.36,
      "predicted_gross_profit": 115401.84,
      "model_used_revenue": "ridge",
      "model_used_COGS": "baseline",
      "model_used_SGA": "ridge"
    },
    {
      "order_date": "2026-01-01",
      "row_type": "future",
      "region": "Australia & Oceania",
      "geo": "APAC",
      "country": "Australia",
      "item_type": "Office Supplies",
      "customer": "ABC Infra",
      "actual_total_revenue": null,
      "actual_COGS": null,
      "actual_SGA": null,
      "actual_gross_profit": null,
      "predicted_total_revenue": 468200.0,
      "predicted_COGS": 352100.0,
      "predicted_SGA": 43800.0,
      "predicted_gross_profit": 116100.0,
      "model_used_revenue": "ridge",
      "model_used_COGS": "baseline",
      "model_used_SGA": "ridge"
    }
  ],
  "summary": {
    "historical_row_count": 60,
    "future_row_count": 0,
    "total_row_count": 60,
    "total_actual_revenue": 17761738.15,
    "total_actual_gross_profit": 4029620.69,
    "total_predicted_revenue": 19501938.69,
    "total_predicted_gross_profit": 5769821.23
  },
  "external_factors_info": "External factors merged from stored file by month: ['CCI', 'CPI', 'Oil', 'GDP', 'Unemployment', 'ROI']."
}
```

| Field | Description |
|-------|-------------|
| `row_type` | `"historical"` = from CSV, `"future"` = synthetically generated |
| `actual_*` | Real values from CSV — always `null` for future rows |
| `predicted_gross_profit` | `predicted_total_revenue - predicted_COGS` |
| `summary.total_actual_revenue` | Sum of actual revenue from historical rows only |
| `summary.total_predicted_revenue` | Sum of predicted revenue — historical + future combined |

---

## 5. Forecast Map

### `POST /forecast-map/{model_id}`

Same as `POST /predict/{model_id}` but accepts multiple rows and returns a regional breakdown.

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "rows": [
    {
      "order_date": "2026-01-01",
      "Region": "Australia & Oceania",
      "Country": "Australia",
      "Item_type": "Beverages",
      "Customer": "MNO Coffee",
      "CCI": 105.0,
      "CPI": 3.2,
      "Oil": 78.5,
      "GDP": 650.0,
      "Unemployment": 4.1,
      "ROI": 5.2
    }
  ]
}
```

**Response**
```json
{
  "model_id": 1,
  "model_name": "revenue_model_v1",
  "predictions": [
    {
      "order_date": "2026-01-01",
      "predicted_total_revenue": 468200.0,
      "predicted_COGS": 352100.0,
      "predicted_SGA": 43800.0,
      "model_used_revenue": "ridge",
      "model_used_COGS": "baseline",
      "model_used_SGA": "ridge"
    }
  ],
  "region_summary": [
    {
      "region": "Australia & Oceania",
      "total_revenue": 468200.0,
      "total_COGS": 352100.0,
      "total_SGA": 43800.0,
      "row_count": 1
    }
  ],
  "forecast_map_base64": null,
  "map_note": "1 region(s) found: Australia & Oceania."
}
```

---

### `POST /forecast-map/from-batch/{model_id}` ⭐ Main production endpoint

Reads SUBLEDGER CSV from Supabase Storage, auto-merges external factors, predicts every row, and returns full day-by-day breakdown with regional summary.

**Path param:** `model_id` (integer)
**Query param:** `batch_id` (integer, required)

No request body needed.

**Response**
```json
{
  "model_id": 1,
  "model_name": "revenue_model_v1",
  "batch_id": 3,
  "row_count": 60,
  "daily_predictions": [
    {
      "order_date": "2021-09-01",
      "region": "Australia & Oceania",
      "geo": "APAC",
      "country": "Australia",
      "item_type": "Office Supplies",
      "customer": "ABC Infra",
      "predicted_total_revenue": 364134.04,
      "predicted_COGS": 248732.2,
      "predicted_SGA": 83133.36,
      "model_used_revenue": "ridge",
      "model_used_COGS": "baseline",
      "model_used_SGA": "ridge"
    }
  ],
  "region_summary": [
    {
      "region": "Australia & Oceania",
      "total_revenue": 19501938.69,
      "total_COGS": 14832117.46,
      "total_SGA": 2900000.0,
      "row_count": 60
    }
  ],
  "forecast_map_base64": null,
  "map_note": "1 region(s): Australia & Oceania.",
  "external_factors_info": "External factors merged from stored file by month: ['CCI', 'CPI', 'Oil', 'GDP', 'Unemployment', 'ROI']."
}
```

---

## 6. Heatmap — Base Endpoints

All heatmaps return a JSON correlation matrix. Frontend renders with Plotly.js. No image is generated server-side.

---

### `GET /heatmap/from-storage`

Reads backend-stored `external_factors.csv`. Returns 6×6 correlation matrix between ext factors + raw monthly data for time-series charts.

No request body or query params needed.

**Response**
```json
{
  "factors_found": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "row_count": 80,
  "date_range_start": "2021-01-01",
  "date_range_end": "2025-12-01",
  "correlation": {
    "factors": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
    "matrix": [
      [ 1.0,  -0.54, -0.40, -0.54,  0.76, -0.70],
      [-0.54,  1.0,   0.58,  0.21, -0.68,  0.19],
      [-0.40,  0.58,  1.0,  -0.06, -0.58, -0.13],
      [-0.54,  0.21, -0.06,  1.0,  -0.57,  0.89],
      [ 0.76, -0.68, -0.58, -0.57,  1.0,  -0.53],
      [-0.70,  0.19, -0.13,  0.89, -0.53,  1.0 ]
    ]
  },
  "monthly_data": [
    { "date": "2021-01-01", "CCI": 107.0, "CPI": 1.1, "Oil": 52.16, "GDP": 533.9, "Unemployment": 6.3, "ROI": 0.1 }
  ],
  "message": "Data ready. 6 factors, 80 months."
}
```

**Plotly.js**
```js
Plotly.newPlot('div', [{
  type: 'heatmap',
  z: res.correlation.matrix,
  x: res.correlation.factors,
  y: res.correlation.factors,
  colorscale: 'RdBu', zmin: -1, zmax: 1
}])
```

---

### `POST /heatmap/external-factors/{model_id}`

Loads SUBLEDGER CSV from batch + merges ext factors. Returns **11×6 rectangular matrix** — SL business columns vs external factors.

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "batch_id": 3,
  "CCI": 105.0,
  "Oil": 95.5
}
```

All ext factor fields are optional. Only pass when you want to override stored values.

**Response**
```json
{
  "sl_columns": ["Total Revenue", "COGS", "SG&A", "Raw Material", "Direct Labor", "Freight", "Storage", "Packaging", "Indirect Labor", "Rent & Utility", "Overhead"],
  "ext_columns": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "correlation": [
    [-0.01, -0.11, -0.08,  0.08,  0.09,  0.09],
    [-0.01, -0.13, -0.04,  0.04,  0.11,  0.06],
    [-0.01, -0.01, -0.10,  0.07,  0.05,  0.07],
    [-0.01, -0.11, -0.06,  0.05,  0.10,  0.07],
    [-0.01, -0.15, -0.01,  0.05,  0.10,  0.07],
    [ 0.02, -0.17,  0.00, -0.03,  0.15,  0.00],
    [-0.01, -0.18,  0.03,  0.00,  0.13,  0.03],
    [ 0.01, -0.13, -0.01, -0.01,  0.12,  0.01],
    [ 0.00, -0.14, -0.04,  0.03,  0.12,  0.05],
    [ 0.00, -0.11, -0.03,  0.02,  0.11,  0.03],
    [ 0.01, -0.14,  0.01, -0.02,  0.11,  0.01]
  ],
  "row_count": 60,
  "message": "Cross-correlation: 11 SL columns × 6 external factors."
}
```

**Plotly.js**
```js
Plotly.newPlot('div', [{
  type: 'heatmap',
  z: res.correlation,
  x: res.ext_columns,
  y: res.sl_columns,
  colorscale: 'RdBu', zmin: -1, zmax: 1
}])
```

---

### `POST /heatmap/full-dataset/{model_id}`

Full 17×17 correlation matrix — all SL business columns + all external factors vs everything.

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "batch_id": 3,
  "CCI": 105.0,
  "Oil": 95.5
}
```

All ext factor fields optional.

**Response**
```json
{
  "columns": ["Total Revenue", "COGS", "SG&A", "Raw Material", "Direct Labor", "Freight", "Storage", "Packaging", "Indirect Labor", "Rent & Utility", "Overhead", "CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "correlation": [
    [1.0, 0.96, 0.80, 0.94, 0.88, 0.66, 0.82, 0.65, 0.93, 0.81, 0.73, -0.01, -0.11, -0.08, 0.08, 0.09, 0.09],
    "... (17 rows total)"
  ],
  "row_count": 60,
  "sl_columns_found": ["Total Revenue", "COGS", "SG&A", "Raw Material", "Direct Labor", "Freight", "Storage", "Packaging", "Indirect Labor", "Rent & Utility", "Overhead"],
  "ext_columns_found": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "message": "Correlation matrix computed. 11 SL columns + 6 ext factors = 17 total features."
}
```

**Plotly.js**
```js
Plotly.newPlot('div', [{
  type: 'heatmap',
  z: res.correlation,
  x: res.columns,
  y: res.columns,
  colorscale: 'RdBu', zmin: -1, zmax: 1
}], { width: 900, height: 900 })
```

---

### `POST /heatmap/upload` — *Admin only*

Upload 1–6 CSV files (one per external factor). Factor name auto-detected from column headers.

**Request:** `multipart/form-data`, field name `files`

**Response**
```json
{
  "factors_loaded": ["CCI", "CPI"],
  "row_count": 75,
  "date_range_start": "2020-01-01",
  "date_range_end": "2025-12-01",
  "heatmap_base64": null,
  "message": "Loaded 2 factor(s): CCI, CPI."
}
```

---

### `POST /heatmap/from-csv` — *Admin only*

Upload the full training or SL CSV. Backend extracts external factor columns from it and saves.

**Request:** `multipart/form-data`, field name `file`

**Response:** Same structure as `/heatmap/upload`

---

### `POST /heatmap/from-subledger/{batch_id}` — *Admin only*

Reads SUBLEDGER CSV already in Supabase Storage for the given batch, extracts external factor columns, and saves.

**Path param:** `batch_id` (integer)

**Response:** Same structure as `/heatmap/upload`

---

### `GET /heatmap/data` — *Admin / Debug*

Returns the stored external factor dataset as raw JSON rows.

**Response**
```json
{
  "row_count": 80,
  "columns": ["date", "CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "rows": [
    { "date": "2021-01-01", "CCI": 107.0, "CPI": 1.1, "Oil": 52.16, "GDP": 533.9, "Unemployment": 6.3, "ROI": 0.1 }
  ]
}
```

---

## 7. Heatmap — What-If Endpoints

These endpoints let the user **test new ext factor values** and see how the correlation matrix changes — without modifying any stored files. All computation is in-memory.

**Two modes:**
- **Append** — adds a new row (current month) with user values, missing factors filled with historical medians
- **Replace-latest** — replaces only the most recent month's ext factor values

---

### 6×6 What-If (ext factors vs each other)

#### `POST /heatmap/from-storage/append`

**Request body** (no `batch_id` needed)
```json
{
  "CCI": 112.0,
  "Oil": 95.5
}
```

**Response:** Same as `GET /heatmap/from-storage` with `row_count` increased by 1.

---

#### `POST /heatmap/from-storage/replace-latest`

**Request body**
```json
{
  "CCI": 112.0,
  "Oil": 95.5
}
```

**Response:** Same as `GET /heatmap/from-storage` — latest month's values replaced.

---

### 11×6 What-If (SL columns vs ext factors)

#### `POST /heatmap/external-factors/append/{model_id}`

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "batch_id": 3,
  "CCI": 112.0,
  "Oil": 95.5
}
```

New row gets median SL column values + your ext factor values.

**Response:** Same as `POST /heatmap/external-factors/{model_id}` with `row_count` increased by 1.

---

#### `POST /heatmap/external-factors/replace-latest/{model_id}`

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "batch_id": 3,
  "CCI": 112.0,
  "Oil": 95.5
}
```

**Response:** Same as `POST /heatmap/external-factors/{model_id}` — latest row's ext values replaced.

---

### 17×17 What-If (full dataset)

#### `POST /heatmap/full-dataset/append/{model_id}`

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "batch_id": 3,
  "CCI": 112.0,
  "Oil": 95.5
}
```

**Response:** Same as `POST /heatmap/full-dataset/{model_id}` with `row_count` increased by 1.

---

#### `POST /heatmap/full-dataset/replace-latest/{model_id}`

**Path param:** `model_id` (integer)

**Request body**
```json
{
  "batch_id": 3,
  "CCI": 112.0,
  "Oil": 95.5
}
```

**Response:** Same as `POST /heatmap/full-dataset/{model_id}` — latest row's ext values replaced.

---

## Quick Reference — All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Train new model from CSV |
| GET | `/datasets` | List datasets |
| GET | `/datasets/{id}` | Get dataset |
| GET | `/datasets/{id}/models` | List models for dataset |
| GET | `/models` | List all models |
| GET | `/models/{id}` | Get model |
| DELETE | `/models/{id}` | Delete model |
| POST | `/predict/{model_id}` | Date-range monthly forecast |
| POST | `/predict/from-batch/{model_id}` | ⭐ Batch CSV prediction — actual vs predicted |
| POST | `/forecast-map/{model_id}` | Multi-row prediction with regional summary |
| POST | `/forecast-map/from-batch/{model_id}` | ⭐ Batch CSV forecast with regional breakdown |
| GET | `/heatmap/from-storage` | 6×6 ext factor correlation |
| POST | `/heatmap/external-factors/{model_id}` | 11×6 SL vs ext factor correlation |
| POST | `/heatmap/full-dataset/{model_id}` | 17×17 full correlation matrix |
| POST | `/heatmap/from-storage/append` | 6×6 — append new row what-if |
| POST | `/heatmap/from-storage/replace-latest` | 6×6 — replace latest row what-if |
| POST | `/heatmap/external-factors/append/{model_id}` | 11×6 — append new row what-if |
| POST | `/heatmap/external-factors/replace-latest/{model_id}` | 11×6 — replace latest row what-if |
| POST | `/heatmap/full-dataset/append/{model_id}` | 17×17 — append new row what-if |
| POST | `/heatmap/full-dataset/replace-latest/{model_id}` | 17×17 — replace latest row what-if |
| POST | `/heatmap/upload` | Upload ext factor CSVs (admin) |
| POST | `/heatmap/from-csv` | Extract ext factors from training CSV (admin) |
| POST | `/heatmap/from-subledger/{batch_id}` | Extract ext factors from batch CSV (admin) |
| GET | `/heatmap/data` | View stored ext factor data (debug) |

---

## Error Codes

| Status | Meaning |
|--------|---------|
| `401` | Missing or invalid JWT token |
| `403` | Accessing another company's data |
| `404` | Resource not found (model, batch, file) |
| `409` | Model name already exists |
| `422` | Invalid input — check request body |
| `500` | Server error — CSV unreadable or model file missing |
