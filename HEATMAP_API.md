# Heatmap API Documentation

Three endpoints for computing and rendering correlation heatmaps between external macro-economic factors and internal business data. All rendering happens on the frontend (Plotly.js) — the backend returns pure JSON matrices.

All endpoints require a Bearer JWT token:
```
Authorization: Bearer <your_token>
```

---

## 1. `GET /heatmap/from-storage`

**What it does:**
Reads the `external_factors.csv` baked into the backend, computes Pearson correlation between the 6 external factors (CCI, CPI, Oil, GDP, Unemployment, ROI), and returns the 6×6 matrix + raw monthly data.

This is the **bottom-right corner** of the full heatmap — external factors correlated with each other.

No request body or query params needed.

```
GET /heatmap/from-storage
Authorization: Bearer <your_token>
```

### Response

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
    { "date": "2021-01-01", "CCI": 107.0, "CPI": 1.1, "Oil": 52.16, "GDP": 533.9, "Unemployment": 6.3, "ROI": 0.1 },
    { "date": "2021-02-01", "CCI": 109.5, "CPI": 1.3, "Oil": 55.72, "GDP": 533.9, "Unemployment": 6.0, "ROI": 0.1 }
  ],
  "message": "Data ready. 6 factors, 80 months."
}
```

### Response fields

| Field | Description |
|-------|-------------|
| `factors_found` | Which of the 6 factor columns were present in the file |
| `row_count` | Number of monthly rows |
| `date_range_start` | Earliest month in the file |
| `date_range_end` | Latest month in the file |
| `correlation.factors` | Factor names — same order for X and Y axis |
| `correlation.matrix` | 6×6 Pearson correlation values (−1 to +1) |
| `monthly_data` | Raw monthly values — use for time-series line charts |

### Plotly.js (frontend)

```js
const res = await fetch('/heatmap/from-storage', {
  headers: { Authorization: `Bearer ${token}` }
}).then(r => r.json())

// Correlation heatmap
Plotly.newPlot('heatmap-div', [{
  type: 'heatmap',
  z: res.correlation.matrix,
  x: res.correlation.factors,
  y: res.correlation.factors,
  colorscale: 'RdBu',
  zmin: -1,
  zmax: 1,
  text: res.correlation.matrix.map(row => row.map(v => v.toFixed(2))),
  texttemplate: '%{text}'
}])

// Time-series line chart
const traces = res.factors_found.map(f => ({
  type: 'scatter',
  mode: 'lines+markers',
  name: f,
  x: res.monthly_data.map(r => r.date),
  y: res.monthly_data.map(r => r[f])
}))
Plotly.newPlot('timeseries-div', traces)
```

### Error responses

| Status | Reason |
|--------|--------|
| `404` | `external_factors.csv` not found in `model_store/` |
| `422` | File has no recognisable factor columns |
| `401` | Missing or invalid JWT token |

---

## 2. `POST /heatmap/external-factors/{model_id}`

**What it does:**
Loads the SUBLEDGER CSV from Supabase Storage for the given batch, merges external factors by year-month, then computes the **cross-correlation** between the 6 external factors and the 11 SL business columns (Total Revenue, COGS, SG&A, cost components).

This is the **top-right corner** of the full heatmap — how each external factor affects each business metric.

Returns a **rectangular matrix** — rows are SL business columns, columns are external factors.

Also supports **optional override values** for live what-if analysis — change a factor value and instantly get the updated matrix without re-uploading any file.

### Path parameter

| Param | Type | Description |
|-------|------|-------------|
| `model_id` | integer | ID of the trained model (used for routing context) |

### Request body

```json
{
  "batch_id": 3,
  "CCI": 105.0,
  "Oil": 80.0
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `batch_id` | ✅ Yes | Batch ID whose SUBLEDGER CSV to load from Supabase Storage |
| `CCI` | ❌ Optional | Override CCI value across all rows for what-if analysis |
| `CPI` | ❌ Optional | Override CPI value |
| `Oil` | ❌ Optional | Override Oil price value |
| `GDP` | ❌ Optional | Override GDP value |
| `Unemployment` | ❌ Optional | Override Unemployment rate value |
| `ROI` | ❌ Optional | Override ROI value |

### Response

```json
{
  "sl_columns": [
    "Total Revenue", "COGS", "SG&A",
    "Raw Material", "Direct Labor", "Freight",
    "Storage", "Packaging", "Indirect Labor",
    "Rent & Utility", "Overhead"
  ],
  "ext_columns": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "correlation": [
    [-0.01, -0.11, -0.08,  0.08,  0.09,  0.09],
    [-0.01, -0.13, -0.04,  0.04,  0.11,  0.06],
    [-0.01, -0.01, -0.10,  0.07,  0.05,  0.07],
    [-0.01, -0.11, -0.06,  0.05,  0.10,  0.07],
    [-0.01, -0.15, -0.01,  0.05,  0.10,  0.07],
    [ 0.02, -0.17,  0.00, -0.03,  0.15, -0.00],
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

### Response fields

| Field | Description |
|-------|-------------|
| `sl_columns` | Y axis labels — SL business columns found in the CSV |
| `ext_columns` | X axis labels — external factor columns after merge |
| `correlation` | Rectangular matrix. `correlation[i][j]` = correlation between `sl_columns[i]` and `ext_columns[j]` |
| `row_count` | Number of CSV rows used for computation |
| `message` | Summary including which overrides were applied |

### Plotly.js (frontend)

```js
const res = await fetch('/heatmap/external-factors/1', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`
  },
  body: JSON.stringify({ batch_id: 3 })
}).then(r => r.json())

Plotly.newPlot('heatmap-div', [{
  type: 'heatmap',
  z: res.correlation,
  x: res.ext_columns,     // X axis — CCI, CPI, Oil...
  y: res.sl_columns,      // Y axis — Total Revenue, COGS...
  colorscale: 'RdBu',
  zmin: -1,
  zmax: 1,
  text: res.correlation.map(row => row.map(v => v.toFixed(2))),
  texttemplate: '%{text}'
}])
```

### Live what-if refresh

User changes Oil price to 95? Just resend with the override — no file re-upload:

```js
body: JSON.stringify({ batch_id: 3, Oil: 95.0 })
```

The entire column is replaced with the new value and correlation is recomputed instantly.

### Error responses

| Status | Reason |
|--------|--------|
| `404` | Batch not found or no SUBLEDGER file for this batch |
| `422` | No SL numeric columns or no external factor columns found |
| `500` | Failed to load CSV from Supabase Storage |
| `401` | Missing or invalid JWT token |

---

## 3. `POST /heatmap/full-dataset/{model_id}`

**What it does:**
Combines everything — loads the SUBLEDGER CSV from Supabase Storage, merges external factors by year-month, and computes the **full square correlation matrix** across all 17 columns (11 SL business columns + 6 external factors).

This is the **complete heatmap** — the full image with all rows and columns.

Supports the same optional override values for live what-if analysis.

### Path parameter

| Param | Type | Description |
|-------|------|-------------|
| `model_id` | integer | ID of the trained model (used for routing context) |

### Request body

```json
{
  "batch_id": 3,
  "CCI": 105.0,
  "Oil": 80.0
}
```

Same fields as `/heatmap/external-factors/{model_id}` — `batch_id` required, all factor overrides optional.

### Response

```json
{
  "columns": [
    "Total Revenue", "COGS", "SG&A",
    "Raw Material", "Direct Labor", "Freight",
    "Storage", "Packaging", "Indirect Labor",
    "Rent & Utility", "Overhead",
    "CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"
  ],
  "correlation": [
    [1.0,  0.96, 0.80, 0.94, 0.88, 0.66, 0.82, 0.65, 0.93, 0.81, 0.73, -0.01, -0.11, -0.08,  0.08,  0.09,  0.09],
    [0.96, 1.0,  0.70, 0.98, 0.93, 0.69, 0.89, 0.66, 0.95, 0.81, 0.74, -0.01, -0.13, -0.04,  0.04,  0.11,  0.06],
    "..."
  ],
  "row_count": 60,
  "sl_columns_found": ["Total Revenue", "COGS", "SG&A", "Raw Material", "Direct Labor", "Freight", "Storage", "Packaging", "Indirect Labor", "Rent & Utility", "Overhead"],
  "ext_columns_found": ["CCI", "CPI", "Oil", "GDP", "Unemployment", "ROI"],
  "message": "Correlation matrix computed. 11 SL columns + 6 ext factors = 17 total features."
}
```

### Response fields

| Field | Description |
|-------|-------------|
| `columns` | Axis labels — same list for both X and Y axis |
| `correlation` | Full square N×N Pearson correlation matrix |
| `row_count` | Number of CSV rows used |
| `sl_columns_found` | Which SL business columns were found in the CSV |
| `ext_columns_found` | Which external factor columns were merged |
| `message` | Summary including override info if any |

### Plotly.js (frontend)

```js
const res = await fetch('/heatmap/full-dataset/1', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`
  },
  body: JSON.stringify({ batch_id: 3 })
}).then(r => r.json())

Plotly.newPlot('heatmap-div', [{
  type: 'heatmap',
  z: res.correlation,
  x: res.columns,
  y: res.columns,
  colorscale: 'RdBu',
  zmin: -1,
  zmax: 1,
  text: res.correlation.map(row => row.map(v => v.toFixed(2))),
  texttemplate: '%{text}'
}], {
  title: 'Feature Correlation Heatmap — Full Dataset',
  width: 900,
  height: 900
})
```

### Error responses

| Status | Reason |
|--------|--------|
| `404` | Batch not found or no SUBLEDGER file for this batch |
| `422` | No numeric columns found after loading CSV |
| `500` | Failed to load CSV from Supabase Storage |
| `401` | Missing or invalid JWT token |

---

## Summary

| Endpoint | Matrix shape | What it shows |
|----------|-------------|---------------|
| `GET /heatmap/from-storage` | 6×6 | External factors vs each other |
| `POST /heatmap/external-factors/{model_id}` | 11×6 | SL business columns vs external factors |
| `POST /heatmap/full-dataset/{model_id}` | 17×17 | Everything vs everything |
