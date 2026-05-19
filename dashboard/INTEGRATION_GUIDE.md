# Integration Guide — Classification

This guide is for the team member working on **classification**. It explains how to save your results so the dashboard picks them up automatically.

---

## Current Status

| Component | Status | Who |
|-----------|--------|-----|
| **Clustering** | Integrated in dashboard (K-Means, country K=8, product K=5) | Done |
| **Forecasting** | Evaluation tables + 2025-2027 forecast chart visible | Run `python dashboard/run_forecasting.py` |
| **Classification** | Placeholder "Pending" badge in Intelligence tab | **You** |

---

## How the Dashboard Discovers Your Results

```
project/
└── data/res/
    └── classification_results.csv    ← You create this file HERE
                        │
                        ▼
            python dashboard/prepare_data.py
                        │
                        ▼
            dashboard auto-detects the file
            and displays your results
```

`prepare_data.py` checks for the existence of `data/res/classification_results.csv`. If found, the dashboard replaces the "Pending" badge with your actual classification data.

---

## Your Task

### Step 1 — Build your classification model

Predict `opportunity_label` (High / Medium / Low) for every (importer, product) pair.

- **Input features:** 21 numeric columns from `world_trade_data_features.csv` (GDP, distance, trade openness, market share, market penetration, growth rates, etc.)
- **Training labels:** Column `opportunity_label` in `world_trade_data_features.csv` (gap-derived: 0=Low, 1=Medium, 2=High)

### Step 2 — Save your results to the EXACT path

At the end of your notebook, add this code. Do NOT change the path or column names.

```python
import pandas as pd

# Build your results DataFrame
results = pd.DataFrame({
    'importer': ...,              # int — country code
    'product': ...,               # int — HS4 product code
    'opportunity_label': ...,     # str — 'High', 'Medium', or 'Low'
})

# Validate before saving
assert results['importer'].dtype in ('int64', 'int32'), 'importer must be integer'
assert results['product'].dtype in ('int64', 'int32'), 'product must be integer'
assert results['opportunity_label'].isin(['High', 'Medium', 'Low']).all(), \
    'opportunity_label must be High, Medium, or Low'

# Save to the exact path the dashboard expects
results.to_csv('../data/res/classification_results.csv', index=False)
print(f'Saved {len(results):,} rows to data/res/classification_results.csv')
```

**Required columns — exact spelling, case-sensitive:**

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `importer` | int | `12` | Must match country codes in `world_trade_data_features.csv` |
| `product` | int | `101` | Must match HS4 codes in `world_trade_data_features.csv` |
| `opportunity_label` | str | `High` | One of: `High`, `Medium`, `Low` (capitalised) |

### Step 3 — Re-run data preparation

From the **project root** (`ML-Export-Project/`):

```bash
python dashboard/prepare_data.py
```

You should see output like:
```
...
--- Classification Data ---
  classification_results.csv found — copying to dashboard
...
✓ All dashboard data prepared.
```

If you don't see the classification line, double-check the file path.

### Step 4 — Verify in the dashboard

```bash
python dashboard/app.py
```

Open http://127.0.0.1:8050 and check:

- **Intelligence tab** → The "Classification — Pending" badge should be **gone**, replaced by your results alongside the clustering visualisations.
- **Opportunities tab** → The opportunity labels will now reflect your ML-based predictions instead of the old gap-derived labels.

### Step 5 — Commit and push

```bash
git add data/res/classification_results.csv
git add notebooks/your_classification_notebook.ipynb
git commit -m "Add ML-based classification results"
git push -u origin feat/classification
```

> `data/res/classification_results.csv` is NOT in `.gitignore` — it will be tracked.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Dashboard still shows "Pending" badge | File not at the right path | Check `data/res/classification_results.csv` exists |
| `prepare_data.py` doesn't mention classification | Wrong column names | Re-run validation code in Step 2 |
| Opportunities tab shows old labels | `prepare_data.py` wasn't re-run | Run it again after placing the CSV |
| Wrong column name error | Case mismatch | Column names must be lowercase: `importer`, `product`, `opportunity_label` |

---

## File Structure (relevant paths)

```
ML-Export-Project/                  ← project root (run everything from here)
├── data/
│   ├── world_trade_data_features.csv   (main data, 512MB — do not commit)
│   │
│   └── res/
│       ├── classification_results.csv  ← ★ YOU CREATE THIS ★
│       │
│       ├── evaluation/
│       │   ├── comparison_task1.csv
│       │   └── comparison_task2.csv
│       │
│       └── dashboard/                  (auto-generated — do not touch)
│           ├── agg_year_sector.csv
│           ├── country_clusters.csv
│           ├── product_clusters.csv
│           ├── cluster_statistics.csv
│           ├── cluster_sector_composition.csv
│           └── ...
│
├── dashboard/
│   ├── app.py                          ← Dash app (5 tabs)
│   ├── prepare_data.py                 ← data pre-computation
│   ├── INTEGRATION_GUIDE.md            ← this file
│   ├── requirements.txt
│   └── assets/style.css
│
├── outputs/
│   ├── country_cluster_profiles.csv
│   └── product_cluster_profiles.csv
│
└── notebooks/
    ├── clustering_notebook.ipynb
    ├── full_forecasting.ipynb
    └── forecasting.ipynb
```

---

## Quick Validation Script

Run this standalone to verify your file before committing:

```python
import pandas as pd

path = 'data/res/classification_results.csv'
cf = pd.read_csv(path)

assert 'importer' in cf.columns, f"Missing column 'importer' in {path}"
assert 'product' in cf.columns, f"Missing column 'product' in {path}"
assert 'opportunity_label' in cf.columns, f"Missing column 'opportunity_label' in {path}"
assert cf['importer'].dtype in ('int64', 'int32'), "'importer' must be integer"
assert cf['product'].dtype in ('int64', 'int32'), "'product' must be integer"

valid = {'High', 'Medium', 'Low'}
labels = set(cf['opportunity_label'].unique())
assert labels.issubset(valid), f"Invalid labels found: {labels - valid}"

print(f"Valid — {len(cf):,} rows")
print(f"Label distribution: {cf['opportunity_label'].value_counts().to_dict()}")
```

---

**Questions?** Ask the team lead.
