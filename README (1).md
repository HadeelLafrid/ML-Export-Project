# 🌍 Forecasting and Identifying Global Export Opportunities for Algerian Exporters

> An end-to-end machine learning system to identify, analyze, and forecast international export opportunities for Algeria — combining clustering, classification, and forecasting into a unified decision-support dashboard.

**ENSIA — Machine Learning Project | Spring 2025-2026**
**Team:** Hadil, Ibtihal, Ilyas, Raouf, Ahmed

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
  - [Clustering](#1-clustering)
  - [Classification](#2-classification)
  - [Forecasting](#3-forecasting)
- [Dashboard](#dashboard)
- [Key Results](#key-results)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Team Contributions](#team-contributions)
- [References](#references)

---

## Project Overview

Algeria's export economy is critically dependent on hydrocarbons: in 2023, oil and gas accounted for approximately **92% of total exports**, with overall revenues estimated at $50–55 billion. This extreme concentration exposes the national economy to global energy price volatility and reinforces the strategic urgency of diversification.

This project builds a complete machine learning system that answers two fundamental questions for Algerian exporters and institutions such as the **Algerian Chamber of Commerce and Industry (CACI)** and the **Ministry of External Commerce**:

1. **Where** should Algeria export its existing competitive products? *(EPI Classification)*
2. **What** new products should Algeria develop to diversify its export basket? *(PDI Classification)*
3. **When** will global demand grow, and by how much? *(Time-Series Forecasting)*

All results are delivered through an interactive **Plotly Dash dashboard** accessible to policymakers, exporters, and institutional stakeholders.

---

## Motivation

Under the directives of President Abdelmadjid Tebboune, Algeria has placed significant emphasis on increasing non-hydrocarbon exports and improving export competitiveness. Despite this political will, non-hydrocarbon exports remain a small share of the total export basket.

Data-driven tools can help bridge this gap by systematically identifying:
- Which global markets are underserved by Algerian exports relative to their potential
- Which new products Algeria has the hidden capability to produce, based on its existing industrial strengths
- How trade volumes will evolve through 2027 to guide forward-looking investment decisions

This project provides exactly these tools, grounded in internationally recognized methodology from the **International Trade Centre (ITC)**.

---

## Dataset

### Sources
| Source | Content |
|--------|---------|
| UN Comtrade | Global trade flows by product, country, and year |
| World Bank | GDP, population, inflation, trade openness |
| ITC Trade Map | Algeria RCA products, export potential indicators |
| CEPII GeoDist | Bilateral distances, colonial links, language |

### Key Statistics
| Metric | Value |
|--------|-------|
| Total rows | 1,501,178 |
| Importing countries | 97 |
| Products (HS4 level) | 1,224 |
| Years covered | 2012 – 2024 (13 years) |
| Features per row | 30+ |

### Feature Groups
- **Trade flows:** `total_value`, `algeria_export_v`, `demand_gap_v`, `algeria_market_share`
- **Economic indicators:** `gdp_usd`, `gdp_per_capita`, `gdp_growth_rate`, `inflation_rate`, `population`, `trade_openness`
- **Gravity model variables:** `distance_km`, `shares_border`, `common_language_off`, `colonial_link`, `common_colonizer`, `is_landlocked`
- **Product-level demand:** `world_demand_growth_3y`, `global_demand_index`, `global_demand_growth`
- **Temporal controls:** `year`, `is_covid_year`

---

## Project Structure

```
ML-Export-Project/
│
├── dashboard/
│   ├── app.py                        ← Plotly Dash application (5 tabs)
│   ├── prepare_data.py               ← Data pre-computation pipeline
│   ├── INTEGRATION_GUIDE.md
│   └── assets/style.css
│
├── data/
│   ├── world_trade_data_features.csv ← Main dataset (512MB — not committed)
│   └── res/
│       ├── classification_results.csv       ← ML classification output
│       ├── algeria_export_forecast_2025_2027.csv
│       ├── country_codes_V202601.csv
│       ├── hs4_full_mapping.csv
│       └── dashboard/               ← Auto-generated (do not edit)
│
├── notebooks/
│   ├── algeria_trade_analysis.ipynb  ← EDA
│   ├── clustering_notebook.ipynb     ← K-Means clustering
│   ├── classification - Copie.ipynb  ← Classification (EPI + PDI)
│   └── full_forecasting.ipynb        ← Prophet/LSTM forecasting
│
├── outputs/
│   ├── country_cluster_profiles.csv
│   └── product_cluster_profiles.csv
│
└── README.md
```

---

## ML Pipeline

### 1. Clustering

**Goal:** Segment countries and products by trade behavior to reveal natural market groupings.

**Method:** K-Means clustering with PCA dimensionality reduction.

| Clustering Task | K | Description |
|----------------|---|-------------|
| Country clusters | 8 | Mature Markets, Emerging Hubs, High-Potential, Resource-Driven, Gateway Economies, Small Open, Frontier Markets, Global Anchors |
| Product clusters | 5 | Low-Value Commodities, High-Value Specialised, Strategic Commodities, Light Manufacturing, Intermediate Goods |

**Evaluation:** Silhouette Score, Davies-Bouldin Index

**Output:** `outputs/country_cluster_profiles.csv`, `outputs/product_cluster_profiles.csv`

---

### 2. Classification

This is the most technically involved component of the project. It comprises three parts: a documented failure (data leakage), a principled fix (ITC methodology), and two classification tracks (EPI and PDI).

#### 2a. Data Leakage — Diagnosed and Fixed

Our initial approach assigned labels using a rule-based function (`label_opportunity()`) that thresholded three variables: `algeria_market_share`, `world_demand_growth_3y`, and `global_demand_index`. These same three variables were simultaneously present in the feature matrix X, creating **target leakage** — the model was not learning economic patterns but reverse-engineering the label formula.

**Evidence:**
- Random Forest accuracy: **1.00** (impossible in real trade prediction)
- XGBoost accuracy: **1.00**
- Logistic Regression: **0.46** (cannot memorize decision boundaries)
- Decision Tree (depth=3, 3 leaking features only): **~1.00** (smoking gun)

**Conclusion:** The 100% accuracy was meaningless. Dropping the leaking features from X is insufficient — the label itself must be redefined using a framework that is independent of any feature in the model.

#### 2b. ITC EPI Methodology — Label Construction

We replaced the rule-based label with the **International Trade Centre Export Potential Indicator (EPI)** framework (Decreux & Spies, ITC 2016), which decomposes export potential into three independent components:

**Supply** — Algeria's projected world market share for product k:
```
Supply(k) = Σ [algeria_export_v_k × ΔGDP_Algeria]
            ──────────────────────────────────────
            Σ_i [export_v_ik × ΔGDP_i]
```

**Demand** — Target market j's projected imports of product k:
```
Demand(j,k) = total_value_jk × (1 + gdp_growth_rate_j)
              clipped at 0 (no negative demand)
```

**Ease of Trade** — How well Algeria trades with market j overall:
```
Ease(j) =    Algeria actual total exports to j
             ──────────────────────────────────
             Σ_k [Supply(k) × total_value_jk]
```

**EPI Score and Label:**
```
EPI(j,k)    = Supply(k) × Ease(j) × Demand(j,k)
Unrealized  = EPI(j,k) − min(actual_exports_jk, EPI(j,k))
Label       = High   if unrealized ≥ 70th percentile ($17,756)
              Medium if unrealized ≥ 30th percentile ($25)
              Low    otherwise
```

The label is constructed from `algeria_export_v`, `total_value`, and GDP projections — none of which appear in the classifier's feature matrix. **Zero leakage confirmed.**

**Time-lagging:** All features are shifted by one year (T-1 → T) to enforce a realistic information constraint: decisions made in January 2022 use 2021 data.

#### 2c. EPI Track — Existing Products

| Scope | 56 products where Algeria has RCA ≥ 1 |
|-------|--------------------------------------|
| Dataset size | 54,865 rows (after lagging) |
| Split | Train 2013–2021 / Val 2022 / Test 2023–2024 |
| Features | 17 (macroeconomic + gravity model) |
| Label balance | 30% Low / 40% Medium / 30% High |

**Results:**

| Model | F1-macro (Val) | F1-macro (Test) |
|-------|---------------|----------------|
| Logistic Regression | 0.18 | — |
| XGBoost | 0.46 | — |
| **Random Forest** | **0.51** | **0.51** |
| Random baseline | 0.38 | 0.38 |

**Top 5 features (XGBoost importance):**

| Feature | Importance | Interpretation |
|---------|-----------|---------------|
| `is_covid_year` | 0.246 | Pandemic disruption dominates |
| `gdp_usd_lag1` | 0.092 | Market size matters most |
| `common_colonizer` | 0.070 | Historical ties beat distance |
| `gdp_growth_rate_lag1` | 0.068 | Economic momentum |
| `distance_km` | 0.068 | Gravity model confirmed |

**EPI Output:** A ranked list of target markets per product. For each of Algeria's 56 competitive products, the model identifies which markets have the largest unrealized export potential gap.

#### 2d. PDI Track — New Products (Diversification)

The PDI track addresses a harder question: which entirely new products should Algeria develop? It is based on Hausmann and Hidalgo's **product space** concept — products frequently exported together by many countries require similar productive capabilities.

**Proximity Matrix:** A 96×96 Jaccard similarity matrix computed over all HS2 chapters:
```
proximity(k, l) = countries with RCA in BOTH k and l
                  ──────────────────────────────────
                  countries with RCA in EITHER k or l
```

**Density:** How close is candidate product k to Algeria's current export basket?
```
Density(Algeria, k) = Σ_l [proximity(k,l) × CA(Algeria,l)]
                      ──────────────────────────────────────
                      Σ_l [proximity(k,l)]
```

Where CA(Algeria, l) = 1 if Algeria has RCA ≥ 1 in chapter l, else 0.

**PDI Score and Label:**
```
PDI(j,k) = Density(k) × Ease(j) × Demand(j,k)
```
Labeled using the same 30/70 percentile approach as EPI.

| Scope | 1,168 candidate products (67 HS2 chapters) |
|-------|-------------------------------------------|
| Dataset size | 717,679 rows (sampled to 150,000 for training) |
| Features | 18 (EPI features + density) |
| Label balance | 33% Low / 33% Medium / 33% High |

**Results:**

| Model | F1-macro (Val) |
|-------|---------------|
| Logistic Regression | 0.20 |
| Random Forest | 0.65 |
| **XGBoost** | **0.68** |

PDI outperforms EPI by +0.17–0.22 F1-macro, almost entirely attributable to the `density` feature providing a direct product-space feasibility signal.

**Top 15 Diversification Recommendations (after feasibility filter):**

| Rank | HS2 | Chapter | Density | Demand Growth | Score |
|------|-----|---------|---------|--------------|-------|
| 1 | 50 | Silk | 0.314 | 101.6% | 0.634 |
| 2 | 53 | Vegetable textile fibres | 0.348 | 70.5% | 0.593 |
| 3 | 05 | Animal products | 0.299 | 73.7% | 0.520 |
| 4 | 10 | Cereals (wheat, barley) | 0.332 | 55.3% | 0.516 |
| 5 | 79 | Zinc | 0.334 | 53.3% | 0.511 |

**Strategic clusters:**
- **Textile value chain** (chapters 50, 53, 55, 58, 60): builds on existing cotton, wool, carpet exports
- **Agricultural processing** (chapters 05, 10, 15): builds on existing agri-food sector
- **Base metals** (chapter 79 Zinc): adjacent to iron, steel, copper exports

**PDI Output:** A prioritized diversification roadmap — which industries Algeria should invest in to replace hydrocarbons in the long run.

---

### 3. Forecasting

**Goal:** Predict future trade volumes and demand trends for 2025–2027 to support forward-looking export strategy.

**Method:** Facebook Prophet and LSTM neural networks.

**Evaluation metrics:** MAE, RMSE, MAPE

**Split:** Train 2012–2021 / Validation 2022 / Test 2023–2024 / Forecast 2025–2027

**Output:** `data/res/algeria_export_forecast_2025_2027.csv`

---

## Dashboard

An interactive **Plotly Dash** dashboard with 5 tabs, designed for use by CACI experts, exporters, and policymakers.

| Tab | Content |
|-----|---------|
| Overview | Trade flow summary, Algeria export profile, yearly trends |
| Explorer | Drill down by country, product, sector, continent |
| Forecasts | 2025–2027 demand predictions with historical comparison |
| Opportunities | EPI/PDI opportunity rankings by market and product |
| Intelligence | Clustering results + Classification ML model outputs |

### Running the Dashboard

```bash
# Step 1 — Prepare all dashboard data
python dashboard/prepare_data.py

# Step 2 — Launch the application
python dashboard/app.py

# Step 3 — Open in browser
# http://127.0.0.1:8050
```

---

## Key Results

| Component | Method | Key Metric | Value |
|-----------|--------|-----------|-------|
| Country clustering | K-Means K=8 | Silhouette Score | — |
| Product clustering | K-Means K=5 | Davies-Bouldin | — |
| EPI classification | Random Forest | F1-macro (test) | **0.51** |
| PDI classification | XGBoost | F1-macro (val) | **0.68** |
| Forecasting | Prophet | MAPE | — |
| Leakage (old method) | XGBoost | Accuracy | 1.00 (**invalid**) |

**Classification output:** 66,934 unique (importer, product) pairs labeled as High, Medium, or Low export opportunity:
- High: 23,606 pairs (35.3%)
- Medium: 29,637 pairs (44.3%)
- Low: 13,691 pairs (20.5%)

---

## Installation

### Requirements

```bash
pip install -r dashboard/requirements.txt
```

### Core dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
plotly>=5.15
dash>=2.12
dash-bootstrap-components>=1.4
prophet
```

### Data files required (not committed due to size)

Place these files in `data/` before running:
- `world_trade_data_features.csv` (512MB — main dataset)
- `world_trade_data_enriched.csv`

Place these in `data/res/`:
- `country_codes_V202601.csv`
- `hs4_full_mapping.csv`
- `algeria_export_forecast_2025_2027.csv`

---

## How to Run

### Run full classification notebook

```bash
# Open in Jupyter
jupyter notebook "notebooks/classification - Copie.ipynb"

# Run all cells — checkpoints will load pre-computed files if available:
# - data/epi_labeled.csv       (EPI labeled dataset)
# - data/pdi_labeled.csv       (PDI labeled dataset)
# - data/proximity_matrix.csv  (96×96 product space matrix)
```

### Generate classification results for dashboard

```bash
# After running the classification notebook:
python dashboard/prepare_data.py
python dashboard/app.py
```

### Run clustering notebook

```bash
jupyter notebook notebooks/clustering_notebook.ipynb
```

### Run forecasting notebook

```bash
jupyter notebook notebooks/full_forecasting.ipynb
```

---

## Team Contributions

| Member | Role |
|--------|------|
| **Hadil** | Classification (EPI + PDI pipeline, leakage diagnosis, ITC methodology, dashboard integration) |
| **Ibtihal** | Dashboard development (Plotly Dash, 5-tab architecture, prepare_data.py) |
| **Ilyas** | Data collection, preprocessing, feature engineering, project lead |
| **Raouf** | Clustering (K-Means, PCA, cluster interpretation) |
| **Ahmed** | Forecasting (Prophet, LSTM, 2025–2027 projections) |

---

## Methodology References

- Decreux, Y. & Spies, J. (2016). *Export Potential and Diversification Assessments: A methodology to identify export opportunities*. International Trade Centre (ITC).
- Hidalgo, C., Klinger, B., Barabasi, A.L., Hausmann, R. (2007). *The product space conditions the development of nations*. Science 317, 482–487.
- Hausmann, R. & Hidalgo, C.A. (2007). *Structural Transformation and Patterns of Comparative Advantage in the Product Space*. Harvard CID Working Paper No. 128.
- Head, K. & Mayer, T. (2014). *Gravity Equations: Workhorse, Toolkit, and Cookbook*. Handbook of International Economics vol. 4.
- Balassa, B. (1965). *Trade Liberalisation and Revealed Comparative Advantage*. The Manchester School.

---

## Limitations and Future Work

**Current limitations:**
- Country sample limited to 97 countries (ITC uses 226) — reduces proximity matrix precision
- No tariff data — full ITC formula requires ITC Market Access Map (not publicly available)
- Proximity computed at HS2 level; ITC uses HS6 for finer granularity
- COVID-19 years (2020–2021) dominate feature importance, potentially masking structural signals

**Future directions:**
- Integrate ITC Market Access Map tariff data for full EPI formula implementation
- Expand proximity matrix to HS6 level with a larger country sample
- Add product complexity index (OEC data) as a PDI feature
- Build real-time data pipeline from UN Comtrade API
- Deploy dashboard on a public server for CACI access
- Apply NLP to trade agreement texts to extract tariff reduction signals

---

## License

This project was developed for academic purposes at ENSIA (Higher National School of Artificial Intelligence), Algeria. All data used is publicly available from UN Comtrade, World Bank, and ITC Trade Map under their respective open data licenses.

---

*"Data-driven export diversification for a stronger Algerian economy."*
