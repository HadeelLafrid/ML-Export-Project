"""Pre-compute aggregates from the large trade CSV for fast dashboard loading."""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RES_DIR = DATA_DIR / 'res'
DASH_DIR = RES_DIR / 'dashboard'
DASH_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = DATA_DIR / 'world_trade_data_features.csv'
COUNTRY_CSV = RES_DIR / 'country_codes_V202601.csv'
PRODUCT_CSV = RES_DIR / 'hs4_full_mapping.csv'

NEEDED_COLS = [
    'year', 'importer', 'product', 'total_value', 'total_qty',
    'algeria_export_v', 'algeria_export_q',
    'demand_gap_v', 'algeria_market_share',
    'sector', 'continent', 'opportunity_label',
    'world_demand_growth_3y', 'global_demand_index', 'global_demand_growth',
    'market_penetration_ratio', 'algeria_product_count', 'algeria_market_count',
    'gdp_usd', 'gdp_per_capita', 'gdp_growth_rate', 'population',
    'trade_openness', 'inflation_rate', 'distance_km',
    'shares_border', 'common_language_off', 'common_language_eth',
    'colonial_link', 'common_colonizer', 'is_landlocked',
]

print("Loading trade data (this may take a moment)...")
df = pd.read_csv(FEATURES_CSV, usecols=NEEDED_COLS)
print(f"  Loaded {len(df):,} rows, {df['importer'].nunique()} countries, {df['product'].nunique()} products")

df['opportunity'] = df['opportunity_label'].map({0: 'Medium', 1: 'Low', 2: 'High'}).fillna('Unknown')

print("Loading country codes...")
countries = pd.read_csv(COUNTRY_CSV)
print("Loading product codes...")
products = pd.read_csv(PRODUCT_CSV)

df = df.merge(countries[['country_code', 'country_name']], left_on='importer', right_on='country_code', how='left')
df = df.drop(columns=['country_code'])

df = df.merge(products[['product_code', 'product_name']], left_on='product', right_on='product_code', how='left')
df = df.drop(columns=['product_code'])

print("Creating aggregates...")

agg_year_sector = (
    df.groupby(['year', 'sector'])
    .agg(
        total_value=('total_value', 'sum'),
        algeria_export_v=('algeria_export_v', 'sum'),
        total_qty=('total_qty', 'sum'),
        importers=('importer', 'nunique'),
        products=('product', 'nunique'),
        demand_gap=('demand_gap_v', 'sum'),
    )
    .reset_index()
)
agg_year_sector.to_csv(DASH_DIR / 'agg_year_sector.csv', index=False)
print(f"  agg_year_sector: {len(agg_year_sector)} rows")

agg_year_continent = (
    df.groupby(['year', 'continent'])
    .agg(
        total_value=('total_value', 'sum'),
        algeria_export_v=('algeria_export_v', 'sum'),
        importers=('importer', 'nunique'),
        products=('product', 'nunique'),
        demand_gap=('demand_gap_v', 'sum'),
    )
    .reset_index()
)
agg_year_continent.to_csv(DASH_DIR / 'agg_year_continent.csv', index=False)
print(f"  agg_year_continent: {len(agg_year_continent)} rows")

agg_year_sector_continent = (
    df.groupby(['year', 'sector', 'continent'])
    .agg(
        total_value=('total_value', 'sum'),
        algeria_export_v=('algeria_export_v', 'sum'),
        demand_gap=('demand_gap_v', 'sum'),
    )
    .reset_index()
)
agg_year_sector_continent.to_csv(DASH_DIR / 'agg_year_sector_continent.csv', index=False)
print(f"  agg_year_sector_continent: {len(agg_year_sector_continent)} rows")

print("Creating top products by sector (latest year)...")
latest_year = df['year'].max()
top_products = (
    df[df['year'] == latest_year]
    .groupby(['sector', 'product', 'product_name'])
    .agg(total_value=('total_value', 'sum'))
    .reset_index()
    .sort_values('total_value', ascending=False)
    .groupby('sector')
    .head(10)
    .reset_index(drop=True)
)
top_products.to_csv(DASH_DIR / 'top_products.csv', index=False)
print(f"  top_products: {len(top_products)} rows")

print("Creating opportunity ranking...")
opp_ranking = (
    df[df['year'] == latest_year]
    .groupby(['importer', 'country_name', 'product', 'product_name', 'sector', 'continent', 'opportunity'])
    .agg(
        total_value=('total_value', 'sum'),
        algeria_export_v=('algeria_export_v', 'sum'),
        demand_gap=('demand_gap_v', 'sum'),
        market_share=('algeria_market_share', 'mean'),
    )
    .reset_index()
    .sort_values('demand_gap', ascending=False)
)
opp_ranking.to_csv(DASH_DIR / 'opportunity_ranking.csv', index=False)
print(f"  opportunity_ranking: {len(opp_ranking)} rows")

print("Creating summary stats...")
summary = {
    'metric': [
        'Total Trade Value (all years)',
        'Algeria Total Exports (all years)',
        'Unique Importers',
        'Unique Products',
        'Sectors Covered',
        'Years Covered',
        'High Opportunity Pairs',
        'Latest Year',
    ],
    'value': [
        f"${df['total_value'].sum():,.0f}",
        f"${df['algeria_export_v'].sum():,.0f}",
        str(df['importer'].nunique()),
        str(df['product'].nunique()),
        str(df['sector'].nunique()),
        f"{int(df['year'].min())} - {int(df['year'].max())}",
        str(len(df[df['opportunity'] == 'High'])),
        str(int(latest_year)),
    ]
}
pd.DataFrame(summary).to_csv(DASH_DIR / 'summary_stats.csv', index=False)

print("Creating yearly trade trend...")
yearly_trend = (
    df.groupby('year')
    .agg(
        total_value=('total_value', 'sum'),
        algeria_export_v=('algeria_export_v', 'sum'),
        importers=('importer', 'nunique'),
        products=('product', 'nunique'),
    )
    .reset_index()
    .sort_values('year')
)
yearly_trend.to_csv(DASH_DIR / 'yearly_trend.csv', index=False)
print(f"  yearly_trend: {len(yearly_trend)} rows")

print("Creating Algeria yearly export profile...")
alg_yearly = (
    df.groupby('year')
    .agg(
        algeria_export_v=('algeria_export_v', 'sum'),
        algeria_present_pairs=('algeria_export_v', lambda x: (x > 0).sum()),
        total_import_demand=('total_value', 'sum'),
    )
    .reset_index()
    .sort_values('year')
)
alg_yearly.to_csv(DASH_DIR / 'algeria_yearly.csv', index=False)
print(f"  algeria_yearly: {len(alg_yearly)} rows")

print("Creating sector demand index (2012=1.0)...")
sector_demand = (
    df.groupby(['year', 'sector'])['total_value'].sum().reset_index()
)
sector_pivot = sector_demand.pivot(index='year', columns='sector', values='total_value')
sector_indexed = (sector_pivot / sector_pivot.iloc[0]).reset_index()
sector_indexed.to_csv(DASH_DIR / 'sector_demand_index.csv', index=False)
print(f"  sector_demand_index: {sector_indexed.shape}")

print("Creating PCA sample for scatter plot...")
pca_sample = df[['importer', 'product', 'sector', 'year', 'continent', 'opportunity_label']].copy()
pca_sample = pca_sample[pca_sample['year'] == df['year'].max()]
pca_sample['key'] = pca_sample['importer'].astype(str) + '_' + pca_sample['product'].astype(str)
pca_sample = pca_sample.groupby('key').first().reset_index().drop(columns=['key'])
pca_sample = pca_sample.sample(min(15000, len(pca_sample)), random_state=42)
pca_sample.to_csv(DASH_DIR / 'pca_sample.csv', index=False)
print(f"  pca_sample: {len(pca_sample)} rows")

print("Creating correlation matrix (numeric features only)...")
corr_cols = [
    'total_value', 'algeria_market_share', 'world_demand_growth_3y',
    'global_demand_index', 'global_demand_growth', 'market_penetration_ratio',
    'algeria_product_count', 'algeria_market_count',
    'gdp_usd', 'gdp_per_capita', 'gdp_growth_rate', 'population',
    'trade_openness', 'inflation_rate', 'distance_km',
    'shares_border', 'common_language_off', 'common_language_eth',
    'colonial_link', 'common_colonizer', 'is_landlocked',
]
available_corr = [c for c in corr_cols if c in df.columns]
if len(available_corr) >= 3:
    corr_data = df[available_corr].sample(min(500000, len(df)), random_state=42)
    corr_matrix = corr_data.corr().round(3)
    corr_matrix.to_csv(DASH_DIR / 'feature_correlation.csv')
    print(f"  feature_correlation: {corr_matrix.shape}")
else:
    print("  Skipping correlation — not enough feature columns available")

print("Copying evaluation CSVs to dashboard...")
import shutil
eval_dir = RES_DIR / 'evaluation'
for f in ['comparison_task1.csv', 'comparison_task2.csv']:
    src = eval_dir / f
    if src.exists():
        dst = DASH_DIR / f
        shutil.copy(src, dst)
        print(f"  Copied {f}")

print("\n--- Clustering ---")
COUNTRY_OUT = Path(__file__).resolve().parent.parent / 'outputs' / 'country_cluster_profiles.csv'
PRODUCT_OUT = Path(__file__).resolve().parent.parent / 'outputs' / 'product_cluster_profiles.csv'

if COUNTRY_OUT.exists() and PRODUCT_OUT.exists():
    country_df = pd.read_csv(COUNTRY_OUT)
    product_df = pd.read_csv(PRODUCT_OUT)

    pc_cols = [f'PC{i}' for i in range(1, 17)]
    country_out = country_df[['importer'] + pc_cols + ['country_cluster']]
    country_out.to_csv(DASH_DIR / 'country_clusters.csv', index=False)
    print(f"  country_clusters: {len(country_out)} rows, K={country_out['country_cluster'].nunique()}")

    product_out = product_df[['product', 'sector'] + pc_cols + ['product_cluster']]
    product_out.to_csv(DASH_DIR / 'product_clusters.csv', index=False)
    print(f"  product_clusters: {len(product_out)} rows, K={product_out['product_cluster'].nunique()}")

    sector_comp = product_df.groupby(['product_cluster', 'sector']).size().reset_index(name='count')
    sector_comp.to_csv(DASH_DIR / 'cluster_sector_composition.csv', index=False)
    print(f"  cluster_sector_composition: {len(sector_comp)} rows")

    cluster_descriptions = {
        'country': ['Mature Markets', 'Emerging Hubs', 'High-Potential', 'Resource-Driven',
                     'Gateway Economies', 'Small Open', 'Frontier Markets', 'Global Anchors'],
        'product': ['Low-Value Commodities', 'High-Value Specialised', 'Strategic Commodities',
                    'Light Manufacturing', 'Intermediate Goods'],
    }
    stats_rows = []
    for cid in range(country_out['country_cluster'].nunique()):
        cnt = len(country_out[country_out['country_cluster'] == cid])
        stats_rows.append({'type': 'country', 'cluster_id': cid,
                           'count': cnt, 'description': cluster_descriptions['country'][cid]})
    for cid in range(product_out['product_cluster'].nunique()):
        cnt = len(product_out[product_out['product_cluster'] == cid])
        stats_rows.append({'type': 'product', 'cluster_id': cid,
                           'count': cnt, 'description': cluster_descriptions['product'][cid]})
    pd.DataFrame(stats_rows).to_csv(DASH_DIR / 'cluster_statistics.csv', index=False)
    print(f"  cluster_statistics: {len(stats_rows)} rows")
else:
    print("  Cluster output CSVs not found — run clustering_notebook.ipynb first")

print("\n--- Forecast Data ---")
FORECAST_CSV = RES_DIR / 'algeria_export_forecast_2025_2027.csv'
if FORECAST_CSV.exists():
    dst = DASH_DIR / 'forecast_data.csv'
    shutil.copy(FORECAST_CSV, dst)
    print(f"  Copied forecast data ({FORECAST_CSV.stat().st_size / 1024:.0f} KB)")
else:
    print("  No forecast data found — run full_forecasting.ipynb first")

print("\n✓ All dashboard data prepared.")
print(f"  Output: {DASH_DIR}")
