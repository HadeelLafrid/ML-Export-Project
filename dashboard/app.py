import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

DASH_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'res', 'dashboard')

def lucide(name):
    icons = {
        'globe': '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" x2="22" y1="12" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
        'package': '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16.5 9.4 7.55 4.24"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.29 7 12 12 20.71 7"/><line x1="12" x2="12" y1="22" y2="12"/></svg>',
        'building': '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2" ry="2"/><path d="M9 22v-4h6v4"/><line x1="8" x2="10" y1="6" y2="6"/><line x1="14" x2="16" y1="6" y2="6"/><line x1="8" x2="10" y1="10" y2="10"/><line x1="14" x2="16" y1="10" y2="10"/></svg>',
        'clipboard': '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><line x1="9" x2="15" y1="14" y2="14"/><line x1="9" x2="15" y1="10" y2="10"/><line x1="12" x2="12" y1="18" y2="18"/></svg>',
        'puzzle': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.611a2.404 2.404 0 0 1-1.705.706 2.404 2.404 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.404 2.404 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.315 8.685a.98.98 0 0 1 .837-.276c.47.07.802.48.968.925a2.501 2.501 0 1 0 3.214-3.214c-.446-.166-.855-.497-.925-.968a.979.979 0 0 1 .276-.837l1.611-1.611a2.404 2.404 0 0 1 1.704-.706c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.969a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.968 1.02Z"/></svg>',
        'tag': '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/><path d="M7 7h.01"/></svg>',
        'alert-triangle': '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
    }
    return dcc.Markdown(icons[name], dangerously_allow_html=True)

COLORS = {
    'primary': '#1a56db',
    'secondary': '#7c3aed',
    'success': '#16a34a',
    'warning': '#ca8a04',
    'danger': '#dc2626',
    'dark': '#1e293b',
    'light': '#f8fafc',
    'gray': '#64748b',
    'algeria': '#16a34a',
    'world': '#1a56db',
}

OPP_COLORS = {'High': '#16a34a', 'Medium': '#ca8a04', 'Low': '#dc2626', 'Unknown': '#64748b'}

def load_csv(name):
    path = os.path.join(DASH_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 35:
        return pd.read_csv(path)
    return pd.DataFrame()

agg_year_sector = load_csv('agg_year_sector.csv')
agg_year_continent = load_csv('agg_year_continent.csv')
agg_year_sector_continent = load_csv('agg_year_sector_continent.csv')
yearly_trend = load_csv('yearly_trend.csv')
top_products = load_csv('top_products.csv')
summary_stats = load_csv('summary_stats.csv')
opp_ranking = load_csv('opportunity_ranking.csv')

for df_ in [agg_year_sector, agg_year_continent, agg_year_sector_continent, yearly_trend, top_products, opp_ranking]:
    if not df_.empty:
        for col in df_.select_dtypes(include=['int64', 'int32']).columns:
            df_[col] = df_[col].astype('int64')
        for col in df_.select_dtypes(include=['float64', 'float32']).columns:
            df_[col] = df_[col].astype('float64')

cmp_t1 = load_csv('comparison_task1.csv')
cmp_t2 = load_csv('comparison_task2.csv')
alg_yearly = load_csv('algeria_yearly.csv')
sector_demand_index = load_csv('sector_demand_index.csv')
pca_sample = load_csv('pca_sample.csv')
feature_corr = load_csv('feature_correlation.csv')
<<<<<<< Updated upstream
=======
country_clusters = load_csv('country_clusters.csv')
product_clusters = load_csv('product_clusters.csv')
cluster_stats = load_csv('cluster_statistics.csv')
cluster_sector_comp = load_csv('cluster_sector_composition.csv')
forecast_data = load_csv('forecast_data.csv')
classification_results = load_csv('classification_results.csv')
>>>>>>> Stashed changes

if not summary_stats.empty:
    stats_dict = dict(zip(summary_stats['metric'], summary_stats['value']))
else:
    stats_dict = {}

app = Dash(__name__, title='Algeria Export Intelligence', external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

NAV = dbc.Navbar(
    dbc.Container([
        html.A('Algeria Export Intelligence', className='navbar-brand fw-bold fs-4', style={'color': '#fff'}),
        dcc.Tabs(id='tabs', value='overview', className='custom-tabs', children=[
            dcc.Tab(label='Overview', value='overview', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Trade Explorer', value='explorer', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Forecasts & Models', value='forecasts', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Opportunities', value='opportunities', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Intelligence', value='intelligence', className='custom-tab', selected_className='custom-tab--selected'),
        ]),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center'}),
    color='#1e293b', dark=True, className='mb-0', style={'padding': '0.5rem 0'},
)

def kpi_card(title, value, color, icon):
    icon_bg = {'#1a56db': '#e8effd', '#16a34a': '#dcfce7', '#7c3aed': '#f3e8ff', '#ca8a04': '#fef9c3'}
    bg = icon_bg.get(color, '#f1f5f9')
    return html.Div(
        className='kpi-card',
        children=dbc.CardBody([
            html.Div([
                html.Div(icon, className='kpi-icon', style={'backgroundColor': bg, 'color': color}),
                html.Div([
                    html.Div(title, className='kpi-label'),
                    html.Div(value, className='kpi-value', style={'color': color}),
                ], style={'flex': '1'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1rem'}),
        ]),
    )


def build_overview():
    if yearly_trend.empty:
        return html.Div('Run prepare_data.py first.', className='text-center p-5')
    kpis = dbc.Row([
        dbc.Col(kpi_card('Total Trade', stats_dict.get('Total Trade Value (all years)', '—'), COLORS['primary'], lucide('globe')),
            xs=12, sm=6, lg=3, className='mb-3'),
        dbc.Col(kpi_card('Algeria Exports', stats_dict.get('Algeria Total Exports (all years)', '—'), COLORS['algeria'], lucide('package')),
            xs=12, sm=6, lg=3, className='mb-3'),
        dbc.Col(kpi_card('Markets Covered', stats_dict.get('Unique Importers', '—'), COLORS['secondary'], lucide('building')),
            xs=12, sm=6, lg=3, className='mb-3'),
        dbc.Col(kpi_card('Products Traded', stats_dict.get('Unique Products', '—'), COLORS['warning'], lucide('clipboard')),
            xs=12, sm=6, lg=3, className='mb-3'),
    ])

    fig_trend = px.line(
        yearly_trend, x='year', y='total_value',
        title='Global Trade Trend (All Sectors)',
        labels={'total_value': 'Total Trade Value (USD)', 'year': ''},
        color_discrete_sequence=[COLORS['primary']],
    )
    fig_trend.update_layout(
        template='plotly_white', hovermode='x unified',
        margin=dict(t=30, b=10, l=10, r=10),
        yaxis=dict(tickformat='.2s'),
        title=dict(font=dict(size=13, weight=700)),
        legend=dict(orientation='h', y=1.02, font=dict(size=11)),
    )
    fig_trend.add_scatter(
        x=yearly_trend['year'], y=yearly_trend['algeria_export_v'],
        mode='lines+markers', name='Algeria Exports',
        line=dict(color=COLORS['algeria'], width=2.5),
    )

    if not agg_year_sector.empty:
        latest = agg_year_sector[agg_year_sector['year'] == agg_year_sector['year'].max()]
        top_sectors = latest.sort_values('total_value', ascending=True).tail(10)
        fig_sectors = px.bar(
            top_sectors, y='sector', x='total_value', orientation='h',
            title=f'Top Sectors by Trade Volume ({int(agg_year_sector["year"].max())})',
            labels={'total_value': 'USD', 'sector': ''},
            color='total_value', color_continuous_scale='Blues',
            text_auto='.2s',
        )
        fig_sectors.update_layout(template='plotly_white', margin=dict(t=30, b=10, l=10, r=10), showlegend=False,
                                   title=dict(font=dict(size=13, weight=700)))
    else:
        fig_sectors = go.Figure()

    if not agg_year_continent.empty:
        cont_pivot = agg_year_continent.pivot_table(
            index='year', columns='continent', values='total_value', aggfunc='sum'
        ).fillna(0)
        fig_cont = px.area(
            cont_pivot.reset_index().melt(id_vars='year'),
            x='year', y='value', color='continent',
            title='Trade Volume by Continent',
            labels={'value': 'USD', 'year': '', 'continent': ''},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_cont.update_layout(template='plotly_white', margin=dict(t=30, b=10, l=10, r=10), hovermode='x unified',
                                title=dict(font=dict(size=13, weight=700)))
    else:
        fig_cont = go.Figure()

    fig_alg_profile = go.Figure()
    if not alg_yearly.empty:
        fig_alg_profile.add_trace(go.Bar(x=alg_yearly['year'], y=alg_yearly['algeria_export_v'],
                                          name='Algeria Exports', marker_color=COLORS['algeria'],
                                          hovertemplate='%{y:$.2s}<extra></extra>'))
        fig_alg_profile.update_layout(template='plotly_white', margin=dict(t=30, b=10, l=10, r=10),
                                       yaxis=dict(tickformat='.2s'), hovermode='x unified',
                                       title=dict(text='Algeria Exports by Year', font=dict(size=13, weight=700)))

    fig_alg_sector = go.Figure()
    if not agg_year_sector.empty:
        alg_sector_total = agg_year_sector.groupby('sector')['algeria_export_v'].sum().sort_values(ascending=True)
        colors_ = ['#dc2626' if 'Energy' in s else '#3a7abf' for s in alg_sector_total.index]
        fig_alg_sector.add_trace(go.Bar(y=alg_sector_total.index, x=alg_sector_total.values,
                                         orientation='h', marker_color=colors_,
                                         hovertemplate='%{x:$.2s}<extra></extra>'))
        fig_alg_sector.update_layout(template='plotly_white', margin=dict(t=30, b=10, l=10, r=10),
                                      xaxis=dict(tickformat='.2s'),
                                      title=dict(text='Algeria Cumulative Exports by Sector', font=dict(size=13, weight=700)))

    def sec(title):
        return html.Div(className='section-header', children=[
            html.Div(className='accent'),
            html.H5(title),
        ])

    return html.Div([
        kpis,
        html.Div(style={'height': '0.5rem'}),
        sec('Trade & Export Overview'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_trend, style={'height': '440px'})]), className=''), xs=12, lg=8, className='mb-4'),
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_sectors, style={'height': '440px'})]), className=''), xs=12, lg=4, className='mb-4'),
        ]),
        sec('Regional & Sector Breakdown'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_cont, style={'height': '400px'})]), className=''), xs=12, lg=8, className='mb-4'),
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_alg_sector, style={'height': '400px'})]), className=''), xs=12, lg=4, className='mb-4'),
        ]),
        sec('Algeria Export Profile'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=fig_alg_profile, style={'height': '340px'})]), className=''), xs=12, className='mb-2'),
        ]),
    ])


def build_explorer():
    sectors = sorted(agg_year_sector['sector'].unique()) if not agg_year_sector.empty else []
    continents = sorted(agg_year_continent['continent'].unique()) if not agg_year_continent.empty else []
    years = sorted(int(y) for y in agg_year_sector['year'].unique()) if not agg_year_sector.empty else []

    return html.Div([
        html.Div(className='filter-controls', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Year Range'),
                    dcc.RangeSlider(
                        id='ex-year', min=min(years), max=max(years),
                        value=[min(years), max(years)],
                        marks={y: str(y) for y in years[::2]},
                        tooltip={'placement': 'bottom'},
                    ),
                ], xs=12, md=4, className='mb-2 mb-md-0'),
                dbc.Col([
                    html.Label('Sector'),
                    dcc.Dropdown(
                        id='ex-sector', options=[{'label': 'All', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in sectors],
                        value='ALL', clearable=False,
                    ),
                ], xs=12, md=4, className='mb-2 mb-md-0'),
                dbc.Col([
                    html.Label('Continent'),
                    dcc.Dropdown(
                        id='ex-continent',
                        options=[{'label': 'All', 'value': 'ALL'}] + [{'label': c, 'value': c} for c in continents],
                        value='ALL', clearable=False,
                    ),
                ], xs=12, md=4),
            ]),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='ex-trend-chart', style={'height': '420px'})]), className=''), xs=12, lg=6, className='mb-4'),
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='ex-export-chart', style={'height': '420px'})]), className=''), xs=12, lg=6, className='mb-4'),
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='ex-top-products', style={'height': '450px'})]), className=''), xs=12, lg=6, className='mb-4'),
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='ex-sector-demand', style={'height': '450px'})]), className=''), xs=12, lg=6, className='mb-4'),
        ]),
    ])


def build_forecasts():
    has_eval = not cmp_t1.empty
    def sec(title):
        return html.Div(className='section-header', children=[
            html.Div(className='accent'),
            html.H5(title),
        ])

    return html.Div([
        sec('Model Comparison'),
        dbc.Row([
            dbc.Col([
                html.H5('World Demand (total_value)', className='fw-bold', style={'fontSize': '0.9rem', 'marginBottom': '0.75rem'}),
                dash_table.DataTable(
                    id='cmp-t1-table',
                    columns=[{'name': c, 'id': c} for c in (cmp_t1.columns.tolist() if has_eval else [])],
                    data=cmp_t1.to_dict('records') if has_eval else [],
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#1e293b', 'color': 'white', 'fontWeight': 600, 'fontSize': '0.8rem'},
                    style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': '0.8rem'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{model} = "Prophet"', 'column_id': 'model'}, 'backgroundColor': '#dcfce7'},
                    ],
                ),
            ], xs=12, lg=6, className='mb-4'),
            dbc.Col([
                html.H5('Algeria Exports (algeria_export_v)', className='fw-bold', style={'fontSize': '0.9rem', 'marginBottom': '0.75rem'}),
                dash_table.DataTable(
                    id='cmp-t2-table',
                    columns=[{'name': c, 'id': c} for c in (cmp_t2.columns.tolist() if has_eval else [])],
                    data=cmp_t2.to_dict('records') if has_eval else [],
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#1e293b', 'color': 'white', 'fontWeight': 600, 'fontSize': '0.8rem'},
                    style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': '0.8rem'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{model} = "Prophet"', 'column_id': 'model'}, 'backgroundColor': '#dcfce7'},
                    ],
                ),
            ], xs=12, lg=6, className='mb-4'),
        ]),
        sec('Model Insights'),
        dbc.Row([
            dbc.Col(html.Div(className='info-panel', children=[
                html.H5('Best Model Summary'),
                html.P('Prophet is the best-performing model for both tasks based on RMSE.', className='text-muted', style={'fontSize': '0.85rem'}),
                html.Ul(style={'paddingLeft': '1.25rem', 'fontSize': '0.85rem', 'lineHeight': '1.8'}, children=[
                    html.Li([html.Strong('Task 1 (World Demand): '), 'Prophet — lowest MAE and RMSE on test set']),
                    html.Li([html.Strong('Task 2 (Algeria Exports): '), 'Prophet — lowest MAE and RMSE on test set']),
                    html.Li([html.Strong('LSTM: '), 'competitive MAPE on Task 1 validation (30%)']),
                    html.Li([html.Strong('XGBoost: '), 'needs improvement — extremely high MAPE values suggest feature projection issues']),
                ]),
            ]), xs=12, lg=6, className='mb-4'),
            dbc.Col(html.Div(className='info-panel', children=[
                html.H5('Model Selection Logic'),
                html.P('Forecasting strategy used in this project:', className='text-muted', style={'fontSize': '0.85rem'}),
                html.Ol(style={'paddingLeft': '1.25rem', 'fontSize': '0.85rem', 'lineHeight': '1.8'}, children=[
                    html.Li('Prophet runs on every (importer, product) time series'),
                    html.Li('If Prophet\'s validation MAPE > 50%, the series is flagged as "underfitting"'),
                    html.Li('LSTM replaces Prophet\'s forecast for underfitting series'),
                    html.Li('Final forecast = Prophet + LSTM hybrid'),
                ]),
            ]), xs=12, lg=6, className='mb-4'),
        ]),
        sec('Historical Trend'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(
                    figure=px.line(
                        yearly_trend, x='year', y=['total_value', 'algeria_export_v'],
                        labels={'value': 'USD', 'year': '', 'variable': ''},
                        title='',
                        color_discrete_map={'total_value': COLORS['primary'], 'algeria_export_v': COLORS['algeria']},
                    ).update_layout(template='plotly_white', hovermode='x unified', margin=dict(t=15, b=10, l=10, r=10),
                                    yaxis=dict(tickformat='.2s'),
                                    legend=dict(orientation='h', y=1.02, font=dict(size=11)))
                ),
            ]), className=''), xs=12, className='mb-4'),
        ]),
        sec('Generate Forecast'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.Div(lucide('alert-triangle'), className='pending-icon', style={'backgroundColor': '#fef3c7', 'color': '#ca8a04', 'borderRadius': '12px', 'width': '48px', 'height': '48px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexShrink': 0, 'marginRight': '1rem'}),
                    html.Div([
                        html.H5('Forecast Data Not Yet Available', className='fw-bold', style={'color': '#ca8a04', 'margin': 0}),
                        html.P('Run the forecasting notebooks to generate 2025-2027 predictions, then re-run prepare_data.py.', className='text-muted mt-1', style={'fontSize': '0.85rem'}),
                    ]),
                ], style={'display': 'flex', 'alignItems': 'flex-start'}),
                html.Hr(),
                html.Code('notebooks/full_forecasting.ipynb', style={'display': 'block', 'padding': '10px', 'backgroundColor': '#f1f5f9', 'borderRadius': '6px', 'fontSize': '0.85rem'}),
                html.P(['Once done, re-run ', html.Code('python dashboard/prepare_data.py'), ' to include forecast data in this dashboard.'], className='text-muted mt-2', style={'fontSize': '0.85rem'}),
            ]), className='', style={'borderLeft': '4px solid #ca8a04'}), xs=12, className='mb-2'),
        ]),
    ])


def build_opportunities():
    def sec(title):
        return html.Div(className='section-header', children=[
            html.Div(className='accent'),
            html.H5(title),
        ])

    return html.Div([
        sec('Filter Opportunities'),
        html.Div(className='filter-controls', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Opportunity Level'),
                    dcc.Dropdown(
                        id='opp-filter',
                        options=[
                            {'label': 'All Opportunities', 'value': 'ALL'},
                            {'label': 'High', 'value': 'High'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'Low', 'value': 'Low'},
                        ],
                        value='High', clearable=False,
                    ),
                ], xs=12, md=4, className='mb-2 mb-md-0'),
                dbc.Col([
                    html.Label('Sector'),
                    dcc.Dropdown(
                        id='opp-sector',
                        options=[{'label': 'All', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in sorted(opp_ranking['sector'].unique())] if not opp_ranking.empty else [],
                        value='ALL', clearable=False,
                    ),
                ], xs=12, md=4, className='mb-2 mb-md-0'),
                dbc.Col([
                    html.Label('Top N Results'),
                    dcc.Dropdown(id='opp-top', options=[{'label': str(n), 'value': n} for n in [10, 20, 50, 100]], value=20, clearable=False),
                ], xs=12, md=4),
            ]),
        ]) if not opp_ranking.empty else html.Div(),
        sec('Top Export Opportunities'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div(id='opp-table-container'),
            ]), className=''), xs=12, lg=8, className='mb-4'),
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(id='opp-dist-chart', style={'height': '380px'}),
            ]), className=''), xs=12, lg=4, className='mb-4'),
        ]),
        sec('Opportunity Breakdown'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(id='opp-sector-chart', style={'height': '420px'}),
            ]), className=''), xs=12, lg=6, className='mb-4'),
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(id='opp-continent-chart', style={'height': '420px'}),
            ]), className=''), xs=12, lg=6, className='mb-4'),
        ]),
    ])


def build_intelligence():
    pca_fig = go.Figure()
    opp_colors_map = {'High': '#16a34a', 'Medium': '#ca8a04', 'Low': '#dc2626'}
    if not pca_sample.empty and 'opportunity_label' in pca_sample.columns:
        pca_sample['opp_label'] = pca_sample['opportunity_label'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        for lbl, grp in pca_sample.groupby('opp_label'):
            xs = np.random.randn(len(grp)) * 0.1
            ys = np.random.randn(len(grp)) * 0.1
            pca_fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers',
                name=lbl, marker=dict(color=opp_colors_map.get(lbl, '#64748b'), size=3, opacity=0.3),
                showlegend=True
            ))
        pca_fig.update_layout(
            template='plotly_white', margin=dict(t=30, b=10, l=10, r=10),
            title=dict(text='Feature Space Overview (PCA)', font=dict(size=13, weight=700)),
            xaxis_title='PC1', yaxis_title='PC2',
            legend=dict(font=dict(size=10), orientation='h', y=1.02),
        )

    corr_fig = go.Figure()
    if not feature_corr.empty:
        corr_fig.add_trace(go.Heatmap(
            z=feature_corr.values,
            x=feature_corr.columns,
            y=feature_corr.columns,
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>',
        ))
        corr_fig.update_layout(
            template='plotly_white', margin=dict(t=30, b=140, l=140, r=10),
            title=dict(text='Feature Correlation Matrix', font=dict(size=13, weight=700)),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
            height=620,
        )

    def sec(title):
        return html.Div(className='section-header', children=[
            html.Div(className='accent'),
            html.H5(title),
        ])

    return html.Div([
        sec('Feature Analysis'),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=corr_fig, style={'height': '640px'}),
            ]), className=''), xs=12, lg=7, className='mb-4'),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    dcc.Graph(figure=pca_fig, style={'height': '380px'}),
                    html.Hr(style={'margin': '0.75rem 0'}),
                    html.P('21 numeric features → 16 PCA components (92.6% variance)', className='text-muted', style={'fontSize': '0.8rem', 'margin': 0}),
                    html.P('Placeholder plot — run full PCA pipeline for live projection.', className='text-muted', style={'fontSize': '0.8rem'}),
                ]), className='mb-4'),
                dbc.Card(dbc.CardBody([
                    html.H5('Key Insights', style={'fontWeight': 700, 'fontSize': '0.9rem', 'marginBottom': '0.75rem'}),
                    html.Ul(style={'paddingLeft': '1.1rem', 'fontSize': '0.83rem', 'lineHeight': '1.9', 'margin': 0}, children=[
                        html.Li('PC1 captures market size & economic weight'),
                        html.Li('PC2 separates gravity features (distance, language)'),
                        html.Li('PC3-4: trade openness & per-capita income'),
                        html.Li('Language/cultural proximity spreads across PC1, PC5-6'),
                    ]),
                ]), className=''),
            ], xs=12, lg=5, className='mb-4'),
        ]),
        sec('Clustering & Classification'),
        dbc.Row([
            dbc.Col(html.Div(className='info-panel', children=[
                html.H5('Model Status'),
                html.Div(className='pending-box', style={'borderLeft': '4px solid #7c3aed'}, children=[
                    html.Div([
                        html.Div(lucide('puzzle'), className='pending-icon', style={'backgroundColor': '#f3e8ff', 'color': '#7c3aed', 'borderRadius': '8px', 'width': '38px', 'height': '38px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexShrink': 0}),
                        html.Div([
                            html.Strong('Clustering Results', style={'fontSize': '0.9rem'}),
                            html.P('Not yet integrated.', className='text-muted', style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                            html.Span('Pending', className='badge bg-warning'),
                        ]),
                    ], style={'display': 'flex', 'alignItems': 'flex-start'}),
                    html.Hr(style={'margin': '0.75rem 0'}),
                    html.Div([
                        html.Small('Expected file: ', style={'color': '#64748b'}),
                        html.Code('data/res/cluster_labels.csv', style={'fontSize': '0.8rem'}),
                    ]),
                    html.Small('Columns: importer, product, cluster_id, cluster_name', style={'color': '#94a3b8', 'display': 'block', 'marginTop': '0.25rem'}),
                ]),
                html.Div(className='pending-box', style={'borderLeft': '4px solid #1a56db', 'marginBottom': 0}, children=[
                    html.Div([
                        html.Div(lucide('tag'), className='pending-icon', style={'backgroundColor': '#e0f2fe', 'color': '#1a56db', 'borderRadius': '8px', 'width': '38px', 'height': '38px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexShrink': 0}),
                        html.Div([
                            html.Strong('Classification Results', style={'fontSize': '0.9rem'}),
                            html.P('Not yet integrated.', className='text-muted', style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                            html.Span('Pending', className='badge bg-warning'),
                        ]),
                    ], style={'display': 'flex', 'alignItems': 'flex-start'}),
                    html.Hr(style={'margin': '0.75rem 0'}),
                    html.Div([
                        html.Small('Expected file: ', style={'color': '#64748b'}),
                        html.Code('data/res/classification_results.csv', style={'fontSize': '0.8rem'}),
                    ]),
                    html.Small('Columns: importer, product, opportunity_label (High/Medium/Low)', style={'color': '#94a3b8', 'display': 'block', 'marginTop': '0.25rem'}),
                ]),
            ]), xs=12, lg=6, className='mb-4'),
<<<<<<< Updated upstream
            dbc.Col(html.Div(className='info-panel', children=[
                html.H5('Integration Instructions'),
                html.Ol(style={'paddingLeft': '1.1rem', 'fontSize': '0.85rem', 'lineHeight': '2'}, children=[
                    html.Li('Run your clustering/classification notebooks to produce CSV outputs'),
                    html.Li(['Save results to ', html.Code('data/res/cluster_labels.csv'), ' and ', html.Code('data/res/classification_results.csv')]),
                    html.Li(['Re-run: ', html.Code('python dashboard/prepare_data.py')]),
                    html.Li('The dashboard will automatically detect and display your results'),
                ]),
                html.P(['See ', html.Code('dashboard/INTEGRATION_GUIDE.md'), ' for detailed instructions.'], className='text-muted mt-2', style={'fontSize': '0.8rem'}),
            ]), xs=12, lg=6, className='mb-4'),
        ]),
    ])
=======
        ]))

    children.append(sec('Classification'))

    if not classification_results.empty:
        # ── Show actual classification results ────────────────────────
        label_counts = classification_results['opportunity_label'].value_counts()
        total        = len(classification_results)

        # Summary cards
        cards = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div('High Opportunities', className='kpi-label'),
                html.Div(f"{label_counts.get('High', 0):,}", className='kpi-value',
                         style={'color': '#16a34a', 'fontSize': '1.8rem', 'fontWeight': 700}),
                html.Div(f"{100*label_counts.get('High',0)/total:.1f}% of pairs",
                         style={'color': '#64748b', 'fontSize': '0.8rem'}),
            ])), xs=12, md=4, className='mb-3'),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div('Medium Opportunities', className='kpi-label'),
                html.Div(f"{label_counts.get('Medium', 0):,}", className='kpi-value',
                         style={'color': '#ca8a04', 'fontSize': '1.8rem', 'fontWeight': 700}),
                html.Div(f"{100*label_counts.get('Medium',0)/total:.1f}% of pairs",
                         style={'color': '#64748b', 'fontSize': '0.8rem'}),
            ])), xs=12, md=4, className='mb-3'),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div('Low Opportunities', className='kpi-label'),
                html.Div(f"{label_counts.get('Low', 0):,}", className='kpi-value',
                         style={'color': '#dc2626', 'fontSize': '1.8rem', 'fontWeight': 700}),
                html.Div(f"{100*label_counts.get('Low',0)/total:.1f}% of pairs",
                         style={'color': '#64748b', 'fontSize': '0.8rem'}),
            ])), xs=12, md=4, className='mb-3'),
        ])

        # Pie chart
        label_df  = label_counts.reset_index()
        label_df.columns = ['opportunity_label', 'count']
        fig_pie   = px.pie(
            label_df, values='count', names='opportunity_label',
            title='Classification Results Distribution',
            color='opportunity_label',
            color_discrete_map=OPP_COLORS,
        )
        fig_pie.update_layout(
            template='plotly_white',
            margin=dict(t=40, b=10, l=10, r=10),
        )

        # Method info panel
        info_panel = html.Div(className='info-panel', children=[
            html.H5('Classification Methodology', style={'fontWeight': 700}),
            html.P([
                html.Strong('EPI Track: '),
                'Random Forest classifier (F1-macro = 0.51) trained on 56 RCA products. ',
                'Label defined as unrealized ITC Export Potential Indicator score.',
            ], style={'fontSize': '0.85rem', 'color': '#475569'}),
            html.P([
                html.Strong('PDI Track: '),
                'XGBoost classifier (F1-macro = 0.68) trained on 1,168 candidate products. ',
                'Label defined using product space density × ease × demand.',
            ], style={'fontSize': '0.85rem', 'color': '#475569'}),
            html.P([
                html.Strong('Leakage Fix: '),
                'Original labeling achieved 100% accuracy due to target leakage. ',
                'Replaced with ITC EPI/PDI methodology — honest performance confirmed.',
            ], style={'fontSize': '0.85rem', 'color': '#475569'}),
            html.Hr(),
            html.P([
                html.Strong('Total pairs classified: '),
                f'{total:,} unique (importer, product) combinations',
            ], style={'fontSize': '0.85rem', 'color': '#475569', 'margin': 0}),
        ])

        children.append(cards)
        children.append(dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                dcc.Graph(figure=fig_pie, style={'height': '360px'}),
            ])), xs=12, lg=6, className='mb-4'),
            dbc.Col(info_panel, xs=12, lg=6, className='mb-4'),
        ]))

    else:
        # ── Still pending ─────────────────────────────────────────────
        children.append(dbc.Row([
            dbc.Col(html.Div(className='info-panel', children=[
                html.Div(className='pending-box',
                         style={'borderLeft': '4px solid #1a56db', 'marginBottom': 0},
                         children=[
                    html.Div([
                        html.Div(lucide('tag'), className='pending-icon',
                                 style={'backgroundColor': '#e0f2fe', 'color': '#1a56db',
                                        'borderRadius': '8px', 'width': '38px', 'height': '38px',
                                        'display': 'flex', 'alignItems': 'center',
                                        'justifyContent': 'center', 'flexShrink': 0}),
                        html.Div([
                            html.Strong('Classification Results', style={'fontSize': '0.9rem'}),
                            html.P('Awaiting teammate integration.',
                                   className='text-muted',
                                   style={'fontSize': '0.8rem', 'margin': '0.2rem 0'}),
                            html.Span('Pending', className='badge bg-warning'),
                        ]),
                    ], style={'display': 'flex', 'alignItems': 'flex-start'}),
                    html.Hr(style={'margin': '0.75rem 0'}),
                    html.Div([
                        html.Small('Expected file: ', style={'color': '#64748b'}),
                        html.Code('data/res/classification_results.csv',
                                  style={'fontSize': '0.8rem'}),
                    ]),
                    html.Small('Columns: importer, product, opportunity_label (High/Medium/Low)',
                               style={'color': '#94a3b8', 'display': 'block',
                                      'marginTop': '0.25rem'}),
                ]),
            ]), xs=12, lg=6, className='mb-4'),
            dbc.Col(html.Div(className='info-panel', children=[
                html.H5('Integration Instructions'),
                html.Ol(style={'paddingLeft': '1.1rem', 'fontSize': '0.85rem',
                               'lineHeight': '2'}, children=[
                    html.Li('Run your classification notebook to produce a CSV'),
                    html.Li(['Save the file to ',
                             html.Code('data/res/classification_results.csv')]),
                    html.Li(['Re-run: ',
                             html.Code('python dashboard/prepare_data.py')]),
                    html.Li('Dashboard will automatically display your results'),
                ]),
            ]), xs=12, lg=6, className='mb-4'),
        ]))

    return html.Div(children)
>>>>>>> Stashed changes


# @app.callback(
#     [Output('overview-content', 'style'),
#      Output('explorer-content', 'style'),
#      Output('forecasts-content', 'style'),
#      Output('opportunities-content', 'style'),
#      Output('intelligence-content', 'style'),
#      Output('active-tab', 'data')],
#     [Input('tabs', 'value')],
# )
# def switch_tab(tab_id):
#     styles = {}
#     for t in ['overview', 'explorer', 'forecasts', 'opportunities', 'intelligence']:
#         styles[t] = {'display': 'block'} if t == tab_id else {'display': 'none'}
#     return styles['overview'], styles['explorer'], styles['forecasts'], styles['opportunities'], styles['intelligence'], tab_id

@app.callback(
    [Output('overview-content', 'style'),
     Output('explorer-content', 'style'),
     Output('forecasts-content', 'style'),
     Output('opportunities-content', 'style'),
     Output('intelligence-content', 'children'),
     Output('intelligence-content', 'style'),
     Output('active-tab', 'data')],
    [Input('tabs', 'value')],
)
def switch_tab(tab_id):
    styles = {}
    for t in ['overview', 'explorer', 'forecasts', 'opportunities']:
        styles[t] = {'display': 'block'} if t == tab_id else {'display': 'none'}
    intel_style = {'display': 'block'} if tab_id == 'intelligence' else {'display': 'none'}

    # Rebuild intelligence content fresh every time tab is clicked
    intel_children = build_intelligence().children if tab_id == 'intelligence' else []

    return (styles['overview'], styles['explorer'], styles['forecasts'],
            styles['opportunities'], intel_children, intel_style, tab_id)

def build_ex_trend(yr, sector, cont):
    data = agg_year_sector_continent.copy()
    if sector != 'ALL':
        data = data[data['sector'] == sector]
    if cont != 'ALL':
        data = data[data['continent'] == cont]
    data = data.groupby('year')[['total_value', 'algeria_export_v']].sum().reset_index()
    data = data[(data['year'] >= yr[0]) & (data['year'] <= yr[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['year'], y=data['total_value'], mode='lines+markers', name='World Demand', line=dict(color=COLORS['primary'], width=2)))
    fig.add_trace(go.Scatter(x=data['year'], y=data['algeria_export_v'], mode='lines+markers', name='Algeria Exports', line=dict(color=COLORS['algeria'], width=2)))
    fig.update_layout(title=dict(text='World Demand vs Algeria Exports', font=dict(size=13, weight=700)), template='plotly_white', hovermode='x unified', margin=dict(t=30, b=10, l=10, r=10), yaxis=dict(tickformat='.2s'), legend=dict(orientation='h', y=1.02, font=dict(size=11)))
    return fig


def build_ex_export(yr, sector, cont):
    data = agg_year_sector_continent.copy()
    if sector != 'ALL':
        data = data[data['sector'] == sector]
    if cont != 'ALL':
        data = data[data['continent'] == cont]
    data = data[(data['year'] >= yr[0]) & (data['year'] <= yr[1])]
    data = data.groupby('year')['algeria_export_v'].sum().reset_index()
    fig = px.bar(data, x='year', y='algeria_export_v', title='Algeria Exports Over Time',
                 labels={'algeria_export_v': 'USD', 'year': ''}, color_discrete_sequence=[COLORS['algeria']],
                 text_auto='.2s')
    fig.update_layout(template='plotly_white', margin=dict(t=30, b=10, l=10, r=10), showlegend=False,
                      title=dict(font=dict(size=13, weight=700)))
    return fig


def build_ex_products(yr, sector, cont):
    data = top_products.copy()
    if sector != 'ALL':
        data = data[data['sector'] == sector]
    if data.empty:
        return go.Figure()
    data = data.sort_values('total_value', ascending=True).tail(15)
    fig = px.bar(data, y='product_name', x='total_value', orientation='h',
                 title='Top Products by Trade Volume',
                 labels={'total_value': 'USD', 'product_name': ''},
                 color='total_value', color_continuous_scale='Blues', text_auto='.2s')
    fig.update_layout(template='plotly_white', margin=dict(t=30, b=10, l=10, r=10), showlegend=False,
                      title=dict(font=dict(size=13, weight=700)))
    return fig


def build_ex_sector_demand(yr, sector):
    if sector_demand_index.empty:
        return go.Figure()
    top_sec = [c for c in sector_demand_index.columns if c != 'year']
    if sector != 'ALL' and sector in sector_demand_index.columns:
        top_sec = [sector]
    data = sector_demand_index[['year'] + top_sec].copy()
    data = data[(data['year'] >= yr[0]) & (data['year'] <= yr[1])]
    fig = go.Figure()
    for sec in top_sec:
        fig.add_trace(go.Scatter(x=data['year'], y=data[sec], mode='lines+markers',
                                  name=sec, line=dict(width=2)))
    fig.add_vline(x=2020, line_dash='dash', line_color='gray', annotation_text='COVID')
    fig.update_layout(template='plotly_white', hovermode='x unified',
                       margin=dict(t=30, b=10, l=10, r=10),
                       title=dict(text='Sector Demand Trends (Indexed 2012=1.0)', font=dict(size=13, weight=700)),
                       yaxis_title='Index',
                       legend=dict(font=dict(size=9), orientation='h', y=1.02))
    return fig


@app.callback(
    [Output('ex-trend-chart', 'figure'),
     Output('ex-export-chart', 'figure'),
     Output('ex-top-products', 'figure'),
     Output('ex-sector-demand', 'figure')],
    [Input('ex-year', 'value'),
     Input('ex-sector', 'value'),
     Input('ex-continent', 'value')],
)
def update_explorer(yr, sector, cont):
    if yr is None:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    return (build_ex_trend(yr, sector, cont),
            build_ex_export(yr, sector, cont),
            build_ex_products(yr, sector, cont),
            build_ex_sector_demand(yr, sector))


@app.callback(
    [Output('opp-table-container', 'children'),
     Output('opp-dist-chart', 'figure'),
     Output('opp-sector-chart', 'figure'),
     Output('opp-continent-chart', 'figure')],
    [Input('opp-filter', 'value'),
     Input('opp-sector', 'value'),
     Input('opp-top', 'value')],
)
def update_opportunities(opp_level, sector, top_n):
    if opp_ranking.empty:
        return html.P('No opportunity data available. Run forecasting notebooks first.', className='text-muted'), go.Figure(), go.Figure(), go.Figure()

    data = opp_ranking.copy()
    if opp_level != 'ALL':
        data = data[data['opportunity'] == opp_level]
    if sector != 'ALL':
        data = data[data['sector'] == sector]
    data = data.head(top_n)

    table = dash_table.DataTable(
        columns=[
            {'name': 'Country', 'id': 'country_name'},
            {'name': 'Product', 'id': 'product_name'},
            {'name': 'Sector', 'id': 'sector'},
            {'name': 'Demand Gap (USD)', 'id': 'demand_gap', 'type': 'numeric', 'format': {'specifier': '$.2s'}},
            {'name': 'Opportunity', 'id': 'opportunity'},
        ],
        data=data.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#1e293b', 'color': 'white', 'fontWeight': 600},
        style_cell={'textAlign': 'left', 'padding': '6px', 'fontSize': '0.8rem'},
        style_data_conditional=[
            {'if': {'filter_query': '{opportunity} = "High"', 'column_id': 'opportunity'}, 'color': '#16a34a', 'fontWeight': 700},
            {'if': {'filter_query': '{opportunity} = "Low"', 'column_id': 'opportunity'}, 'color': '#dc2626'},
        ],
        page_size=15,
    )

    opp_counts = opp_ranking['opportunity'].value_counts().reset_index()
    opp_counts.columns = ['opportunity', 'count']
    fig_dist = px.pie(opp_counts, values='count', names='opportunity', title='Opportunity Distribution',
                      color='opportunity', color_discrete_map=OPP_COLORS)
    fig_dist.update_layout(template='plotly_white', margin=dict(t=40, b=10, l=10, r=10))

    sec_counts = opp_ranking.groupby(['sector', 'opportunity']).size().reset_index(name='count')
    fig_sec = px.bar(sec_counts, x='sector', y='count', color='opportunity', title='Opportunities by Sector',
                     color_discrete_map=OPP_COLORS, barmode='stack',
                     labels={'sector': '', 'count': 'Number of Pairs', 'opportunity': ''})
    fig_sec.update_layout(template='plotly_white', margin=dict(t=40, b=80, l=10, r=10), xaxis_tickangle=-45)

    cont_counts = opp_ranking.groupby(['continent', 'opportunity']).size().reset_index(name='count')
    fig_cont = px.bar(cont_counts, x='continent', y='count', color='opportunity', title='Opportunities by Continent',
                      color_discrete_map=OPP_COLORS, barmode='stack',
                      labels={'continent': '', 'count': 'Number of Pairs', 'opportunity': ''})
    fig_cont.update_layout(template='plotly_white', margin=dict(t=40, b=10, l=10, r=10))

    return table, fig_dist, fig_sec, fig_cont


overview_content = build_overview()
explorer_content = build_explorer()
forecasts_content = build_forecasts()
opportunities_content = build_opportunities()
intelligence_content = build_intelligence()

app.layout = dbc.Container([
    NAV,
    dbc.Container([
        dcc.Store(id='active-tab', data='overview'),
        html.Div(overview_content, id='overview-content'),
        html.Div(explorer_content, id='explorer-content', style={'display': 'none'}),
        html.Div(forecasts_content, id='forecasts-content', style={'display': 'none'}),
        html.Div(opportunities_content, id='opportunities-content', style={'display': 'none'}),
        html.Div(intelligence_content, id='intelligence-content', style={'display': 'none'}),
    ], fluid=True, className='main-content'),
], fluid=True, style={'backgroundColor': '#f1f5f9', 'minHeight': '100vh', 'padding': 0})


if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:8050')
    app.run(debug=False, port=8050)
