import streamlit as st
st.set_page_config(page_title="Prévision SARIMA", page_icon="🟣")

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_auth
require_auth("SARIMA")

import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
from utils import get_connection, sidebar_filters
from holidays_utils import add_holiday_feature, get_holiday_info_text, resolve_country_code
warnings.filterwarnings("ignore")

st.title("📈 Dashboard Prévision des Ventes - SARIMA")

df, product, depot_id, depot_sel, date_range, selected_country = sidebar_filters()

if df is None or len(df) == 0:
    st.error(f"❌ Aucune vente trouvée pour le produit **{product or 'Tous'}** / dépôt **{depot_sel}**.")
    st.stop()

def prepare_and_smooth(df):
    df_d = (df.groupby(pd.Grouper(key='date_time', freq='D'))['quantity']
            .sum().reset_index()
            .rename(columns={'date_time': 'ds', 'quantity': 'y'}))
    df_d = df_d.set_index('ds').asfreq('D').reset_index()
    df_d['y'] = df_d['y'].fillna(0)
    n_nonzero = (df_d['y'] > 0).sum()
    window = 7 if n_nonzero >= 60 else (3 if n_nonzero >= 20 else 1)
    if window > 1:
        df_d['y'] = df_d['y'].rolling(window=window, min_periods=1, center=True).mean()
    mean_y, std_y = df_d['y'].mean(), df_d['y'].std()
    if std_y > 0:
        df_d['y'] = df_d['y'].clip(lower=0, upper=mean_y + 3 * std_y)
    return df_d.dropna(subset=['y']).reset_index(drop=True)

df_model = prepare_and_smooth(df)

# ── Jours fériés → variable exogène réelle ────
country_code = resolve_country_code(selected_country or "FR")
df_model = add_holiday_feature(df_model, country_code)
st.caption(get_holiday_info_text(country_code))
n_holidays_train = int(df_model['is_holiday'].sum())
st.info(f"📅 **{n_holidays_train} jour(s) férié(s)** détectés dans la série — "
        f"ils sont intégrés comme **variable exogène** dans SARIMA.")

label_produit = f"{product or 'Tous'} / {depot_sel}"
n_pts = len(df_model)
st.info(f"📅 [{label_produit}] Série lissée (journalière) — **{n_pts} points**")

MIN_POINTS = 10
if n_pts < MIN_POINTS:
    st.warning(f"⚠️ Seulement {n_pts} points disponibles (minimum requis : {MIN_POINTS}).")
    st.stop()

if n_pts < 60:
    df_model = (df_model.set_index('ds')
                .resample('W')
                .agg({'y': 'sum', 'is_holiday': 'max'})
                .reset_index())
    st.info(f"🔁 Données agrégées par semaine ({len(df_model)} semaines)")
    freq_label = "hebdomadaire"
    seasonal_period = 52
else:
    freq_label = "journalière"
    seasonal_period = 7

n_pts = len(df_model)
if n_pts < MIN_POINTS:
    st.warning(f"⚠️ Pas assez de données même après agrégation ({n_pts} points).")
    st.stop()

split_index = max(1, int(n_pts * 0.8))
train_df = df_model.iloc[:split_index].copy()
test_df  = df_model.iloc[split_index:].copy()
if len(test_df) == 0:
    train_df = df_model.iloc[:-1].copy()
    test_df  = df_model.iloc[-1:].copy()

# ── Recherche du meilleur ordre SARIMA ────────
@st.cache_data
def find_best_sarima_order(train_y_tuple, train_exog_tuple, s, small=False):
    train_y    = list(train_y_tuple)
    train_exog = np.array(train_exog_tuple).reshape(-1, 1)
    best_aic, best_order, best_seasonal = np.inf, (1, 1, 1), (0, 0, 0)

    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 2 if small else 3)
    P_range, D_range, Q_range = range(0, 2), range(0, 2), range(0, 2)

    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            if (p + d + q + P + D + Q) == 0:
                                continue
                            try:
                                m = SARIMAX(
                                    train_y,
                                    exog=train_exog,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False)
                                if m.aic < best_aic:
                                    best_aic      = m.aic
                                    best_order    = (p, d, q)
                                    best_seasonal = (P, D, Q)
                            except Exception:
                                continue
    return best_order, best_seasonal

is_small = n_pts < 60
with st.spinner("🔍 Recherche du meilleur ordre SARIMA (p,d,q)(P,D,Q)[s]..."):
    best_order, best_seasonal = find_best_sarima_order(
        tuple(train_df['y'].values),
        tuple(train_df['is_holiday'].values),
        s=seasonal_period,
        small=is_small,
    )

st.success(
    f"✅ Meilleur ordre SARIMA : {best_order} × {best_seasonal}[{seasonal_period}]"
    f"  —  jours fériés intégrés comme exogène"
)

# ── Entraînement + prédiction sur le test ─────
@st.cache_data
def train_sarima(train_y_tuple, train_exog_tuple, test_exog_tuple,
                 order, seasonal_order, s):
    train_y    = list(train_y_tuple)
    train_exog = np.array(train_exog_tuple).reshape(-1, 1)
    test_exog  = np.array(test_exog_tuple).reshape(-1, 1)
    model_fit  = SARIMAX(
        train_y,
        exog=train_exog,
        order=order,
        seasonal_order=(*seasonal_order, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    fc = model_fit.forecast(steps=len(test_exog), exog=test_exog)
    return np.array(fc.values if hasattr(fc, 'values') else fc)

with st.spinner("⏳ Entraînement SARIMA..."):
    test_preds = train_sarima(
        tuple(train_df['y'].values),
        tuple(train_df['is_holiday'].values),
        tuple(test_df['is_holiday'].values),
        best_order, best_seasonal, seasonal_period,
    )

y_true = test_df['y'].values
y_pred = test_preds

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('nan')

daily_avg         = df_model['y'].mean()
daily_avg_display = daily_avg / 7 if freq_label == "hebdomadaire" else daily_avg
mae_pct  = (mae  / daily_avg_display * 100) if daily_avg_display > 0 else float('nan')
rmse_pct = (rmse / daily_avg_display * 100) if daily_avg_display > 0 else float('nan')

# ── Prévisions futures avec exog jours fériés ─
@st.cache_data
def make_full_forecast_sarima(full_y_tuple, full_exog_tuple, order, seasonal_order,
                               s, last_date, freq_label, country_code):
    full_y    = list(full_y_tuple)
    full_exog = np.array(full_exog_tuple).reshape(-1, 1)

    model_full = SARIMAX(
        full_y,
        exog=full_exog,
        order=order,
        seasonal_order=(*seasonal_order, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    target       = pd.Timestamp("2026-12-31")
    freq         = 'W' if freq_label == "hebdomadaire" else 'D'
    future_steps = max(1, (target - last_date).days // (7 if freq == 'W' else 1))
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_steps,
        freq=freq,
    )

    from holidays_utils import get_holidays_df
    hdf = get_holidays_df(country_code)
    holiday_dates = set(hdf["ds"].dt.normalize()) if not hdf.empty else set()
    future_exog = np.array(
        [1 if d.normalize() in holiday_dates else 0 for d in future_dates],
        dtype=float,
    ).reshape(-1, 1)

    fc     = model_full.forecast(steps=future_steps, exog=future_exog)
    fc_arr = np.array(fc.values if hasattr(fc, 'values') else fc)
    return pd.DataFrame({'ds': future_dates, 'yhat': fc_arr, 'is_holiday': future_exog.flatten()})

with st.spinner("⏳ Génération des prévisions futures..."):
    forecast = make_full_forecast_sarima(
        tuple(df_model['y'].values),
        tuple(df_model['is_holiday'].values),
        best_order, best_seasonal, seasonal_period,
        df_model['ds'].max(), freq_label, country_code,
    )

forecast['month'] = forecast['ds'].dt.to_period('M')
monthly_forecast  = forecast.groupby('month')['yhat'].sum().reset_index()
monthly_forecast['month_str'] = monthly_forecast['month'].astype(str)

today = pd.Timestamp.today().normalize()
future_monthly = monthly_forecast[
    monthly_forecast['month'].apply(lambda p: p.to_timestamp()) >= today.replace(day=1)
].copy()

next_month_qty   = future_monthly['yhat'].iloc[0]  if len(future_monthly) >= 1 else float('nan')
next_month_label = future_monthly['month_str'].iloc[0] if len(future_monthly) >= 1 else "—"
best_month_qty   = future_monthly['yhat'].max()    if not future_monthly.empty else float('nan')
best_month_label = future_monthly.loc[future_monthly['yhat'].idxmax(), 'month_str'] if not future_monthly.empty else "—"
trend = ("📈 Hausse" if len(future_monthly) >= 2
         and future_monthly['yhat'].iloc[-1] > future_monthly['yhat'].iloc[0]
         else "📉 Baisse")

def get_quality(r2, mape):
    if r2 >= 0.85 and mape <= 10:   return "#28a745", "🟢 Excellent"
    elif r2 >= 0.70 and mape <= 20: return "#ffc107", "🟡 Bon"
    elif r2 >= 0.50 and mape <= 50: return "#fd7e14", "🟠 Moyen"
    else:                           return "#dc3545", "🔴 Faible"

color, label = get_quality(r2, mape)
r2_pct   = max(0.0, min(r2 if not np.isnan(r2) else 0, 1.0)) * 100
mape_val = mape if not np.isnan(mape) else 100
mape_bar = max(0.0, 100.0 - min(mape_val, 100.0))

chart_start = pd.Timestamp("2024-01-01")
chart_end   = pd.Timestamp("2026-12-31")
if df_model['ds'].max() < chart_start:
    chart_start = df_model['ds'].min()

train_chart      = train_df[train_df['ds'] >= chart_start]
test_chart       = test_df[(test_df['ds'] >= chart_start) & (test_df['ds'] <= chart_end)]
test_preds_s     = pd.Series(test_preds, index=test_df['ds'])
test_preds_chart = test_preds_s[(test_preds_s.index >= chart_start) & (test_preds_s.index <= chart_end)]
forecast_chart   = forecast[(forecast['ds'] >= chart_start) & (forecast['ds'] <= chart_end)]

tab1, tab2, tab3 = st.tabs(["📈 Graphique", "📊 Performance", "📌 KPIs"])

# =========================
# TAB 1 — Graphique
# =========================
with tab1:
    st.subheader(f"Historique + Prévision — {label_produit} ({freq_label})")
    fig = go.Figure()

    if not train_chart.empty:
        fig.add_trace(go.Scatter(
            x=train_chart['ds'], y=train_chart['y'],
            mode='lines', name='Train', line=dict(color='blue', width=1.5)))

    if not test_chart.empty:
        fig.add_trace(go.Scatter(
            x=test_chart['ds'], y=test_chart['y'],
            mode='lines', name='Test (réel)', line=dict(color='orange', width=1.5)))

    if not test_preds_chart.empty:
        fig.add_trace(go.Scatter(
            x=test_preds_chart.index, y=test_preds_chart.values,
            mode='lines', name='Prédiction (test)', line=dict(color='purple', dash='dash', width=1.5)))

    if not forecast_chart.empty:
        fig.add_trace(go.Scatter(
            x=forecast_chart['ds'], y=forecast_chart['yhat'],
            mode='lines', name='Prévision future', line=dict(color='green', width=2)))

    holidays_in_chart = df_model[
        (df_model['is_holiday'] == 1) &
        (df_model['ds'] >= chart_start) &
        (df_model['ds'] <= chart_end)
    ]
    if not holidays_in_chart.empty:
        fig.add_trace(go.Scatter(
            x=holidays_in_chart['ds'], y=holidays_in_chart['y'],
            mode='markers', name='Jour férié (historique)',
            marker=dict(symbol='star', color='red', size=8),
        ))

    holidays_forecast = forecast_chart[forecast_chart['is_holiday'] == 1]
    if not holidays_forecast.empty:
        fig.add_trace(go.Scatter(
            x=holidays_forecast['ds'], y=holidays_forecast['yhat'],
            mode='markers', name='Jour férié (prévu)',
            marker=dict(symbol='star', color='darkred', size=8, opacity=0.7),
        ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Quantité",
        xaxis=dict(range=[chart_start, chart_end]),
        template="plotly_white", hovermode="x unified", height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2 — Performance
# =========================
with tab2:
    st.subheader("Indicateurs de Performance du Modèle")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE",  f"{mae:.2f}",  help="Erreur absolue moyenne.")
    col2.metric("RMSE", f"{rmse:.2f}", help="Erreur quadratique — pénalise les grandes erreurs.")
    col3.metric("MAPE", f"{mape_val:.2f}%", help="Erreur en % par rapport aux vraies valeurs.")
    col4.metric("R²",   f"{r2:.4f}" if not np.isnan(r2) else "N/A", help="Proportion de variance expliquée.")
    col5.metric("Moy. vente / jour", f"{daily_avg_display:.2f}", help="Référence pour interpréter MAE et RMSE.")
    st.caption(
        f"💡 MAE = **{mae_pct:.1f}%** de la vente moyenne journalière · "
        f"RMSE = **{rmse_pct:.1f}%** — "
        f"{'✅ Erreur faible, modèle précis' if mae_pct < 20 else '⚠️ Erreur élevée par rapport à la moyenne'}."
    )

    st.divider()
    st.markdown("#### 📊 Erreurs vs Moyenne des ventes journalières")
    mae_pct_of_avg  = (mae  / daily_avg_display * 100) if daily_avg_display > 0 else float('nan')
    rmse_pct_of_avg = (rmse / daily_avg_display * 100) if daily_avg_display > 0 else float('nan')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**MAE ({mae:.2f})** représente **{mae_pct_of_avg:.1f}%** de la moyenne journalière ({daily_avg_display:.2f})")
        mae_bar_pct = min(mae_pct_of_avg, 100) if not np.isnan(mae_pct_of_avg) else 0
        mae_color = "#28a745" if mae_pct_of_avg <= 10 else "#ffc107" if mae_pct_of_avg <= 25 else "#dc3545"
        st.markdown(f"""
        <div style="background:#e0e0e0; border-radius:10px; height:22px; width:100%;">
            <div style="background:{mae_color}; width:{mae_bar_pct:.1f}%; height:22px; border-radius:10px;"></div>
        </div>
        <p style="text-align:right; font-size:12px; color:gray;">{'✅ Faible erreur' if mae_pct_of_avg <= 10 else '⚠️ Erreur modérée' if mae_pct_of_avg <= 25 else '❌ Erreur élevée'}</p>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"**RMSE ({rmse:.2f})** représente **{rmse_pct_of_avg:.1f}%** de la moyenne journalière ({daily_avg_display:.2f})")
        rmse_bar_pct = min(rmse_pct_of_avg, 100) if not np.isnan(rmse_pct_of_avg) else 0
        rmse_color = "#28a745" if rmse_pct_of_avg <= 10 else "#ffc107" if rmse_pct_of_avg <= 25 else "#dc3545"
        st.markdown(f"""
        <div style="background:#e0e0e0; border-radius:10px; height:22px; width:100%;">
            <div style="background:{rmse_color}; width:{rmse_bar_pct:.1f}%; height:22px; border-radius:10px;"></div>
        </div>
        <p style="text-align:right; font-size:12px; color:gray;">{'✅ Faible erreur' if rmse_pct_of_avg <= 10 else '⚠️ Erreur modérée' if rmse_pct_of_avg <= 25 else '❌ Erreur élevée'}</p>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### Qualité du modèle : {label}")
    col_r2, col_mape = st.columns(2)
    with col_r2:
        st.markdown("**R² — Coefficient de détermination**")
        r2_label = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        r2_text  = ('≥ 0.85 Excellent' if not np.isnan(r2) and r2 >= 0.85
                    else '≥ 0.70 Bon'  if not np.isnan(r2) and r2 >= 0.70
                    else '≥ 0.50 Moyen' if not np.isnan(r2) and r2 >= 0.50
                    else '< 0.50 Faible')
        st.markdown(f"""
        <div style="background:#e0e0e0; border-radius:10px; height:22px; width:100%;">
            <div style="background:{color}; width:{r2_pct:.1f}%; height:22px; border-radius:10px;"></div>
        </div>
        <p style="text-align:right; font-size:13px; color:gray;">R² = {r2_label} &nbsp;|&nbsp; {r2_text}</p>
        """, unsafe_allow_html=True)
    with col_mape:
        st.markdown("**MAPE — Erreur absolue en %**")
        mape_text = ('≤ 10% Excellent' if mape_val <= 10
                     else '≤ 20% Bon'  if mape_val <= 20
                     else '≤ 50% Moyen' if mape_val <= 50
                     else '> 50% Faible')
        st.markdown(f"""
        <div style="background:#e0e0e0; border-radius:10px; height:22px; width:100%;">
            <div style="background:{color}; width:{mape_bar:.1f}%; height:22px; border-radius:10px;"></div>
        </div>
        <p style="text-align:right; font-size:13px; color:gray;">MAPE = {mape_val:.2f}% &nbsp;|&nbsp; {mape_text}</p>
        """, unsafe_allow_html=True)

    st.divider()
    if label == "🟢 Excellent":   st.success("✅ SARIMA est très bien adapté à ce produit !")
    elif label == "🟡 Bon":       st.info("ℹ️ Bonne performance. SARIMA est fiable pour ce produit.")
    elif label == "🟠 Moyen":     st.warning("⚠️ Performance moyenne. Essayez xgboost ou LSTM.")
    else:                          st.error("❌ Performance faible. Essayez xgboost ou LSTM pour ce produit.")

# =========================
# TAB 3 — KPIs
# =========================
with tab3:
    st.subheader("📌 KPIs Clés")
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Prochain mois ({next_month_label})", f"{next_month_qty:,.0f} unités",
              help="Quantité totale prévue pour le mois qui vient.")
    k2.metric(f"Meilleur mois prévu ({best_month_label})", f"{best_month_qty:,.0f} unités",
              help="Le mois avec la plus forte prévision sur toute la période future.")
    k3.metric("Tendance générale", trend,
              help="Comparaison entre la prévision du premier et du dernier mois disponible.")

    st.divider()
    st.subheader("📅 Prévisions par mois à venir")

    if future_monthly.empty:
        st.info("Aucune prévision mensuelle disponible au-delà d'aujourd'hui.")
    else:
        future_monthly = future_monthly.copy()
        future_monthly['days_in_month'] = future_monthly['month'].apply(
            lambda p: p.to_timestamp().days_in_month
        )
        future_monthly['moyenne_jour'] = future_monthly['yhat'] / future_monthly['days_in_month']
        future_monthly['lower'] = (future_monthly['yhat'] - rmse * future_monthly['days_in_month']).clip(lower=0)
        future_monthly['upper'] = future_monthly['yhat'] + rmse * future_monthly['days_in_month']

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=future_monthly['month_str'],
            y=future_monthly['yhat'].round(0),
            name='Total prévu',
            marker_color='#6c63ff',
            opacity=0.85,
        ))
        fig_monthly.update_layout(
            template="plotly_white",
            xaxis_title="Mois", yaxis_title="Total prévu (quantité)",
            height=420, hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0), bargap=0.3,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        display_df = future_monthly[['month_str', 'yhat', 'moyenne_jour', 'lower', 'upper']].copy()
        display_df.columns = ['Mois', 'Total prévu', 'Moyenne/jour', 'Borne basse', 'Borne haute']
        display_df[['Total prévu', 'Moyenne/jour', 'Borne basse', 'Borne haute']] = \
            display_df[['Total prévu', 'Moyenne/jour', 'Borne basse', 'Borne haute']].round(2)
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Exporter les prévisions mensuelles (CSV)",
            data=csv,
            file_name="previsions_mensuelles_sarima.csv",
            mime="text/csv"
        )

# ✅ CLÉ CORRIGÉE — maintenant cohérente avec Comparaison Modèles
sess_key = f"sarima_{product}_{depot_id}"
st.session_state[sess_key] = {
    "MAE":       mae,
    "RMSE":      rmse,
    "MAPE":      mape,
    "R2":        r2,
    "quality":   label,
    "future":    forecast[['ds', 'yhat']].copy(),
    "product":   product,
    "depot_id":  depot_id,
    "depot_sel": depot_sel,
    "freq":      freq_label,
}

# =========================
# 🤖 ANALYSE IA
# =========================
from analyse.generateur import generer_analyse_sarima

st.divider()
col_btn = st.columns([2, 2, 2])
with col_btn[1]:
    btn = st.button("🤖 Analyse Intelligente", use_container_width=True, type="primary")

if btn:
    filtres = {
        "Produit"                 : product or "Tous",
        "Dépôt"                   : depot_sel,
        "Pays"                    : selected_country,
        "Fréquence"               : freq_label,
        "Ordre SARIMA (p,d,q)"    : str(best_order),
        "Saisonnalité (P,D,Q)[s]" : f"{best_seasonal}[{seasonal_period}]",
    }
    metriques = {
        "MAE"                                 : f"{mae:.2f} ({mae_pct:.1f}% de la moyenne journalière)",
        "RMSE"                                : f"{rmse:.2f} ({rmse_pct:.1f}% de la moyenne journalière)",
        "MAPE"                                : f"{mape_val:.2f}%",
        "R²"                                  : f"{r2:.4f}" if not np.isnan(r2) else "N/A",
        "Qualité du modèle"                   : label,
        "Moyenne vente/jour"                  : f"{daily_avg_display:.2f}",
        "Nombre de points"                    : n_pts,
        f"Prévision {next_month_label}"       : f"{next_month_qty:,.0f} unités",
        f"Meilleur mois ({best_month_label})" : f"{best_month_qty:,.0f} unités",
        "Tendance générale"                   : trend,
        "Jours fériés intégrés (exogène)"     : n_holidays_train,
        "Pays (jours fériés)"                 : country_code,
    }
    with st.spinner("🧠 Analyse en cours..."):
        analyse = generer_analyse_arima(filtres, metriques)
    with st.container(border=True):
        st.markdown(analyse)
    st.download_button(
        label="⬇️ Télécharger l'analyse",
        data=analyse,
        file_name=f"analyse_sarima_{product}_{depot_sel}.txt",
        mime="text/plain"
    )