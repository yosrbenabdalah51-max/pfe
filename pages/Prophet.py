import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
from utils import get_connection

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prophet Dashboard", page_icon="📊", layout="wide")
st.title("📈 Prévision de ventes avec Prophet")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT ref_product, quantity, date_time FROM sales", conn)
        conn.close()
        df['date_time'] = pd.to_datetime(df['date_time'])
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur de connexion : {e}")
        st.stop()

df = load_data()

product = st.session_state.get("product", None)
if product is None:
    st.warning("⚠️ Choisissez un produit depuis la page principale")
    st.stop()

# =========================
# Préparation + Lissage des données (Daily)
# =========================
@st.cache_data
def prepare_and_smooth(df, product):
    df_prod = df[df['ref_product'] == product].copy()

    df_d = (df_prod
            .groupby(pd.Grouper(key='date_time', freq='D'))['quantity']
            .sum()
            .reset_index()
            .rename(columns={'date_time': 'ds', 'quantity': 'y'}))

    df_d = df_d.set_index('ds').asfreq('D').reset_index()
    df_d['y'] = df_d['y'].replace(0, np.nan)
    df_d['y'] = df_d['y'].interpolate(method='linear')
    df_d['y'] = df_d['y'].fillna(method='bfill').fillna(method='ffill')

    # Rolling sur 7 jours
    df_d['y'] = df_d['y'].rolling(window=7, min_periods=1, center=True).mean()

    mean_y = df_d['y'].mean()
    std_y  = df_d['y'].std()
    df_d['y'] = df_d['y'].clip(lower=mean_y - 3 * std_y, upper=mean_y + 3 * std_y)

    df_d = df_d.dropna(subset=['y'])
    df_d = df_d[df_d['y'] > 0].reset_index(drop=True)
    return df_d

df_model = prepare_and_smooth(df, product)

st.info(f"📅 Série lissée (journalière, rolling 7 jours) — **{len(df_model)} points**")

if len(df_model) < 30:
    st.warning("⚠️ Pas assez de données pour Prophet avec ce produit.")
    st.stop()

# =========================
# Train / Test split 80/20
# =========================
split_index = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_index]
test_df  = df_model.iloc[split_index:]

# =========================
# Entraînement Prophet sur train
# =========================
@st.cache_resource
def train_prophet(train_df):
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0
    )
    m.fit(train_df)
    return m

with st.spinner("⏳ Entraînement Prophet..."):
    model = train_prophet(train_df)

# =========================
# Prédictions sur test
# =========================
forecast_test = model.predict(test_df[['ds']])
y_true = test_df['y'].values
y_pred = forecast_test['yhat'].values

# =========================
# Métriques
# =========================
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('nan')

# =========================
# Modèle complet pour forecast futur
# =========================
@st.cache_resource
def train_full_prophet(df_model):
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0
    )
    m.fit(df_model)
    return m

@st.cache_data
def make_forecast(_model, last_date):
    target       = pd.Timestamp("2026-12-31")
    future_days  = max(1, (target - last_date).days)
    future       = _model.make_future_dataframe(periods=future_days, freq='D')
    return _model.predict(future)

with st.spinner("⏳ Génération des prévisions futures..."):
    model_full = train_full_prophet(df_model)
    forecast   = make_forecast(model_full, df_model['ds'].max())

# =========================
# Qualité du modèle
# =========================
def get_quality(r2, mape):
    if r2 >= 0.85 and mape <= 10:
        return "#28a745", "🟢 Excellent"
    elif r2 >= 0.70 and mape <= 20:
        return "#ffc107", "🟡 Bon"
    elif r2 >= 0.50 and mape <= 50:
        return "#fd7e14", "🟠 Moyen"
    else:
        return "#dc3545", "🔴 Faible"

color, label = get_quality(r2, mape)
r2_pct   = max(0.0, min(r2, 1.0)) * 100
mape_bar = max(0.0, 100.0 - min(mape, 100.0))

# =========================
# Filtrage graphique : 2024 → 2026
# =========================
chart_start = pd.Timestamp("2024-01-01")
chart_end   = pd.Timestamp("2026-12-31")

train_chart        = train_df[train_df['ds'] >= chart_start]
test_chart         = test_df[(test_df['ds'] >= chart_start) & (test_df['ds'] <= chart_end)]
forecast_test_chart = forecast_test[
    (forecast_test['ds'] >= chart_start) & (forecast_test['ds'] <= chart_end)]
forecast_future    = forecast[
    (forecast['ds'] > df_model['ds'].max()) &
    (forecast['ds'] >= chart_start) &
    (forecast['ds'] <= chart_end)]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Graphique",
    "📊 Performance",
    "📋 Prévisions",
    "📌 KPIs"
])

# ── Tab 1 : Graphique ──────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Historique & Prévision — {product} (2024–2026)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_chart['ds'], y=train_chart['y'],
        mode='lines', name='Train',
        line=dict(color='#1f77b4', width=1.5)))

    fig.add_trace(go.Scatter(
        x=test_chart['ds'], y=test_chart['y'],
        mode='lines', name='Test (réel)',
        line=dict(color='orange', width=1.5)))

    fig.add_trace(go.Scatter(
        x=forecast_test_chart['ds'], y=forecast_test_chart['yhat'],
        mode='lines', name='Prédiction (test)',
        line=dict(color='purple', width=1.5, dash='dash')))

    fig.add_trace(go.Scatter(
        x=forecast_future['ds'], y=forecast_future['yhat'],
        mode='lines', name='Prévision future',
        line=dict(color='#ff7f0e', width=2)))

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_future['ds'], forecast_future['ds'][::-1]]),
        y=pd.concat([forecast_future['yhat_upper'], forecast_future['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(255,127,14,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name='Intervalle confiance'))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Quantité",
        xaxis=dict(range=[chart_start, chart_end]),
        template="plotly_white", hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 : Performance ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Indicateurs de Performance du Modèle")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE",  f"{mae:.2f}",   help="Erreur absolue moyenne")
    col2.metric("RMSE", f"{rmse:.2f}",  help="Racine erreur quadratique moyenne")
    col3.metric("MAPE", f"{mape:.2f}%", help="Erreur absolue en %")
    col4.metric("R²",   f"{r2:.4f}",   help="Coefficient de détermination (1 = parfait)")

    st.divider()
    st.markdown(f"### Qualité du modèle : {label}")

    col_r2, col_mape = st.columns(2)

    with col_r2:
        st.markdown("**R² — Coefficient de détermination**")
        st.markdown(f"""
        <div style="background:#e0e0e0; border-radius:10px; height:22px; width:100%;">
            <div style="background:{color}; width:{r2_pct:.1f}%; height:22px; border-radius:10px;"></div>
        </div>
        <p style="text-align:right; font-size:13px; color:gray;">R² = {r2:.4f} &nbsp;|&nbsp;
        {'≥ 0.85 Excellent' if r2 >= 0.85 else '≥ 0.70 Bon' if r2 >= 0.70 else '≥ 0.50 Moyen' if r2 >= 0.50 else '< 0.50 Faible'}</p>
        """, unsafe_allow_html=True)

    with col_mape:
        st.markdown("**MAPE — Erreur absolue en %**")
        st.markdown(f"""
        <div style="background:#e0e0e0; border-radius:10px; height:22px; width:100%;">
            <div style="background:{color}; width:{mape_bar:.1f}%; height:22px; border-radius:10px;"></div>
        </div>
        <p style="text-align:right; font-size:13px; color:gray;">MAPE = {mape:.2f}% &nbsp;|&nbsp;
        {'≤ 10% Excellent' if mape <= 10 else '≤ 20% Bon' if mape <= 20 else '≤ 50% Moyen' if mape <= 50 else '> 50% Faible'}</p>
        """, unsafe_allow_html=True)

    st.divider()
    if label == "🟢 Excellent":
        st.success("✅ Prophet est très bien adapté à ce produit !")
    elif label == "🟡 Bon":
        st.info("ℹ️ Bonne performance. Prophet est fiable pour ce produit.")
    elif label == "🟠 Moyen":
        st.warning("⚠️ Performance moyenne. Essayez ARIMA ou LSTM pour de meilleurs résultats.")
    else:
        st.error("❌ Performance faible. Essayez ARIMA ou LSTM pour ce produit.")

# ── Tab 3 : Tableau de prévision ───────────────────────────────────────────────
with tab3:
    st.subheader("Tableau de prévision (50 derniers points)")
    st.dataframe(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(50),
        height=400, use_container_width=True)

# ── Tab 4 : KPIs ───────────────────────────────────────────────────────────────
with tab4:
    st.subheader("KPIs Clés")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Dernière prévision",      f"{forecast['yhat'].iloc[-1]:.2f}")
    kpi2.metric("Moyenne prévision (30j)", f"{forecast['yhat'].iloc[-30:].mean():.2f}")
    kpi3.metric("Max prévision",           f"{forecast['yhat'].max():.2f}")