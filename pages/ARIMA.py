import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
from utils import get_connection, sidebar_product_selector

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prévision ARIMA", page_icon="🟢")
st.title("📈 Dashboard Prévision des Ventes - ARIMA")

# =========================
# Sidebar — Sélecteur produit
# =========================
product = sidebar_product_selector()

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

# =========================
# Préparation + Lissage des données (Daily)
# =========================
@st.cache_data
def prepare_and_smooth(df, product):
    # Filtrer par produit si pas "Tous"
    if product is not None:
        df = df[df['ref_product'] == product].copy()

    df_d = (df
            .groupby(pd.Grouper(key='date_time', freq='D'))['quantity']
            .sum()
            .reset_index()
            .rename(columns={'date_time': 'ds', 'quantity': 'y'}))

    df_d = df_d.set_index('ds').asfreq('D').reset_index()
    df_d['y'] = df_d['y'].replace(0, np.nan)
    df_d['y'] = df_d['y'].interpolate(method='linear')
    df_d['y'] = df_d['y'].fillna(method='bfill').fillna(method='ffill')

    df_d['y'] = df_d['y'].rolling(window=7, min_periods=1, center=True).mean()

    mean_y = df_d['y'].mean()
    std_y  = df_d['y'].std()
    df_d['y'] = df_d['y'].clip(lower=mean_y - 3 * std_y, upper=mean_y + 3 * std_y)

    df_d = df_d.dropna(subset=['y'])
    df_d = df_d[df_d['y'] > 0].reset_index(drop=True)
    return df_d

df_model = prepare_and_smooth(df, product)

label_produit = product if product else "Tous les produits"
st.info(f"📅 [{label_produit}] Série lissée (journalière, rolling 7 jours) — **{len(df_model)} points**")

if len(df_model) < 30:
    st.warning("⚠️ Pas assez de données pour ARIMA avec ce produit.")
    st.stop()

# =========================
# Train / Test split 80/20
# =========================
split_index = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_index]
test_df  = df_model.iloc[split_index:]

# =========================
# Recherche automatique du meilleur ordre ARIMA (AIC)
# =========================
@st.cache_data
def find_best_arima_order(train_y):
    best_aic   = np.inf
    best_order = (1, 1, 1)
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    m = ARIMA(train_y, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic   = m.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order

with st.spinner("🔍 Recherche du meilleur ordre ARIMA (p,d,q)..."):
    best_order = find_best_arima_order(train_df['y'].values)

st.success(f"✅ Meilleur ordre ARIMA : {best_order}")

# =========================
# Entraînement + prédictions test
# =========================
@st.cache_data
def train_arima(train_y, order, test_len):
    model_fit = ARIMA(train_y, order=order).fit()
    fc = model_fit.forecast(steps=test_len)
    return np.array(fc.values if hasattr(fc, 'values') else fc)

with st.spinner("⏳ Entraînement ARIMA..."):
    test_preds = train_arima(train_df['y'].values, best_order, len(test_df))

# =========================
# Métriques
# =========================
y_true = test_df['y'].values
y_pred = test_preds

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('nan')

# =========================
# Forecast complet jusqu'au 31/12/2026
# =========================
@st.cache_data
def make_full_forecast(full_y, order, last_date):
    model_full   = ARIMA(full_y, order=order).fit()
    target       = pd.Timestamp("2026-12-31")
    future_steps = max(1, (target - last_date).days)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=future_steps, freq='D')
    fc     = model_full.forecast(steps=future_steps)
    fc_arr = np.array(fc.values if hasattr(fc, 'values') else fc)
    return pd.DataFrame({'ds': future_dates, 'yhat': fc_arr})

with st.spinner("⏳ Génération des prévisions futures..."):
    forecast = make_full_forecast(df_model['y'].values, best_order, df_model['ds'].max())

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

train_chart      = train_df[train_df['ds'] >= chart_start]
test_chart       = test_df[(test_df['ds'] >= chart_start) & (test_df['ds'] <= chart_end)]
test_preds_s     = pd.Series(test_preds, index=test_df['ds'])
test_preds_chart = test_preds_s[(test_preds_s.index >= chart_start) & (test_preds_s.index <= chart_end)]
forecast_chart   = forecast[(forecast['ds'] >= chart_start) & (forecast['ds'] <= chart_end)]

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
    st.subheader(f"Historique + Prévision — {label_produit} (2024–2026)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_chart['ds'], y=train_chart['y'],
        mode='lines', name='Train',
        line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(
        x=test_chart['ds'], y=test_chart['y'],
        mode='lines', name='Test (réel)',
        line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(
        x=test_preds_chart.index, y=test_preds_chart.values,
        mode='lines', name='Prédiction (test)',
        line=dict(color='purple', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(
        x=forecast_chart['ds'], y=forecast_chart['yhat'],
        mode='lines', name='Prévision future',
        line=dict(color='green', width=2)))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Quantité",
        xaxis=dict(range=[chart_start, chart_end]),
        template="plotly_white", hovermode="x unified", height=500)
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
        st.success("✅ ARIMA est très bien adapté à ce produit !")
    elif label == "🟡 Bon":
        st.info("ℹ️ Bonne performance. ARIMA est fiable pour ce produit.")
    elif label == "🟠 Moyen":
        st.warning("⚠️ Performance moyenne. Essayez Prophet ou LSTM pour de meilleurs résultats.")
    else:
        st.error("❌ Performance faible. Essayez Prophet ou LSTM pour ce produit.")

# ── Tab 3 : Tableau de prévision ───────────────────────────────────────────────
with tab3:
    st.subheader("Tableau de prévision (50 derniers points)")
    st.dataframe(forecast.tail(50), use_container_width=True)

# ── Tab 4 : KPIs ───────────────────────────────────────────────────────────────
with tab4:
    st.subheader("KPIs Clés")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Dernière prévision",      f"{forecast['yhat'].iloc[-1]:.2f}")
    kpi2.metric("Moyenne prévision (30j)", f"{forecast['yhat'].iloc[-30:].mean():.2f}")
    kpi3.metric("Max prévision",           f"{forecast['yhat'].max():.2f}")