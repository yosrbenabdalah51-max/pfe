import streamlit as st
import pandas as pd
import mysql.connector
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prévision ARIMA", page_icon="🟢")
st.title("📈 Dashboard Prévision des Ventes - ARIMA")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="streamlit_user",
            password="Password123@",
            database="pfe"
        )
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
# Préparation + Lissage des données
# =========================
@st.cache_data
def prepare_and_smooth(df, product):
    df_prod = df[df['ref_product'] == product].copy()

    df_w = (df_prod
            .groupby(pd.Grouper(key='date_time', freq='W'))['quantity']
            .sum()
            .reset_index()
            .rename(columns={'date_time': 'ds', 'quantity': 'y'}))

    df_w = df_w.set_index('ds').asfreq('W').reset_index()
    df_w['y'] = df_w['y'].replace(0, np.nan)
    df_w['y'] = df_w['y'].interpolate(method='linear')
    df_w['y'] = df_w['y'].fillna(method='bfill').fillna(method='ffill')

    df_w['y'] = df_w['y'].rolling(window=4, min_periods=1, center=True).mean()

    mean_y = df_w['y'].mean()
    std_y  = df_w['y'].std()
    df_w['y'] = df_w['y'].clip(lower=mean_y - 3 * std_y, upper=mean_y + 3 * std_y)

    df_w = df_w.dropna(subset=['y'])
    df_w = df_w[df_w['y'] > 0].reset_index(drop=True)
    return df_w

df_model = prepare_and_smooth(df, product)

st.info(f"📅 Série lissée (hebdomadaire, rolling 4 semaines) — **{len(df_model)} points**")

if len(df_model) < 20:
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
    future_steps = max(1, int((target - last_date).days / 7))
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                 periods=future_steps, freq='W')
    fc     = model_full.forecast(steps=future_steps)
    fc_arr = np.array(fc.values if hasattr(fc, 'values') else fc)
    return pd.DataFrame({'ds': future_dates, 'yhat': fc_arr})

with st.spinner("⏳ Génération des prévisions futures..."):
    forecast = make_full_forecast(df_model['y'].values, best_order, df_model['ds'].max())

# =========================
# Graphique
# =========================
st.subheader(f"📈 Historique + Prévision - {product}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df['ds'], y=train_df['y'],
    mode='lines+markers', name='Train', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'],
    mode='lines+markers', name='Test (réel)', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=test_df['ds'], y=test_preds,
    mode='lines', name='Prédiction (test)', line=dict(color='purple', dash='dash')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
    mode='lines', name='Prévision future', line=dict(color='green')))
fig.update_layout(xaxis_title="Date", yaxis_title="Quantité",
    template="plotly_white", hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# =========================
# Métriques de Performance
# =========================
st.divider()
st.subheader("📊 Indicateurs de Performance du Modèle")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE",  f"{mae:.2f}",   help="Erreur absolue moyenne")
col2.metric("RMSE", f"{rmse:.2f}",  help="Racine erreur quadratique moyenne")
col3.metric("MAPE", f"{mape:.2f}%", help="Erreur absolue en %")
col4.metric("R²",   f"{r2:.4f}",   help="Coefficient de détermination (1 = parfait)")

# =========================
# Qualité du modèle combinée R² + MAPE
# =========================
st.divider()

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

if label == "🟢 Excellent":
    st.success("✅ ARIMA est très bien adapté à ce produit !")
elif label == "🟡 Bon":
    st.info("ℹ️ Bonne performance. ARIMA est fiable pour ce produit.")
elif label == "🟠 Moyen":
    st.warning("⚠️ Performance moyenne. Essayez Prophet ou LSTM pour de meilleurs résultats.")
else:
    st.error("❌ Performance faible. Essayez Prophet ou LSTM pour ce produit.")

st.divider()
st.subheader("📋 Tableau de prévision (50 derniers points)")
st.dataframe(forecast.tail(50), use_container_width=True)
st.divider()
st.metric("📌 Dernière prévision", f"{forecast['yhat'].iloc[-1]:.2f}")