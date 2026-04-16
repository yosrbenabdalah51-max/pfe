import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from utils import get_connection, sidebar_product_selector

warnings.filterwarnings("ignore")

st.set_page_config(page_title="LSTM", page_icon="🔴")
st.title("📈 Prévision avec LSTM")

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

SEQ_LENGTH = 30

if len(df_model) < SEQ_LENGTH + 30:
    st.warning("⚠️ Pas assez de données pour LSTM avec ce produit.")
    st.stop()

# =========================
# Scale
# =========================
scaler      = MinMaxScaler()
data_scaled = scaler.fit_transform(df_model['y'].values.reshape(-1, 1))

# =========================
# Train / Test split 80/20
# =========================
split_index = int(len(data_scaled) * 0.8)

# =========================
# Séquences
# =========================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(data_scaled, SEQ_LENGTH)
X_train, y_train = X_all[:split_index], y_all[:split_index]
X_test,  y_test  = X_all[split_index:], y_all[split_index:]

# =========================
# Entraînement LSTM
# =========================
@st.cache_resource
def train_lstm(X_train, y_train):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True,
             input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.1),
        LSTM(32, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_train, y_train,
              epochs=100, batch_size=16,
              validation_split=0.1,
              callbacks=[es], verbose=0)
    return model

with st.spinner("⏳ Entraînement LSTM..."):
    model = train_lstm(X_train, y_train)

# =========================
# Prédictions test
# =========================
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

test_dates = df_model['ds'].iloc[split_index + SEQ_LENGTH:].values

# =========================
# Métriques
# =========================
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('nan')

# =========================
# Forecast jusqu'au 31/12/2026
# =========================
@st.cache_data
def make_forecast(_model, data_scaled, last_date, _scaler):
    target       = pd.Timestamp("2026-12-31")
    future_steps = max(1, (target - last_date).days)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=future_steps, freq='D')
    preds    = []
    last_seq = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    for _ in range(future_steps):
        p        = _model.predict(last_seq, verbose=0)[0][0]
        preds.append(p)
        last_seq = np.append(last_seq[:, 1:, :], [[[p]]], axis=1)
    fc_inv = _scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return pd.DataFrame({'ds': future_dates, 'yhat': fc_inv.flatten()})

with st.spinner("⏳ Génération des prévisions futures..."):
    forecast = make_forecast(model, data_scaled, df_model['ds'].max(), scaler)

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

train_dates       = df_model['ds'].iloc[:split_index + SEQ_LENGTH]
train_values      = df_model['y'].iloc[:split_index + SEQ_LENGTH]
train_chart_mask  = train_dates >= chart_start
train_dates_chart = train_dates[train_chart_mask]
train_vals_chart  = train_values[train_chart_mask]

test_dates_pd    = pd.DatetimeIndex(test_dates)
test_chart_mask  = (test_dates_pd >= chart_start) & (test_dates_pd <= chart_end)
test_dates_chart = test_dates_pd[test_chart_mask]
y_true_chart     = y_true[test_chart_mask]
y_pred_chart     = y_pred[test_chart_mask]

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
        x=train_dates_chart, y=train_vals_chart,
        mode='lines', name='Train',
        line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(
        x=test_dates_chart, y=y_true_chart,
        mode='lines', name='Test (réel)',
        line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(
        x=test_dates_chart, y=y_pred_chart,
        mode='lines', name='Prédiction (test)',
        line=dict(color='purple', dash='dash', width=1.5)))
    fig.add_trace(go.Scatter(
        x=forecast_chart['ds'], y=forecast_chart['yhat'],
        mode='lines', name='Prévision future',
        line=dict(color='red', width=2)))
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
        st.success("✅ LSTM est très bien adapté à ce produit !")
    elif label == "🟡 Bon":
        st.info("ℹ️ Bonne performance. LSTM est fiable pour ce produit.")
    elif label == "🟠 Moyen":
        st.warning("⚠️ Performance moyenne. Essayez Prophet ou ARIMA pour de meilleurs résultats.")
    else:
        st.error("❌ Performance faible. Essayez Prophet ou ARIMA pour ce produit.")

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