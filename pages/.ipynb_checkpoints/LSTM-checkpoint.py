import streamlit as st
import pandas as pd
import mysql.connector
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="LSTM", page_icon="🔴")
st.title("📈 Prévision avec LSTM")

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

SEQ_LENGTH = 8

if len(df_model) < SEQ_LENGTH + 15:
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
              epochs=100, batch_size=8,
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
    future_steps = max(1, int((target - last_date).days / 7))
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                 periods=future_steps, freq='W')
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
# Graphique
# =========================
st.subheader(f"📈 Historique + Prévision - {product}")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_model['ds'].iloc[:split_index + SEQ_LENGTH],
    y=df_model['y'].iloc[:split_index + SEQ_LENGTH],
    mode='lines+markers', name='Train', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_dates, y=y_true,
    mode='lines+markers', name='Test (réel)', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=test_dates, y=y_pred,
    mode='lines', name='Prédiction (test)', line=dict(color='purple', dash='dash')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
    mode='lines', name='Prévision future', line=dict(color='red')))
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
    st.success("✅ LSTM est très bien adapté à ce produit !")
elif label == "🟡 Bon":
    st.info("ℹ️ Bonne performance. LSTM est fiable pour ce produit.")
elif label == "🟠 Moyen":
    st.warning("⚠️ Performance moyenne. Essayez Prophet ou ARIMA pour de meilleurs résultats.")
else:
    st.error("❌ Performance faible. Essayez Prophet ou ARIMA pour ce produit.")

st.divider()
st.subheader("📋 Tableau de prévision (50 derniers points)")
st.dataframe(forecast.tail(50), use_container_width=True)
st.divider()
st.metric("📌 Dernière prévision", f"{forecast['yhat'].iloc[-1]:.2f}")