import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

# ── Imports modèles ───────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from db_utils import get_connection, sidebar_product_selector
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Comparaison Modèles", page_icon="📊", layout="wide")
st.title("📊 Comparaison des Modèles de Prévision")
st.markdown("Évaluation côte à côte de **ARIMA**, **XGBoost** et **LSTM** sur les mêmes données.")

# =========================
# Sidebar
# =========================
product = sidebar_product_selector()
label_produit = product if product else "Tous les produits"

# =========================
# Chargement données
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
# Préparation + Lissage
# =========================
@st.cache_data
def prepare_and_smooth(df, product):
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
    df_d['y'] = df_d['y'].bfill().ffill()
    df_d['y'] = df_d['y'].rolling(window=7, min_periods=1, center=True).mean()
    mean_y = df_d['y'].mean()
    std_y  = df_d['y'].std()
    df_d['y'] = df_d['y'].clip(lower=mean_y - 3 * std_y, upper=mean_y + 3 * std_y)
    df_d = df_d.dropna(subset=['y'])
    df_d = df_d[df_d['y'] > 0].reset_index(drop=True)
    return df_d

df_model = prepare_and_smooth(df, product)

if len(df_model) < 60:
    st.warning("⚠️ Pas assez de données pour comparer les modèles.")
    st.stop()

split_index = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_index]
test_df  = df_model.iloc[split_index:]
y_true   = test_df['y'].values

st.info(f"📅 [{label_produit}] — **{len(df_model)} points** | Train: {len(train_df)} | Test: {len(test_df)}")

# =========================
# Métriques helper
# =========================
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('nan')
    return mae, rmse, mape, r2

def get_quality(r2, mape):
    if r2 >= 0.85 and mape <= 10:
        return "🟢 Excellent"
    elif r2 >= 0.70 and mape <= 20:
        return "🟡 Bon"
    elif r2 >= 0.50 and mape <= 50:
        return "🟠 Moyen"
    else:
        return "🔴 Faible"

results = {}

# =========================
# ── ARIMA ────────────────
# =========================
with st.spinner("⏳ Entraînement ARIMA..."):
    @st.cache_data
    def run_arima(train_y, test_len):
        best_aic, best_order = np.inf, (1, 1, 1)
        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        m = ARIMA(train_y, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic, best_order = m.aic, (p, d, q)
                    except Exception:
                        continue
        model_fit = ARIMA(train_y, order=best_order).fit()
        fc = model_fit.forecast(steps=test_len)
        preds = np.array(fc.values if hasattr(fc, 'values') else fc)
        return preds, best_order

    arima_preds, arima_order = run_arima(train_df['y'].values, len(test_df))
    arima_preds = np.clip(arima_preds, 0, None)
    mae, rmse, mape, r2 = compute_metrics(y_true, arima_preds)
    results['ARIMA'] = {
        'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2,
        'Qualité': get_quality(r2, mape),
        'preds': arima_preds,
        'Note': f"Ordre {arima_order} — Linéaire, stable sur séries simples"
    }

# =========================
# ── XGBoost ──────────────
# =========================
with st.spinner("⏳ Entraînement XGBoost..."):
    FEATURE_COLS = [
        'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear',
        'month', 'quarter', 'year', 'is_weekend',
        'lag_1', 'lag_7', 'lag_14', 'lag_21', 'lag_28',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
        'rolling_std_7',  'rolling_std_14',  'rolling_std_28',
    ]

    def create_features(df):
        d = df.copy()
        d['dayofweek']  = d['ds'].dt.dayofweek
        d['dayofmonth'] = d['ds'].dt.day
        d['dayofyear']  = d['ds'].dt.dayofyear
        d['weekofyear'] = d['ds'].dt.isocalendar().week.astype(int)
        d['month']      = d['ds'].dt.month
        d['quarter']    = d['ds'].dt.quarter
        d['year']       = d['ds'].dt.year
        d['is_weekend'] = (d['dayofweek'] >= 5).astype(int)
        for lag in [1, 7, 14, 21, 28]:
            d[f'lag_{lag}'] = d['y'].shift(lag)
        for w in [7, 14, 28]:
            d[f'rolling_mean_{w}'] = d['y'].shift(1).rolling(w).mean()
            d[f'rolling_std_{w}']  = d['y'].shift(1).rolling(w).std()
        return d

    @st.cache_resource
    def run_xgboost(df_model, split_index):
        df_feat = create_features(df_model).dropna().reset_index(drop=True)
        tr = df_feat.iloc[:split_index]
        te = df_feat.iloc[split_index:]
        xgb = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
            early_stopping_rounds=50, eval_metric='rmse'
        )
        xgb.fit(tr[FEATURE_COLS], tr['y'],
                eval_set=[(tr[FEATURE_COLS], tr['y']), (te[FEATURE_COLS], te['y'])],
                verbose=False)
        preds = np.clip(xgb.predict(te[FEATURE_COLS]), 0, None)
        return preds, xgb, te

    xgb_preds, xgb_model, xgb_test = run_xgboost(df_model, split_index)
    # Aligner avec y_true (XGBoost drop les premières lignes à cause des lags)
    n = min(len(xgb_preds), len(y_true))
    mae, rmse, mape, r2 = compute_metrics(y_true[-n:], xgb_preds[-n:])
    results['XGBoost'] = {
        'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2,
        'Qualité': get_quality(r2, mape),
        'preds': xgb_preds,
        'Note': "Features temporelles + lags — Capture relations non-linéaires"
    }

# =========================
# ── LSTM ─────────────────
# =========================
with st.spinner("⏳ Entraînement LSTM..."):
    SEQ_LENGTH = 30

    @st.cache_resource
    def run_lstm(df_model, split_index):
        scaler      = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_model['y'].values.reshape(-1, 1))

        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i + seq_len])
                y.append(data[i + seq_len])
            return np.array(X), np.array(y)

        X_all, y_all = create_sequences(data_scaled, SEQ_LENGTH)
        X_train, y_train = X_all[:split_index], y_all[:split_index]
        X_test,  y_test  = X_all[split_index:], y_all[split_index:]

        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
            Dropout(0.1),
            LSTM(32, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=16,
                  validation_split=0.1, callbacks=[es], verbose=0)

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        return y_pred, y_true_inv

    lstm_preds, lstm_true = run_lstm(df_model, split_index)
    mae, rmse, mape, r2 = compute_metrics(lstm_true, lstm_preds)
    results['LSTM'] = {
        'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2,
        'Qualité': get_quality(r2, mape),
        'preds': lstm_preds,
        'Note': "Séquences temporelles — Capture dépendances longues"
    }

# =========================
# ── Tableau comparatif ────
# =========================
st.divider()
st.subheader("📋 Tableau Comparatif des Performances")

rows = []
for name, m in results.items():
    rows.append({
        'Modèle':       name,
        'MAE':          f"{m['MAE']:.2f}",
        'RMSE':         f"{m['RMSE']:.2f}",
        'MAPE (%)':     f"{m['MAPE']:.2f}",
        'R²':           f"{m['R²']:.4f}",
        'Qualité':      m['Qualité'],
        'Interprétation': m['Note'],
    })

df_cmp = pd.DataFrame(rows)

# Mise en évidence du meilleur modèle (R² max)
best_model = max(results, key=lambda k: results[k]['R²'])

def highlight_best(row):
    if row['Modèle'] == best_model:
        return ['background-color: #d4edda; font-weight: bold'] * len(row)
    return [''] * len(row)

st.dataframe(
    df_cmp.style.apply(highlight_best, axis=1),
    use_container_width=True,
    hide_index=True
)
st.success(f"🏆 Meilleur modèle sur ce produit : **{best_model}** (R² = {results[best_model]['R²']:.4f})")

# =========================
# ── Graphiques radar + barres
# =========================
st.divider()
st.subheader("📈 Visualisation Comparative")

col_bar, col_radar = st.columns(2)

with col_bar:
    st.markdown("**R² par modèle** (plus haut = meilleur)")
    fig_r2 = go.Figure(go.Bar(
        x=list(results.keys()),
        y=[results[m]['R²'] for m in results],
        marker_color=['#28a745' if m == best_model else '#1f77b4' for m in results],
        text=[f"{results[m]['R²']:.4f}" for m in results],
        textposition='outside'
    ))
    fig_r2.update_layout(
        yaxis=dict(range=[0, 1.1]), template='plotly_white',
        height=320, showlegend=False,
        yaxis_title="R²"
    )
    st.plotly_chart(fig_r2, use_container_width=True)

with col_radar:
    st.markdown("**MAPE (%) par modèle** (plus bas = meilleur)")
    fig_mape = go.Figure(go.Bar(
        x=list(results.keys()),
        y=[results[m]['MAPE'] for m in results],
        marker_color=['#28a745' if results[m]['MAPE'] == min(results[mm]['MAPE'] for mm in results) else '#ff7f0e' for m in results],
        text=[f"{results[m]['MAPE']:.2f}%" for m in results],
        textposition='outside'
    ))
    fig_mape.update_layout(
        template='plotly_white', height=320, showlegend=False,
        yaxis_title="MAPE (%)"
    )
    st.plotly_chart(fig_mape, use_container_width=True)

# =========================
# ── Graphique prédictions test superposées
# =========================
st.divider()
st.subheader("📉 Prédictions sur le jeu de test")

fig_test = go.Figure()

# Réel (ARIMA a la même longueur que test_df)
fig_test.add_trace(go.Scatter(
    x=test_df['ds'].values, y=y_true,
    mode='lines', name='Réel',
    line=dict(color='black', width=2)
))

colors = {'ARIMA': 'green', 'XGBoost': '#1f77b4', 'LSTM': 'red'}
for name, m in results.items():
    preds = m['preds']
    n = len(preds)
    x_vals = test_df['ds'].values[-n:]
    fig_test.add_trace(go.Scatter(
        x=x_vals, y=preds,
        mode='lines', name=name,
        line=dict(color=colors[name], dash='dash', width=1.5)
    ))

fig_test.update_layout(
    xaxis_title="Date", yaxis_title="Quantité",
    template='plotly_white', hovermode='x unified', height=450,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
st.plotly_chart(fig_test, use_container_width=True)

# =========================
# ── Recommandation finale
# =========================
st.divider()
st.subheader("💡 Recommandation")

best = results[best_model]
interp = {
    'ARIMA':   "Idéal pour les séries stationnaires et linéaires. Rapide à entraîner.",
    'XGBoost': "Excellent pour capturer les patterns non-linéaires avec des features riches.",
    'LSTM':    "Meilleur pour les dépendances temporelles longues et complexes."
}

st.info(f"""
**Modèle recommandé pour [{label_produit}] : {best_model}**

- R² = **{best['R²']:.4f}** | MAPE = **{best['MAPE']:.2f}%** | MAE = **{best['MAE']:.2f}** | RMSE = **{best['RMSE']:.2f}**
- Qualité : {best['Qualité']}
- {interp[best_model]}
""")