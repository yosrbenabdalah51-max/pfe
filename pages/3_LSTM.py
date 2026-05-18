import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import get_connection, sidebar_filters
from holidays_utils import add_holiday_feature, get_holiday_info_text, resolve_country_code
from auth import require_auth
require_auth("LSTM")
warnings.filterwarnings("ignore")

st.set_page_config(page_title="LSTM", page_icon="🔴")
st.title("📈 Prévision avec LSTM")

EXCLUDED_DEPOT_IDS = {8, 41, 57}
SEQ_LENGTH = 30

# =========================
# CHARGEMENT COMPLET
# =========================
@st.cache_data(ttl=300)
def load_data():
    try:
        conn = get_connection()
        excluded = ",".join(str(i) for i in EXCLUDED_DEPOT_IDS)
        df = pd.read_sql(f"""
            SELECT s.ref_product, s.quantity, s.price, s.date_time,
                   s.depot_id,
                   d.name       AS depot_name,
                   c.name       AS country_name
            FROM sales s
            LEFT JOIN depot   d ON s.depot_id   = d.depot_id
            LEFT JOIN country c ON d.country_id = c.id
            WHERE s.depot_id NOT IN ({excluded})
        """, conn)
        conn.close()
        df["date_time"]   = pd.to_datetime(df["date_time"])
        df["ref_product"] = df["ref_product"].astype(int)
        df["total"]       = df["quantity"] * df["price"]
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur de connexion : {e}")
        st.stop()

df_full = load_data()

# =========================
# SIDEBAR + FILTRES
# =========================
df, product, depot_id, depot_sel, date_range, selected_country = sidebar_filters()

if df is None or len(df) == 0:
    st.error(f"❌ Aucune vente trouvée pour le produit **{product or 'Tous'}** / dépôt **{depot_sel}**.")
    st.stop()

# =========================
# PRÉPARATION + LISSAGE
# =========================
def prepare_and_smooth(df, product, depot_id):
    df_d = (df
            .groupby(pd.Grouper(key="date_time", freq="D"))["quantity"]
            .sum()
            .reset_index()
            .rename(columns={"date_time": "ds", "quantity": "y"}))
    df_d = df_d.set_index("ds").asfreq("D").reset_index()
    df_d["y"] = df_d["y"].fillna(0)
    n_nonzero = (df_d["y"] > 0).sum()
    window = 7 if n_nonzero >= 60 else (3 if n_nonzero >= 20 else 1)
    if window > 1:
        df_d["y"] = df_d["y"].rolling(window=window, min_periods=1, center=True).mean()
    mean_y = df_d["y"].mean()
    std_y  = df_d["y"].std()
    if std_y > 0:
        df_d["y"] = df_d["y"].clip(lower=0, upper=mean_y + 3 * std_y)
    df_d = df_d.dropna(subset=["y"]).reset_index(drop=True)
    return df_d

df_model = prepare_and_smooth(df, product, depot_id)

if df_model is None or len(df_model) == 0:
    st.error("❌ Aucune donnée disponible après lissage pour ce produit/dépôt.")
    st.stop()

label_produit = f"{product} / {depot_sel}" if product else f"Tous / {depot_sel}"
st.info(f"📅 [{label_produit}] Série lissée (journalière) — **{len(df_model)} points**")

if len(df_model) < SEQ_LENGTH + 10:
    st.warning(f"⚠️ Pas assez de données ({len(df_model)} points). Minimum requis : {SEQ_LENGTH + 10}.")
    st.stop()

# Moyenne journalière des ventes
avg_daily_sales = df_model["y"].mean()

# =========================
# SCALE
# =========================
scaler      = MinMaxScaler()
data_scaled = scaler.fit_transform(df_model["y"].values.reshape(-1, 1))

# =========================
# TRAIN / TEST SPLIT 80/20
# =========================
split_index = int(len(data_scaled) * 0.8)

# =========================
# SÉQUENCES
# =========================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(data_scaled, SEQ_LENGTH)

if len(X_all) == 0:
    st.warning("⚠️ Pas assez de données pour créer des séquences.")
    st.stop()

X_train, y_train = X_all[:split_index], y_all[:split_index]
X_test,  y_test  = X_all[split_index:], y_all[split_index:]

if len(X_train) == 0:
    st.warning("⚠️ Pas assez de données d'entraînement.")
    st.stop()

# =========================
# ENTRAÎNEMENT LSTM
# =========================
@st.cache_resource
def train_lstm(X_train, y_train):
    model = Sequential([
        LSTM(64, activation="relu", return_sequences=True,
             input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.1),
        LSTM(32, activation="relu"),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(X_train, y_train,
              epochs=100, batch_size=16,
              validation_split=0.1,
              callbacks=[es], verbose=0)
    return model

with st.spinner("⏳ Entraînement LSTM..."):
    model = train_lstm(X_train, y_train)

# =========================
# PRÉDICTIONS TEST
# =========================
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_dates = df_model["ds"].iloc[split_index + SEQ_LENGTH:].values

# =========================
# MÉTRIQUES
# =========================
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float("nan")

mae_pct_of_avg  = (mae  / avg_daily_sales * 100) if avg_daily_sales > 0 else float('nan')
rmse_pct_of_avg = (rmse / avg_daily_sales * 100) if avg_daily_sales > 0 else float('nan')

# =========================
# FORECAST jusqu'au 31/12/2026
# =========================
@st.cache_data
def make_forecast(_model, data_scaled_tuple, last_date, _scaler):
    data_sc  = np.array(data_scaled_tuple).reshape(-1, 1)
    target   = pd.Timestamp("2026-12-31")
    future_steps = max(1, (target - last_date).days)
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_steps, freq="D"
    )
    preds    = []
    last_seq = data_sc[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    for _ in range(future_steps):
        p        = _model.predict(last_seq, verbose=0)[0][0]
        preds.append(p)
        last_seq = np.append(last_seq[:, 1:, :], [[[p]]], axis=1)
    fc_inv = _scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    fc_inv = np.clip(fc_inv, 0, None)
    return pd.DataFrame({"ds": future_dates, "yhat": fc_inv.flatten()})

with st.spinner("⏳ Génération des prévisions futures..."):
    forecast = make_forecast(
        model,
        tuple(data_scaled.flatten()),
        df_model["ds"].max(),
        scaler
    )

# =========================
# AUTO-SAVE FORECAST
# =========================
save_dir = "forecasts"
os.makedirs(save_dir, exist_ok=True)
ref_tag      = str(product) if product else "all"
depot_id_tag = str(st.session_state.get("depot_id", "all"))
save_path    = os.path.join(save_dir, f"forecast_{ref_tag}_{depot_id_tag}.csv")
forecast_to_save = forecast.copy()
forecast_to_save["ref_product"] = ref_tag
forecast_to_save["depot_id"]    = depot_id_tag
forecast_to_save.to_csv(save_path, index=False)
st.success(f"✅ Prévisions sauvegardées → {save_path}")

# =========================
# QUALITÉ DU MODÈLE
# =========================
def get_quality(r2, mape):
    if r2 >= 0.85 and mape <= 10:   return "#28a745", "🟢 Excellent"
    elif r2 >= 0.70 and mape <= 20: return "#ffc107", "🟡 Bon"
    elif r2 >= 0.50 and mape <= 50: return "#fd7e14", "🟠 Moyen"
    else:                           return "#dc3545", "🔴 Faible"

color, label = get_quality(r2, mape)
r2_pct   = max(0.0, min(r2 if not np.isnan(r2) else 0, 1.0)) * 100
mape_val = mape if not np.isnan(mape) else 100
mape_bar = max(0.0, 100.0 - min(mape_val, 100.0))

# =========================
# FILTRAGE GRAPHIQUE 2024 → 2026
# =========================
chart_start = pd.Timestamp("2024-01-01")
chart_end   = pd.Timestamp("2026-12-31")
if df_model["ds"].max() < chart_start:
    chart_start = df_model["ds"].min()

train_dates       = df_model["ds"].iloc[:split_index + SEQ_LENGTH]
train_values      = df_model["y"].iloc[:split_index + SEQ_LENGTH]
train_chart_mask  = train_dates >= chart_start
train_dates_chart = train_dates[train_chart_mask]
train_vals_chart  = train_values[train_chart_mask]

test_dates_pd    = pd.DatetimeIndex(test_dates)
test_chart_mask  = (test_dates_pd >= chart_start) & (test_dates_pd <= chart_end)
test_dates_chart = test_dates_pd[test_chart_mask]
y_true_chart     = y_true[test_chart_mask]
y_pred_chart     = y_pred[test_chart_mask]

forecast_chart = forecast[
    (forecast["ds"] >= chart_start) & (forecast["ds"] <= chart_end)
]

# =========================
# Prévisions mensuelles
# =========================
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
trend = "📈 Hausse" if len(future_monthly) >= 2 and future_monthly['yhat'].iloc[-1] > future_monthly['yhat'].iloc[0] else "📉 Baisse"

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📈 Graphique", "📊 Performance", "📌 KPIs"])

with tab1:
    st.subheader(f"Historique + Prévision — {label_produit} (2024–2026)")
    fig = go.Figure()
    if len(train_dates_chart) > 0:
        fig.add_trace(go.Scatter(
            x=train_dates_chart, y=train_vals_chart,
            mode="lines", name="Train",
            line=dict(color="blue", width=1.5)))
    if len(test_dates_chart) > 0:
        fig.add_trace(go.Scatter(
            x=test_dates_chart, y=y_true_chart,
            mode="lines", name="Test (réel)",
            line=dict(color="orange", width=1.5)))
        fig.add_trace(go.Scatter(
            x=test_dates_chart, y=y_pred_chart,
            mode="lines", name="Prédiction (test)",
            line=dict(color="purple", dash="dash", width=1.5)))
    if not forecast_chart.empty:
        fig.add_trace(go.Scatter(
            x=forecast_chart["ds"], y=forecast_chart["yhat"],
            mode="lines", name="Prévision future",
            line=dict(color="red", width=2)))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Quantité",
        xaxis=dict(range=[chart_start, chart_end]),
        template="plotly_white", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2 — Performance
# =========================
with tab2:
    st.subheader("Indicateurs de Performance du Modèle")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE",  f"{mae:.2f}",
                help="Erreur absolue moyenne — comparez avec la moyenne journalière ci-dessous.")
    col2.metric("RMSE", f"{rmse:.2f}",
                help="Erreur quadratique — pénalise les grandes erreurs.")
    col3.metric("MAPE", f"{mape_val:.2f}%",
                help="Erreur en % par rapport aux vraies valeurs.")
    col4.metric("R²",   f"{r2:.4f}" if not np.isnan(r2) else "N/A",
                help="Proportion de variance expliquée. Plus proche de 1 = meilleur modèle.")
    col5.metric("Moy. vente / jour", f"{avg_daily_sales:.2f}",
                help="Référence pour interpréter MAE et RMSE.")

    # Comparaison MAE / RMSE vs moyenne journalière
    st.divider()
    st.markdown("#### 📊 Erreurs vs Moyenne des ventes journalières")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**MAE ({mae:.2f})** représente **{mae_pct_of_avg:.1f}%** de la moyenne journalière ({avg_daily_sales:.2f})")
        mae_bar_pct = min(mae_pct_of_avg, 100) if not np.isnan(mae_pct_of_avg) else 0
        mae_color = "#28a745" if mae_pct_of_avg <= 10 else "#ffc107" if mae_pct_of_avg <= 25 else "#dc3545"
        st.markdown(f"""
        <div style="background:#e0e0e0; border-radius:10px; height:22px; width:100%;">
            <div style="background:{mae_color}; width:{mae_bar_pct:.1f}%; height:22px; border-radius:10px;"></div>
        </div>
        <p style="text-align:right; font-size:12px; color:gray;">{'✅ Faible erreur' if mae_pct_of_avg <= 10 else '⚠️ Erreur modérée' if mae_pct_of_avg <= 25 else '❌ Erreur élevée'}</p>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"**RMSE ({rmse:.2f})** représente **{rmse_pct_of_avg:.1f}%** de la moyenne journalière ({avg_daily_sales:.2f})")
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
    if label == "🟢 Excellent":   st.success("✅ LSTM est très bien adapté à ce produit !")
    elif label == "🟡 Bon":       st.info("ℹ️ Bonne performance. LSTM est fiable pour ce produit.")
    elif label == "🟠 Moyen":     st.warning("⚠️ Performance moyenne. Essayez Prophet ou ARIMA.")
    else:                          st.error("❌ Performance faible. Essayez Prophet ou ARIMA pour ce produit.")

# =========================
# TAB 3 — KPIs + Prévisions mensuelles
# =========================
with tab3:
    st.subheader("📌 KPIs Clés")
    k1, k2, k3 = st.columns(3)
    k1.metric(
        f"Prochain mois ({next_month_label})",
        f"{next_month_qty:,.0f} unités",
        help="Quantité totale prévue pour le mois qui vient."
    )
    k2.metric(
        f"Meilleur mois prévu ({best_month_label})",
        f"{best_month_qty:,.0f} unités",
        help="Le mois avec la plus forte prévision sur toute la période future."
    )
    k3.metric(
        "Tendance générale",
        trend,
        help="Comparaison entre la prévision du premier et du dernier mois disponible."
    )
    st.divider()
    st.subheader("📅 Prévisions par mois à venir")

    if future_monthly.empty:
        st.info("Aucune prévision mensuelle disponible au-delà d'aujourd'hui.")
    else:
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=future_monthly['month_str'],
            y=future_monthly['yhat'].round(0),
            name='Total prévu',
            marker_color='#e74c3c',
            opacity=0.85,
        ))
        fig_monthly.update_layout(
            template="plotly_white",
            xaxis_title="Mois",
            yaxis_title="Quantité prévue",
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            bargap=0.3,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        display_monthly = future_monthly[['month_str', 'yhat']].copy()
        display_monthly.columns = ['Mois', 'Quantité prévue']
        display_monthly['Quantité prévue'] = display_monthly['Quantité prévue'].round(1)
        csv_monthly = display_monthly.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Exporter prévisions mensuelles (CSV)",
            data=csv_monthly,
            file_name=f"previsions_mensuelles_lstm_{ref_tag}_{depot_id_tag}.csv",
            mime="text/csv"
        )



# =========================
# ✅ SAUVEGARDE SESSION_STATE
# =========================
freq_label = "hebdomadaire" if len(df_model) < 60 else "journalière"

sess_key = f"lstm_{product}_{depot_id}"
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
# ============================================
# 🤖 ANALYSE IA
# ============================================
from analyse.generateur import generer_analyse_lstm

st.divider()
col_btn = st.columns([2, 2, 2])
with col_btn[1]:
    btn = st.button(" 🤖 Analyse Intelligente", use_container_width=True, type="primary")

if btn:
    filtres = {
        "Produit"         : product or "Tous",
        "Dépôt"           : depot_sel,
        "Pays"            : selected_country,
        "Fréquence"       : freq_label,
        "Modèle"          : "LSTM (réseau de neurones récurrent)",
        "Séquence (steps)": SEQ_LENGTH,
    }

    metriques = {
        "MAE"                               : f"{mae:.2f} ({mae_pct_of_avg:.1f}% de la moyenne journalière)",
        "RMSE"                              : f"{rmse:.2f} ({rmse_pct_of_avg:.1f}% de la moyenne journalière)",
        "MAPE"                              : f"{mape_val:.2f}%",
        "R²"                                : f"{r2:.4f}" if not np.isnan(r2) else "N/A",
        "Qualité du modèle"                 : label,
        "Moyenne ventes/jour"               : f"{avg_daily_sales:.2f}",
        "Nombre de points"                  : len(df_model),
        f"Prévision {next_month_label}"     : f"{next_month_qty:,.0f} unités",
        f"Meilleur mois ({best_month_label})": f"{best_month_qty:,.0f} unités",
        "Tendance générale"                 : trend,
    }

    with st.spinner("🧠 Analyse en cours..."):
        analyse = generer_analyse_lstm(filtres, metriques)

    with st.container(border=True):
        st.markdown(analyse)

    st.download_button(
        label="⬇️ Télécharger l'analyse",
        data=analyse,
        file_name=f"analyse_lstm_{product}_{depot_sel}.txt",
        mime="text/plain"
    )