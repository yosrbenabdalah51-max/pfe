import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from utils import get_connection, sidebar_filters

warnings.filterwarnings("ignore")

st.set_page_config(page_title="XGBoost Dashboard", page_icon="📊", layout="wide")
st.title("📈 Prévision de ventes avec XGBoost")


# =========================
# SIDEBAR + FILTRES — via utils.sidebar_filters()
# =========================
df, product, depot_id, depot_sel, date_range, selected_country = sidebar_filters()

if df is None or len(df) == 0:
    st.error(f"❌ Aucune vente trouvée pour le produit **{product or 'Tous'}** / dépôt **{depot_sel}**.")
    st.stop()

# =========================
# Préparation + Lissage
# =========================
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

label_produit = f"{product or 'Tous'} / {depot_sel}"
n_pts = len(df_model)
st.info(f"📅 [{label_produit}] Série lissée (journalière) — **{n_pts} points**")

MIN_POINTS = 10
if n_pts < MIN_POINTS:
    st.warning(f"⚠️ Seulement {n_pts} points disponibles (minimum requis : {MIN_POINTS}).")
    st.stop()

if n_pts < 60:
    df_model = df_model.set_index('ds').resample('W')['y'].sum().reset_index()
    df_model = df_model.reset_index(drop=True)
    st.info(f"🔁 Données agrégées par semaine ({len(df_model)} semaines)")
    freq_label = "hebdomadaire"
    LAG_LIST  = [1, 2, 4, 6, 8]
    ROLL_WINS = [2, 4, 8]
else:
    freq_label = "journalière"
    LAG_LIST  = [1, 7, 14, 21, 28]
    ROLL_WINS = [7, 14, 28]

n_pts = len(df_model)
if n_pts < MIN_POINTS:
    st.warning(f"⚠️ Pas assez de données même après agrégation ({n_pts} points).")
    st.stop()

# =========================
# Feature Engineering
# =========================
def create_features(df, lag_list, roll_wins):
    d = df.copy()
    d['dayofweek']  = d['ds'].dt.dayofweek
    d['dayofmonth'] = d['ds'].dt.day
    d['dayofyear']  = d['ds'].dt.dayofyear
    d['weekofyear'] = d['ds'].dt.isocalendar().week.astype(int)
    d['month']      = d['ds'].dt.month
    d['quarter']    = d['ds'].dt.quarter
    d['year']       = d['ds'].dt.year
    d['is_weekend'] = (d['dayofweek'] >= 5).astype(int)
    for lag in lag_list:
        d[f'lag_{lag}'] = d['y'].shift(lag)
    for w in roll_wins:
        d[f'rolling_mean_{w}'] = d['y'].shift(1).rolling(w).mean()
        d[f'rolling_std_{w}']  = d['y'].shift(1).rolling(w).std()
    return d

df_feat = create_features(df_model, LAG_LIST, ROLL_WINS).dropna().reset_index(drop=True)

FEATURE_COLS = (
    ['dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear',
     'month', 'quarter', 'year', 'is_weekend'] +
    [f'lag_{l}' for l in LAG_LIST] +
    [f'rolling_mean_{w}' for w in ROLL_WINS] +
    [f'rolling_std_{w}'  for w in ROLL_WINS]
)

if len(df_feat) < 10:
    st.warning("⚠️ Pas assez de données après feature engineering.")
    st.stop()

split_index = int(len(df_feat) * 0.8)
train_feat = df_feat.iloc[:split_index]
test_feat  = df_feat.iloc[split_index:]
if len(test_feat) == 0:
    train_feat = df_feat.iloc[:-1]
    test_feat  = df_feat.iloc[-1:]

X_train, y_train = train_feat[FEATURE_COLS], train_feat['y']
X_test,  y_test  = test_feat[FEATURE_COLS],  test_feat['y']

# =========================
# Entraînement XGBoost
# =========================
@st.cache_resource
def train_xgboost(X_train_tuple, y_train_tuple, X_test_tuple, y_test_tuple, cols):
    X_tr = pd.DataFrame(list(X_train_tuple), columns=cols)
    y_tr = np.array(y_train_tuple)
    X_te = pd.DataFrame(list(X_test_tuple), columns=cols)
    y_te = np.array(y_test_tuple)
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
        early_stopping_rounds=50, eval_metric='rmse'
    )
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_te, y_te)], verbose=False)
    return model

with st.spinner("⏳ Entraînement XGBoost..."):
    model = train_xgboost(
        tuple(map(tuple, X_train.values)), tuple(y_train.values),
        tuple(map(tuple, X_test.values)),  tuple(y_test.values),
        FEATURE_COLS
    )

y_pred = np.clip(model.predict(X_test), 0, None)
y_true = y_test.values

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('nan')

# Moyenne journalière des ventes (sur toute la série historique)
avg_daily_sales = df_model['y'].mean()

# =========================
# Prévision Future
# =========================
@st.cache_data
def make_future_forecast(_model, df_model_vals, df_model_dates, lag_list, roll_wins, feat_cols, rmse_val):
    history = pd.DataFrame({'ds': list(df_model_dates), 'y': list(df_model_vals)})
    last_date = pd.Timestamp(history['ds'].max())
    target = pd.Timestamp("2026-12-31")
    future_days = max(1, (target - last_date).days)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
    preds = []
    for fd in future_dates:
        tmp = pd.DataFrame({'ds': [fd], 'y': [np.nan]})
        extended = pd.concat([history, tmp], ignore_index=True)
        extended = create_features(extended, list(lag_list), list(roll_wins))
        row = extended.iloc[-1].copy()
        for col in list(feat_cols):
            if pd.isna(row[col]):
                row[col] = history['y'].iloc[-1]
        pred = max(float(_model.predict(pd.DataFrame([row[list(feat_cols)]]))[0]), 0)
        preds.append(pred)
        history = pd.concat([history, pd.DataFrame({'ds': [fd], 'y': [pred]})], ignore_index=True)
    return pd.DataFrame({
        'ds': future_dates, 'yhat': preds,
        'yhat_lower': [max(0, p - rmse_val) for p in preds],
        'yhat_upper': [p + rmse_val for p in preds],
    })

with st.spinner("⏳ Génération des prévisions futures..."):
    forecast_future_df = make_future_forecast(
        model, tuple(df_model['y'].values), tuple(df_model['ds'].values),
        tuple(LAG_LIST), tuple(ROLL_WINS), tuple(FEATURE_COLS), rmse
    )

# =========================
# Qualité
# =========================
def get_quality(r2, mape):
    if r2 >= 0.90 and mape <= 10:   return "#28a745", "🟢 Excellent"
    elif r2 >= 0.70 and mape <= 20: return "#ffc107", "🟡 Bon"
    elif r2 >= 0.50 and mape <= 50: return "#fd7e14", "🟠 Moyen"
    else:                           return "#dc3545", "🔴 Faible"

color, label = get_quality(r2, mape)
r2_pct   = max(0.0, min(r2 if not np.isnan(r2) else 0, 1.0)) * 100
mape_val = mape if not np.isnan(mape) else 100
mape_bar = max(0.0, 100.0 - min(mape_val, 100.0))

chart_start = pd.Timestamp("2024-01-01")
chart_end   = df_model['ds'].max() + pd.Timedelta(days=90)
if df_model['ds'].max() < chart_start:
    chart_start = df_model['ds'].min()

train_chart = train_feat[train_feat['ds'] >= chart_start]
test_chart  = test_feat[(test_feat['ds'] >= chart_start) & (test_feat['ds'] <= chart_end)].copy()
test_chart['yhat'] = np.clip(model.predict(test_chart[FEATURE_COLS]), 0, None)
forecast_future_chart = forecast_future_df[forecast_future_df['ds'] <= chart_end]

# =========================
# Prévisions mensuelles agrégées (style ARIMA)
# =========================
forecast_future_df['month'] = forecast_future_df['ds'].dt.to_period('M')
monthly_forecast = forecast_future_df.groupby('month')['yhat'].sum().reset_index()
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
tab1, tab2, tab3 = st.tabs(["📈 Graphique", "📊 Performance", "📌 KPIs & Prévisions mensuelles"])

with tab1:
    st.subheader(f"Historique & Prévision — {label_produit} (2024–2026)")
    fig = go.Figure()
    if not train_chart.empty:
        fig.add_trace(go.Scatter(x=train_chart['ds'], y=train_chart['y'],
            mode='lines', name='Train', line=dict(color='#1f77b4', width=1.5)))
    if not test_chart.empty:
        fig.add_trace(go.Scatter(x=test_chart['ds'], y=test_chart['y'],
            mode='lines', name='Test (réel)', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=test_chart['ds'], y=test_chart['yhat'],
            mode='lines', name='Prédiction (test)', line=dict(color='purple', width=1.5, dash='dash')))
    if not forecast_future_chart.empty:
        fig.add_trace(go.Scatter(x=forecast_future_chart['ds'], y=forecast_future_chart['yhat'],
            mode='lines', name='Prévision future', line=dict(color='#ff7f0e', width=2)))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_future_chart['ds'], forecast_future_chart['ds'][::-1]]),
            y=pd.concat([forecast_future_chart['yhat_upper'], forecast_future_chart['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name='Intervalle confiance'))
    fig.update_layout(xaxis_title="Date", yaxis_title="Quantité",
        xaxis=dict(range=[chart_start, chart_end]),
        template="plotly_white", hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2 — Performance (sans feature importance)
# =========================
with tab2:
    st.subheader("Indicateurs de Performance du Modèle")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE",  f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAPE", f"{mape_val:.2f}%")
    col4.metric("R²",   f"{r2:.4f}" if not np.isnan(r2) else "N/A")
    col5.metric("Moy. ventes/jour", f"{avg_daily_sales:.2f}")

    # Comparaison MAE / RMSE vs moyenne journalière
    st.divider()
    st.markdown("#### 📊 Erreurs vs Moyenne des ventes journalières")

    mae_pct_of_avg  = (mae  / avg_daily_sales * 100) if avg_daily_sales > 0 else float('nan')
    rmse_pct_of_avg = (rmse / avg_daily_sales * 100) if avg_daily_sales > 0 else float('nan')

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
    if label == "🟢 Excellent":   st.success("✅ XGBoost est très bien adapté à ce produit !")
    elif label == "🟡 Bon":       st.info("ℹ️ Bonne performance. XGBoost est fiable pour ce produit.")
    elif label == "🟠 Moyen":     st.warning("⚠️ Performance moyenne. Essayez LSTM ou un tuning des hyperparamètres.")
    else:                          st.error("❌ Performance faible. Essayez LSTM ou enrichissez les features.")

# =========================
# TAB 3 — KPIs + Prévisions mensuelles (style ARIMA)
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
    st.subheader("📅 Détail mois par mois")

    if future_monthly.empty:
        st.info("Aucune prévision mensuelle disponible au-delà d'aujourd'hui.")
    else:
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=future_monthly['month_str'],
            y=future_monthly['yhat'].round(0),
            text=future_monthly['yhat'].round(0).astype(int),
            textposition='outside',
            marker_color='#1f77b4',
            opacity=0.85,
            name='Quantité prévue'
        ))
        fig_monthly.update_layout(
            template="plotly_white",
            xaxis_title="Mois",
            yaxis_title="Quantité prévue",
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            bargap=0.3
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        display_monthly = future_monthly[['month_str', 'yhat']].copy()
        display_monthly.columns = ['Mois', 'Quantité prévue']
        display_monthly['Quantité prévue'] = display_monthly['Quantité prévue'].round(1)
        csv_monthly = display_monthly.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Exporter prévisions mensuelles (CSV)",
            data=csv_monthly,
            file_name="previsions_mensuelles_xgboost.csv",
            mime="text/csv"
        )

# =========================
# ✅ SAUVEGARDE SESSION_STATE pour la page Comparaison
# =========================
sess_key = f"xgboost_{product}_{depot_id}"
st.session_state[sess_key] = {
    "MAE":       mae,
    "RMSE":      rmse,
    "MAPE":      mape,
    "R2":        r2,
    "quality":   label,
    "future":    forecast_future_df[['ds', 'yhat']].copy(),
    "product":   product,
    "depot_id":  depot_id,
    "depot_sel": depot_sel,
    "freq":      freq_label,
}
# ============================================
# 🤖 ANALYSE IA
# ============================================
from analyse.generateur import generer_analyse_xgboost

st.divider()
st.markdown("### 🤖 Analyse intelligente de cette page")
col_btn = st.columns([2, 2, 2])
with col_btn[1]:
    btn = st.button(" Analyse Intelligente  ", use_container_width=True, type="primary")

if btn:
    filtres = {
        "Produit"           : product or "Tous",
        "Dépôt"             : depot_sel,
        "Pays"              : selected_country,
        "Fréquence"         : freq_label,
        "Modèle"            : "XGBoost",
    }

    metriques = {
        "MAE"                               : f"{mae:.2f}",
        "RMSE"                              : f"{rmse:.2f}",
        "MAPE"                              : f"{mape_val:.2f}%",
        "R²"                                : f"{r2:.4f}" if not np.isnan(r2) else "N/A",
        "Qualité du modèle"                 : label,
        "Moyenne ventes/jour"               : f"{avg_daily_sales:.2f}",
        "Nombre de points"                  : n_pts,
        f"Prévision {next_month_label}"     : f"{next_month_qty:,.0f} unités",
        f"Meilleur mois ({best_month_label})": f"{best_month_qty:,.0f} unités",
        "Tendance générale"                 : trend,
    }

    with st.spinner("🧠 Analyse en cours..."):
        analyse = generer_analyse_xgboost(filtres, metriques)

    with st.container(border=True):
        st.markdown(analyse)

    st.download_button(
        label="⬇️ Télécharger l'analyse",
        data=analyse,
        file_name=f"analyse_xgboost_{product}_{depot_sel}.txt",
        mime="text/plain"
    )