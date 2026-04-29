import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from db_utils import get_connection, sidebar_product_selector, sidebar_depot_selector

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Comparaison Modèles", page_icon="📊", layout="wide")

# =========================
# CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }
.page-title { font-size: 30px; font-weight: 800; color: #1a1a2e; margin-bottom: 4px; }
.page-sub { font-size: 14px; color: #9ca3af; margin-bottom: 24px; }
.section-title {
    font-size: 15px; font-weight: 700; color: #1a1a2e;
    margin: 24px 0 12px 0; padding-left: 10px;
    border-left: 4px solid #6c63ff;
}
.model-card {
    background: #ffffff; border-radius: 16px; padding: 20px;
    box-shadow: 0 2px 16px rgba(108,99,255,0.1);
    border-top: 4px solid var(--c);
}
.model-card .mc-name { font-size: 18px; font-weight: 800; color: var(--c); margin-bottom: 12px; }
.model-card .mc-quality { font-size: 13px; font-weight: 600; margin-bottom: 14px; }
.mc-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.mc-metric { background: #f8f7ff; border-radius: 10px; padding: 10px 12px; }
.mc-metric .lbl { font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
.mc-metric .val { font-size: 18px; font-weight: 800; color: #1a1a2e; }
.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white; font-size: 12px; font-weight: 700;
    padding: 4px 12px; border-radius: 20px;
    margin-left: 10px; vertical-align: middle;
}
.warn-box {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-radius: 14px; padding: 20px 24px;
    color: #92400e; font-weight: 600;
    font-size: 15px; text-align: center; margin: 40px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 Comparaison des Modèles de Prévision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Résultats issus des pages <b>ARIMA</b>, <b>XGBoost</b> et <b>LSTM</b> '
    '— aucun ré-entraînement ici.</div>',
    unsafe_allow_html=True
)

# =========================
# Sidebar
# =========================
product = sidebar_product_selector()

if product is None:
    st.markdown("""
    <div class="warn-box">
        ⚠️ Veuillez sélectionner un produit spécifique dans la sidebar.<br>
        <span style="font-weight:400; font-size:13px;">
        La comparaison nécessite un produit précis pour être pertinente.
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

depot_id, depot_sel, zone_name = sidebar_depot_selector(product)
st.sidebar.caption(f"Produit: {product} | Dépôt: {depot_sel}")
label_produit = f"{product} / {depot_sel}"

# =========================
# Lecture session_state
# =========================
MODEL_KEYS = {
    "ARIMA":   f"arima_{product}_{depot_id}",
    "XGBoost": f"xgboost_{product}_{depot_id}",
    "LSTM":    f"lstm_{product}_{depot_id}",
}

results = {}
missing = []

for model_name, key in MODEL_KEYS.items():
    if key in st.session_state:
        results[model_name] = st.session_state[key]
    else:
        missing.append(model_name)

if missing:
    missing_str = ", ".join([f"**{m}**" for m in missing])
    st.warning(
        f"⚠️ Les résultats de {missing_str} ne sont pas encore disponibles pour "
        f"le produit **{label_produit}**.\n\n"
        f"Veuillez d'abord exécuter {'ces pages' if len(missing) > 1 else 'cette page'} "
        f"avec le même produit et dépôt, puis revenez ici."
    )
    if not results:
        st.stop()

# =========================
# Normalisation → journalier
# Si un modèle a tourné sur des données hebdomadaires (freq="hebdomadaire"),
# ses yhat sont des totaux/semaine. On divise par 7 et on interpole en daily
# pour comparer sur la même échelle que l'historique réel journalier.
# =========================
def to_daily(future_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = future_df.copy()
    if freq == "hebdomadaire":
        # Diviser par 7 pour avoir une quantité journalière moyenne
        df["yhat"] = df["yhat"] / 7.0
        # Interpoler pour avoir un point par jour (au lieu d'un par semaine)
        df = (df.set_index("ds")
                .resample("D")["yhat"]
                .interpolate(method="linear")
                .reset_index())
    return df

# =========================
# Helpers affichage
# =========================
DARK_BG    = "#0e1117"
DARK_PLOT  = "#161b22"
DARK_GRID  = "#21262d"
DARK_TEXT  = "#e6edf3"
HIST_COLOR = "#c9d1d9"

MODEL_COLORS = {
    "ARIMA":   "#f97316",
    "XGBoost": "#a855f7",
    "LSTM":    "#22c55e",
}
MODEL_DASH = {
    "ARIMA":   "dash",
    "XGBoost": "dashdot",
    "LSTM":    "dot",
}
MODEL_COLORS_CARDS = {
    "ARIMA":   "#f97316",
    "XGBoost": "#a855f7",
    "LSTM":    "#10b981",
}

def fmt_r2(v):
    return "N/A" if (v is None or np.isnan(v)) else f"{v:.3f}"

def fmt_mape(v):
    return "N/A" if (v is None or np.isnan(v)) else f"{v:.1f}%"

# =========================
# Meilleur modèle (R² le plus élevé)
# =========================
best_model = max(
    results,
    key=lambda k: results[k]["R2"] if not np.isnan(results[k]["R2"]) else -999
)

# =========================
# Chargement historique réel journalier
# =========================
@st.cache_data(ttl=300)
def load_history(ref_product, depot_id):
    try:
        conn = get_connection()
        conditions, params = [], {}
        if ref_product is not None:
            conditions.append("ref_product = %(ref)s")
            params["ref"] = int(ref_product)
        if depot_id is not None and depot_id != "all":
            conditions.append("depot_id = %(depot)s")
            params["depot"] = int(depot_id)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        df = pd.read_sql(
            f"SELECT quantity, date_time FROM sales {where}",
            conn, params=params if params else None
        )
        conn.close()
        df["date_time"] = pd.to_datetime(df["date_time"])
        df_d = (df.groupby(pd.Grouper(key="date_time", freq="D"))["quantity"]
                .sum().reset_index()
                .rename(columns={"date_time": "ds", "quantity": "y"}))
        df_d = df_d.set_index("ds").asfreq("D").reset_index()
        df_d["y"] = df_d["y"].fillna(0)
        n_nonzero = (df_d["y"] > 0).sum()
        window = 7 if n_nonzero >= 60 else (3 if n_nonzero >= 20 else 1)
        if window > 1:
            df_d["y"] = df_d["y"].rolling(window=window, min_periods=1, center=True).mean()
        mean_y, std_y = df_d["y"].mean(), df_d["y"].std()
        if std_y > 0:
            df_d["y"] = df_d["y"].clip(lower=0, upper=mean_y + 3 * std_y)
        return df_d.dropna(subset=["y"]).reset_index(drop=True)
    except Exception as e:
        st.error(f"⚠️ Erreur chargement historique : {e}")
        return pd.DataFrame(columns=["ds", "y"])

df_hist_full = load_history(product, depot_id)
df_hist      = df_hist_full[df_hist_full["ds"] >= pd.Timestamp("2024-01-01")].copy()
hist_end     = df_hist_full["ds"].max() if not df_hist_full.empty else pd.Timestamp("2025-01-01")

# =========================
# Debug — décommenter pour vérifier les échelles
# =========================
# with st.expander("🔍 Debug — échelles brutes par modèle"):
#     for name, m in results.items():
#         f = m["future"]
#         freq = m.get("freq", "journalière")
#         st.write(f"**{name}** | freq={freq} | n={len(f)} | "
#                  f"min={f['yhat'].min():.2f} | max={f['yhat'].max():.2f} | "
#                  f"mean={f['yhat'].mean():.2f} | "
#                  f"première={f['ds'].iloc[0].date()} | dernière={f['ds'].iloc[-1].date()}")

# =========================
# SECTION 1 — GRAPHIQUE PRINCIPAL
# =========================
st.markdown(
    '<div class="section-title">📈 Historique réel & Comparaison des prévisions (quantité / jour)</div>',
    unsafe_allow_html=True
)

fig_main = go.Figure()

# Séparateur historique / futur
fig_main.add_shape(
    type="line", xref="x", yref="paper",
    x0=hist_end.isoformat(), x1=hist_end.isoformat(),
    y0=0, y1=1,
    line=dict(color="#58a6ff", width=1.5, dash="dash"),
    layer="above"
)
fig_main.add_annotation(
    x=hist_end.isoformat(), xref="x",
    y=0.97, yref="paper",
    text="<b>▶ Prévisions</b>",
    showarrow=False, xanchor="left",
    font=dict(color="#58a6ff", size=11),
    bgcolor="rgba(0,0,0,0)"
)

# Historique réel
if not df_hist.empty:
    fig_main.add_trace(go.Scatter(
        x=df_hist["ds"], y=df_hist["y"],
        mode="lines+markers", name="Réel",
        line=dict(color=HIST_COLOR, width=2),
        marker=dict(size=5, color=HIST_COLOR, symbol="circle"),
        hovertemplate="<b>Réel</b><br>%{x|%Y-%m-%d}<br>Qté/j : %{y:,.2f}<extra></extra>"
    ))

# Prévisions futures normalisées
for name, m in results.items():
    freq      = m.get("freq", "journalière")
    future_df = to_daily(m["future"].copy(), freq)
    future_df = future_df[
        (future_df["ds"] > hist_end) &
        (future_df["ds"] <= pd.Timestamp("2026-12-31"))
    ]
    if future_df.empty:
        continue

    is_best    = name == best_model
    col        = MODEL_COLORS.get(name, "#ffffff")
    dash_style = MODEL_DASH.get(name, "dash")
    lw         = 2.5 if is_best else 1.8
    suffix     = " 🏆" if is_best else ""
    freq_note  = " (÷7/j)" if freq == "hebdomadaire" else ""

    fig_main.add_trace(go.Scatter(
        x=future_df["ds"], y=future_df["yhat"],
        mode="lines",
        name=f"{name}{suffix}{freq_note}",
        line=dict(color=col, dash=dash_style, width=lw),
        legendgroup=name,
        hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>Prév/j : %{{y:,.2f}}<extra></extra>"
    ))

# Annotation meilleur modèle
fig_main.add_annotation(
    x=0.99, y=0.05, xref="paper", yref="paper",
    text=f"🏆 <b>{best_model}</b>  R²={fmt_r2(results[best_model]['R2'])}  MAPE={fmt_mape(results[best_model]['MAPE'])}",
    showarrow=False, xanchor="right",
    font=dict(size=11, color=DARK_TEXT),
    bgcolor="rgba(88,166,255,0.15)",
    bordercolor="#58a6ff", borderwidth=1, borderpad=6
)

fig_main.update_layout(
    template="plotly_dark",
    height=500,
    hovermode="x unified",
    xaxis=dict(
        title=None,
        showgrid=True, gridcolor=DARK_GRID, gridwidth=1,
        tickfont=dict(color=DARK_TEXT, size=12),
        tickformat="%Y-%m",
        range=["2024-01-01", "2026-12-31"],
    ),
    yaxis=dict(
        title="Quantité / jour",
        showgrid=True, gridcolor=DARK_GRID, gridwidth=1,
        zeroline=False,
        tickfont=dict(color=DARK_TEXT, size=12),
    ),
    legend=dict(
        orientation="v", x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        bgcolor="rgba(22,27,34,0.85)",
        bordercolor=DARK_GRID, borderwidth=1,
        font=dict(size=12, color=DARK_TEXT),
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor=DARK_PLOT,
    paper_bgcolor=DARK_BG,
    font=dict(color=DARK_TEXT),
    title=dict(
        text=f"<b>Comparaison Multi-Modèles : {label_produit}</b>",
        font=dict(size=16, color=DARK_TEXT),
        x=0.01
    ),
)

st.plotly_chart(fig_main, use_container_width=True)

# =========================
# SECTION 2 — CARDS PERFORMANCES
# =========================
st.markdown(
    '<div class="section-title">📋 Performances par modèle</div>',
    unsafe_allow_html=True
)

cols_cards = st.columns(len(results))
for i, (name, m) in enumerate(results.items()):
    is_best  = name == best_model
    r2_val   = m["R2"]
    mape_val = m["MAPE"] if not np.isnan(m["MAPE"]) else 100
    freq     = m.get("freq", "journalière")
    with cols_cards[i]:
        st.markdown(f"""
        <div class="model-card" style="--c:{MODEL_COLORS_CARDS.get(name,'#6c63ff')}">
          <div class="mc-name">{name}
            {'<span class="best-badge">🏆 Meilleur</span>' if is_best else ''}
          </div>
          <div class="mc-quality">
            {m['quality']}
            <span style="font-size:11px;color:#9ca3af;font-weight:400"> · {freq}</span>
          </div>
          <div class="mc-metrics">
            <div class="mc-metric">
              <div class="lbl">R²</div>
              <div class="val">{'N/A' if np.isnan(r2_val) else f'{r2_val:.3f}'}</div>
            </div>
            <div class="mc-metric">
              <div class="lbl">MAPE</div>
              <div class="val">{mape_val:.1f}%</div>
            </div>
            <div class="mc-metric">
              <div class="lbl">MAE</div>
              <div class="val">{m['MAE']:.2f}</div>
            </div>
            <div class="mc-metric">
              <div class="lbl">RMSE</div>
              <div class="val">{m['RMSE']:.2f}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================
# SECTION 5 — RECOMMANDATION
# =========================
st.markdown(
    '<div class="section-title">💡 Recommandation</div>',
    unsafe_allow_html=True
)

interp = {
    "ARIMA":   "Idéal pour les séries stationnaires et linéaires. Rapide à entraîner.",
    "XGBoost": "Excellent pour capturer les patterns non-linéaires avec des features riches.",
    "LSTM":    "Meilleur pour les dépendances temporelles longues et complexes.",
}

best = results[best_model]
st.success(f"""
**🏆 Modèle recommandé pour le produit {label_produit} : {best_model}**

R² = **{fmt_r2(best['R2'])}** | MAPE = **{fmt_mape(best['MAPE'])}** | MAE = **{best['MAE']:.2f}** | RMSE = **{best['RMSE']:.2f}** | {best['quality']}

{interp.get(best_model, '')}
""")

if missing:
    st.info(
        "ℹ️ Modèles non encore chargés : "
        + ", ".join(missing)
        + ". Lancez leurs pages respectives pour les inclure."
    )