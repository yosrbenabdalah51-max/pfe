import streamlit as st
st.set_page_config(page_title="Comparaison Modèles", page_icon="📊", layout="wide")

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_auth, user_topbar   # ← importer user_topbar
require_auth("Comparaison Modèles")

# ✅ Affiche nom utilisateur + bouton déconnexion EN DEHORS du sidebar
user_topbar()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from utils import get_connection, sidebar_filters
warnings.filterwarnings("ignore")

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 Comparaison des Modèles de Prévision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Résultats issus des pages <b>SARIMA</b>, <b>XGBoost</b> et <b>LSTM</b> '
    '— aucun ré-entraînement ici.</div>',
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
df, product, depot_id, depot_sel, date_range, selected_country = sidebar_filters()

label_produit = str(product) if product is not None else "Tous"

if df is None or len(df) == 0:
    st.error(f"❌ Aucune vente trouvée pour le produit **{label_produit}** / dépôt **{depot_sel}**.")
    st.stop()

# =========================
# Lecture session_state
# =========================
MODEL_KEYS = {
    "SARIMA":   f"sarima_{product}_{depot_id}",
    "XGBoost":  f"xgboost_{product}_{depot_id}",
    "LSTM":     f"lstm_{product}_{depot_id}",
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
# =========================
def to_daily(future_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = future_df.copy()
    if freq == "hebdomadaire":
        df["yhat"] = df["yhat"] / 7.0
        df = (df.set_index("ds")
                .resample("D")["yhat"]
                .interpolate(method="linear")
                .reset_index())
    return df

# =========================
# Constantes visuelles
# =========================
DARK_BG   = "#0e1117"
DARK_PLOT = "#161b22"
DARK_GRID = "#21262d"
DARK_TEXT = "#e6edf3"
HIST_COLOR = "#c9d1d9"

MODEL_COLORS = {
    "SARIMA":  "#f97316",
    "XGBoost": "#a855f7",
    "LSTM":    "#22c55e",
}
MODEL_COLORS_CARDS = {
    "SARIMA":  "#f97316",
    "XGBoost": "#a855f7",
    "LSTM":    "#10b981",
}
MODEL_DASH = {
    "SARIMA":  "dash",
    "XGBoost": "dashdot",
    "LSTM":    "dot",
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
# SECTION 1 — GRAPHIQUE PRINCIPAL
# =========================
st.markdown(
    '<div class="section-title">📈 Historique réel & Comparaison des prévisions (quantité / jour)</div>',
    unsafe_allow_html=True
)

fig_main = go.Figure()

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

if not df_hist.empty:
    fig_main.add_trace(go.Scatter(
        x=df_hist["ds"], y=df_hist["y"],
        mode="lines+markers", name="Réel",
        line=dict(color=HIST_COLOR, width=2),
        marker=dict(size=5, color=HIST_COLOR, symbol="circle"),
        hovertemplate="<b>Réel</b><br>%{x|%Y-%m-%d}<br>Qté/j : %{y:,.2f}<extra></extra>"
    ))

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
# SECTION 3 — TABLEAU COMPARATIF
# =========================
st.markdown(
    '<div class="section-title">📊 Tableau comparatif des métriques</div>',
    unsafe_allow_html=True
)

rows = []
for name, m in results.items():
    rows.append({
        "Modèle":   name,
        "R²":       fmt_r2(m["R2"]),
        "MAPE":     fmt_mape(m["MAPE"]),
        "MAE":      f"{m['MAE']:.2f}",
        "RMSE":     f"{m['RMSE']:.2f}",
        "Qualité":  m["quality"],
        "Fréquence": m.get("freq", "journalière"),
        "Meilleur": "🏆" if name == best_model else "",
    })

df_table = pd.DataFrame(rows)
st.dataframe(df_table, use_container_width=True, hide_index=True)

# =========================
# SECTION 4 — GRAPHIQUE BARRES MÉTRIQUES
# =========================
st.markdown(
    '<div class="section-title">📉 Comparaison visuelle MAE / RMSE / MAPE</div>',
    unsafe_allow_html=True
)

model_names = list(results.keys())
mae_vals    = [results[n]["MAE"]  for n in model_names]
rmse_vals   = [results[n]["RMSE"] for n in model_names]
mape_vals   = [results[n]["MAPE"] if not np.isnan(results[n]["MAPE"]) else 0 for n in model_names]
colors      = [MODEL_COLORS.get(n, "#6c63ff") for n in model_names]

col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    fig_mae = go.Figure(go.Bar(
        x=model_names, y=mae_vals,
        marker_color=colors, text=[f"{v:.2f}" for v in mae_vals],
        textposition="outside"
    ))
    fig_mae.update_layout(
        title="MAE (↓ meilleur)", template="plotly_white",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="MAE"
    )
    st.plotly_chart(fig_mae, use_container_width=True)

with col_b2:
    fig_rmse = go.Figure(go.Bar(
        x=model_names, y=rmse_vals,
        marker_color=colors, text=[f"{v:.2f}" for v in rmse_vals],
        textposition="outside"
    ))
    fig_rmse.update_layout(
        title="RMSE (↓ meilleur)", template="plotly_white",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="RMSE"
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with col_b3:
    fig_mape = go.Figure(go.Bar(
        x=model_names, y=mape_vals,
        marker_color=colors, text=[f"{v:.1f}%" for v in mape_vals],
        textposition="outside"
    ))
    fig_mape.update_layout(
        title="MAPE % (↓ meilleur)", template="plotly_white",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="MAPE %"
    )
    st.plotly_chart(fig_mape, use_container_width=True)

# =========================
# SECTION 5 — PRÉVISIONS MENSUELLES COMPARÉES
# =========================
st.markdown(
    '<div class="section-title">📅 Prévisions mensuelles comparées</div>',
    unsafe_allow_html=True
)

today = pd.Timestamp.today().normalize()
fig_monthly = go.Figure()

for name, m in results.items():
    freq      = m.get("freq", "journalière")
    future_df = to_daily(m["future"].copy(), freq)
    future_df = future_df[future_df["ds"] > hist_end].copy()
    if future_df.empty:
        continue
    future_df["month"] = future_df["ds"].dt.to_period("M")
    monthly = (future_df.groupby("month")["yhat"].sum().reset_index())
    monthly["month_str"] = monthly["month"].astype(str)
    monthly = monthly[
        monthly["month"].apply(lambda p: p.to_timestamp()) >= today.replace(day=1)
    ]
    if monthly.empty:
        continue
    fig_monthly.add_trace(go.Scatter(
        x=monthly["month_str"], y=monthly["yhat"].round(0),
        mode="lines+markers", name=name,
        line=dict(color=MODEL_COLORS.get(name, "#6c63ff"), width=2),
        marker=dict(size=7),
        hovertemplate=f"<b>{name}</b><br>%{{x}}<br>Total : %{{y:,.0f}}<extra></extra>"
    ))

fig_monthly.update_layout(
    template="plotly_white", height=380,
    hovermode="x unified",
    xaxis_title="Mois", yaxis_title="Quantité totale prévue",
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", y=1.1, x=0),
)
st.plotly_chart(fig_monthly, use_container_width=True)

# =========================
# SECTION 6 — RECOMMANDATION
# =========================
st.markdown(
    '<div class="section-title">💡 Recommandation</div>',
    unsafe_allow_html=True
)

interp = {
    "SARIMA":  "Idéal pour les séries stationnaires et linéaires avec saisonnalité. Rapide à entraîner.",
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

# =========================
# 🤖 ANALYSE IA
# =========================
from analyse.generateur import generer_analyse_comparaison

st.divider()
col_btn = st.columns([2, 2, 2])
with col_btn[1]:
    btn = st.button("🤖 Analyse Intelligente", use_container_width=True, type="primary")

if btn:
    filtres = {
        "Produit":         label_produit,
        "Dépôt":           depot_sel,
        "Pays":            selected_country,
        "Meilleur modèle": best_model,
    }
    metriques = {}
    for name, m in results.items():
        metriques[f"{name} — R²"]        = fmt_r2(m["R2"])
        metriques[f"{name} — MAPE"]      = fmt_mape(m["MAPE"])
        metriques[f"{name} — MAE"]       = f"{m['MAE']:.2f}"
        metriques[f"{name} — RMSE"]      = f"{m['RMSE']:.2f}"
        metriques[f"{name} — Qualité"]   = m["quality"]
        metriques[f"{name} — Fréquence"] = m.get("freq", "journalière")

    with st.spinner("🧠 Analyse en cours..."):
        analyse = generer_analyse_comparaison(filtres, metriques)

    with st.container(border=True):
        st.markdown(analyse)

    st.download_button(
        label="⬇️ Télécharger l'analyse",
        data=analyse,
        file_name=f"analyse_comparaison_{label_produit}_{depot_sel}.txt",
        mime="text/plain"
    )