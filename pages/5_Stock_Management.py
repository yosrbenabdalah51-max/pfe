import streamlit as st
st.set_page_config(page_title="Stock Management", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import get_connection

st.title(" Stock Management")
st.caption("Sélection automatique du meilleur modèle par produit · Analyse stock · Alertes rupture / surstock")

MODEL_COLORS = {
    "arima":   "#6c63ff",
    "xgboost": "#f59e0b",
    "lstm":    "#e74c3c",
}
MODEL_ICONS = {
    "arima":   "🟢",
    "xgboost": "📊",
    "lstm":    "🔴",
}

def get_all_model_results():
    results = {}
    for key, val in st.session_state.items():
        for model in ["arima", "xgboost", "lstm"]:
            if key.startswith(f"{model}_") and isinstance(val, dict) and "R2" in val:
                product  = val.get("product")
                depot_id = val.get("depot_id")
                combo    = (str(product), str(depot_id))
                results.setdefault(combo, {})
                results[combo][model] = val
    return results

all_results = get_all_model_results()

if not all_results:
    st.warning("⚠️ Aucun modèle n'a encore été exécuté. Lance au moins un modèle (ARIMA, XGBoost ou LSTM) depuis la sidebar.")
    st.stop()

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.header("⚙️ Paramètres")
    combo_options = sorted(all_results.keys())
    combo_labels  = [f"Produit {p} / Dépôt {d}" for p, d in combo_options]
    selected_idx  = st.selectbox("Produit / Dépôt", range(len(combo_labels)),
                                 format_func=lambda i: combo_labels[i])
    selected_combo = combo_options[selected_idx]
    lead_time    = st.number_input("Délai réappro (jours)", min_value=1, max_value=60, value=7)
    horizon_days = st.selectbox("Horizon analyse (jours)", [7, 14, 30, 60, 90], index=2)
    stock_actuel = st.number_input("Stock actuel (unités)", min_value=0, value=100)

# =============================================================
# MEILLEUR MODÈLE
# =============================================================
combo_models = all_results[selected_combo]

def model_score(data):
    r2 = data.get("R2", float('-inf'))
    r2 = r2 if (r2 is not None and not np.isnan(r2)) else float('-inf')
    return r2

best_model_name = max(combo_models, key=lambda m: model_score(combo_models[m]))
best_data       = combo_models[best_model_name]
product         = best_data["product"]
depot_id        = best_data["depot_id"]
depot_sel       = best_data.get("depot_sel", depot_id)

# =============================================================
# BANDEAU MEILLEUR MODÈLE
# =============================================================
st.markdown("### 🏆 Meilleur modèle sélectionné pour ce produit")
color = MODEL_COLORS.get(best_model_name, "#888")
icon  = MODEL_ICONS.get(best_model_name, "🔵")
st.markdown(f"""
<div style="border:2px solid {color};border-radius:12px;padding:16px 24px;
            text-align:center;background:rgba(108,99,255,0.06);
            max-width:300px;margin:auto">
    <div style="font-size:18px;font-weight:700;color:{color}">{icon} {best_model_name.upper()}</div>
    <div style="font-size:12px;background:{color};color:white;border-radius:8px;
                padding:3px 12px;display:inline-block;margin-top:8px">✓ Modèle utilisé</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

missing = [m for m in ["arima", "xgboost", "lstm"] if m not in combo_models]
if missing:
    st.info(f"ℹ️ Modèles non encore exécutés : **{', '.join(m.upper() for m in missing)}**")

# =============================================================
# PRÉVISIONS
# =============================================================
forecast_df = best_data.get("future")
if forecast_df is None or forecast_df.empty:
    st.error("❌ Aucune prévision disponible pour ce modèle.")
    st.stop()

forecast_df = forecast_df.copy()
forecast_df["ds"]   = pd.to_datetime(forecast_df["ds"])
forecast_df["yhat"] = forecast_df["yhat"].clip(lower=0)

today            = pd.Timestamp.today().normalize()
forecast_horizon = forecast_df[forecast_df["ds"] >= today].head(horizon_days).copy()

if forecast_horizon.empty:
    st.warning("⚠️ Aucune prévision disponible sur la période demandée.")
    st.stop()

# =============================================================
# HISTORIQUE VENTES
# =============================================================
@st.cache_data(ttl=300)
def load_history(ref_product, depot_id_val):
    try:
        conn = get_connection()
        conditions, params = [], {}
        if ref_product is not None and str(ref_product) != "None":
            conditions.append("ref_product = %(ref)s")
            params["ref"] = int(ref_product)
        if depot_id_val is not None and str(depot_id_val) not in ("None", "all"):
            conditions.append("depot_id = %(depot)s")
            params["depot"] = int(depot_id_val)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        df = pd.read_sql(f"""
            SELECT DATE(date_time) AS date, SUM(quantity) AS quantity
            FROM sales {where}
            GROUP BY DATE(date_time)
            ORDER BY date
        """, conn, params=params if params else None)
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.warning(f"⚠️ Historique non disponible : {e}")
        return pd.DataFrame(columns=["date", "quantity"])

with st.spinner("Chargement historique..."):
    history = load_history(product, depot_id)

# =============================================================
# SEUILS
# =============================================================
if not history.empty:
    hist_qty      = history["quantity"].values
    q1_h          = np.quantile(hist_qty, 0.25)
    q2_h          = np.quantile(hist_qty, 0.50)
    q3_h          = np.quantile(hist_qty, 0.75)
    iqr_h         = q3_h - q1_h
    safety_stock  = int(round(q1_h * lead_time))
    stock_optimal = int(round(q2_h * lead_time))
    reorder_point = int(round(q3_h * lead_time))
    stock_max     = int(round((q3_h + 1.5 * iqr_h) * lead_time))
else:
    yhat_mean     = forecast_horizon["yhat"].mean()
    safety_stock  = int(round(yhat_mean * lead_time * 0.5))
    stock_optimal = int(round(yhat_mean * lead_time))
    reorder_point = int(round(yhat_mean * lead_time * 1.5))
    stock_max     = int(round(yhat_mean * lead_time * 2.5))

# =============================================================
# DEMANDE PRÉVUE
# =============================================================
yhat_vals    = forecast_horizon["yhat"].values
demand_total = int(round(float(np.sum(yhat_vals))))
demand_moy_j = round(float(np.mean(yhat_vals)), 1)

# =============================================================
# DIAGNOSTIC
# =============================================================
stock_fin_horizon = stock_actuel - demand_total

if stock_fin_horizon < 0:
    risque = "🔴 Rupture de stock prévue"
    risk_color = "#e74c3c"; risk_bg = "rgba(231,76,60,0.1)"
elif stock_fin_horizon < safety_stock:
    risque = "🟠 Risque de rupture"
    risk_color = "#e67e22"; risk_bg = "rgba(230,126,34,0.1)"
elif stock_fin_horizon <= stock_optimal:
    risque = "🟢 Niveau optimal"
    risk_color = "#27ae60"; risk_bg = "rgba(39,174,96,0.1)"
elif stock_fin_horizon <= stock_max:
    risque = "🟡 Surstock léger"
    risk_color = "#f39c12"; risk_bg = "rgba(243,156,18,0.1)"
else:
    risque = "🟡 Surstock important"
    risk_color = "#f39c12"; risk_bg = "rgba(243,156,18,0.1)"

qte_cmd  = max(0, reorder_point - stock_fin_horizon)
excedent = max(0, stock_fin_horizon - stock_max)

if qte_cmd > 0:    action = f"→ Commander <b>{qte_cmd:,}</b> unités minimum"
elif excedent > 0: action = f"→ Excédent de <b>{excedent:,}</b> unités à réduire"
else:              action = "→ Aucune action requise ✓"

# =============================================================
# KPIs
# =============================================================
st.markdown("---")
st.markdown(f"### 📊 Analyse stock — Produit **{product}** / Dépôt **{depot_sel}** · Modèle : {MODEL_ICONS.get(best_model_name,'')} **{best_model_name.upper()}**")

k1, k2, k3, k4, k5, k6 = st.columns(6)
metrics = [
    (k1, "Stock actuel",               f"{stock_actuel:,}",      "#4a9eff", "unités disponibles"),
    (k2, f"Demande ({horizon_days}j)",  f"{demand_total:,}",      MODEL_COLORS.get(best_model_name,'#888'), f"Σ prévision {best_model_name.upper()}"),
    (k3, "Stock fin horizon",           f"{stock_fin_horizon:,}", "#4a9eff" if stock_fin_horizon >= 0 else "#e74c3c", f"après {horizon_days}j"),
    (k4, "Sécurité (Q1)",              f"{safety_stock:,}",       "#e74c3c", f"Q1×{lead_time}j"),
    (k5, "Optimal (Q2)",               f"{stock_optimal:,}",      "#27ae60", f"Q2×{lead_time}j"),
    (k6, "ROP (Q3)",                   f"{reorder_point:,}",      "#f39c12", f"Q3×{lead_time}j"),
]
for col, title, value, color, subtitle in metrics:
    with col:
        st.markdown(f"""
        <div style="background:#0f1117;border:1px solid #2a2d3a;border-radius:10px;
                    padding:14px;text-align:center;">
            <div style="font-size:10px;color:#6b7280;text-transform:uppercase;
                        letter-spacing:.08em;margin-bottom:5px">{title}</div>
            <div style="font-size:22px;font-weight:700;color:{color}">{value}</div>
            <div style="font-size:10px;color:#6b7280;margin-top:3px">{subtitle}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background:{risk_bg};border-left:4px solid {risk_color};
            padding:14px 18px;border-radius:0 8px 8px 0;margin-bottom:20px">
    <span style="color:{risk_color};font-size:17px;font-weight:600">{risque}</span><br>
    Stock actuel : <b>{stock_actuel:,}</b> unités &nbsp;|&nbsp;
    Demande prévue ({horizon_days}j) : <b>{demand_total:,}</b> unités &nbsp;|&nbsp;
    Stock restant : <b>{stock_fin_horizon:,}</b> unités &nbsp;|&nbsp;
    Moy/jour : <b>{demand_moy_j}</b> &nbsp;|&nbsp; {action}
</div>
""", unsafe_allow_html=True)

# =============================================================
# FONCTION SIMULATION CORRIGÉE AVEC LEAD TIME
# =============================================================
def simulate_stock(stock_depart, forecast_df, reorder_point, stock_optimal, lead_time):
    """
    Simulation réaliste :
    - La commande est passée quand stock < ROP
    - La livraison arrive après lead_time jours
    - La quantité commandée = stock_optimal - stock_actuel (pas ROP entier)
    - Une seule commande en attente à la fois
    """
    n          = len(forecast_df)
    stock_c    = stock_depart
    livraisons = {}   # {jour_livraison: quantite}
    
    stock_sim  = []
    cmd_sim    = []
    cmd_en_cours = False   # évite de passer plusieurs commandes simultanées

    for i, (_, row) in enumerate(forecast_df.iterrows()):

        # 1. Réceptionner la livraison si elle arrive aujourd'hui
        if i in livraisons:
            stock_c += livraisons[i]
            cmd_en_cours = False

        # 2. Consommer la demande du jour
        stock_c -= row["yhat"]

        # 3. Passer une commande si stock < ROP et pas de commande déjà en cours
        cmd = 0
        if stock_c < reorder_point and not cmd_en_cours:
            cmd = max(0, stock_optimal - stock_c)   # commander juste ce qu'il faut
            jour_livraison = i + lead_time
            if jour_livraison < n:                  # livraison dans l'horizon
                livraisons[jour_livraison] = livraisons.get(jour_livraison, 0) + cmd
            cmd_en_cours = True

        stock_sim.append(max(0, round(stock_c)))
        cmd_sim.append(round(cmd))

    return stock_sim, cmd_sim

# =============================================================
# TABS
# =============================================================
tab1, tab2 = st.tabs([
    "📈 Évolution stock simulée",
    "📋 Tableau journalier",
])

# ── TAB 1 ──────────────────────────────────────────────────
with tab1:
    st.subheader(f"Simulation du stock sur {horizon_days} jours — {best_model_name.upper()}")

    stock_sim, cmd_sim = simulate_stock(
        stock_actuel, forecast_horizon, reorder_point, stock_optimal, lead_time
    )

    forecast_horizon = forecast_horizon.copy()
    forecast_horizon["stock_simule"] = stock_sim
    forecast_horizon["commande"]     = cmd_sim
    forecast_horizon["date_label"]   = forecast_horizon["ds"].dt.strftime("%d/%m")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f"Demande prévue ({best_model_name.upper()})",
        x=forecast_horizon["date_label"],
        y=forecast_horizon["yhat"].round(1),
        marker_color=MODEL_COLORS.get(best_model_name, "#888"),
        opacity=0.85, offsetgroup=1, width=0.35
    ))

    fig.add_trace(go.Bar(
        name="Stock simulé",
        x=forecast_horizon["date_label"],
        y=forecast_horizon["stock_simule"],
        marker_color="#a78bfa",
        opacity=0.85, offsetgroup=2, width=0.35
    ))

    # Marqueurs de commandes passées
    cmd_days = forecast_horizon[forecast_horizon["commande"] > 0]
    if not cmd_days.empty:
        fig.add_trace(go.Scatter(
            name="Commande passée",
            x=cmd_days["date_label"],
            y=cmd_days["stock_simule"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#00d4ff"),
        ))

    for y_val, hname, hcolor in [
        (safety_stock,  f"Sécurité={safety_stock}",  "#e74c3c"),
        (stock_optimal, f"Optimal={stock_optimal}",   "#27ae60"),
        (reorder_point, f"ROP={reorder_point}",       "#f59e0b"),
        (stock_max,     f"Max={stock_max}",           "#9b59b6"),
    ]:
        fig.add_hline(
            y=y_val, line_dash="dot", line_color=hcolor, line_width=2,
            annotation_text=hname, annotation_position="top left",
            annotation_font_color=hcolor, annotation_font_size=11
        )

    fig.update_layout(
        barmode="group", template="plotly_dark", height=500,
        xaxis_title="Jour", yaxis_title="Quantité (unités)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        bargap=0.2, bargroupgap=0.05,
        xaxis=dict(tickangle=-45, nticks=min(len(forecast_horizon), 30))
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"🔺 Triangle bleu = commande passée · Livraison reçue après {lead_time} jours · "
            f"Quantité commandée = stock optimal ({stock_optimal}) - stock courant")

# ── TAB 2 ──────────────────────────────────────────────────
with tab2:
    st.subheader("Tableau journalier — Stock simulé & Alertes")

    stock_sim2, cmd_sim2 = simulate_stock(
        stock_actuel, forecast_horizon, reorder_point, stock_optimal, lead_time
    )

    rows     = []
    stock_c  = stock_actuel
    livraisons2 = {}
    cmd_en_cours2 = False

    for i, (_, row) in enumerate(forecast_horizon.iterrows()):

        # Réception livraison
        livraison_recue = livraisons2.get(i, 0)
        if livraison_recue > 0:
            stock_c += livraison_recue
            cmd_en_cours2 = False

        stock_av  = round(stock_c, 1)
        demande_j = round(float(row["yhat"]), 1)
        stock_c  -= demande_j

        cmd = 0
        if stock_c < reorder_point and not cmd_en_cours2:
            cmd = max(0, stock_optimal - stock_c)
            jour_livraison = i + lead_time
            if jour_livraison < len(forecast_horizon):
                livraisons2[jour_livraison] = livraisons2.get(jour_livraison, 0) + cmd
            cmd_en_cours2 = True

        stock_ap = max(0, round(stock_c, 1))

        if stock_ap <= 0:              etat = "🔴 Rupture"
        elif stock_ap < safety_stock:  etat = "🟠 Risque rupture"
        elif stock_ap <= stock_optimal:etat = "🟢 Optimal"
        elif stock_ap <= stock_max:    etat = "🟡 Surstock léger"
        else:                          etat = "🟡 Surstock"

        rows.append({
            "Date":                row["ds"].strftime("%d/%m/%Y"),
            "Demande prévue":      demande_j,
            "Livraison reçue":     livraison_recue if livraison_recue > 0 else "—",
            "Stock début jour":    stock_av,
            "Stock fin jour":      stock_ap,
            "Commande passée":     f"{round(cmd)} (J+{lead_time})" if cmd > 0 else "—",
            "État":                etat,
        })

    df_table = pd.DataFrame(rows)

    def color_etat(val):
        if "Rupture"   in str(val): return "background-color:rgba(231,76,60,0.2);color:#e74c3c;font-weight:600"
        elif "Risque"  in str(val): return "background-color:rgba(230,126,34,0.15);color:#e67e22;font-weight:600"
        elif "Optimal" in str(val): return "background-color:rgba(39,174,96,0.15);color:#27ae60;font-weight:600"
        elif "Surstock"in str(val): return "background-color:rgba(243,156,18,0.15);color:#f39c12;font-weight:600"
        return ""

    st.dataframe(
        df_table.style.applymap(color_etat, subset=["État"]),
        use_container_width=True, height=460
    )

    csv = df_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter CSV",
        data=csv,
        file_name=f"stock_journalier_{product}_{depot_id}_{horizon_days}j.csv",
        mime="text/csv"
    )
    # ============================================
# 🤖 ANALYSE IA
# ============================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyse.generateur import generer_analyse_comparaison

st.divider()

col_btn = st.columns([2, 2, 2])
with col_btn[1]:
    btn = st.button("🤖 Analyse Intelligente", use_container_width=True, type="primary")

if btn:
    filtres = {
        "Produit"          : str(product),
        "Dépôt"            : str(depot_sel),
        "Modèle utilisé"   : best_model_name.upper(),
        "Horizon analyse"  : f"{horizon_days} jours",
        "Délai réappro"    : f"{lead_time} jours",
    }

    metriques = {
        "Stock actuel"              : f"{stock_actuel} unités",
        "Demande prévue totale"     : f"{demand_total} unités",
        "Moyenne journalière"       : f"{demand_moy_j} unités/jour",
        "Stock fin horizon"         : f"{stock_fin_horizon} unités",
        "Diagnostic"                : risque,
        "Seuil sécurité"            : f"{safety_stock} unités",
        "Stock optimal"             : f"{stock_optimal} unités",
        "Point de réappro (ROP)"    : f"{reorder_point} unités",
        "Stock maximum"             : f"{stock_max} unités",
        "Action recommandée"        : f"Commander {qte_cmd} unités" if qte_cmd > 0 else "Aucune commande requise",
        "Excédent"                  : f"{excedent} unités" if excedent > 0 else "Aucun",
    }

    with st.spinner("🧠 Analyse en cours..."):
        analyse = generer_analyse_comparaison(filtres, metriques)

    with st.container(border=True):
        st.markdown(analyse)

    st.download_button(
        label="⬇️ Télécharger l'analyse",
        data=analyse,
        file_name=f"analyse_stock_{product}_{depot_sel}.txt",
        mime="text/plain"
    )