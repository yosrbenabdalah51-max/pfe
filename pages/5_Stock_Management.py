import streamlit as st
st.set_page_config(page_title="Stock Management", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import get_connection
from auth import require_auth
require_auth("Stock Management")

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

# =============================================================
# RÉCUPÉRATION DES RÉSULTATS MODÈLES
# =============================================================
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
    st.warning("⚠️ Aucun modèle n'a encore été exécuté. Lance au moins un modèle (ARIMA, XGBoost ou LSTM).")
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

# =============================================================
# ESTIMATION STOCK — cohérente avec les seuils, robuste aux données anciennes
# =============================================================
def estimer_stock(history: pd.DataFrame, lead_time: int = 7, periode_jours: int = 30) -> tuple:
    """
    Estime le stock actuel ≈ stock optimal théorique
    = ROP (Q3 × lead_time) + Q2 × lead_time

    Fallback automatique si les 30 derniers jours calendaires sont vides :
    on utilise les 30 dernières lignes disponibles dans l'historique.

    Retourne (estime, recent_qty, fallback_utilise)
    """
    if history.empty:
        return 0, pd.Series(dtype=float), False

    today  = pd.Timestamp.today().normalize()
    recent = history[history["date"] >= today - pd.Timedelta(days=periode_jours)]["quantity"]
    fallback_utilise = False

    if recent.empty:
        recent = history.tail(periode_jours)["quantity"]
        fallback_utilise = True

    if recent.empty:
        return 0, pd.Series(dtype=float), fallback_utilise

    q2 = np.quantile(recent.values, 0.50)
    q3 = np.quantile(recent.values, 0.75)

    reorder_point_estime = int(round(q3 * lead_time))
    estime               = int(round(reorder_point_estime + q2 * lead_time))
    return max(0, estime), recent, fallback_utilise

# =============================================================
# SIDEBAR — Paramètres
# =============================================================
with st.sidebar:
    st.header("⚙️ Paramètres")

    combo_options = sorted(all_results.keys())
    combo_labels  = [f"Produit {p} / Dépôt {d}" for p, d in combo_options]
    selected_idx  = st.selectbox(
        "Produit / Dépôt", range(len(combo_labels)),
        format_func=lambda i: combo_labels[i]
    )
    selected_combo = combo_options[selected_idx]

    lead_time = st.number_input("Délai réappro (jours)", min_value=1, max_value=60, value=7)

    st.markdown("#### 📅 Horizon d'analyse")
    today_date = pd.Timestamp.today().normalize()

    date_fin = st.date_input(
        "Analyser jusqu'au",
        value=(today_date + pd.Timedelta(days=30)).date(),
        min_value=(today_date + pd.Timedelta(days=1)).date(),
        max_value=pd.Timestamp("2026-12-31").date(),
        help="Choisissez librement la date de fin d'analyse. Les prévisions disponibles vont jusqu'à la date générée par vos modèles."
    )

    horizon_days = (pd.Timestamp(date_fin) - today_date).days
    st.caption(f"📊 Soit **{horizon_days} jour(s)** à partir d'aujourd'hui")

    if horizon_days > 90:
        st.warning(
            f"⚠️ Horizon long ({horizon_days} jours).\n\n"
            "La fiabilité des prévisions diminue avec le temps. "
            "Au-delà de 90 jours, les chiffres sont indicatifs."
        )

# =============================================================
# MEILLEUR MODÈLE
# =============================================================
combo_models = all_results[selected_combo]

def model_score(data):
    r2 = data.get("R2", float('-inf'))
    return r2 if (r2 is not None and not np.isnan(r2)) else float('-inf')

best_model_name = max(combo_models, key=lambda m: model_score(combo_models[m]))
best_data       = combo_models[best_model_name]
product         = best_data["product"]
depot_id        = best_data["depot_id"]
depot_sel       = best_data.get("depot_sel") or depot_id

# Chargement historique
with st.spinner("Chargement historique..."):
    history = load_history(product, depot_id)

# =============================================================
# SEUILS — basés sur tout l'historique disponible
# =============================================================
if not history.empty:
    hist_qty = history["quantity"].values
    q1_h     = np.quantile(hist_qty, 0.25)
    q2_h     = np.quantile(hist_qty, 0.50)
    q3_h     = np.quantile(hist_qty, 0.75)
    iqr_h    = q3_h - q1_h

    safety_stock  = int(round(q1_h * lead_time))
    reorder_point = int(round(q3_h * lead_time))
    stock_optimal = int(round(reorder_point + q2_h * lead_time))
    stock_max     = int(round(reorder_point + (q3_h + 1.5 * iqr_h) * lead_time))
else:
    forecast_df_tmp = best_data.get("future", pd.DataFrame())
    yhat_mean     = forecast_df_tmp["yhat"].mean() if not forecast_df_tmp.empty else 100
    safety_stock  = int(round(yhat_mean * lead_time * 0.5))
    reorder_point = int(round(yhat_mean * lead_time * 1.5))
    stock_optimal = int(round(reorder_point + yhat_mean * lead_time))
    stock_max     = int(round(reorder_point + yhat_mean * lead_time * 2.0))

# =============================================================
# STOCK ESTIMÉ
# =============================================================
stock_estime, recent_qty_used, fallback_utilise = estimer_stock(
    history, lead_time=lead_time, periode_jours=30
)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📦 Stock actuel")

    if not history.empty:
        if fallback_utilise:
            st.warning(
                "⚠️ Pas de ventes dans les 30 derniers jours calendaires.\n\n"
                "Estimation basée sur les **dernières données disponibles** dans l'historique."
            )
        if not recent_qty_used.empty:
            q2_disp = round(np.quantile(recent_qty_used.values, 0.50), 1)
            q3_disp = round(np.quantile(recent_qty_used.values, 0.75), 1)
            st.info(
                f"📊 Stock estimé ≈ stock optimal théorique\n\n"
                f"ROP (Q3={q3_disp} × {lead_time}j) + Q2 ({q2_disp} × {lead_time}j) "
                f"= **{stock_estime:,} unités**"
            )
        else:
            st.info("📊 Historique insuffisant — stock estimé à 0.")
    else:
        st.info("📊 Aucun historique disponible — stock estimé à 0.")

    stock_actuel = st.number_input(
        "Confirmer / Corriger le stock (unités)",
        min_value=0,
        value=stock_estime,
        help=(
            "Estimé automatiquement depuis les quantiles de ventes × délai réappro. "
            "Cohérent avec les seuils affichés. Corrigez si vous connaissez la valeur réelle."
        )
    )
    if stock_actuel != stock_estime:
        st.warning(f"⚠️ Valeur modifiée manuellement ({stock_actuel:,} unités)")
    else:
        st.success("✅ Valeur estimée automatiquement")

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

date_fin_ts      = pd.Timestamp(date_fin)
forecast_horizon = forecast_df[
    (forecast_df["ds"] >= today_date) &
    (forecast_df["ds"] <= date_fin_ts)
].copy()

if forecast_horizon.empty:
    st.error(
        f"❌ Aucune prévision disponible entre aujourd'hui et le {date_fin.strftime('%d/%m/%Y')}.\n\n"
        "Les prévisions de vos modèles ne couvrent pas cette période. "
        "Réduisez l'horizon ou relancez vos modèles avec un horizon plus long."
    )
    st.stop()

# Avertir si les prévisions s'arrêtent avant la date choisie
derniere_prevision = forecast_horizon["ds"].max()
if derniere_prevision < date_fin_ts:
    st.warning(
        f"⚠️ Les prévisions disponibles s'arrêtent au **{derniere_prevision.strftime('%d/%m/%Y')}**, "
        f"avant votre date cible ({date_fin.strftime('%d/%m/%Y')}. "
        "Relancez vos modèles avec un horizon plus long pour couvrir toute la période."
    )
    horizon_days = (derniere_prevision - today_date).days

# =============================================================
# DEMANDE PRÉVUE
# =============================================================
yhat_vals    = forecast_horizon["yhat"].values
demand_total = int(round(float(np.sum(yhat_vals))))
demand_moy_j = round(float(np.mean(yhat_vals)), 1)

# =============================================================
# SIMULATION UNIQUE — partagée entre KPIs, graphique et tableau
# =============================================================
def simulate_stock(stock_depart, forecast_df, reorder_point, stock_optimal, lead_time):
    """
    Simule l'évolution du stock jour par jour avec déclenchement automatique des commandes.
    Retourne (stock_sim, cmd_sim, stock_fin_reel)
    """
    n            = len(forecast_df)
    stock_c      = float(stock_depart)
    livraisons   = {}
    stock_sim    = []
    cmd_sim      = []
    cmd_en_cours = False

    for i, (_, row) in enumerate(forecast_df.iterrows()):
        if i in livraisons:
            stock_c += livraisons[i]
            cmd_en_cours = False

        stock_c -= float(row["yhat"])

        cmd = 0
        if stock_c < reorder_point and not cmd_en_cours:
            cmd = max(0, stock_optimal - stock_c)
            jour_livraison = i + lead_time
            if jour_livraison < n:
                livraisons[jour_livraison] = livraisons.get(jour_livraison, 0) + cmd
            cmd_en_cours = True

        stock_sim.append(max(0, round(stock_c)))
        cmd_sim.append(round(cmd))

    stock_fin_reel = stock_sim[-1] if stock_sim else 0
    return stock_sim, cmd_sim, stock_fin_reel


stock_sim, cmd_sim, stock_fin_reel = simulate_stock(
    stock_actuel, forecast_horizon, reorder_point, stock_optimal, lead_time
)

forecast_horizon = forecast_horizon.copy()
forecast_horizon["stock_simule"] = stock_sim
forecast_horizon["commande"]     = cmd_sim
forecast_horizon["date_label"]   = forecast_horizon["ds"].dt.strftime("%d/%m")

# =============================================================
# DIAGNOSTIC — basé sur stock_fin_reel (avec réappros)
# =============================================================
if stock_fin_reel < 0:
    risque     = "🔴 Rupture de stock prévue"
    risk_color = "#e74c3c"; risk_bg = "rgba(231,76,60,0.1)"
elif stock_fin_reel < safety_stock:
    risque     = "🟠 Risque de rupture"
    risk_color = "#e67e22"; risk_bg = "rgba(230,126,34,0.1)"
elif stock_fin_reel <= reorder_point:
    risque     = "🟡 Proche du ROP — Commander bientôt"
    risk_color = "#f39c12"; risk_bg = "rgba(243,156,18,0.1)"
elif stock_fin_reel <= stock_optimal:
    risque     = "🟢 Niveau optimal"
    risk_color = "#27ae60"; risk_bg = "rgba(39,174,96,0.1)"
elif stock_fin_reel <= stock_max:
    risque     = "🟡 Surstock léger"
    risk_color = "#f39c12"; risk_bg = "rgba(243,156,18,0.1)"
else:
    risque     = "🟡 Surstock important"
    risk_color = "#f39c12"; risk_bg = "rgba(243,156,18,0.1)"

qte_cmd  = max(0, stock_optimal - stock_fin_reel) if stock_fin_reel < reorder_point else 0
excedent = max(0, stock_fin_reel - stock_max)

if qte_cmd > 0:    action = f"→ Commander <b>{qte_cmd:,}</b> unités minimum"
elif excedent > 0: action = f"→ Excédent de <b>{excedent:,}</b> unités à réduire"
else:              action = "→ Aucune action requise ✓"

# =============================================================
# KPIs
# =============================================================
st.markdown("---")
st.markdown(
    f"### 📊 Analyse stock — Produit **{product}** / Dépôt **{depot_sel}** · "
    f"Modèle : {MODEL_ICONS.get(best_model_name,'')} **{best_model_name.upper()}** · "
    f"Horizon : jusqu'au **{date_fin.strftime('%d/%m/%Y')}** ({horizon_days}j)"
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
metrics = [
    (k1, "Stock actuel",               f"{stock_actuel:,}",
         "#4a9eff",                                               "unités disponibles"),
    (k2, f"Demande ({horizon_days}j)",  f"{demand_total:,}",
         MODEL_COLORS.get(best_model_name, '#888'),               f"Σ prévision {best_model_name.upper()}"),
    (k3, "Stock fin simulé",           f"{stock_fin_reel:,}",
         "#4a9eff" if stock_fin_reel >= 0 else "#e74c3c",         f"au {date_fin.strftime('%d/%m/%Y')} (avec réappro)"),
    (k4, "Sécurité (Q1)",              f"{safety_stock:,}",
         "#e74c3c",                                               f"Q1×{lead_time}j"),
    (k5, "ROP (Q3)",                   f"{reorder_point:,}",
         "#f39c12",                                               f"Q3×{lead_time}j"),
    (k6, "Optimal (ROP+Q2)",           f"{stock_optimal:,}",
         "#27ae60",                                               f"ROP+Q2×{lead_time}j"),
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
    Stock simulé au {date_fin.strftime('%d/%m/%Y')} : <b>{stock_fin_reel:,}</b> unités &nbsp;|&nbsp;
    Moy/jour : <b>{demand_moy_j}</b> &nbsp;|&nbsp; {action}
</div>
""", unsafe_allow_html=True)

# =============================================================
# TABS
# =============================================================
tab1, tab2 = st.tabs(["📈 Évolution stock simulée", "📋 Tableau journalier"])

with tab1:
    st.subheader(
        f"Simulation du stock du {today_date.strftime('%d/%m/%Y')} "
        f"au {date_fin.strftime('%d/%m/%Y')} — {best_model_name.upper()} "
        f"({horizon_days} jours)"
    )

    # Pour les grands horizons, on agrège par semaine pour la lisibilité
    if horizon_days > 60:
        fh_plot = forecast_horizon.copy()
        fh_plot["semaine"] = fh_plot["ds"].dt.to_period("W").apply(lambda r: r.start_time)
        fh_agg = fh_plot.groupby("semaine").agg(
            yhat=("yhat", "sum"),
            stock_simule=("stock_simule", "last"),
            commande=("commande", "sum"),
        ).reset_index()
        fh_agg["date_label"] = fh_agg["semaine"].dt.strftime("S%U\n%d/%m")
        x_axis   = fh_agg["date_label"]
        y_demand = fh_agg["yhat"].round(1)
        y_stock  = fh_agg["stock_simule"]
        y_cmd    = fh_agg["commande"]
        note_agg = "📌 Données agrégées par semaine pour lisibilité (horizon > 60 jours)"
    else:
        x_axis   = forecast_horizon["date_label"]
        y_demand = forecast_horizon["yhat"].round(1)
        y_stock  = forecast_horizon["stock_simule"]
        y_cmd    = forecast_horizon["commande"]
        note_agg = None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Demande prévue ({best_model_name.upper()})",
        x=x_axis, y=y_demand,
        marker_color=MODEL_COLORS.get(best_model_name, "#888"),
        opacity=0.85, offsetgroup=1, width=0.35
    ))
    fig.add_trace(go.Bar(
        name="Stock simulé",
        x=x_axis, y=y_stock,
        marker_color="#a78bfa", opacity=0.85, offsetgroup=2, width=0.35
    ))

    # Triangles commandes
    mask_cmd = y_cmd > 0
    if mask_cmd.any():
        fig.add_trace(go.Scatter(
            name="Commande passée",
            x=x_axis[mask_cmd], y=y_stock[mask_cmd],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#00d4ff"),
        ))

    for y_val, hname, hcolor in [
        (safety_stock,  f"Sécurité={safety_stock}",   "#e74c3c"),
        (reorder_point, f"ROP={reorder_point}",        "#f59e0b"),
        (stock_optimal, f"Optimal={stock_optimal}",    "#27ae60"),
        (stock_max,     f"Max={stock_max}",            "#9b59b6"),
    ]:
        fig.add_hline(
            y=y_val, line_dash="dot", line_color=hcolor, line_width=2,
            annotation_text=hname, annotation_position="top left",
            annotation_font_color=hcolor, annotation_font_size=11
        )

    fig.update_layout(
        barmode="group", template="plotly_dark", height=500,
        xaxis_title="Période", yaxis_title="Quantité (unités)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        bargap=0.2, bargroupgap=0.05,
        xaxis=dict(tickangle=-45, nticks=min(len(x_axis), 40))
    )
    st.plotly_chart(fig, use_container_width=True)

    if note_agg:
        st.info(note_agg)
    st.info(
        f"🔺 Triangle bleu = commande passée · Livraison reçue après {lead_time} jours · "
        f"Quantité commandée = stock optimal ({stock_optimal}) - stock courant"
    )

with tab2:
    st.subheader(
        f"Tableau journalier — Stock simulé & Alertes "
        f"({today_date.strftime('%d/%m/%Y')} → {date_fin.strftime('%d/%m/%Y')})"
    )

    rows        = []
    stock_c2    = float(stock_actuel)
    livraisons2 = {}
    cmd_en2     = False

    for i, (_, row) in enumerate(forecast_horizon.iterrows()):
        livraison_recue = livraisons2.get(i, 0)
        if livraison_recue > 0:
            stock_c2 += livraison_recue
            cmd_en2 = False

        stock_av  = round(stock_c2, 1)
        demande_j = round(float(row["yhat"]), 1)
        stock_c2 -= demande_j

        cmd = 0
        if stock_c2 < reorder_point and not cmd_en2:
            cmd = max(0, stock_optimal - stock_c2)
            jour_livraison = i + lead_time
            if jour_livraison < len(forecast_horizon):
                livraisons2[jour_livraison] = livraisons2.get(jour_livraison, 0) + cmd
            cmd_en2 = True

        stock_ap = max(0, round(stock_c2, 1))

        if stock_ap <= 0:               etat = "🔴 Rupture"
        elif stock_ap < safety_stock:   etat = "🟠 Risque rupture"
        elif stock_ap <= reorder_point: etat = "🟡 Proche ROP"
        elif stock_ap <= stock_optimal: etat = "🟢 Optimal"
        elif stock_ap <= stock_max:     etat = "🟡 Surstock léger"
        else:                           etat = "🟡 Surstock"

        rows.append({
            "Date":             row["ds"].strftime("%d/%m/%Y"),
            "Demande prévue":   demande_j,
            "Livraison reçue":  livraison_recue if livraison_recue > 0 else "—",
            "Stock début jour": stock_av,
            "Stock fin jour":   stock_ap,
            "Commande passée":  f"{round(cmd)} (J+{lead_time})" if cmd > 0 else "—",
            "État":             etat,
        })

    df_table = pd.DataFrame(rows)

    def color_etat(val):
        if "Rupture"    in str(val): return "background-color:rgba(231,76,60,0.2);color:#e74c3c;font-weight:600"
        elif "Risque"   in str(val): return "background-color:rgba(230,126,34,0.15);color:#e67e22;font-weight:600"
        elif "Proche"   in str(val): return "background-color:rgba(243,156,18,0.15);color:#f39c12;font-weight:600"
        elif "Optimal"  in str(val): return "background-color:rgba(39,174,96,0.15);color:#27ae60;font-weight:600"
        elif "Surstock" in str(val): return "background-color:rgba(243,156,18,0.15);color:#f39c12;font-weight:600"
        return ""

    st.dataframe(
        df_table.style.applymap(color_etat, subset=["État"]),
        use_container_width=True, height=460
    )
    csv = df_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter CSV", data=csv,
        file_name=f"stock_journalier_{product}_{depot_id}_{today_date.strftime('%d%m%Y')}_au_{date_fin.strftime('%d%m%Y')}.csv",
        mime="text/csv"
    )

# =============================================================
# 🤖 ANALYSE IA
# =============================================================
from analyse.generateur import generer_analyse_stock
 
st.divider()
col_btn = st.columns([2, 2, 2])
with col_btn[1]:
    btn = st.button("🤖 Analyse Intelligente", use_container_width=True, type="primary")
 
if btn:
    filtres = {
        "Produit"          : str(product),
        "Dépôt"            : str(depot_sel),
        "Modèle utilisé"   : best_model_name.upper(),
        "Horizon analyse"  : f"{horizon_days} jours (jusqu'au {date_fin.strftime('%d/%m/%Y')})",
        "Délai réappro"    : f"{lead_time} jours",
        "Source stock"     : (
            "Estimé automatiquement (fallback sur dernières données)"
            if fallback_utilise and stock_actuel == stock_estime
            else "Estimé automatiquement"
            if stock_actuel == stock_estime
            else "Corrigé manuellement"
        ),
    }
    metriques = {
        # ── KPIs ──────────────────────────────────────────────
        "Stock estimé automatiquement"  : f"{stock_estime:,} unités",
        "Stock actuel utilisé"          : f"{stock_actuel:,} unités",
        "Demande prévue totale"         : f"{demand_total:,} unités sur {horizon_days} jours",
        "Moyenne journalière"           : f"{demand_moy_j} unités/jour",
        "Stock fin horizon (simulé)"    : f"{stock_fin_reel:,} unités au {date_fin.strftime('%d/%m/%Y')} (après réappros automatiques)",
 
        # ── Seuils ────────────────────────────────────────────
        "Seuil de sécurité (SS = Q1 × délai)"        : f"{safety_stock:,} unités — en dessous = risque rupture",
        "Point de réappro (ROP = Q3 × délai)"         : f"{reorder_point:,} unités — déclenche une commande automatique",
        "Stock optimal (ROP + Q2 × délai)"            : f"{stock_optimal:,} unités — cible lors d'une commande",
        "Stock maximum (ROP + (Q3+1.5×IQR) × délai)" : f"{stock_max:,} unités — au-delà = surstock",
 
        # ── Diagnostic ────────────────────────────────────────
        "Diagnostic"                    : risque,
        "Action recommandée"            : f"Commander {qte_cmd:,} unités" if qte_cmd > 0 else "Aucune commande requise",
        "Excédent"                      : f"{excedent:,} unités à réduire" if excedent > 0 else "Aucun excédent",
 
        # ── Simulation ────────────────────────────────────────
        "Nombre de commandes simulées"  : f"{sum(1 for c in cmd_sim if c > 0)} commande(s) automatique(s) déclenchée(s) sur l'horizon",
        "Total commandé (simulé)"       : f"{sum(c for c in cmd_sim if c > 0):,} unités au total sur l'horizon",
        "Stock actuel vs ROP"           : (
            f"{stock_actuel:,} > {reorder_point:,} → au-dessus du ROP, pas de commande immédiate nécessaire"
            if stock_actuel > reorder_point
            else f"{stock_actuel:,} ≤ {reorder_point:,} → en dessous du ROP, commande urgente recommandée"
        ),
        "Stock actuel vs Seuil sécurité": (
            f"{stock_actuel:,} > {safety_stock:,} → au-dessus du seuil de sécurité"
            if stock_actuel > safety_stock
            else f"{stock_actuel:,} ≤ {safety_stock:,} → EN DESSOUS du seuil de sécurité, risque élevé"
        ),
    }
 
    with st.spinner("🧠 Analyse en cours..."):
        analyse = generer_analyse_stock(filtres, metriques)
    with st.container(border=True):
        st.markdown(analyse)
    st.download_button(
        label="⬇️ Télécharger l'analyse",
        data=analyse,
        file_name=f"analyse_stock_{product}_{depot_sel}_{date_fin.strftime('%d%m%Y')}.txt",
        mime="text/plain"
    )
 