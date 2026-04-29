import streamlit as st
st.set_page_config(page_title="Stock Management", page_icon="📦", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, glob
from utils import get_connection

st.title("📦 Stock Management")
st.caption("Analyse combinée : Historique des ventes + Meilleur modèle de prévision par produit + Quartiles")

FORECASTS_DIR = "forecasts"

# =============================================================
# CONVENTION DES NOMS DE FICHIERS PAR MODÈLE
# ARIMA   → forecast_arima_{ref}_{depot}.csv
# XGBoost → forecast_xgboost_{ref}_{depot}.csv
# LSTM    → forecast_{ref}_{depot}.csv
# =============================================================
MODEL_REGISTRY = {
    "ARIMA":   {"prefix": "forecast_arima_",   "color": "#6c63ff", "icon": "🟢"},
    "XGBoost": {"prefix": "forecast_xgboost_", "color": "#f59e0b", "icon": "📊"},
    "LSTM":    {"prefix": "forecast_",          "color": "#10b981", "icon": "🔴"},
}

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.header("⚙️ Paramètres")
    lead_time    = st.number_input("Délai réappro (jours)", min_value=1, max_value=60, value=7)
    horizon_days = st.selectbox("Horizon analyse (jours)", [7, 14, 30, 60, 90], index=2)
    st.markdown("---")
    st.markdown(f"""
**Formules seuils**

| Niveau | Calcul |
|---|---|
| 🔴 Sécurité | Q1 hist × {lead_time}j |
| 🟢 Optimal  | Q2 hist × {lead_time}j |
| 🟠 ROP      | Q3 hist × {lead_time}j |
| 🟡 Max      | (Q3+1.5×IQR) × {lead_time}j |
| 📦 Besoin   | Σ yhat meilleur modèle {horizon_days}j |

*Seuils = quartiles de l'historique réel*
*Besoin = prévision du meilleur modèle (R² max)*
    """)
    st.markdown("---")
    st.markdown("**🔄 Générer des prévisions**")
    if st.button("ARIMA",   use_container_width=True): st.switch_page("pages/1_ARIMA.py")
    if st.button("XGBoost", use_container_width=True): st.switch_page("pages/2_XGBoost.py")
    if st.button("LSTM",    use_container_width=True): st.switch_page("pages/3_LSTM.py")

# =============================================================
# VÉRIFICATION FORECASTS DIR
# =============================================================
if not os.path.exists(FORECASTS_DIR):
    st.warning("⚠️ Aucune prévision disponible. Lance d'abord un modèle de prévision.")
    st.stop()

# =============================================================
# DÉCOUVERTE DE TOUS LES FICHIERS DISPONIBLES
# Stratégie : on liste les combos (ref, depot) qui ont au moins
# un fichier de prévision, tous modèles confondus.
# On privilégie LSTM pour la liste des combos car c'est le naming
# par défaut ; on complète avec ARIMA et XGBoost.
# =============================================================
def discover_combos():
    """
    Retourne un dict  {(ref, depot): {model_name: filepath}}
    pour tous les fichiers de prévision trouvés.
    """
    combos = {}
    for model_name, cfg in MODEL_REGISTRY.items():
        prefix = cfg["prefix"]
        # Pour LSTM le prefix est "forecast_" — on doit exclure arima/xgboost
        if model_name == "LSTM":
            pattern = os.path.join(FORECASTS_DIR, "forecast_*.csv")
            files   = [f for f in glob.glob(pattern)
                       if "arima" not in os.path.basename(f)
                       and "xgboost" not in os.path.basename(f)]
        else:
            pattern = os.path.join(FORECASTS_DIR, f"{prefix}*.csv")
            files   = glob.glob(pattern)

        for f in files:
            base  = os.path.basename(f).replace(prefix, "").replace(".csv", "")
            parts = base.rsplit("_", 1)
            if len(parts) == 2:
                key = (parts[0], parts[1])   # (ref, depot)
                combos.setdefault(key, {})
                combos[key][model_name] = f

    return combos

combos = discover_combos()

if not combos:
    st.warning("⚠️ Aucune prévision trouvée. Lance d'abord ARIMA, XGBoost ou LSTM.")
    st.stop()

# =============================================================
# SÉLECTION PRODUIT / DÉPÔT
# =============================================================
st.markdown("### 🎯 Sélection Produit / Dépôt")
combo_labels = [f"{ref}_{depot}" for ref, depot in sorted(combos.keys())]
selected_label = st.selectbox("Produit / Dépôt", combo_labels)

ref_sel, depot_sel_str = selected_label.rsplit("_", 1)
sel_key    = (ref_sel, depot_sel_str)
sel_models = combos[sel_key]   # {model_name: filepath}

# =============================================================
# CHARGEMENT ET COMPARAISON DES MÉTRIQUES
# Chaque CSV peut contenir des colonnes "r2" et "mape"
# (sauvegardées par les pages modèle). On lit le meilleur.
# =============================================================
@st.cache_data(ttl=60)
def load_forecast(path):
    df = pd.read_csv(path, parse_dates=["ds"])
    df["yhat"] = df["yhat"].clip(lower=0)
    return df

def get_model_r2(df):
    """Lit le R² depuis la colonne 'r2' du CSV si elle existe."""
    if "r2" in df.columns:
        val = df["r2"].dropna()
        if len(val) > 0:
            return float(val.iloc[0])
    return float("-inf")   # pas de métrique → classé dernier

def get_model_mape(df):
    if "mape" in df.columns:
        val = df["mape"].dropna()
        if len(val) > 0:
            return float(val.iloc[0])
    return float("nan")

# ── Chargement de tous les modèles disponibles pour ce combo ──
loaded_models = {}
for model_name, filepath in sel_models.items():
    df_fc = load_forecast(filepath)
    if not df_fc.empty:
        loaded_models[model_name] = {
            "df":    df_fc,
            "r2":    get_model_r2(df_fc),
            "mape":  get_model_mape(df_fc),
            "color": MODEL_REGISTRY[model_name]["color"],
            "icon":  MODEL_REGISTRY[model_name]["icon"],
        }

if not loaded_models:
    st.error("❌ Aucune prévision valide trouvée pour ce produit/dépôt.")
    st.stop()

# ── Sélection automatique du meilleur modèle (R² le plus élevé) ──
best_model_name = max(loaded_models, key=lambda k: loaded_models[k]["r2"])
best            = loaded_models[best_model_name]

# =============================================================
# BANDEAU : modèle sélectionné + comparaison disponibles
# =============================================================
qual_label = lambda r2, mape: (
    "🟢 Excellent" if (not np.isnan(r2) and r2 >= 0.85 and not np.isnan(mape) and mape <= 10)
    else "🟡 Bon"  if (not np.isnan(r2) and r2 >= 0.70)
    else "🟠 Moyen" if (not np.isnan(r2) and r2 >= 0.50)
    else "🔴 Faible" if not np.isnan(r2)
    else "⚪ Inconnu"
)

r2_disp   = f"{best['r2']:.4f}"   if not np.isnan(best['r2'])   and best['r2'] != float('-inf') else "N/A"
mape_disp = f"{best['mape']:.2f}%" if not np.isnan(best['mape']) else "N/A"

st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(167,139,250,0.08));
            border:1px solid rgba(108,99,255,0.3);border-radius:14px;
            padding:16px 20px;margin-bottom:16px">
    <span style="font-size:18px;font-weight:800;color:#6c63ff">
        🏆 Meilleur modèle pour ce produit : {best['icon']} {best_model_name}
    </span><br>
    <span style="font-size:13px;color:#6b7280">
        R² = <b style="color:#1a1a2e">{r2_disp}</b> &nbsp;|&nbsp;
        MAPE = <b style="color:#1a1a2e">{mape_disp}</b> &nbsp;|&nbsp;
        {qual_label(best['r2'], best['mape'])}
    </span>
</div>
""", unsafe_allow_html=True)

# Affichage compact des autres modèles disponibles
if len(loaded_models) > 1:
    cols_m = st.columns(len(loaded_models))
    for i, (mname, mdata) in enumerate(loaded_models.items()):
        is_best = mname == best_model_name
        r2_s  = f"{mdata['r2']:.4f}"    if mdata['r2'] != float('-inf') and not np.isnan(mdata['r2']) else "N/A"
        mape_s= f"{mdata['mape']:.2f}%" if not np.isnan(mdata['mape']) else "N/A"
        border = f"border:2px solid {mdata['color']}" if is_best else "border:1px solid #e5e7eb"
        badge  = ' <span style="background:#6c63ff;color:white;font-size:10px;padding:2px 7px;border-radius:10px;margin-left:6px">✓ Utilisé</span>' if is_best else ""
        with cols_m[i]:
            st.markdown(f"""
            <div style="background:#fff;border-radius:10px;{border};padding:12px 14px;text-align:center">
                <div style="font-size:14px;font-weight:700;color:{mdata['color']}">{mdata['icon']} {mname}{badge}</div>
                <div style="font-size:12px;color:#6b7280;margin-top:4px">R² {r2_s} | MAPE {mape_s}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
else:
    missing_models = [m for m in MODEL_REGISTRY if m not in loaded_models]
    if missing_models:
        st.info(f"ℹ️ Modèles non encore générés pour ce produit : **{', '.join(missing_models)}**. "
                f"Lance-les depuis la sidebar pour une comparaison complète.")

# ── Prévision retenue = meilleur modèle ──
forecast = best["df"].sort_values("ds").head(horizon_days).copy()

# =============================================================
# CHARGEMENT HISTORIQUE VENTES (filtré produit + dépôt)
# =============================================================
@st.cache_data(ttl=300)
def load_history(ref_product, depot_id):
    try:
        conn = get_connection()
        conditions, params = [], {}

        if ref_product != "all":
            conditions.append("ref_product = %(ref)s")
            params["ref"] = int(ref_product)
        if depot_id != "all":
            conditions.append("depot_id = %(depot)s")
            params["depot"] = int(depot_id)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        df = pd.read_sql(f"""
            SELECT DATE(date_time) AS date, SUM(quantity) AS quantity
            FROM sales
            {where}
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
    history = load_history(ref_sel, depot_sel_str)

if history.empty:
    st.error("❌ Aucun historique de ventes trouvé pour ce produit/dépôt.")
    st.stop()

st.caption(f"""
📅 Historique : **{history['date'].min().date()}** → **{history['date'].max().date()}**
({len(history)} jours) &nbsp;|&nbsp;
🔮 Prévision **{best_model_name}** : **{forecast['ds'].min().date()}** → **{forecast['ds'].max().date()}**
({len(forecast)} jours) — *horizon : {horizon_days}j*
""")

# =============================================================
# CALCUL QUARTILES SUR HISTORIQUE RÉEL
# =============================================================
hist_qty = history["quantity"].values
q1_h     = np.quantile(hist_qty, 0.25)
q2_h     = np.quantile(hist_qty, 0.50)
q3_h     = np.quantile(hist_qty, 0.75)
iqr_h    = q3_h - q1_h

safety_stock  = int(round(q1_h * lead_time))
stock_optimal = int(round(q2_h * lead_time))
reorder_point = int(round(q3_h * lead_time))
stock_max     = int(round((q3_h + 1.5 * iqr_h) * lead_time))

# Besoin prévu depuis le meilleur modèle
yhat         = forecast["yhat"].values
demand       = int(round(float(np.sum(yhat))))
demand_moy_j = round(float(np.mean(yhat)), 1)

# =============================================================
# DIAGNOSTIC
# =============================================================
if demand < safety_stock:
    risque = "🔴 Rupture imminente";  risk_color = "#e74c3c"; risk_bg = "rgba(231,76,60,0.1)"
elif demand < stock_optimal:
    risque = "🟠 Risque de rupture";  risk_color = "#e67e22"; risk_bg = "rgba(230,126,34,0.1)"
elif demand <= reorder_point:
    risque = "🟢 Niveau optimal";     risk_color = "#27ae60"; risk_bg = "rgba(39,174,96,0.1)"
elif demand <= stock_max:
    risque = "🟡 Surstock léger";     risk_color = "#f39c12"; risk_bg = "rgba(243,156,18,0.1)"
else:
    risque = "🟡 Surstock important"; risk_color = "#f39c12"; risk_bg = "rgba(243,156,18,0.1)"

qte_cmd  = max(0, reorder_point - demand)
excedent = max(0, demand - stock_max)

# =============================================================
# KPIs
# =============================================================
st.markdown("---")
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, title, value, color, subtitle=""):
    with col:
        st.markdown(f"""
        <div style="background:#0f1117;border:1px solid #2a2d3a;border-radius:10px;
                    padding:16px;text-align:center;">
            <div style="font-size:11px;color:#6b7280;text-transform:uppercase;
                        letter-spacing:.1em;margin-bottom:6px">{title}</div>
            <div style="font-size:24px;font-weight:600;color:{color}">{value}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px">{subtitle}</div>
        </div>""", unsafe_allow_html=True)

kpi(k1, f"Besoin prévu ({horizon_days}j)", f"{demand:,}",        "#4a9eff",  f"Σ yhat {best_model_name}")
kpi(k2, "Sécurité (Q1)",                  f"{safety_stock:,}",  "#e74c3c",  f"Q1×{lead_time}j")
kpi(k3, "Optimal (Q2)",                   f"{stock_optimal:,}", "#27ae60",  f"Q2×{lead_time}j")
kpi(k4, "ROP (Q3)",                       f"{reorder_point:,}", "#f39c12",  f"Q3×{lead_time}j")
kpi(k5, "Max toléré",                     f"{stock_max:,}",     "#9b59b6",  f"(Q3+IQR)×{lead_time}j")

st.markdown("<br>", unsafe_allow_html=True)

if qte_cmd > 0:    action = f"→ Commander <b>{qte_cmd}</b> unités minimum"
elif excedent > 0: action = f"→ Excédent de <b>{excedent}</b> unités à réduire"
else:               action = "→ Aucune action requise ✓"

st.markdown(f"""
<div style="background:{risk_bg};border-left:4px solid {risk_color};
            padding:14px 18px;border-radius:0 8px 8px 0;margin-bottom:20px">
    <span style="color:{risk_color};font-size:17px;font-weight:600">{risque}</span><br>
    Besoin prévu <b>{best_model_name}</b> ({horizon_days}j) : <b>{demand}</b> unités &nbsp;|&nbsp;
    Demande moy/jour : <b>{demand_moy_j}</b> unités &nbsp;|&nbsp; {action}
</div>
""", unsafe_allow_html=True)

# =============================================================
# ONGLETS
# =============================================================
tab1, tab2 = st.tabs([
    "📊 Graphique comparatif",
    "📋 Tableau journalier"
])

# ── TAB 1 ────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Demande prévue ({best_model_name}) vs Seuils de stock (historique)")

    fc_d = forecast[["ds", "yhat"]].copy()
    fc_d.columns = ["date", "demande_prevue"]
    fc_d["demande_prevue"] = fc_d["demande_prevue"].round(0)

    # Simulation stock jour par jour
    stock_courant = stock_optimal
    stock_simule  = []
    commandes_sim = []
    for _, row in fc_d.iterrows():
        stock_courant -= row["demande_prevue"]
        cmd = 0
        if stock_courant < reorder_point:
            cmd            = reorder_point
            stock_courant += cmd
        stock_simule.append(max(0, round(stock_courant)))
        commandes_sim.append(cmd)

    fc_d["stock_simule"] = stock_simule
    fc_d["commande"]     = commandes_sim
    fc_d["date_label"]   = fc_d["date"].dt.strftime("%a %d/%m")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"📦 Demande prévue ({best_model_name})",
        x=fc_d["date_label"], y=fc_d["demande_prevue"],
        marker_color=best["color"], marker_line_width=0,
        offsetgroup=1, width=0.35
    ))
    fig.add_trace(go.Bar(
        name="🏭 Stock simulé",
        x=fc_d["date_label"], y=fc_d["stock_simule"],
        marker_color="#a78bfa", marker_line_width=0,
        offsetgroup=2, width=0.35
    ))

    for y_val, hline_name, color in [
        (safety_stock,  f"🔴 Sécurité = {safety_stock}",  "#e74c3c"),
        (stock_optimal, f"🟢 Optimal = {stock_optimal}",   "#27ae60"),
        (reorder_point, f"🟠 ROP = {reorder_point}",       "#f39c12"),
        (stock_max,     f"🟡 Max = {stock_max}",           "#9b59b6"),
    ]:
        fig.add_hline(
            y=y_val, line_dash="dot", line_color=color, line_width=2,
            annotation_text=hline_name, annotation_position="top left",
            annotation_font_color=color, annotation_font_size=11
        )

    fig.update_layout(
        barmode="group", template="plotly_dark", height=520,
        xaxis_title="Jour", yaxis_title="Quantité (unités)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)"),
        bargap=0.2, bargroupgap=0.05,
        xaxis=dict(tickangle=-45, nticks=min(len(fc_d), 30))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
    **Lecture :**
    Barres colorées = demande prévue par jour (**{best_model_name}**, meilleur modèle) sur **{horizon_days} jours** |
    🟣 Barres violettes = stock simulé après consommation + réappro auto |
    Lignes pointillées = seuils calculés depuis les **quartiles de l'historique réel**
    """)

# ── TAB 2 ────────────────────────────────────────────────────
with tab2:
    st.subheader("Tableau journalier — Demande prévue, Stock prévu, Résultat")
    st.caption(f"Seuils depuis l'historique réel | Prévisions : **{best_model_name}** (R² = {r2_disp}) | Horizon : {horizon_days}j")

    fc_tbl = forecast[["ds", "yhat"]].copy()
    fc_tbl.columns = ["date", "demande_prevue"]
    fc_tbl["demande_prevue"] = fc_tbl["demande_prevue"].round(0).astype(int)

    rows          = []
    stock_courant = stock_optimal

    for _, row in fc_tbl.iterrows():
        demande_jour  = int(row["demande_prevue"])
        stock_avant   = int(round(stock_courant))
        stock_courant -= demande_jour
        commande = 0
        if stock_courant < reorder_point:
            commande       = int(reorder_point)
            stock_courant += commande
        stock_apres = max(0, int(round(stock_courant)))

        if stock_apres < safety_stock:
            resultat = "🔴 Rupture imminente"
        elif stock_apres < stock_optimal:
            resultat = "🟠 Risque de rupture"
        elif stock_apres <= reorder_point:
            resultat = "🟢 Niveau optimal"
        elif stock_apres <= stock_max:
            resultat = "🟡 Surstock léger"
        else:
            resultat = "🟡 Surstock important"

        rows.append({
            "Date":             row["date"].strftime("%d/%m/%Y"),
            "Demande prévue":   demande_jour,
            "Stock début jour": stock_avant,
            "Stock prévu":      stock_apres,
            "Commande générée": commande if commande > 0 else "—",
            "Résultat":         resultat,
        })

    df_table = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1: date_debut = st.date_input("Du", value=fc_tbl["date"].min().date())
    with c2: date_fin   = st.date_input("Au", value=fc_tbl["date"].max().date())

    mask = (
        pd.to_datetime(df_table["Date"], format="%d/%m/%Y").dt.date >= date_debut
    ) & (
        pd.to_datetime(df_table["Date"], format="%d/%m/%Y").dt.date <= date_fin
    )
    df_filtered = df_table[mask]

    def color_result(val):
        if "Rupture imminente" in str(val): return "background-color:rgba(231,76,60,0.2);color:#e74c3c;font-weight:600"
        elif "Risque"          in str(val): return "background-color:rgba(230,126,34,0.15);color:#e67e22;font-weight:600"
        elif "optimal"         in str(val): return "background-color:rgba(39,174,96,0.15);color:#27ae60;font-weight:600"
        elif "Surstock"        in str(val): return "background-color:rgba(243,156,18,0.15);color:#f39c12;font-weight:600"
        return ""

    st.dataframe(
        df_filtered.style.applymap(color_result, subset=["Résultat"]),
        use_container_width=True, height=460
    )

    st.markdown(f"""
    **Colonnes :**
    - **Demande prévue** = quantité prévue par **{best_model_name}** pour le jour
    - **Stock début jour** = stock disponible avant consommation
    - **Stock prévu** = stock restant après consommation de la demande
    - **Commande générée** = quantité à commander si stock < ROP
    - **Résultat** = diagnostic automatique coloré
    """)

    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter CSV", csv,
                           f"stock_journalier_{selected_label}_{horizon_days}j.csv", "text/csv")
    with c_exp2:
        st.markdown(f"""
        <div style="background:#0f1117;border:1px solid #2a2d3a;
                    border-radius:8px;padding:12px;font-size:13px">
        Modèle utilisé : <b style="color:{best['color']}">{best_model_name}</b>
        (R² = {r2_disp} | MAPE = {mape_disp})<br>
        Stock initial : <b>{stock_optimal}</b> unités (= Q2 historique × {lead_time}j)<br>
        Réappro auto quand stock &lt; ROP (<b>{reorder_point}</b>)<br>
        Horizon analysé : <b>{horizon_days}</b> jours
        </div>""", unsafe_allow_html=True)