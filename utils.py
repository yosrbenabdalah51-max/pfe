import mysql.connector
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =========================
# Connexion BD
# =========================
def get_connection():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    return conn

def get_data(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data

# =========================
# Sidebar — Sélecteur produit
# =========================
@st.cache_data(ttl=600)
def load_product_ranking():
    """
    ✅ OPTIMISÉ — agrégation faite côté SQL, pas en Python.
    Charge seulement les totaux par produit, pas toutes les lignes.
    """
    conn = get_connection()
    df = pd.read_sql("""
        SELECT
            ref_product,
            SUM(quantity * price) AS total
        FROM sales
        GROUP BY ref_product
        ORDER BY total DESC
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=600)
def load_depots_for_product(ref_product):
    """
    ✅ NOUVEAU — charge les dépôts disponibles pour un produit donné.
    Requête légère, résultat mis en cache.
    """
    conn = get_connection()
    if ref_product is None:
        # Tous les dépôts
        df = pd.read_sql("""
            SELECT DISTINCT d.depot_id, d.name AS depot_name, c.name AS zone
            FROM depot d
            JOIN country c ON d.country_id = c.id
            ORDER BY d.name
        """, conn)
    else:
        df = pd.read_sql("""
            SELECT DISTINCT d.depot_id, d.name AS depot_name, c.name AS zone
            FROM sales s
            JOIN depot   d ON s.depot_id   = d.depot_id
            JOIN country c ON d.country_id = c.id
            WHERE s.ref_product = %(ref)s
            ORDER BY d.name
        """, conn, params={"ref": int(ref_product)})
    conn.close()
    return df

def sidebar_product_selector():
    """
    Affiche dans le sidebar un selectbox de produits triés par ventes.
    Retourne le produit sélectionné (int) ou None si 'Tous'.
    """
    product_rank = load_product_ranking()

    labels = ["🌐 Tous les produits"] + [
        f"{row['ref_product']}  (💰 {int(row['total'])})"
        for _, row in product_rank.iterrows()
    ]

    st.sidebar.header("🎯 Produit")
    selected_label = st.sidebar.selectbox("Choisir un produit", labels)

    if selected_label == "🌐 Tous les produits":
        product = None
    else:
        product = int(product_rank[
            product_rank.apply(
                lambda x: f"{x['ref_product']}  (💰 {int(x['total'])})" == selected_label,
                axis=1
            )
        ]["ref_product"].values[0])

    st.session_state["product"] = product
    return product

def sidebar_depot_selector(ref_product):
    """
    ✅ NOUVEAU — sélecteur dépôt dans la sidebar, séparé du sélecteur produit.
    À appeler après sidebar_product_selector() dans la page LSTM.
    """
    df_depots = load_depots_for_product(ref_product)

    if df_depots.empty:
        st.sidebar.warning("Aucun dépôt trouvé.")
        st.session_state["depot_id"] = "all"
        return None, "Tous", "—"

    depot_options = df_depots["depot_name"].tolist()
    depot_sel     = st.sidebar.selectbox("🏭 Dépôt", depot_options)
    depot_row     = df_depots[df_depots["depot_name"] == depot_sel].iloc[0]
    depot_id      = int(depot_row["depot_id"])
    zone_name     = depot_row["zone"]

    st.session_state["depot_id"]   = depot_id
    st.session_state["depot_name"] = depot_sel
    st.session_state["zone_name"]  = zone_name
    st.sidebar.caption(f"🌍 Zone : {zone_name}")

    return depot_id, depot_sel, zone_name