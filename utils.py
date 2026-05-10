import mysql.connector
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =========================
# Dépôts exclus
# =========================
EXCLUDED_DEPOT_IDS = {8, 41, 57}

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
# Chargement noms produits depuis la BD
# =========================
@st.cache_data(ttl=3600)
def load_product_names():
    """Retourne un dict {ref_product (int): name (str)} depuis la table product."""
    conn = get_connection()
    df = pd.read_sql("SELECT ref_product, name FROM product", conn)
    conn.close()
    df["ref_product"] = df["ref_product"].astype(int)
    return dict(zip(df["ref_product"], df["name"]))

# =========================
# Chargement données ventes
# =========================
@st.cache_data(ttl=600)
def load_product_ranking():
    conn = get_connection()
    excluded = ",".join(str(i) for i in EXCLUDED_DEPOT_IDS)
    df = pd.read_sql(f"""
        SELECT ref_product, SUM(quantity * price) AS total
        FROM sales
        WHERE depot_id NOT IN ({excluded})
        GROUP BY ref_product
        ORDER BY total DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=600)
def load_depots_for_product(ref_product):
    conn = get_connection()
    excluded = ",".join(str(i) for i in EXCLUDED_DEPOT_IDS)
    if ref_product is None:
        df = pd.read_sql(f"""
            SELECT DISTINCT d.depot_id, d.name AS depot_name, c.name AS zone
            FROM depot d
            JOIN country c ON d.country_id = c.id
            WHERE d.depot_id NOT IN ({excluded})
            ORDER BY d.name
        """, conn)
    else:
        df = pd.read_sql(f"""
            SELECT DISTINCT d.depot_id, d.name AS depot_name, c.name AS zone
            FROM sales s
            JOIN depot   d ON s.depot_id   = d.depot_id
            JOIN country c ON d.country_id = c.id
            WHERE s.ref_product = %(ref)s
              AND d.depot_id NOT IN ({excluded})
            ORDER BY d.name
        """, conn, params={"ref": int(ref_product)})
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_full_data():
    conn = get_connection()
    excluded = ",".join(str(i) for i in EXCLUDED_DEPOT_IDS)
    df = pd.read_sql(f"""
        SELECT s.ref_product, s.quantity, s.price, s.date_time,
               s.depot_id,
               d.name  AS depot_name,
               c.name  AS country_name
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


# =========================
# Sidebar unifié
# =========================
def sidebar_filters():
    """
    Sidebar : Date + Produit (avec nom depuis BD) + Pays + Dépôt.
    Retourne (df_filtered, product, depot_id, depot_sel, date_range, selected_country).
    """
    df_full      = load_full_data()
    product_names = load_product_names()   # {ref_product: name} depuis la BD

    # --- Date ---
    st.sidebar.header("🔎 Filtrage")
    date_range = st.sidebar.date_input(
        "Date",
        [df_full["date_time"].min(), df_full["date_time"].max()]
    )

    # --- Produit ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Produit")
    product_rank_sb = (df_full.groupby("ref_product")["total"]
                       .sum().reset_index()
                       .sort_values("total", ascending=False))

    def make_label(row):
        ref = int(row["ref_product"])
        nom = product_names.get(ref, "Inconnu")
        return f"#{ref} — {nom}  (💰 {int(row['total']):,})"

    product_rank_sb["label"] = product_rank_sb.apply(make_label, axis=1)

    labels_sb = ["🌐 Tous les produits"] + product_rank_sb["label"].tolist()
    selected_label = st.sidebar.selectbox("Choisir un produit", labels_sb)

    if selected_label == "🌐 Tous les produits":
        product = None
    else:
        product = int(selected_label.split("—")[0].replace("#", "").strip())

    st.session_state["product"] = product

    # --- Pays ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Pays")
    country_options = ["🌐 Tous les pays"] + sorted(df_full["country_name"].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Choisir un pays", country_options)
    if selected_country == "🌐 Tous les pays":
        selected_country = None

    # --- Dépôt ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏭 Dépôt")
    if selected_country:
        depot_list = df_full[df_full["country_name"] == selected_country]["depot_name"].dropna().unique().tolist()
    else:
        depot_list = df_full["depot_name"].dropna().unique().tolist()

    depot_options = ["🏭 Tous les dépôts"] + sorted(depot_list)
    selected_depot = st.sidebar.selectbox("Choisir un dépôt", depot_options)
    if selected_depot == "🏭 Tous les dépôts":
        selected_depot = None

    # Résolution depot_id / depot_sel
    if selected_depot is not None:
        row = df_full[df_full["depot_name"] == selected_depot].iloc[0]
        depot_id  = int(row["depot_id"])
        depot_sel = selected_depot
        zone_name = selected_country or ""
    else:
        depot_id  = "all"
        depot_sel = "Tous"
        zone_name = selected_country or "—"

    st.session_state["depot_id"]   = depot_id
    st.session_state["depot_name"] = depot_sel
    st.session_state["zone_name"]  = zone_name

    prod_label = product_names.get(product, f"#{product}") if product else "Tous"
    st.sidebar.caption(f"Produit: {prod_label} | Dépôt: {depot_sel}")

    # --- Filtres ---
    df = df_full.copy()
    if len(date_range) == 2:
        df = df[
            (df["date_time"] >= pd.to_datetime(date_range[0])) &
            (df["date_time"] <= pd.to_datetime(date_range[1]))
        ]
    if product is not None:
        df = df[df["ref_product"] == product]
    if selected_country is not None:
        df = df[df["country_name"] == selected_country]
    if selected_depot is not None:
        df = df[df["depot_name"] == selected_depot]

    return df, product, depot_id, depot_sel, date_range, selected_country