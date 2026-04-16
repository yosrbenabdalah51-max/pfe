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
@st.cache_data
def load_product_ranking():
    """Charge tous les produits triés par ventes décroissantes."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT ref_product, quantity, price FROM sales", conn
    )
    conn.close()
    df['total'] = df['quantity'] * df['price']
    product_rank = (df.groupby('ref_product')['total']
                    .sum()
                    .reset_index()
                    .sort_values(by='total', ascending=False))
    return product_rank


def sidebar_product_selector():
    """
    Affiche dans le sidebar un selectbox de produits triés par ventes.
    La première option est 'Tous' (entraîne le modèle sur tous les produits).
    Retourne le produit sélectionné (str) ou None si 'Tous'.
    Stocke aussi la valeur dans st.session_state['product'].
    """
    product_rank = load_product_ranking()

    # Construction des labels : "Tous" en premier, puis produits triés
    labels = ["🌐 Tous les produits"] + [
        f"{row['ref_product']}  (💰 {int(row['total'])})"
        for _, row in product_rank.iterrows()
    ]

    st.sidebar.header("🎯 Produit")
    selected_label = st.sidebar.selectbox("Choisir un produit", labels)

    if selected_label == "🌐 Tous les produits":
        product = None  # signifie "tous"
    else:
        product = product_rank[
            product_rank.apply(
                lambda x: f"{x['ref_product']}  (💰 {int(x['total'])})" == selected_label,
                axis=1
            )
        ]["ref_product"].values[0]

    st.session_state["product"] = product
    return product