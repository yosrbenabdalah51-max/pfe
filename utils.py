import streamlit as st
import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

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
# PRODUCTS LIST 🔥
# =========================
@st.cache_data(ttl=600)
def load_product_list():
    conn = get_connection()

    query = """
    SELECT 
        p.ref_product,
        p.name,
        COALESCE(SUM(s.quantity * s.price), 0) as total_sales
    FROM product p
    LEFT JOIN sales s 
        ON p.ref_product = s.ref_product
    GROUP BY p.ref_product, p.name
    ORDER BY total_sales DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # nettoyage
    df["ref_product"] = df["ref_product"].astype(str)
    df["name"] = df["name"].fillna("Unknown")

    df["label"] = df["ref_product"] + " (" + df["name"] + ")"

    return df
# =========================
# SIDEBAR FILTER (SMART)
# =========================
def sidebar_product_filter():
    df = load_product_list()

    st.sidebar.title(" Produits")

    selected = st.sidebar.selectbox(
        "Choisir un produit",
        df["label"]
    )

    product = df.loc[
        df["label"] == selected,
        "ref_product"
    ].values[0]

    st.session_state["product"] = product

    return product
    # =========================
# DAILY SERIES (FIXED + STRONG)
# =========================
@st.cache_data(ttl=600)
def prepare_daily_series(product):
    conn = get_connection()

    query = """
    SELECT 
        DATE(date_time) as ds,
        SUM(quantity) as y
    FROM sales
    WHERE ref_product = %s
    GROUP BY DATE(date_time)
    ORDER BY ds
    """

    df = pd.read_sql(query, conn, params=(product,))

    conn.close()  
    if df.empty:
        return df

    df["ds"] = pd.to_datetime(df["ds"])

    # fill missing days (CRUCIAL 🔥)
    df = df.set_index("ds").asfreq("D").fillna(0).reset_index()

    return df



