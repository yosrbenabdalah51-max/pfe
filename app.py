import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
from utils import get_connection

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Vision Analytics",
    page_icon="📊",
    layout="wide"
)

# =========================
# STYLE CSS
# =========================
st.markdown("""
<style>
.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: gray;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}
.big-title {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        conn = get_connection()
        
        df = pd.read_sql(
            "SELECT ref_product, quantity, price, date_time FROM sales",
            conn
        )
        conn.close()
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['total'] = df['quantity'] * df['price']
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur de connexion à la base de données : {e}")
        st.stop()

df = load_data()

# =========================
# SIDEBAR DATE ONLY
# =========================
st.sidebar.header("🔎 Filtrage")

date_range = st.sidebar.date_input(
    "Date",
    [df['date_time'].min(), df['date_time'].max()]
)

if len(date_range) == 2:
    df = df[
        (df['date_time'] >= pd.to_datetime(date_range[0])) &
        (df['date_time'] <= pd.to_datetime(date_range[1]))
    ]

# =========================
# TITLE
# =========================
st.markdown('<p class="big-title">📊 Vision Analytics Dashboard</p>', unsafe_allow_html=True)

# =========================
# VÉRIFICATION DONNÉES VIDES
# =========================
if df.empty:
    st.warning("⚠️ Aucune donnée trouvée pour la période sélectionnée. Veuillez modifier les filtres.")
    st.stop()

# =========================
# 🎯 SELECT PRODUIT (TOP)
# =========================
product_rank = df.groupby('ref_product')['total'].sum().reset_index()
product_rank = product_rank.sort_values(by='total', ascending=False)

if product_rank.empty:
    st.warning("⚠️ Aucun produit trouvé pour la période sélectionnée.")
    st.stop()

# 🧠 label avec valeur
try:
    product_rank["label"] = product_rank.apply(
        lambda x: f"{x['ref_product']}  (💰 {int(x['total'])})", axis=1
    ).astype(str)
except Exception as e:
    st.error(f"⚠️ Erreur lors de la création des labels produits : {e}")
    st.stop()

selected_label = st.selectbox(
    "🎯 Choisir un produit",
    product_rank["label"]
)

# Extraction du nom du produit
selected_product = product_rank[
    product_rank["label"] == selected_label
]["ref_product"].values[0]

# Stockage
st.session_state["product"] = selected_product

st.success(f"Produit sélectionné: {selected_product}")



# =========================
# KPI CARDS
# =========================
total_sales = df['total'].sum()
total_products = df['ref_product'].nunique()
total_quantity = df['quantity'].sum()
avg_sale = df['total'].mean()

col1, col2, col3, col4 = st.columns(4)

def card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

with col1:
    card("TOTAL SALES", f"{total_sales:,.0f}")

with col2:
    card("TOTAL PRODUCTS", total_products)

with col3:
    card("TOTAL UNITS", f"{total_quantity:,}")

with col4:
    card("AVERAGE SALE", f"{avg_sale:,.2f}")

# =========================
# GRAPH 1
# =========================
colA, colB = st.columns([2, 1])

with colA:
    st.subheader("📈 Performance globale")

    df_time = df.groupby(df['date_time'].dt.to_period("M"))['total'].sum().reset_index()
    df_time['date_time'] = df_time['date_time'].astype(str)

    fig1 = px.area(df_time, x='date_time', y='total')
    fig1.update_layout(template="plotly_white", height=400)

    st.plotly_chart(fig1, use_container_width=True)

# =========================
# TOP PRODUITS
# =========================
with colB:
    st.subheader("🏆 Top Produits")

    top_products = product_rank.head(5)

    fig2 = px.bar(
        top_products,
        x='total',
        y='ref_product',
        orientation='h',
        color='ref_product'
    )

    fig2.update_layout(height=400)

    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TABLE + PIE
# =========================
colC, colD = st.columns([2, 1])

with colC:
    st.subheader("📋 Top 5 Produits")

    top_table = df.groupby('ref_product').agg({
        'total': 'sum',
        'quantity': 'sum'
    }).reset_index()

    top_table = top_table.sort_values(by='total', ascending=False).head(5)

    st.dataframe(top_table, use_container_width=True)

with colD:
    st.subheader("🥧 Répartition")

    fig3 = px.pie(
        top_products,
        names='ref_product',
        values='total'
    )

    st.plotly_chart(fig3, use_container_width=True)