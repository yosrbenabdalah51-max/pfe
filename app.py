import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from db_utils import get_connection

st.set_page_config(page_title="Vision Analytics", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Sidebar auto-expand on hover */
[data-testid="stSidebar"] {
    transition: width 0.3s ease !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0px !important;
    min-width: 0px !important;
    overflow: hidden;
}
section[data-testid="stSidebar"][aria-expanded="false"]:hover {
    width: 350px !important;
    min-width: 350px !important;
    overflow: visible;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15);
}

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }

.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}
.metric-title { font-size: 14px; color: gray; }
.metric-value { font-size: 28px; font-weight: bold; }
.big-title    { font-size: 28px; font-weight: bold; }

.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 18px 0 24px 0;
}
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px 14px;
    box-shadow: 0 2px 12px rgba(108,99,255,0.1);
    border-top: 3px solid #6c63ff;
    text-align: center;
}
.kpi-icon  { font-size: 22px; margin-bottom: 6px; }
.kpi-label { font-size: 10px; font-weight: 600; color: #9ca3af;
             text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.kpi-value { font-size: 16px; font-weight: 800; color: #1a1a2e; line-height: 1.2; }

.dash-section {
    font-size: 15px; font-weight: 700; color: #1a1a2e;
    margin: 22px 0 10px 0;
    padding-left: 10px;
    border-left: 4px solid #6c63ff;
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
        df = pd.read_sql("""
            SELECT s.id, s.order_id, s.ref_product, s.is_pack,
                   s.quantity, s.price, s.depot_id, s.date_time,
                   d.name        AS depot_name,
                   c.name        AS country_name,
                   p.name        AS product_name,
                   p.sub_category_id,
                   cat.category_name,
                   cat.sub_category_name
            FROM sales s
            LEFT JOIN depot    d   ON s.depot_id        = d.depot_id
            LEFT JOIN country  c   ON d.country_id      = c.id
            LEFT JOIN product  p   ON s.ref_product     = p.ref_product
            LEFT JOIN category  cat ON p.sub_category_id = cat.sub_category_id
        """, conn)
        conn.close()
        df['date_time']   = pd.to_datetime(df['date_time'])
        df['total']       = df['quantity'] * df['price']
        df['ref_product'] = df['ref_product'].astype(int)
        df['year']        = df['date_time'].dt.year.astype(str)
        df['month']       = df['date_time'].dt.to_period('M').astype(str)
        df['dayofweek']   = df['date_time'].dt.day_name()
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur : {e}")
        st.stop()

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔎 Filtrage")
date_range = st.sidebar.date_input("Date", [df['date_time'].min(), df['date_time'].max()])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Produit")
product_rank_sb = (df.groupby('ref_product')['total'].sum()
                   .reset_index().sort_values('total', ascending=False))
product_rank_sb["label"] = product_rank_sb.apply(
    lambda x: f"{x['ref_product']}  (💰 {int(x['total']):,})", axis=1).astype(str)

labels_sb = ["🌐 Tous les produits"] + product_rank_sb["label"].tolist()
selected_label = st.sidebar.selectbox("Choisir un produit", labels_sb)

if selected_label == "🌐 Tous les produits":
    selected_product = None
else:
    selected_product = int(float(selected_label.split("  ")[0]))

st.session_state["product"] = selected_product

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Pays")
country_options = ["🌐 Tous les pays"] + sorted(df['country_name'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Choisir un pays", country_options)
if selected_country == "🌐 Tous les pays":
    selected_country = None

st.sidebar.markdown("---")
st.sidebar.subheader("🏭 Dépôt")
if selected_country:
    depot_list = df[df['country_name'] == selected_country]['depot_name'].dropna().unique().tolist()
else:
    depot_list = df['depot_name'].dropna().unique().tolist()
depot_options = ["🏭 Tous les dépôts"] + sorted(depot_list)
selected_depot = st.sidebar.selectbox("Choisir un dépôt", depot_options)
if selected_depot == "🏭 Tous les dépôts":
    selected_depot = None

# =========================
# FILTERS
# =========================
df_f = df.copy()
if len(date_range) == 2:
    df_f = df_f[
        (df_f['date_time'] >= pd.to_datetime(date_range[0])) &
        (df_f['date_time'] <= pd.to_datetime(date_range[1]))
    ]
if selected_product is not None:
    df_f = df_f[df_f['ref_product'] == selected_product]
if selected_country is not None:
    df_f = df_f[df_f['country_name'] == selected_country]
if selected_depot is not None:
    df_f = df_f[df_f['depot_name'] == selected_depot]

if df_f.empty:
    st.warning("⚠️ Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# =========================
# TITLE + KPIs originaux
# =========================
st.markdown('<p class="big-title">📊 Vision Analytics Dashboard</p>', unsafe_allow_html=True)
if selected_product:
    st.success(f"Produit sélectionné : {selected_product}")

total_sales    = df_f['total'].sum()
total_products = df_f['ref_product'].nunique()
total_quantity = df_f['quantity'].sum()
avg_sale       = df_f['total'].mean()

col1, col2, col3, col4 = st.columns(4)
def card(col, title, value):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

card(col1, "TOTAL SALES",    f"{total_sales:,.0f}")
card(col2, "TOTAL PRODUCTS", total_products)
card(col3, "TOTAL UNITS",    f"{total_quantity:,}")
card(col4, "AVERAGE SALE",   f"{avg_sale:,.2f}")

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🏠 Vue Globale", "📊 Dashboard Analytique"])

# ── TAB 1 original ────────────────────────
with tab1:
    product_rank = df_f.groupby('ref_product')['total'].sum().reset_index()
    product_rank = product_rank.sort_values('total', ascending=False)
    top_products = product_rank.head(5)

    st.subheader("📈 Performance globale")
    df_time = df_f.groupby(df_f['date_time'].dt.to_period("M"))['total'].sum().reset_index()
    df_time['date_time'] = df_time['date_time'].astype(str)
    fig1 = px.area(df_time, x='date_time', y='total')
    fig1.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig1, use_container_width=True)

    colC, colD = st.columns([2, 1])
    with colC:
        st.subheader("📋 Top 5 Produits")
        top_table = (df_f.groupby('ref_product')
                     .agg(total=('total','sum'), quantity=('quantity','sum'))
                     .reset_index().sort_values('total', ascending=False).head(5))
        st.dataframe(top_table, use_container_width=True)
    with colD:
        st.subheader("🥧 Répartition")
        fig3 = px.pie(top_products, names='ref_product', values='total')
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2 Dashboard Analytique ────────────
with tab2:

    COLORS = ['#6c63ff','#a78bfa','#f59e0b','#10b981','#f43f5e','#3b82f6','#8b5cf6']
    TW = dict(template="plotly_white", margin=dict(l=0,r=0,t=10,b=0))

    # ── INSIGHTS ──────────────────────────────
    nb_cmd       = df_f['order_id'].nunique()
    best_country = df_f.groupby('country_name')['total'].sum().idxmax()
    best_product = df_f.groupby('ref_product')['total'].sum().idxmax()
    best_depot   = df_f.groupby('depot_name')['total'].sum().idxmax()
    best_year    = df_f.groupby('year')['total'].sum().idxmax()

    st.markdown('<div class="dash-section">💡 Insights clés</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-icon">🛒</div>
        <div class="kpi-label">Commandes</div><div class="kpi-value">{nb_cmd:,}</div></div>
      <div class="kpi-card"><div class="kpi-icon">🌍</div>
        <div class="kpi-label">Meilleur pays</div><div class="kpi-value">{best_country}</div></div>
      <div class="kpi-card"><div class="kpi-icon">🏷️</div>
        <div class="kpi-label">Meilleur produit</div><div class="kpi-value">#{best_product}</div></div>
      <div class="kpi-card"><div class="kpi-icon">🏭</div>
        <div class="kpi-label">Meilleur dépôt</div><div class="kpi-value">{best_depot}</div></div>
      <div class="kpi-card"><div class="kpi-icon">📅</div>
        <div class="kpi-label">Meilleure année</div><div class="kpi-value">{best_year}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart 1 : CA + Qté par mois ───────────
    st.markdown('<div class="dash-section">📈 Évolution CA & Quantités par mois</div>', unsafe_allow_html=True)
    df_monthly = df_f.groupby('month').agg(total=('total','sum'), quantity=('quantity','sum')).reset_index()
    fig_evo = make_subplots(specs=[[{"secondary_y": True}]])
    fig_evo.add_trace(go.Bar(x=df_monthly['month'], y=df_monthly['total'],
        name="CA", marker_color='#6c63ff', opacity=0.85), secondary_y=False)
    fig_evo.add_trace(go.Scatter(x=df_monthly['month'], y=df_monthly['quantity'],
        name="Quantités", line=dict(color='#f59e0b', width=2.5),
        mode='lines+markers', marker=dict(size=5)), secondary_y=True)
    fig_evo.update_layout(**TW, height=320,
        legend=dict(orientation='h', y=1.1, x=0), bargap=0.3)
    fig_evo.update_yaxes(title_text="CA",  secondary_y=False)
    fig_evo.update_yaxes(title_text="Qté", secondary_y=True)
    st.plotly_chart(fig_evo, use_container_width=True)

    # ── Chart 2 : Qté par produit + CA par pays ──
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="dash-section">📦 Quantité vendue par produit (Top 15)</div>', unsafe_allow_html=True)
        df_pq = (df_f.groupby('ref_product')['quantity']
                 .sum().reset_index()
                 .sort_values('quantity', ascending=True).tail(15))
        fig_pq = px.bar(df_pq, x='quantity', y=df_pq['ref_product'].astype(str),
                        orientation='h', text='quantity',
                        color='quantity', color_continuous_scale=['#ede9fe','#6c63ff'])
        fig_pq.update_traces(texttemplate='%{text:,}', textposition='inside', textfont_color='white')
        fig_pq.update_layout(**TW, height=420, coloraxis_showscale=False,
                             xaxis_title="Quantité", yaxis_title="Produit")
        st.plotly_chart(fig_pq, use_container_width=True)

    with c2:
        st.markdown('<div class="dash-section">🌍 CA & Quantité par pays</div>', unsafe_allow_html=True)
        df_pays = (df_f.groupby('country_name')
                   .agg(ca=('total','sum'), qty=('quantity','sum'))
                   .reset_index().sort_values('ca', ascending=False))
        fig_pays = go.Figure()
        fig_pays.add_trace(go.Bar(name='CA', x=df_pays['country_name'], y=df_pays['ca'],
            marker_color='#6c63ff', yaxis='y'))
        fig_pays.add_trace(go.Scatter(name='Quantité', x=df_pays['country_name'], y=df_pays['qty'],
            mode='lines+markers', line=dict(color='#f59e0b', width=2.5),
            marker=dict(size=8), yaxis='y2'))
        fig_pays.update_layout(**TW, height=420,
            yaxis=dict(title='CA'),
            yaxis2=dict(title='Quantité', overlaying='y', side='right'),
            legend=dict(orientation='h', y=1.05, x=0), bargap=0.3)
        st.plotly_chart(fig_pays, use_container_width=True)

    # ── Chart 3 : Produit (x) × Dépôt (couleur) ── axe x = produit, représenté en radar par dépôt
    st.markdown('<div class="dash-section">🏭 Quantité par produit et dépôt — Vue Radar par dépôt</div>', unsafe_allow_html=True)

    top10_prod = (df_f.groupby('ref_product')['quantity']
                  .sum().nlargest(10).index.tolist())
    df_dp = (df_f[df_f['ref_product'].isin(top10_prod)]
             .groupby(['depot_name','ref_product'])['quantity']
             .sum().reset_index())
    df_dp['ref_product'] = df_dp['ref_product'].astype(str)

    depots = df_dp['depot_name'].unique().tolist()
    produits = df_dp['ref_product'].unique().tolist()

    fig_radar = go.Figure()
    for i, depot in enumerate(depots):
        df_d = df_dp[df_dp['depot_name'] == depot]
        vals = [df_d[df_d['ref_product'] == p]['quantity'].sum() for p in produits]
        vals_closed = vals + [vals[0]]
        cats_closed = produits + [produits[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill='toself', name=depot,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            fillcolor=COLORS[i % len(COLORS)].replace('#','rgba(').replace('ff','ff,0.12)') if False else COLORS[i % len(COLORS)],
            opacity=0.7
        ))
    fig_radar.update_layout(
        **TW, height=420,
        polar=dict(
            radialaxis=dict(visible=True, gridcolor='#e5e7eb', linecolor='#e5e7eb'),
            angularaxis=dict(gridcolor='#e5e7eb', linecolor='#e5e7eb')
        ),
        legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center'),
        showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Chart 4 : Pack vs Standard + Treemap ──────────────────
    c3, c4 = st.columns([1, 2])

    with c3:
        st.markdown('<div class="dash-section">📦 Pack vs Standard</div>', unsafe_allow_html=True)
        df_pack = df_f.copy()
        df_pack['type'] = df_pack['is_pack'].apply(lambda x: 'Pack' if x > 0 else 'Standard')
        df_pack_agg = df_pack.groupby('type').agg(
            ca=('total','sum'), qty=('quantity','sum')
        ).reset_index()

        fig_pack = go.Figure()
        fig_pack.add_trace(go.Pie(
            labels=df_pack_agg['type'],
            values=df_pack_agg['ca'],
            hole=0.55,
            marker=dict(colors=['#6c63ff','#f59e0b']),
            textinfo='label+percent',
            textfont=dict(size=13, family='Plus Jakarta Sans'),
            hovertemplate="<b>%{label}</b><br>CA: %{value:,.0f}<br>Part: %{percent}<extra></extra>"
        ))
        fig_pack.update_layout(**TW, height=360,
            annotations=[dict(text='Type', x=0.5, y=0.5,
                             font=dict(size=14, color='#1a1a2e', family='Plus Jakarta Sans'),
                             showarrow=False)])
        st.plotly_chart(fig_pack, use_container_width=True)

    with c4:
        st.markdown('<div class="dash-section">🌿 Treemap — Catégorie → Sous-catégorie → CA</div>', unsafe_allow_html=True)
        df_tree = df_f.dropna(subset=['category_name','sub_category_name']).copy()
        df_tree_agg = (df_tree.groupby(['category_name','sub_category_name'])
                       .agg(ca=('total','sum'), qty=('quantity','sum'))
                       .reset_index())

        fig_tree = px.treemap(
            df_tree_agg,
            path=['category_name', 'sub_category_name'],
            values='ca',
            color='ca',
            color_continuous_scale=['#ede9fe','#6c63ff','#4c1d95'],
            custom_data=['qty'],
            hover_data={'ca': True}
        )
        fig_tree.update_traces(
            texttemplate="<b>%{label}</b><br>CA: %{value:,.0f}",
            textfont=dict(size=12, family='Plus Jakarta Sans'),
            hovertemplate="<b>%{label}</b><br>CA: %{value:,.0f}<br>Qté: %{customdata[0]:,}<extra></extra>"
        )
        fig_tree.update_layout(**TW, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_tree, use_container_width=True)