import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from utils import sidebar_product_filter

product = sidebar_product_filter()

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Stock Management", page_icon="📦", layout="wide")



st.title("Smart Stock Dashboard")

# =========================
# SIDEBAR
# =========================
product = sidebar_product_filter()

if product is None:
    st.stop()

# =========================
# LOAD DATA
# =========================
df = prepare_daily_series(product)

if df.empty or len(df) < 5:
    st.warning("Not enough data to display.")
    st.stop()

# =========================
# KPIs
# =========================
avg     = df["y"].mean()
min_val = df["y"].min()
max_val = df["y"].max()

# =========================
# STOCK LOGIC
# =========================
q1 = df["y"].quantile(0.25)
q3 = df["y"].quantile(0.75)

if avg < q1:
    status      = "Overstock"
    status_icon = "warning"
elif avg > q3:
    status      = "Shortage Risk"
    status_icon = "error"
else:
    status      = "Stable"
    status_icon = "success"

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Demand Chart", "KPI Overview", "Insights & Recommendations"])

# ── Graph ──
with tab1:
    st.subheader(f"Demand Evolution — {product}")

    last_date = df["ds"].max()
    df_plot   = df[df["ds"] >= last_date - pd.Timedelta(days=180)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot["ds"],
        y=df_plot["y"],
        mode="lines",
        name="Daily Demand",
        line=LINE_HISTORY,
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.07)",
    ))

    # Rolling average
    df_plot = df_plot.copy()
    df_plot['rolling'] = df_plot['y'].rolling(14, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=df_plot["ds"],
        y=df_plot["rolling"],
        mode="lines",
        name="14-Day Average",
        line=dict(color="#00d4aa", width=2, dash="dot"),
    ))

    fig = apply_plotly_layout(fig, "Last 6 Months — Daily Demand")
    fig.update_layout(xaxis_title="Date", yaxis_title="Quantity")

    st.plotly_chart(fig, use_container_width=True)

# ── KPIs ──
with tab2:
    st.subheader("Stock KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Demand", f"{avg:.2f}")
    c2.metric("Min Demand",     f"{min_val:.2f}")
    c3.metric("Max Demand",     f"{max_val:.2f}")
    c4.metric("Stock Status",   status)

    st.divider()

    if status == "Shortage Risk":
        st.warning("High demand variability detected — risk of stockout.")
    elif status == "Overstock":
        st.error("Demand is below average — overstock risk detected.")
    else:
        st.success("Demand levels are balanced. No immediate action required.")

# ── Insights ──
with tab3:
    st.subheader("Smart Insights")

    # Status badge
    if status_icon == "success":
        st.success(f"Stock Status: {status}")
    elif status_icon == "warning":
        st.warning(f"Stock Status: {status}")
    else:
        st.error(f"Stock Status: {status}")

    # Trend analysis
    trend = np.polyfit(range(len(df)), df["y"], 1)[0]

    st.divider()

    if trend > 0:
        st.info("Demand is on an upward trend. Consider increasing safety stock levels.")
    elif trend < 0:
        st.warning("Demand is declining. Review procurement volumes to avoid excess inventory.")
    else:
        st.success("Demand is steady. Current stock levels appear appropriate.")

    st.divider()
    st.info("Recommendation: Maintain a safety stock buffer given observed demand variability.")

