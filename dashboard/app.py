"""
dashboard/app.py
----------------
Streamlit dashboard showing:
  - Live streaming simulation with drift detection
  - Model performance comparison (static vs adaptive)
  - Drift event timeline
  - SHAP feature importance
  - Retrain log table

Run: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ── Page config
st.set_page_config(
    page_title="Fraud Drift Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px; border-radius: 12px;
        border: 1px solid #0f3460; text-align: center;
    }
    .drift-alert {
        background: linear-gradient(135deg, #7f0000, #b71c1c);
        padding: 12px 20px; border-radius: 8px;
        color: white; font-weight: bold;
    }
    .retrain-alert {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        padding: 12px 20px; border-radius: 8px;
        color: white; font-weight: bold;
    }
    .stMetric label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders
@st.cache_data
def load_results():
    reports = "reports"
    results = {}
    for fname in ["static_model_metrics", "adaptive_model_metrics", "drift_events", "retrain_log"]:
        path = os.path.join(reports, f"{fname}.csv")
        if os.path.exists(path):
            results[fname] = pd.read_csv(path)
        else:
            results[fname] = pd.DataFrame()
    return results


@st.cache_data
def load_dataset():
    path = "data/creditcard_with_drift.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/fraud.png", width=60)
st.sidebar.title("🔍 Fraud Drift Monitor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview Dashboard",
    "📈 Performance Comparison",
    "🚨 Drift Events",
    "🔄 Retraining Log",
    "📋 Data Explorer"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Adaptive Fraud Detection")
st.sidebar.markdown("**Detectors:** ADWIN | DDM | PageHinkley")
st.sidebar.markdown("**Model:** XGBoost + Auto-Retrain")

# ── Load data
results = load_results()
df_raw = load_dataset()

static_df = results.get("static_model_metrics", pd.DataFrame())
adaptive_df = results.get("adaptive_model_metrics", pd.DataFrame())
drift_df = results.get("drift_events", pd.DataFrame())
retrain_df = results.get("retrain_log", pd.DataFrame())

has_results = not static_df.empty


# ════════════════════════════════════════════════
#  PAGE: Overview Dashboard
# ════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.title("📊 Adaptive Fraud Detection — Overview")
    st.markdown("Real-time model monitoring with automated drift detection and retraining.")

    if not has_results:
        st.warning("⚠️ No experiment results found. Run `python run_experiment.py` first.")
        st.code("python run_experiment.py", language="bash")
    else:
        # ── KPI Row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Static Model\nMean ROC-AUC",
                      f"{static_df['roc_auc'].mean():.3f}")
        with c2:
            st.metric("Adaptive Model\nMean ROC-AUC",
                      f"{adaptive_df['roc_auc'].mean():.3f}",
                      delta=f"+{adaptive_df['roc_auc'].mean() - static_df['roc_auc'].mean():.3f}")
        with c3:
            st.metric("Drift Events\nDetected", len(drift_df[drift_df.get("drift_type", pd.Series()) == "drift"]) if not drift_df.empty else 0)
        with c4:
            st.metric("Retraining\nEvents", len(retrain_df) if not retrain_df.empty else 0)
        with c5:
            if not df_raw.empty:
                st.metric("Total Samples\nProcessed", f"{len(df_raw):,}")

        st.markdown("---")

        # ── F1 comparison chart
        st.subheader("F1 Score Over Time: Static vs Adaptive")
        fig = go.Figure()
        chunks = list(range(len(static_df)))
        window = 8

        fig.add_trace(go.Scatter(
            x=chunks, y=static_df["f1"].rolling(window, min_periods=1).mean(),
            name="Static Model", line=dict(color="#EF5350", width=2.5),
            fill="tozeroy", fillcolor="rgba(239,83,80,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=chunks, y=adaptive_df["f1"].rolling(window, min_periods=1).mean(),
            name="Adaptive Model", line=dict(color="#42A5F5", width=2.5),
            fill="tozeroy", fillcolor="rgba(66,165,245,0.08)"
        ))

        # Drift markers
        if not drift_df.empty:
            drift_actual = drift_df[drift_df["drift_type"] == "drift"] if "drift_type" in drift_df.columns else drift_df
            for _, row in drift_actual.iterrows():
                fig.add_vline(x=row.get("chunk_index", 0), line_color="orange",
                              line_dash="dot", line_width=1.5,
                              annotation_text="Drift", annotation_position="top")

        if not retrain_df.empty:
            for _, row in retrain_df.iterrows():
                fig.add_vline(x=row.get("chunk_index", 0), line_color="limegreen",
                              line_dash="dash", line_width=2,
                              annotation_text="Retrain", annotation_position="top right")

        fig.update_layout(
            height=400, template="plotly_dark",
            xaxis_title="Chunk Index (time →)",
            yaxis_title="F1 Score (Fraud Class)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=40, t=20, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Dataset distribution
        if not df_raw.empty:
            st.subheader("Dataset Distribution by Period")
            col1, col2 = st.columns(2)
            with col1:
                period_counts = df_raw["Period"].value_counts()
                fig_pie = px.pie(values=period_counts.values, names=period_counts.index,
                                 title="Sample Distribution",
                                 color_discrete_sequence=["#42A5F5", "#EF5350", "#66BB6A"])
                fig_pie.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fraud_by_period = df_raw.groupby("Period")["Class"].mean().reset_index()
                fig_bar = px.bar(fraud_by_period, x="Period", y="Class",
                                 title="Fraud Rate by Period",
                                 color="Period",
                                 color_discrete_sequence=["#42A5F5", "#EF5350", "#66BB6A"])
                fig_bar.update_layout(template="plotly_dark", height=300,
                                      yaxis_tickformat=".1%", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════
#  PAGE: Performance Comparison
# ════════════════════════════════════════════════
elif page == "📈 Performance Comparison":
    st.title("📈 Performance Comparison")

    if not has_results:
        st.warning("Run `python run_experiment.py` first to generate results.")
    else:
        metric = st.selectbox("Select Metric", ["f1", "roc_auc", "avg_precision", "error_rate"])
        window = st.slider("Smoothing Window", 1, 20, 8)

        fig = make_subplots(rows=1, cols=1)
        chunks = list(range(len(static_df)))

        fig.add_trace(go.Scatter(
            x=chunks, y=static_df[metric].rolling(window, min_periods=1).mean(),
            name=f"Static ({metric})", line=dict(color="#EF5350", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=chunks, y=adaptive_df[metric].rolling(window, min_periods=1).mean(),
            name=f"Adaptive ({metric})", line=dict(color="#42A5F5", width=2)
        ))

        fig.update_layout(
            height=500, template="plotly_dark",
            title=f"{metric.upper()} Over Time (rolling avg = {window})",
            xaxis_title="Chunk Index",
            yaxis_title=metric,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("Summary Statistics")
        summary = pd.DataFrame({
            "Metric": ["ROC-AUC", "Avg Precision", "F1 Score", "Error Rate"],
            "Static Model": [
                f"{static_df['roc_auc'].mean():.4f}",
                f"{static_df['avg_precision'].mean():.4f}",
                f"{static_df['f1'].mean():.4f}",
                f"{static_df['error_rate'].mean():.4f}"
            ],
            "Adaptive Model": [
                f"{adaptive_df['roc_auc'].mean():.4f}",
                f"{adaptive_df['avg_precision'].mean():.4f}",
                f"{adaptive_df['f1'].mean():.4f}",
                f"{adaptive_df['error_rate'].mean():.4f}"
            ],
            "Improvement": [
                f"+{adaptive_df['roc_auc'].mean() - static_df['roc_auc'].mean():.4f}",
                f"+{adaptive_df['avg_precision'].mean() - static_df['avg_precision'].mean():.4f}",
                f"+{adaptive_df['f1'].mean() - static_df['f1'].mean():.4f}",
                f"{adaptive_df['error_rate'].mean() - static_df['error_rate'].mean():.4f}"
            ]
        })
        st.dataframe(summary, use_container_width=True)


# ════════════════════════════════════════════════
#  PAGE: Drift Events
# ════════════════════════════════════════════════
elif page == "🚨 Drift Events":
    st.title("🚨 Drift Detection Events")

    if drift_df.empty:
        st.warning("No drift events found. Run the experiment first.")
    else:
        drift_only = drift_df[drift_df.get("drift_type", pd.Series("drift", index=drift_df.index)) == "drift"] if "drift_type" in drift_df.columns else drift_df
        warn_only = drift_df[drift_df.get("drift_type", pd.Series()) == "warning"] if "drift_type" in drift_df.columns else pd.DataFrame()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Drift Events", len(drift_only))
        col2.metric("Warning Events", len(warn_only))
        col3.metric("Detectors Active", drift_df["detector"].nunique() if "detector" in drift_df.columns else 0)

        # Drift by detector
        if "detector" in drift_df.columns:
            fig = px.histogram(drift_only, x="detector", title="Drift Events by Detector",
                               color="detector",
                               color_discrete_map={"ADWIN": "#FF5722", "DDM": "#9C27B0", "PageHinkley": "#2196F3"})
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("All Drift Events")
        st.dataframe(drift_df, use_container_width=True)


# ════════════════════════════════════════════════
#  PAGE: Retraining Log
# ════════════════════════════════════════════════
elif page == "🔄 Retraining Log":
    st.title("🔄 Adaptive Retraining Log")

    if retrain_df.empty:
        st.warning("No retraining events found. Run the experiment first.")
    else:
        st.metric("Total Retraining Events", len(retrain_df))

        # Before vs After ROC-AUC
        fig = go.Figure()
        fig.add_trace(go.Bar(x=retrain_df["retrain_id"], y=retrain_df["before_roc_auc"],
                             name="Before Retrain", marker_color="#EF9A9A"))
        fig.add_trace(go.Bar(x=retrain_df["retrain_id"], y=retrain_df["after_roc_auc"],
                             name="After Retrain", marker_color="#80CBC4"))
        fig.update_layout(
            barmode="group", template="plotly_dark", height=400,
            title="ROC-AUC Before vs After Each Retraining Event",
            xaxis_title="Retrain #", yaxis_title="ROC-AUC"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed Log")
        st.dataframe(retrain_df, use_container_width=True)


# ════════════════════════════════════════════════
#  PAGE: Data Explorer
# ════════════════════════════════════════════════
elif page == "📋 Data Explorer":
    st.title("📋 Dataset Explorer")

    if df_raw.empty:
        st.warning("Dataset not found.")
    else:
        st.write(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

        period_filter = st.multiselect("Filter by Period", df_raw["Period"].unique().tolist(),
                                       default=df_raw["Period"].unique().tolist())
        filtered = df_raw[df_raw["Period"].isin(period_filter)]

        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox("Feature to explore", [f"V{i}" for i in range(1, 6)] + ["Amount"])
        with col2:
            st.write("")

        fig = px.histogram(filtered, x=feature, color="Period",
                           nbins=50, title=f"Distribution of {feature} by Period",
                           barmode="overlay", opacity=0.7,
                           color_discrete_map={"pre_drift": "#42A5F5",
                                               "post_drift": "#EF5350",
                                               "recovery": "#66BB6A"})
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sample Data")
        st.dataframe(filtered.head(100), use_container_width=True)
