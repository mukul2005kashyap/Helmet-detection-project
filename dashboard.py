import os
import glob
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Helmet Safety Dashboard",
    page_icon="🪖",
    layout="wide",
)

st.title("🪖  Helmet Safety Compliance Dashboard")
st.caption("Analytics from Helmet Safety Detection System")

st.sidebar.header("⚙️ Configuration")
log_dir = st.sidebar.text_input("Log directory", value="outputs/logs")
snapshot_dir = st.sidebar.text_input("Snapshot directory", value="outputs/snapshots")

@st.cache_data
def load_frame_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_track_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_violation_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

frame_log_path   = os.path.join(log_dir, "frame_compliance_log.csv")
track_log_path   = os.path.join(log_dir, "track_summary_log.csv")
violation_path   = os.path.join(log_dir, "violation_log.csv")

if not os.path.exists(frame_log_path):
    st.warning(f"No frame log found at `{frame_log_path}`. Run the detector first.")
    st.stop()

df_frames    = load_frame_log(frame_log_path)
df_tracks    = load_track_log(track_log_path) if os.path.exists(track_log_path) else pd.DataFrame()
df_violations = load_violation_log(violation_path) if os.path.exists(violation_path) else pd.DataFrame()

total_frames   = len(df_frames)
total_persons  = df_frames["total_persons"].sum()
total_compliant = df_frames["compliant"].sum()
total_violations = df_frames["non_compliant"].sum()
avg_rate = (total_compliant / total_persons * 100) if total_persons > 0 else 100.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Frames Processed", f"{total_frames:,}")
col2.metric("Total Person-Frames", f"{total_persons:,}")
col3.metric("✅ Compliant", f"{total_compliant:,}")
col4.metric("❌ Violations", f"{total_violations:,}",
            delta=f"-{total_violations}" if total_violations else None,
            delta_color="inverse")
col5.metric("Avg Compliance", f"{avg_rate:.1f}%")

st.divider()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_frames["timestamp_s"], y=df_frames["compliance_rate_%"],
    mode="lines", name="Compliance %",
    line=dict(color="#00e676", width=2),
))
fig.add_trace(go.Bar(
    x=df_frames["timestamp_s"], y=df_frames["non_compliant"],
    name="Violations", marker_color="rgba(255, 60, 60, 0.5)",
    yaxis="y2",
))
fig.update_layout(
    xaxis_title="Time (s)",
    yaxis=dict(title="Compliance %", range=[0, 105]),
    yaxis2=dict(title="# Violations", overlaying="y", side="right"),
    legend=dict(x=0.01, y=0.99),
    height=380,
    template="plotly_dark",
)
st.subheader("📈 Compliance Timeline")
st.plotly_chart(fig, use_container_width=True)

if not df_tracks.empty:
    st.subheader("👤 Per-Track Compliance")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig2 = px.bar(
            df_tracks.sort_values("compliance_rate_%"),
            x="track_id", y="compliance_rate_%",
            color="compliance_rate_%",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[0, 100],
            labels={"compliance_rate_%": "Compliance %", "track_id": "Track ID"},
            title="Compliance Rate per Worker Track",
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.dataframe(
            df_tracks[["track_id", "total_frames", "compliance_rate_%", "violation_saved"]]
            .sort_values("compliance_rate_%"),
            use_container_width=True,
        )

if not df_violations.empty:
    st.subheader("🚨 Violation Log")
    st.dataframe(df_violations, use_container_width=True)

st.subheader("📸 Violation Snapshots")

snapshots = sorted(glob.glob(os.path.join(snapshot_dir, "*.jpg")))
if not snapshots:
    st.info("No snapshots found yet.")
else:
    n_cols = 4
    rows = [snapshots[i:i+n_cols] for i in range(0, len(snapshots), n_cols)]
    for row in rows[:3]:
        cols = st.columns(n_cols)
        for col, path in zip(cols, row):
            with col:
                img = Image.open(path)
                col.image(img, caption=Path(path).name, use_column_width=True)

    if len(snapshots) > 12:
        st.caption(f"Showing 12 of {len(snapshots)} snapshots.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Helmet Safety Detection System**  \nBuilt with YOLOv8 + OpenCV")