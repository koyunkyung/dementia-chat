# Lifelog_Predictor.py
# ------------------------------------------------------------
# Streamlit app for visualizing lifelog-based time-series
# from a JSON config (cases: activity_seq, met_5min,
# sleep_hr_seq, sleep_hypno_seq, sleep_rmssd_seq, daily16, mmse13).
# ------------------------------------------------------------
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# THEME / PALETTE
# =========================
PALETTE = {
    "bg": "#FFFBF5",          # 매우 연한 따뜻한 크림색
    "card": "#FFFFFF",
    "border": "#FFEFE0",      # 부드러운 주황색
    "primary": "#FF7A2F",     # 쨍하고 친근한 주황색
    "primary_dark": "#F06B20",# 호버/그림자용 더 어두운 주황색
    "accent": "#FFDABF",      # 연한 살구색/주황색 하이라이트
    "soft": "#FFF4EC",        # 매우 부드러운 주황색/살구색 배경
    "text": "#2C2A28",        # 따뜻한 배경에 잘 보이는 짙은 갈색/회색
    "muted": "#756F6A",
}

# Inject light CSS to apply palette
def inject_css():
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            background: {PALETTE["bg"]};
            color: {PALETTE["text"]};
        }}
        .stApp {{
            background: {PALETTE["bg"]};
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}
        .soft-card {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 14px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        }}
        .soft-header {{
            background: {PALETTE["soft"]};
            border: 1px solid {PALETTE["border"]};
            color: {PALETTE["text"]};
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            font-weight: 600;
        }}
        .pill {{
            display: inline-block;
            background: {PALETTE["accent"]};
            color: {PALETTE["text"]};
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.85rem;
            margin-left: 0.4rem;
        }}
        .metric-card {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            padding: 0.85rem 1rem;
            border-radius: 12px;
        }}
        a, .stMarkdown a {{
            color: {PALETTE["primary_dark"]} !important;
            text-decoration-thickness: 2px !important;
        }}
        .stButton > button {{
            background: {PALETTE["primary"]};
            color: white;
            border-radius: 12px;
            border: 0;
        }}
        .stButton > button:hover {{
            background: {PALETTE["primary_dark"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Helpers
# =========================
def to_288_continuous(vec: List[float]) -> np.ndarray:
    """Resample any length to 288 using linear interpolation."""
    a = np.asarray(vec, dtype=float).ravel()
    if a.size == 288:
        return a
    if a.size == 0:
        return np.full(288, np.nan)
    xp = np.linspace(0, 1, a.size)
    xq = np.linspace(0, 1, 288)
    return np.interp(xq, xp, a)

def to_288_discrete(vec: List[float]) -> np.ndarray:
    """Down/upsample categorical (e.g., activity class, hypnogram) to 288 using nearest index."""
    a = np.asarray(vec).ravel()
    if a.size == 288:
        return a
    if a.size == 0:
        return np.full(288, np.nan)
    idx = np.floor(np.linspace(0, a.size - 1, 288)).astype(int)
    return a[idx]

def moving_avg(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    return pd.Series(a).rolling(k, center=True, min_periods=1).mean().to_numpy()

def normalize_series(a: np.ndarray, mode: str) -> np.ndarray:
    if mode == "z-score":
        m, s = np.nanmean(a), np.nanstd(a)
        return (a - m) / (s + 1e-8)
    if mode == "min-max":
        lo, hi = np.nanmin(a), np.nanmax(a)
        return (a - lo) / (hi - lo + 1e-8)
    return a

def time_axis_5min() -> pd.DatetimeIndex:
    # 24h with 5-min slots: 288 points
    return pd.date_range("00:00", "23:55", freq="5min")

def parse_cases(obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return id -> case dict"""
    cases = {}
    raw_cases = obj.get("cases", [])
    for c in raw_cases:
        cid = c.get("id", f"case_{len(cases)+1}")
        cases[cid] = c
    return cases

def build_case_dataframe(case: Dict[str, Any], smooth_k: int, norm_mode: str):
    # Prepare each channel → length 288 arrays
    activity_seq = to_288_discrete(case.get("activity_seq", []))
    met_5min    = to_288_continuous(case.get("met_5min", []))
    hr_seq      = to_288_continuous(case.get("sleep_hr_seq", []))
    hypno_seq   = to_288_discrete(case.get("sleep_hypno_seq", []))
    rmssd_seq   = to_288_continuous(case.get("sleep_rmssd_seq", []))

    # Optional smoothing
    met_s   = moving_avg(met_5min, smooth_k)
    hr_s    = moving_avg(hr_seq, smooth_k)
    rmssd_s = moving_avg(rmssd_seq, smooth_k)

    # Optional normalization (keep categoricals intact)
    met_s   = normalize_series(met_s, norm_mode)
    hr_s    = normalize_series(hr_s, norm_mode)
    rmssd_s = normalize_series(rmssd_s, norm_mode)

    t = time_axis_5min()
    df = pd.DataFrame({
        "time": t,
        "activity_cls": activity_seq.astype(float),
        "met": met_s.astype(float),
        "hr": hr_s.astype(float),
        "hypno": hypno_seq.astype(float),
        "rmssd": rmssd_s.astype(float),
    })
    return df

def hypno_ticks():
    # 1=deep(N3), 2=light(N1/N2), 3=REM, 4=awake
    return {1: "Deep (N3)", 2: "Light (N1/N2)", 3: "REM", 4: "Awake"}

def activity_ticks():
    # 0=non-wear, 1=rest, 2=inactive, 3=low, 4=medium
    return {0: "Non-wear", 1: "Rest", 2: "Inactive", 3: "Low", 4: "Medium"}

def default_feature_names_16() -> List[str]:
    # If you have your project-specific names, put them here (length must be 16)
    return [
        "activity_steps", "activity_high", "activity_medium", "activity_low",
        "activity_inactive", "activity_daily_movement", "activity_score", "activity_total",
        "sleep_duration", "sleep_efficiency", "sleep_rmssd", "sleep_hr_average",
        "sleep_deep", "sleep_light", "sleep_rem", "sleep_awake",
    ]

# =========================
# Plotters (Plotly)
# =========================
def plot_line(x, y, name, color, yaxis="y", dash=None):
    return go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=dict(color=color, width=2, dash=dash) if dash else dict(color=color, width=2),
        yaxis=yaxis, hovertemplate="%{x|%H:%M} — %{y:.3f}<extra>"+name+"</extra>",
    )

def plot_step(x, y, name, color, yaxis="y"):
    # step-like by using mode='lines' with shape='hv'
    return go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=dict(color=color, width=2, shape="hv"),
        yaxis=yaxis, hovertemplate="%{x|%H:%M} — %{y:.0f}<extra>"+name+"</extra>",
    )

def fig_layout(title: str, height: int = 380, legend: bool = True):
    return dict(
        title=title,
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["card"],
        margin=dict(l=40, r=20, t=60, b=40),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0) if legend else None,
        xaxis=dict(showgrid=True, gridcolor=PALETTE["border"]),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["border"]),
    )

def render_activity_panel(df: pd.DataFrame):
    x = df["time"]
    fig = go.Figure()
    fig.add_trace(plot_step(x, df["activity_cls"], "Activity class", PALETTE["primary"]))
    fig.update_layout(**fig_layout("Activity class (5-min)", height=320))
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(activity_ticks().keys()),
        ticktext=list(activity_ticks().values()),
        rangemode="tozero"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    fig2 = go.Figure()
    fig2.add_trace(plot_line(x, df["met"], "MET (smoothed/normalized)", PALETTE["primary_dark"]))
    fig2.update_layout(**fig_layout("MET (1-min → resampled)", height=320))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

def render_sleep_panel(df: pd.DataFrame):
    x = df["time"]

    fig_hr = go.Figure()
    fig_hr.add_trace(plot_line(x, df["hr"], "Sleep HR", PALETTE["primary"]))
    fig_hr.add_trace(plot_line(x, df["rmssd"], "Sleep RMSSD", PALETTE["primary_dark"], dash="dot"))
    fig_hr.update_layout(**fig_layout("Sleep: HR & RMSSD (smoothed/normalized)", height=360))
    st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

    fig_h = go.Figure()
    fig_h.add_trace(plot_step(x, df["hypno"], "Hypnogram", PALETTE["accent"]))
    fig_h.update_layout(**fig_layout("Sleep Hypnogram (5-min)", height=300))
    fig_h.update_yaxes(
        tickmode="array",
        tickvals=list(hypno_ticks().keys()),
        ticktext=list(hypno_ticks().values()),
        rangemode="tozero"
    )
    st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

def render_overview_panel(df: pd.DataFrame):
    x = df["time"]
    fig = go.Figure()
    fig.add_trace(plot_step(x, df["activity_cls"], "Activity class", PALETTE["primary"]))
    fig.add_trace(plot_line(x, df["met"], "MET", PALETTE["primary_dark"]))
    fig.add_trace(plot_line(x, df["hr"], "Sleep HR", "#8D6E63", yaxis="y2"))
    fig.add_trace(plot_step(x, df["hypno"], "Hypnogram", "#6D4C41", yaxis="y3"))
    fig.update_layout(
        **fig_layout("Overview (24h)", height=520),
        xaxis=dict(domain=[0.05, 0.98]),
        yaxis=dict(title="Activity / MET", side="left"),
        yaxis2=dict(title="HR (norm)", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="Hypno", overlaying="y", side="right", position=1.0, showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================
# App
# =========================
st.set_page_config(
    page_title="Lifelog Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.markdown(f"""
<div class="soft-header">
🧠 Lifelog-based Time-series Explorer <span class="pill">24h · 5-min bins (288)</span>
</div>
""", unsafe_allow_html=True)

# Sidebar — load JSON
with st.sidebar:
    st.markdown("### ⚙️ Config 입력")
    uploaded = st.file_uploader("Upload config JSON", type=["json"])
    config_text = st.text_area(
        "또는 JSON을 그대로 붙여넣기",
        placeholder='{\n  "cases": [ ... ]\n}',
        height=220,
    )
    # Optional: read ./config.json if exists
    use_local = st.checkbox("현재 폴더의 config.json 사용", value=False)

    # Controls
    st.markdown("---")
    st.markdown("### 🎛️ 시각화 옵션")
    smooth_k = st.slider("이동평균 창 크기 (슬롯)", 1, 21, 5, step=2)
    norm_mode = st.selectbox("정규화", ["off", "z-score", "min-max"], index=1)

# Read config
config_obj = None
if uploaded is not None:
    config_obj = json.loads(uploaded.read().decode("utf-8"))
elif use_local and Path("config.json").exists():
    config_obj = json.loads(Path("config.json").read_text(encoding="utf-8"))
elif config_text.strip():
    config_obj = json.loads(config_text)

if not config_obj:
    st.info("좌측에서 JSON을 업로드하거나 붙여넣으면, 케이스별 시계열을 바로 볼 수 있어요.")
    st.stop()

cases = parse_cases(config_obj)
if not cases:
    st.warning("`cases`가 비어 있습니다. JSON 구조를 확인해주세요.")
    st.stop()

# Case picker
case_id = st.selectbox("케이스 선택", list(cases.keys()), index=0)
case = cases[case_id]

# Small summary header
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2.4])
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Case ID", case_id)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Seq length (nominal)", "288 slots")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Smoothing", f"{smooth_k} slots")
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Normalization", norm_mode)
    st.markdown("</div>", unsafe_allow_html=True)

# Build DataFrame for selected case
df = build_case_dataframe(case, smooth_k=smooth_k, norm_mode=norm_mode)

# Tabs
tab_overview, tab_activity, tab_sleep, tab_features = st.tabs(
    ["🔎 Overview", "🏃 Activity & MET", "😴 Sleep", "📊 Daily 16 & MMSE13"]
)

with tab_overview:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    render_overview_panel(df)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_activity:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    render_activity_panel(df)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_sleep:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    render_sleep_panel(df)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_features:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    # Daily16
    d16 = case.get("daily16", [])
    names16 = default_feature_names_16()
    if len(d16) == 16:
        feat_df = pd.DataFrame({"feature": names16, "value": d16})
    else:
        feat_df = pd.DataFrame({"feature": [f"feature_{i+1}" for i in range(len(d16))], "value": d16})

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.subheader("Daily 16")
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # MMSE 13 (raw)
    mmse13 = case.get("mmse13", [])
    with c2:
        st.subheader("MMSE 13 (raw)")
        if mmse13:
            mmse_df = pd.DataFrame({"Q": [f"Q{i+1}" for i in range(len(mmse13))], "value": mmse13})
            st.dataframe(mmse_df, use_container_width=True, hide_index=True)
        else:
            st.write("값이 없습니다.")

    # Download CSV for the 24h expanded time-series
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ 24h 시계열 CSV 다운로드",
        data=csv,
        file_name=f"{case_id}_timeseries_24h.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)
