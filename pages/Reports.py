# pages/Lifelog_Predictor.py
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from pages.lifelog.demrisk_predictor import DementiaRiskPredictor

# -------------------------
# 페이지/테마 기본 설정
# -------------------------
st.set_page_config(
    page_title="당신의 소중한 말벗 또랑이",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Matplotlib 한글 폰트 설정: 둥글둥글 & 깔끔한 계열 우선 ---
from matplotlib import font_manager as fm
import tempfile, requests  # 인터넷 불가 환경이면 requests 부분은 자동으로 건너뜀

KOREAN_FONT_CANDIDATES = [
    # 프로젝트에 폰트를 동봉했다면 여기 경로로 추가 (권장)
    ("assets/fonts/NanumSquareRoundR.ttf", "NanumSquareRound"),
    ("assets/fonts/GowunDodum-Regular.ttf", "GowunDodum"),
    # 시스템 기본 한글 폰트들 (mac / win / linux)
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "Apple SD Gothic Neo"),
    ("C:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
    ("/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf", "NanumSquareRound"),
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic"),
]

@st.cache_resource
def get_kfont(size=12, weight="bold"):
    """Matplotlib에 한글 폰트를 등록하고 FontProperties 반환"""
    # 1) 로컬/시스템 폰트 우선
    for path, _name in KOREAN_FONT_CANDIDATES:
        try:
            p = Path(path)
            if p.exists():
                fm.fontManager.addfont(str(p))
                prop = fm.FontProperties(fname=str(p), size=size, weight=weight)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                return prop
        except Exception:
            pass

    # 2) (선택) 인터넷 가능 시 고운돋움 다운로드
    try:
        url = "https://github.com/googlefonts/gowun-dodum/raw/main/fonts/ttf/GowunDodum-Regular.ttf"
        r = requests.get(url, timeout=5)
        if r.ok:
            tmp = Path(tempfile.gettempdir()) / "GowunDodum-Regular.ttf"
            tmp.write_bytes(r.content)
            fm.fontManager.addfont(str(tmp))
            prop = fm.FontProperties(fname=str(tmp), size=size, weight=weight)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    except Exception:
        pass

    # 3) 최후의 보루(영문 폰트) - 한글 출력은 여전히 제한됨
    return fm.FontProperties(size=size, weight=weight)


# -------------------------
# 색상 팔레트 (그대로 사용)
# -------------------------
PALETTE = {
    "bg": "#FFFBF5",
    "card": "#FFFFFF",
    "border": "#FFEFE0",
    "primary": "#FF7A2F",
    "primary_dark": "#F06B20",
    "accent": "#FFDABF",
    "soft": "#FFF4EC",
    "text": "#2C2A28",
    "muted": "#756F6A",
    "success": "#34D399",
    "warning": "#FBBF24",
}

# PALETTE 아래에 추가
COLORS = {
    "activity": PALETTE["primary"],        # 활동 라인/스텝
    "met_line": PALETTE["primary_dark"],   # MET 라인
    "met_fill": PALETTE["primary_dark"],   # MET 채우기
    "hr": PALETTE["primary"],              # 수면 HR 라인
    "rmssd": PALETTE["primary_dark"],      # RMSSD 라인(점선)
    "hypno": PALETTE["accent"],            # Hypnogram
}


# -------------------------
# CSS (테마 유지 + 설명 카드만 추가)
# -------------------------
BASE_FONT = 18
st.markdown(
    f"""
<style>
  @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

  html, body, [class*="css"] {{
    background-color: {PALETTE['bg']} !important;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI',
                 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif !important;
    font-size: {BASE_FONT}px !important;
    color: {PALETTE['text']};
    font-weight: 500;
  }}

  /* 헤더 카드 */
  .header-card {{
    background: linear-gradient(180deg, {PALETTE['soft']} 0%, #FFFFFF 100%);
    border: 2px solid {PALETTE['border']};
    border-radius: 22px;
    padding: 20px 22px;
    text-align: center;
    margin-bottom: 18px;
  }}
  .header-title {{ font-size: 1.9em; font-weight: 800; color: {PALETTE['text']}; margin: 0; }}
  .header-sub   {{ font-size: 1.05em; color: {PALETTE['muted']}; margin-top: 6px; font-weight: 600; }}

  /* 설명 박스 */
  .note {{
    background:{PALETTE["soft"]};
    border:2px solid {PALETTE["border"]};
    border-radius:14px;
    padding:12px 14px;
    margin-top:10px;
    color:{PALETTE["text"]};
  }}
</style>
""",
    unsafe_allow_html=True,
)

# Matplotlib 경량 테마
plt.rcParams.update({
    "figure.facecolor": PALETTE["card"],
    "axes.facecolor":   PALETTE["card"],
    "axes.edgecolor":   PALETTE["border"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["muted"],
    "ytick.color":      PALETTE["muted"],
    "grid.color":       PALETTE["border"],
    "text.color":       PALETTE["text"],
    "axes.titleweight": "semibold",
})

# ---------------------------
# 유틸 (길이 288 보장)
# ---------------------------
def _to_288_continuous(vec: List[float]) -> np.ndarray:
    a = np.asarray(vec, dtype=float).ravel()
    if a.size == 0: return np.full(288, np.nan)
    if a.size == 288: return a
    xp = np.linspace(0, 1, a.size)
    xq = np.linspace(0, 1, 288)
    return np.interp(xq, xp, a)

def _to_288_discrete(vec: List[float]) -> np.ndarray:
    a = np.asarray(vec).ravel()
    if a.size == 0: return np.full(288, np.nan)
    if a.size == 288: return a
    idx = np.floor(np.linspace(0, a.size - 1, 288)).astype(int)
    return a[idx]

def _time_axis_5min() -> pd.DatetimeIndex:
    return pd.date_range("00:00", "23:55", freq="5min")

HYPNO = {1:"Deep(N3)", 2:"Light", 3:"REM", 4:"Awake"}
ACTCL = {0:"Non-wear", 1:"Rest", 2:"Inactive", 3:"Low", 4:"Medium"}

def case_to_df(case: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "time":        _time_axis_5min(),
        "activity_cls":_to_288_discrete(case.get("activity_seq", [])).astype(float),
        "met":         _to_288_continuous(case.get("met_5min",   [])).astype(float),
        "hr":          _to_288_continuous(case.get("sleep_hr_seq", [])).astype(float),
        "hypno":       _to_288_discrete(case.get("sleep_hypno_seq", [])).astype(float),
        "rmssd":       _to_288_continuous(case.get("sleep_rmssd_seq",[])).astype(float),
    })

# ---------------------------
# Plotters
# ---------------------------
def _beautify_axes(ax):
    ax.grid(True, alpha=0.65, linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_activity_tab(df: pd.DataFrame):
    """
    Activity 탭: Activity class(step) + MET(line)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = df["time"]

    # 1) Activity class (step)
    axes[0].step(
        x, df["activity_cls"], where="post",
        color=COLORS["activity"], linewidth=2.0, label="Activity class"
    )
    axes[0].set_ylabel("Activity")
    axes[0].set_yticks(list(ACTCL.keys()), list(ACTCL.values()))
    axes[0].legend(loc="upper right")
    _beautify_axes(axes[0])

    # 2) MET (line + soft fill)
    axes[1].plot(
        x, df["met"],
        color=COLORS["met_line"], linewidth=2.0, label="MET"
    )
    axes[1].fill_between(
        x, df["met"], df["met"].min(),
        color=COLORS["met_fill"], alpha=0.08
    )
    axes[1].set_ylabel("MET")
    axes[1].legend(loc="upper right")
    _beautify_axes(axes[1])

    # X-axis formatting
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time of day")

    fig.suptitle("Activity (24h · 5-min bins)", y=0.995)
    st.pyplot(fig, clear_figure=True)

    st.markdown(f"""
<div class="note">
• <b>Activity Class</b>: 0(Non-wear)–4(Medium) 단계의 활동 수준(5분 간격)<br/>
• <b>MET</b>: 대사당량 추정치(선이 높을수록 에너지 소비↑)<br/>
</div>
""", unsafe_allow_html=True)

def plot_sleep_tab(df: pd.DataFrame):
    """
    Sleep 탭: HR & RMSSD(line) + Hypnogram(step)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = df["time"]

    # 1) HR & RMSSD (lines + soft fill)
    axes[0].plot(
        x, df["hr"],
        color=COLORS["hr"], linewidth=2.0, label="Sleep HR"
    )
    axes[0].plot(
        x, df["rmssd"],
        color=COLORS["rmssd"], linewidth=2.0, linestyle="-.", label="RMSSD"
    )
    axes[0].fill_between(
        x, df["hr"], df["hr"].min(),
        color=COLORS["hr"], alpha=0.06
    )
    axes[0].fill_between(
        x, df["rmssd"], df["rmssd"].min(),
        color=COLORS["rmssd"], alpha=0.04
    )
    axes[0].set_ylabel("HR / RMSSD")
    axes[0].legend(loc="upper right")
    _beautify_axes(axes[0])

    # 2) Hypnogram (step)
    axes[1].step(
        x, df["hypno"], where="post",
        color=COLORS["rmssd"], linewidth=2.0, label="Hypnogram"
    )
    axes[1].set_ylabel("Stage")
    axes[1].set_yticks(list(HYPNO.keys()), list(HYPNO.values()))
    axes[1].legend(loc="upper right")
    _beautify_axes(axes[1])

    # X-axis formatting
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time of day")

    fig.suptitle("Sleep (24h · 5-min bins)", y=0.995)
    st.pyplot(fig, clear_figure=True)

    st.markdown(f"""
<div class="note">
• <b>Sleep HR</b>: 수면 중 심박수(낮을수록 안정)<br/>
• <b>RMSSD</b>: 심박변이도 지표(높을수록 회복/자율신경 유연성 양호)<br/>
• <b>Hypnogram</b>: Deep → Light → REM → Awake 단계 분포
</div>
""", unsafe_allow_html=True)



####################################################################################
# 레포트 관련 함수들
####################################################################################

# --- 새로 추가: 5각형(요약 레이더) 그림 생성 ---
def make_pentagon(scores, labels=None):
    """
    scores: 길이 5, 각 항목 0~1.
    labels: 각 꼭짓점 라벨(옵션).
    """
    scores = np.clip(np.asarray(scores, dtype=float), 0, 1)

    # 위쪽이 꼭짓점이 되도록 +π/2 회전
    n = 5
    ang = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/2
    outer = np.column_stack([np.cos(ang), np.sin(ang)])         # 단위 오각형
    inner = outer * scores[:, None]                             # 점수 반영 오각형

    fp = get_kfont(size=12, weight="bold")
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=180)

    # 동심 오각형 + 방사선
    for r in [0.25, 0.5, 0.75, 1.0]:
        ring = outer * r
        ax.plot(
            np.r_[ring[:,0], ring[0,0]], np.r_[ring[:,1], ring[0,1]],
            color=PALETTE["border"], linewidth=0.9
        )
    for i in range(n):
        ax.plot([0, outer[i,0]], [0, outer[i,1]],
                color=PALETTE["border"], linewidth=0.8)

    # 바깥 폴리곤
    ax.fill(np.r_[outer[:,0], outer[0,0]], np.r_[outer[:,1], outer[0,1]],
            facecolor=PALETTE["soft"], edgecolor=PALETTE["accent"], linewidth=2)

    # 점수 폴리곤
    ax.fill(np.r_[inner[:,0], inner[0,0]], np.r_[inner[:,1], inner[0,1]],
            facecolor=PALETTE["primary"], alpha=0.32,
            edgecolor=PALETTE["primary_dark"], linewidth=2)

    # 라벨 (한글 폰트 적용)
    for i, lab in enumerate(labels):
        ax.text(
            outer[i,0]*1.17, outer[i,1]*1.17, str(lab),
            ha="center", va="center", fontproperties=fp,
            color=PALETTE["muted"]
        )

    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    return fig


# --- 새로 추가: Reports 탭 렌더러 ---
def plot_report_tab(df: pd.DataFrame, cases: Dict[str, Any], case_id: str):
    """
    오른쪽 상단: 오각형(요약 레이더)
    왼쪽: 문장이 여러 줄 들어가는 DataFrame 요약
    - JSON에 선택적으로 다음 키를 둘 수 있어요:
      "report_text": ["문장1 ...", "문장2 ..."],
      "report_scores": [0.6, 0.7, 0.5, 0.8, 0.4]  # 0~1, 5개
    """
    left, right = st.columns([0.60, 0.40], gap="large")

    # -------- 왼쪽: 텍스트 DataFrame --------
    with left:
        st.subheader("텍스트 리포트")
        texts = cases[case_id].get("report_text")

        if texts and isinstance(texts, (list, tuple)):
            rows = [{"항목": f"노트 {i+1}", "설명": str(t)} for i, t in enumerate(texts)]
        else:
            # 기본 요약 자동 생성(간결)
            mm = cases[case_id].get("mmse13", [])
            mm_total = int(np.nansum(mm)) if len(mm) else 0
            mm_max = int(2 * len(mm)) if len(mm) else 0
            rows = [
                {"항목": "활동 요약",
                 "설명": f"평균 활동 단계 {np.nanmean(df['activity_cls']):.2f}/4, MET 평균 {np.nanmean(df['met']):.2f}."},
                {"항목": "수면/자율신경",
                 "설명": f"수면 HR {np.nanmean(df['hr']):.1f} bpm, RMSSD {np.nanmean(df['rmssd']):.1f} ms."},
                {"항목": "수면 단계",
                 "설명": f"REM 비율 {np.nanmean(df['hypno']==3)*100:.1f}%."},
                {"항목": "MMSE",
                 "설명": f"총점 {mm_total} / {mm_max}."}
            ]

        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

    # -------- 오른쪽: 오각형(상단 배치) --------
    with right:
        st.subheader("DSM-5 인지기능 요약")
        # 점수: JSON 제공 없으면 간단 정규화로 계산
        scores = cases[case_id].get("report_scores")
        if not (isinstance(scores, (list, tuple)) and len(scores) == 5):
            act_s  = float(np.nanmean(df["activity_cls"]) / 4.0)                                   # ↑양호
            met_s  = float(np.clip((np.nanmean(df["met"]) - 1.0) / (5.5 - 1.0), 0, 1))             # ↑양호
            hr_s   = float(1.0 - np.clip((np.nanmean(df["hr"]) - 50.0) / (90.0 - 50.0), 0, 1))     # ↓양호
            rmssd_s= float(np.clip((np.nanmean(df["rmssd"]) - 15.0) / (70.0 - 15.0), 0, 1))        # ↑양호
            rem_s  = float(np.clip((np.nanmean(df["hypno"] == 3) - 0.10) / (0.30 - 0.10), 0, 1))   # ↑양호
            scores = [act_s, met_s, hr_s, rmssd_s, rem_s]

        fig = make_pentagon(scores, labels=["기억력", "언어능력", "정서적 안정성", "계산능력", "시공간 파악 능력"])
        # 오른쪽 상단 배치: 상단에 바로 렌더링(이 열에서 첫 컴포넌트로 표시)
        st.pyplot(fig, use_container_width=True)

####################################################################################


# ---------------------------
# JSON 로드 & normal 선택
# ---------------------------



HERE = Path(__file__).resolve()
default_json = HERE.with_name("user_config2.json")

cfg: Dict[str, Any] = {}
if default_json.exists():
    cfg = json.loads(default_json.read_text(encoding="utf-8"))
else:
    up = st.file_uploader("user_config2.json 업로드", type=["json"])
    if up:
        cfg = json.loads(up.read().decode("utf-8"))
if not cfg:
    st.stop()

raw_cases = cfg.get("cases", [])
cases = {c.get("id", f"case_{i+1}"): c for i, c in enumerate(raw_cases)}
if not cases:
    st.error("`cases`가 비어 있습니다. user_config2.json을 확인해 주세요.")
    st.stop()

case_id = "normal" if "normal" in cases else next(iter(cases.keys()))
df = case_to_df(cases[case_id])

# ---------------------------
# 헤더 + 탭 2개
# ---------------------------
tab_act, tab_sleep, tab_report = st.tabs(["🏃 Activity", "😴 Sleep", "🔖 Reports"])
with tab_act:
    plot_activity_tab(df)
with tab_sleep:
    plot_sleep_tab(df)
with tab_report:
    plot_report_tab(df, cases, case_id)



##################################################################################
# --- MMSE 칩/위험도 게이지용 CSS 보강 ---
st.markdown(f"""
<style>
  .chip {{
    display:inline-block; padding:6px 10px; border-radius:999px;
    font-weight:700; font-size:0.9em; line-height:1; margin-left:8px;
    border:2px solid {PALETTE["border"]};
  }}
  .chip.ok    {{ background:{PALETTE["accent"]}; color:{PALETTE["text"]}; }}
  .chip.warn  {{ background:{PALETTE["soft"]};   color:{PALETTE["text"]}; }}
  .chip.bad   {{ background:#FFE5E5; color:#9B1C1C; border-color:#FECACA; }}

  .mmse-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 12px; border:1.5px solid {PALETTE["border"]};
    background:{PALETTE["card"]}; border-radius:12px; margin-bottom:8px;
  }}
  .mmse-q {{ font-weight:700; color:{PALETTE["text"]}; }}
  .mmse-score {{ color:{PALETTE["muted"]}; font-weight:700; }}

  .progress {{
    width:100%; height:22px; border-radius:14px; overflow:hidden;
    border:2px solid {PALETTE["border"]}; background:{PALETTE["card"]};
  }}
  .bar {{
    height:100%;
    background: linear-gradient(90deg, {PALETTE["accent"]} 0%, {PALETTE["primary"]} 70%);
    width:0%;
  }}
  .risk-card {{
    background: linear-gradient(180deg, {PALETTE["soft"]} 0%, #FFFFFF 100%);
    border:2px solid {PALETTE["border"]}; border-radius:16px; padding:16px;
  }}
</style>
""", unsafe_allow_html=True)


# --- MMSE 질문 레이블(예시 13개) ---
MMSE_LABELS = [
    "시간 지남력", "장소 지남력", "주의집중/계산"
]

def render_mmse_panel(mm: List[float], top_k: int = 3):
    """왼쪽: MMSE 상위 N개 항목을 칩과 함께 요약 표기"""
    if not mm:
        st.info("MMSE 데이터가 없습니다.")
        return
    total = float(np.nansum(mm))
    maxpt = 2.0 * len(mm)  # 점수 스케일(예: 항목별 0~2)
    st.caption(f"총점: **{int(total)} / {int(maxpt)}**")

    # 앞에서부터 top_k만 표시
    for i, score in list(enumerate(mm))[:top_k]:
        status = "정답" if score >= 2 else ("부분 정답" if score == 1 else "오답")
        cls = "ok" if score >= 2 else ("warn" if score == 1 else "bad")
        st.markdown(
            f'<div class="mmse-row">'
            f'  <div class="mmse-q">{MMSE_LABELS[i] if i < len(MMSE_LABELS) else f"문항 {i+1}"}</div>'
            f'  <div>'
            f'    <span class="mmse-score">{int(score)}/2</span>'
            f'    <span class="chip {cls}">{status}</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

@st.cache_resource
def get_dementia_predictor():
    base = Path(__file__).resolve().parent
    return DementiaRiskPredictor(
        ts_model_path=str(base / "best_dementia_model_full.pth"),
        mmse_model_path=str(base / "mmse_rf.pkl"),
        ts_weight=0.60,
        threshold=0.45,
    )

# [REPLACE] 휴리스틱 대신 실제 모델로 예측
def _pad_or_trim(arr, length, fill=np.nan, dtype=float):
    a = np.asarray(arr, dtype=dtype).ravel()
    if a.size < length:
        a = np.concatenate([a, np.full(length - a.size, fill, dtype=dtype)])
    elif a.size > length:
        a = a[:length]
    return a

def predict_risk_with_model(case: Dict[str, Any]) -> float:
    """
    DementiaRiskPredictor로 위험 확률[0~1] 반환.
    입력은 user_config2.json의 'normal' 케이스 딕셔너리 그대로 사용.
    길이가 맞지 않으면 안전하게 패드/자릅니다.
    """
    predictor = get_dementia_predictor()

    a   = _pad_or_trim(case.get("activity_seq",    []), 288)   # int로 캐스팅은 predictor 내부 전처리에서 수행
    m   = _pad_or_trim(case.get("met_5min",        []), 288)
    hr  = _pad_or_trim(case.get("sleep_hr_seq",    []), 288)
    hy  = _pad_or_trim(case.get("sleep_hypno_seq", []), 288)
    rm  = _pad_or_trim(case.get("sleep_rmssd_seq", []), 288)
    d16 = _pad_or_trim(case.get("daily16",         []),  16)
    mm  = _pad_or_trim(case.get("mmse13",          []),  13)

    out = predictor.predict_one(
        activity_seq=a,
        met_5min=m,
        sleep_hr_seq=hr,
        sleep_hypno_seq=hy,
        sleep_rmssd_seq=rm,
        daily16=d16,
        mmse13=mm,
    )
    return float(out["risk_probability"])



# ---------------------------
# 하단 요약: MMSE ↔ 위험도
# ---------------------------
st.markdown("---")
left, right = st.columns(2, gap="large")

with left:
    render_mmse_panel(cases[case_id].get("mmse13", []), top_k=3)

with right:
    risk = predict_risk_with_model(cases[case_id])
    pct = int(round(risk * 100))

    # 숫자 + 게이지
    st.markdown(f'<div class="risk-card"><h3 style="margin:0 0 8px 0;">위험도: <b>{pct}%</b></h3>', unsafe_allow_html=True)
    st.markdown('<div class="progress"><div class="bar" id="riskbar"></div></div></div>', unsafe_allow_html=True)
    # 진행바 폭을 동적으로 설정
    st.markdown(f"""
    <script>
      const el = window.parent.document.getElementById('riskbar');
      if (el) {{ el.style.width = '{pct}%'; }}
    </script>
    """, unsafe_allow_html=True)

    # 간단 해석
    label = "낮음" if pct < 30 else ("보통" if pct < 60 else "높음")
    st.caption(f"현재 추정치는 **{label}** 범위입니다. 실제 진단/평가는 의료진 상담을 권장합니다.")





