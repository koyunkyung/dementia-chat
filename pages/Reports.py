# pages/Lifelog_Predictor.py
import json
from torch import nn
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

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
st.sidebar.markdown(
    f"""
<style>
/* 코드 글씨체 및 스타일 모방 (original_utterance_raw/std 부분) */
.code-like-block {{
    background-color: {PALETTE['soft']};
    border-left: 5px solid {PALETTE['primary']};
    padding: 10px 15px;
    margin-bottom: 10px;
    border-radius: 5px;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 0.95em;
    color: #000;
    line-height: 1.5;
}}
.code-like-block strong {{
    color: {PALETTE['primary_dark']};
}}
.code-like-block em {{
    color: {PALETTE['muted']};
}}

/* 사이드바 전체 배경 & 경계선 */
section[data-testid="stSidebar"] {{
  background: {PALETTE['soft']};
  border-right: 1px solid {PALETTE['border']};
}}

/* 네비 리스트 패딩 */
[data-testid="stSidebarNav"] ul {{ padding: 8px 10px; }}

/* 네비 링크 기본 모양 */
[data-testid="stSidebarNav"] a {{
  display:block;
  padding:12px 14px;
  border-radius:14px;
  font-weight:800;
  color:{PALETTE['text']};
  border:2px solid transparent;
  transition:all .15s ease;
}}

/* 호버 */
[data-testid="stSidebarNav"] a:hover {{
  background:{PALETTE['soft']};
  border-color:{PALETTE['border']};
  transform:translateX(2px);
}}

/* 현재 선택된 페이지 */
[data-testid="stSidebarNav"] a[aria-current="page"] {{
  background:linear-gradient(180deg, {PALETTE['soft']} 0%, #FFFFFF 100%);
  border:2px solid {PALETTE['border']};
  box-shadow:0 6px 18px rgba(255,122,47,.08);
  color:{PALETTE['text']};
}}

/* 상단 작은 브랜드 박스(선택 사항) */
.sidebar-brand {{
  background:#FFFFFF;
  border:2px solid {PALETTE['border']};
  border-radius:16px;
  padding:14px 16px;
  margin:12px 12px 6px;
  font-weight:900;
}}
.sidebar-brand .badge {{
  display:inline-block; padding:4px 8px; border-radius:999px;
  background:{PALETTE['accent']}; color:{PALETTE['text']};
  border:1.5px solid {PALETTE['border']}; font-size:.8em; font-weight:800;
}}
</style>

""",
    unsafe_allow_html=True,
)

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

st.markdown(f"""
<style>
  /* --- 위험도 카드 높이/정렬 고정 --- */
  .risk-card {{
    background: linear-gradient(180deg, {PALETTE["soft"]} 0%, #FFFFFF 100%);
    border:2px solid {PALETTE["border"]};
    border-radius:16px;
    padding:18px 22px;
    min-height:120px;                 /* ← 두 카드 높이 동일 */
    display:flex;                      /* 세로 배치 고정 */
    flex-direction:column;
    gap:10px;
    box-shadow:0 8px 20px rgba(255,122,47,0.06);
  }}
  .risk-head {{                         /* 제목 ↔ 칩 가로 정렬 */
    display:flex; justify-content:space-between; align-items:center;
  }}
  .risk-title {{
    font-weight:800; font-size:1.06em; color:{PALETTE['text']};
  }}
  .chip.pct {{                          /* 퍼센트 칩 폭 고정 */
    min-width:66px; text-align:center;
    background:{PALETTE['accent']}; border-color:{PALETTE['border']};
  }}
  .progress {{                          /* 트랙 두께/모서리 통일 */
    width:100%; height:22px; border-radius:14px; overflow:hidden;
    border:2px solid {PALETTE["border"]}; background:{PALETTE["card"]};
  }}
  .bar {{
    height:100%;
    background: linear-gradient(90deg, {PALETTE["accent"]} 0%, {PALETTE["primary"]} 70%);
    width:0%;
  }}
</style>
""", unsafe_allow_html=True)


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
        st.caption("마지막 발화 분석 결과")
        st.markdown(f"<div class='code-like-block'>아침에 눈 뜨니 허리가 살짝 뻐근하네. 따뜻한 물 한 컵 천천히 마셨지. 달력에 표시해 둔 돌잔치 다시 한 번 확인했어. 혈압 재고, 약도 빠뜨리지 않고 챙겨 먹었어. 택시 부를까, 버스 탈까 잠깐 고민했지.</div>", unsafe_allow_html=True)

        # --- 9칸 메트릭 그리드 CSS (PALETTE 색감과 통일) ---
        st.markdown(f"""
        <style>
        .metric-grid {{
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 8px;
        }}
        .metric-card {{
        background: #FFFFFF; border: 2px solid {PALETTE['border']};
        border-radius: 16px; padding: 12px; text-align: center;
        box-shadow: 0 6px 18px rgba(255,122,47,.08); transition: transform .12s ease, box-shadow .12s ease;
        }}
        .metric-card:hover {{
        transform: translateY(-2px); box-shadow: 0 10px 24px rgba(255,122,47,.12);
        }}
        .metric-label {{
        font-size: 0.92em; color: {PALETTE['muted']}; font-weight: 800;
        }}
        .metric-value {{
        margin-top: 6px; display: inline-block; padding: 6px 10px; 
        color: {PALETTE['text']};font-weight: 900;
        }}
        </style>
        """, unsafe_allow_html=True)

        # --- 9개 항목 (예시 값) ---
        metrics = [
            ("사건 구체성", "0"),
            ("자서전적 기억 변수", "1"),
            ("같은 말 반복", "0"),
            ("시간적 구체성", "1"),
            ("공간적 구체성", "1"),
            ("우울/무기력", "0"),
            ("불안/초조", "0"),
            ("감정 조절 문제", "0"),
            ("", ""),
        ]
        # --- HTML 렌더 ---
        grid_html = "<div class='metric-grid'>" + "".join(
            f"<div class='metric-card'><div class='metric-label'>{k}</div><div class='metric-value'>{v}</div></div>"
            for k, v in metrics
        ) + "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

    # -------- 오른쪽: 오각형(상단 배치) --------
    with right:
        st.markdown(
            "<h3 style='text-align:center; margin: 0;'>DSM-5 인지기능 요약</h3>",
            unsafe_allow_html=True
        )
        # 점수: JSON 제공 없으면 간단 정규화로 계산
        scores = cases[case_id].get("report_scores")
        if not (isinstance(scores, (list, tuple)) and len(scores) == 5):
            act_s  = 0.5                                   # ↑양호
            met_s  = 2             # ↑양호
            hr_s   = 2
            rmssd_s= 2
            rem_s  = 2
            scores = [act_s, met_s, hr_s, rmssd_s, rem_s]

        fig = make_pentagon(scores, labels=["기억력", "언어능력", "정서적 안정성", "계산능력", "시공간\n파악 능력"])
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

# ─────────────────────────────────────────────────────────────
# BeHealthy 탭: 위험도 카드 + 저위험군 코칭(운동/식단/두뇌활동 + 오늘의 미션)
# ─────────────────────────────────────────────────────────────

def render_low_risk_tips(lifelog_pct: float, speech_pct: float, cutoff: int = 40):
    """저위험군(둘 다 cutoff 미만)에게 긍정 강화 + 생활 습관 팁 제공"""
    is_low = (lifelog_pct < cutoff) and (speech_pct < cutoff)
    if not is_low:
        return

    # 날짜 고정 랜덤: 하루에 한 문장/미션 고정
    seed = int(pd.Timestamp.today().strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)

    positive_lines = [
        "지금처럼만 유지하면 충분해요. 작은 루틴이 큰 차이를 만듭니다!",
        "아주 좋아요! 오늘도 뇌가 좋아하는 생활 한 가지를 선택해볼까요?",
        "안정적인 패턴이 보입니다. 스스로를 칭찬해주세요 🙌",
    ]
    daily_missions = [
        "가벼운 스트레칭 10분",
        "빠르게 걷기 15분",
        "채소 2가지 이상 곁들이기",
        "설탕 음료 대신 물 2잔 더 마시기",
        "크로스워드/스도쿠 1판",
        "친구/가족과 통화 10분",
    ]
    st.markdown(f"<div class='coach-msg'>{rng.choice(positive_lines)}</div>", unsafe_allow_html=True)

    tab_ex, tab_food, tab_brain = st.tabs(["💪 운동", "🥗 식단", "🧠 두뇌활동"])

    with tab_ex:
        st.markdown(
            "<ul class='tip-ul'>"
            "<li>하루 총 <b>6,000~8,000보</b> 또는 <b>중강도 20–30분</b> 목표</li>"
            "<li>앉아있는 시간이 길면 <b>한 시간마다 2–3분</b> 일어나 움직이기</li>"
            "<li>수면 3시간 전 격한 운동은 피하고, 낮 시간대에 활동량 확보</li>"
            "</ul>",
            unsafe_allow_html=True,
        )
    with tab_food:
        st.markdown(
            "<ul class='tip-ul'>"
            "<li><b>채소·통곡물·견과류</b> 위주의 간단한 지중해식 구성</li>"
            "<li>가공육/과도한 당류는 <b>주 2회 이하</b>로 줄이기</li>"
            "<li>저녁은 가볍게, 취침 3시간 전 과식 피하기</li>"
            "</ul>",
            unsafe_allow_html=True,
        )
    with tab_brain:
        st.markdown(
            "<ul class='tip-ul'>"
            "<li><b>새로운 것</b>을 배우는 짧은 활동(예: 악보/단어/퍼즐)</li>"
            "<li>양손을 쓰는 과제(요리·정리·간단한 악기)로 <b>집중+협응</b> 자극</li>"
            "<li>하루 한 번 <b>대화/전화</b>로 사회적 상호작용 유지</li>"
            "</ul>",
            unsafe_allow_html=True,
        )

    # 오늘의 미션(칩 스타일 재활용)
    mission = rng.choice(daily_missions)
    st.markdown(
        f"<div style='margin-top:10px;'>"
        f"<span class='chip ok'>오늘의 미션</span> "
        f"<span class='chip'>{mission}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

def render_risk_card(title: str, percent: float):
    pct = int(np.clip(percent, 0, 100))
    st.markdown(
        f"""
<div class="risk-card">
  <div class="risk-head">
    <div class="risk-title">{title}</div>
    <span class="chip pct ok">{pct}%</span>
  </div>
  <div class="progress"><div class="bar" style="width:{pct}%;"></div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_behealthy_tab(lifelog_pct: int, speech_pct: int, cutoff: int = 40):

    # 저위험군이면 코칭 카드 노출 (기존 render_low_risk_tips 사용)
    render_low_risk_tips(lifelog_pct, speech_pct, cutoff=cutoff)

    # 저위험군이 아니면 간단 안내만
    if not ((lifelog_pct < cutoff) and (speech_pct < cutoff)):
        st.info(f"현재 기준(cutoff={cutoff}%)으로 저위험군이 아닙니다. "
                "그래도 생활 습관 관리가 가장 중요해요! 위험도가 낮아지면 맞춤 코칭이 자동 표시됩니다.")

# ---------------------------
# 헤더 + 탭 2개
# ---------------------------
tab_act, tab_sleep, tab_report, tab_health = st.tabs(
    ["🏃 Activity", "😴 Sleep", "🔖 Reports", "🌿 BeHealthy"]
)

with tab_act:
    plot_activity_tab(df)

with tab_sleep:
    plot_sleep_tab(df)

with tab_report:
    plot_report_tab(df, cases, case_id)

with tab_health:
    # 위험도 값은 실제 예측 결과가 있으면 그 값을 쓰고, 없으면 기본값 사용
    lifelog_pct = int(cases[case_id].get("risk_lifelog_pct", 32))
    speech_pct  = int(cases[case_id].get("risk_speech_pct", 28))
    render_behealthy_tab(lifelog_pct, speech_pct, cutoff=40)



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


# ─────────────────────────────────────────────────────────────
# 저위험군 코칭 위젯
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .coach-card {{
    background: linear-gradient(180deg, {PALETTE["soft"]} 0%, #FFFFFF 100%);
    border: 2px solid {PALETTE["border"]};
    border-radius: 16px;
    padding: 16px;
    margin-top: 12px;
    box-shadow: 0 8px 20px rgba(255,122,47,0.06);
  }}
  .coach-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }}
  .coach-title {{ font-weight:900; color:{PALETTE["text"]}; }}
  .coach-badge {{
    display:inline-block; padding:6px 10px; border-radius:999px;
    background:{PALETTE["accent"]}; border:2px solid {PALETTE["border"]};
    font-weight:800; color:{PALETTE["text"]};
  }}
  .coach-msg {{ color:{PALETTE["muted"]}; font-weight:700; margin:6px 0 10px; }}
  .tip-ul {{ margin: 0 0 6px 0; padding-left: 18px; }}
</style>
""", unsafe_allow_html=True)




# ---------------------------
# 하단 요약: MMSE ↔ 위험도
# ---------------------------
st.markdown("---")
left, right = st.columns(2, gap="large")

with left:
    render_mmse_panel(cases[case_id].get("mmse13", []), top_k=3)

with right:
    render_risk_card("라이프로그 데이터로 예측한 치매 위험도", 32)
    render_risk_card("발화 데이터로 예측한 치매 위험도", 28)





