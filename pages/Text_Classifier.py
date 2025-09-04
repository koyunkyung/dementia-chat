import streamlit as st
import pandas as pd
# -------------------------
# 페이지/테마 기본 설정
# -------------------------
st.set_page_config(
    page_title="당신의 소중한 말벗 또랑이",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar 미니 스타일 (배경 · 네비 링크 · 선택 강조) ---
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
    "success": "#34D399",     # 대비를 위한 초록색
    "warning": "#FBBF24",     # 대비를 위한 노란색
}
st.sidebar.markdown(
    f"""
<style>
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


st.title("Fact Memory")
st.markdown("Check How the Fact Memory Operates")

# -------------------------
# CSS 스타일 주입 (새로운 폰트, 색상, 글씨 크기 적용)
# -------------------------
st.markdown(
    f"""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

    html, body, [class*="css"] {{
      background-color: {PALETTE['bg']} !important;
      font-family: 'Pretendard', sans-serif !important;
      color: {PALETTE['text']};
    }}

    .fact-title {{
      font-size: 1.8em; /* 크기 조정 */
      font-weight: 1000;
      color: {PALETTE['text']}; /* 주요 색상 */
      line-height: 1.7;
      margin-bottom: 0.5em;
    }}
    .tag-title {{
      font-size: 1.1em; /* 크기 조정 */
      font-weight: 600;
      color: {PALETTE['muted']};
      margin-bottom: 0.8em;
    }}
    .fact-tag-text {{
        font-size: 1em; /* 태그 내용 글씨 크기 조정 */
        font-weight: 500;
        line-height: 1.6;
    }}

    /* 타임스탬프 박스 스타일 */
    .timestamp-box {{
        background-color: #E5E5E5; /* 회색 톤으로 변경 */
        color: {PALETTE['text']};
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 0.9em;
        font-weight: 500;
        margin-top: 10px;
        display: inline-block;
    }}

    /* 코드 글씨체 및 스타일 모방 (original_utterance_raw/std 부분) */
    .code-like-block {{
        background-color: {PALETTE['soft']};
        border-left: 5px solid {PALETTE['primary']};
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 5px;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: 0.95em;
        color: {PALETTE['text']};
        line-height: 1.5;
    }}
    .code-like-block strong {{
        color: {PALETTE['primary_dark']};
    }}
    .code-like-block em {{
        color: {PALETTE['muted']};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# CSV 파일 읽기
try:
    df = pd.read_csv('fact_memory.csv')
except FileNotFoundError:
    st.error("`fact_memory.csv` 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    st.stop()
    
# 각 행을 반복하며 정보 카드 생성
for index, row in df.iterrows():
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # 좌측에 타임스탬프와 태그를 배치
        st.markdown(f'<p class="tag-title">Tag</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="fact-tag-text"><strong>타입:</strong> {row["type"]} <br><strong>주제:</strong> {row["topic"]} <br><strong>시간:</strong> {row["time_reference"]}</p>', unsafe_allow_html=True)
        
        # 시간은 시까지만 표시
        timestamp_formatted = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
        st.markdown(f'<div class="timestamp-box">⏰ {timestamp_formatted}</div>', unsafe_allow_html=True)
        
    with col2:
        # 우측에 주요 정보(원문, 번역, 요약)를 배치
        st.markdown(f'<p class="fact-title">fact_{index+1}: {row["summary"]}</p>', unsafe_allow_html=True)

        
        # 서비스 내부에서 일어나고 있는 것을 보여주는 코드 스타일 모방
        st.markdown("**원문 (사투리):**")
        st.markdown(f"<div class='code-like-block'><strong>{row['original_utterance_raw']}</strong></div>", unsafe_allow_html=True)
        
        # 표준어 번역
        st.markdown("**번역 (표준어):**")
        st.markdown(f"<div class='code-like-block'><em>{row['original_utterance_std']}</em></div>", unsafe_allow_html=True)

    st.markdown("---") # 각 팩트 카드를 구분하는 선
