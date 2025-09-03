import streamlit as st
from datetime import datetime
import random
import time

# -------------------------
# 페이지/테마 기본 설정
# -------------------------
st.set_page_config(
    page_title="당신의 소중한 말벗 또랑이",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# 색상 팔레트 (따뜻한 주황색 테마)
# -------------------------
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

# -------------------------
# 사이드바 (글자 크기 조절)
# -------------------------
with st.sidebar:
    mood = st.selectbox(
        "오늘 기분은 어떠신가요?",
        ["😊 기쁘다", "😰 불안하다", "😭 우울하다", "☹️ 기분이 오락가락한다"],
        index=0,
        help="기분을 선택하시면 또랑이가 대화 분위기를 맞추는 데 참고해요.",
    )

    st.markdown("<br>", unsafe_allow_html=True) # 여백

    # 글자 크기 조절 옵션
    font_choice = st.radio(
        "글자 크기",
        ["보통", "크게", "아주 크게"],
        index=0,
        horizontal=True,
        key='font_size_choice'
    )

# 선택된 글자 크기에 따라 실제 픽셀 크기 결정
font_size_map = { "보통": 18, "크게": 21, "아주 크게": 24 }
BASE_FONT = font_size_map[st.session_state.font_size_choice]
CHAT_FONT = BASE_FONT + 1 # 채팅 글자는 약간 더 크게

# -------------------------
# CSS 스타일링 (노인친화적 폰트 및 디자인 적용)
# -------------------------
st.markdown(
    f"""
<style>
  /* 웹 폰트 '프리텐다드' 불러오기 */
  @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

  html, body, [class*="css"] {{
    background-color: {PALETTE['bg']} !important;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif !important;
    font-size: {BASE_FONT}px !important;
    color: {PALETTE['text']};
    font-weight: 500; /* 기본 글자 두께를 약간 더 굵게 */
  }}

  /* 메인 헤더 카드 */
  .header-card {{
    background: linear-gradient(180deg, {PALETTE['soft']} 0%, #FFFFFF 100%);
    border: 2px solid {PALETTE['border']};
    border-radius: 22px;
    padding: 24px 24px 20px 24px;
    text-align: center;
    margin-bottom: 20px;
  }}
  .header-title {{
    font-size: 2.0em; /* 상대 크기로 조절 */
    font-weight: 800;
    color: {PALETTE['text']} !important;
    line-height: 1.25;
    margin: 0;
  }}
  .header-sub {{
    font-size: 1.1em;
    color: {PALETTE['muted']};
    margin-top: 8px;
    font-weight: 600;
  }}

  /* 사이드바 */
  div[data-testid="stSidebar"] > div:first-child {{
    background-color: #FEFCF9;
    border-right: 2px solid {PALETTE['border']};
  }}
  div[data-testid="stSidebarContent"] {{
    padding: 20px 18px;
  }}

  /* 큰 시작 버튼 */
  .stButton > button {{
    width: 100%;
    min-height: 58px;
    border-radius: 16px;
    border: 0;
    font-size: 1.1em;
    font-weight: 700;
    background-color: {PALETTE['primary']};
    color: white;
    transition: all .1s ease-in;
    box-shadow: 0 3px 0 {PALETTE['primary_dark']};
  }}
  .stButton > button:hover {{
    background-color: {PALETTE['primary_dark']};
    transform: translateY(-2px);
    box-shadow: 0 5px 0 {PALETTE['primary_dark']};
  }}
  .stButton > button:active {{
    transform: translateY(1px);
    box-shadow: 0 2px 0 {PALETTE['primary_dark']};
  }}


  /* 채팅 말풍선 */
  .bubble {{
    display: inline-block; /* 내용물 크기에 맞게 조절 */
    padding: 16px 20px;
    border-radius: 20px;
    margin: 8px 0;
    max-width: 95%;
    line-height: 1.6;
    font-size: {CHAT_FONT}px;
    font-weight: 600;
    text-align: left;
  }}
  .bubble-container {{
      display: flex;
      margin-bottom: 10px;
  }}
  .bubble-container.user {{
      justify-content: flex-end;
  }}
  .bubble-container.assistant {{
      justify-content: flex-start;
  }}

  .bubble-user {{
    background-color: {PALETTE['soft']} !important;
    border: 2px solid {PALETTE['border']} !important;
    color: {PALETTE['text']} !important;
  }}
  .bubble-bot {{
    background-color: {PALETTE['primary']} !important;
    border: 2px solid {PALETTE['primary_dark']} !important;
    color: white !important;
  }}
  .role-tag {{
    font-weight: 800;
    color: {PALETTE['muted']};
    margin-bottom: 4px;
    font-size: 0.9em;
  }}
  .role-tag.user {{ text-align: right; }}
  .role-tag.assistant {{ text-align: left; }}


  /* 입력창 */
  .stChatInput > div > div {{
    border: 3px solid {PALETTE['border']};
    border-radius: 16px;
    font-size: 1.1em;
  }}
  .stChatInput > div > div:focus-within {{
    border-color: {PALETTE['primary']};
  }}

  /* 통계 타일 */
  [data-testid="metric-container"] {{
    background: {PALETTE['soft']};
    border: 2px solid {PALETTE['border']};
    border-radius: 18px;
    padding: 16px;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# 세션 메시지 초기화
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요👋 오늘 하루는 어떠셨나요? 저와 함께 소소한 이야기를 나누어요."}]

# -------------------------
# 채팅 영역
# -------------------------
# 채팅 히스토리 출력
for m in st.session_state.messages:
    role = m["role"]
    tag = "똥강아지 🐾" if role == "assistant" else "나 👤"
    container_class = "assistant" if role == "assistant" else "user"
    bubble_class = "bubble-bot" if role == "assistant" else "bubble-user"

    with st.container():
        st.markdown(f'<div class="role-tag {container_class}">{tag}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble-container {container_class}"><div class="bubble {bubble_class}">{m["content"]}</div></div>', unsafe_allow_html=True)


# -------------------------
# 입력창
# -------------------------
user_text = st.chat_input("하고 싶은 이야기를 편하게 적어주세요. (예: 오늘 점심에는 김치찌개를 먹었어.)")
if user_text:
    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": user_text})

    # AI 응답 생성 (실제로는 LLM/RAG 연결)
    with st.spinner("또랑이가 생각을 정리하고 있어요..."):
        time.sleep(random.uniform(0.5, 1.2))

    followups = [
        "그때 어떤 기분이 드셨는지 궁금해요.",
        "혹시 함께 있었던 분이 계셨나요?",
        "그 말씀을 들으니 비슷한 경험이 떠오르네요.",
        "이야기해 주셔서 정말 고마워요. 조금 더 자세히 들려주실 수 있을까요?",
        "참 재미있는 일이었겠어요! 더 듣고 싶어요.",
    ]
    answer = (
        f"'{user_text}' 라고 말씀해주셨군요. 경청해서 잘 들었어요. 😊\n\n"
        f"음... {random.choice(followups)}\n\n"
        f"혹시 그때의 기억과 관련해서 떠오르는 사진이나 메모가 있다면 함께 이야기 나눠도 좋아요. "
        f"오늘 하루도 정말 수고 많으셨어요!"
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# -------------------------
# 간단 통계
# -------------------------
# 사용자가 메시지를 한 번 이상 보냈을 때만 통계 표시
if len([m for m in st.session_state.messages if m["role"] == "user"]) > 0:
    st.markdown("### ✨ 오늘의 대화 기록")
    cols = st.columns(3)
    with cols[0]:
        st.metric("총 대화 수", f"{len(st.session_state.messages)}개")
    with cols[1]:
        st.metric("내 메시지", f"{len([m for m in st.session_state.messages if m['role']=='user'])}개")
    with cols[2]:
        st.metric("오늘의 기분", mood.split()[0])
