
# Streamlit Wrapper for Dementia Chat (Modular)

이 폴더에는 기존 `main.py`, `model.py`를 그대로 사용하면서 **Streamlit** UI로 동작하도록 하는 `streamlit_app.py`가 추가되어 있습니다.

## 설치

```bash
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

> `requirements.txt`는 원본 의존성이고, `requirements_streamlit.txt`에는 `streamlit`만 추가되어 있습니다.

## 실행

```bash
# 이 디렉토리( main.py 가 있는 곳 )에서
streamlit run streamlit_app.py
```

## 주요 기능

- 처음 1턴은 **주의력(Attention)** 질문 자동 제시/채점
- 사용자 발화 → **사투리→표준어 정규화** → **사실 추출/일관성 판정** → **공감형 후속질문 생성**
- **일기장**: 체크리스트 5문항 → 자동 주제 3개 → 주제별 Q&A → 1문장 요약 저장
- **CSV 내보내기**: `fact_memory.csv`, `diary_sessions.csv`, `diary_messages.csv`
- 전체 대화 로그를 DataFrame으로 확인

## 환경 변수

- `OPENAI_API_KEY`를 반드시 설정해야 합니다. (좌측 사이드바에서도 입력 가능)
- 필요 시 `CONFIG_TODAY_OVERRIDE`를 `YYYY-MM-DD` 형태로 설정해 날짜 채점 기준을 고정할 수 있습니다.

## 폴더 구조

```
dementia-chat-modular/
├── main.py
├── model.py
├── run.py
├── prompts/
│   └── few_shot_empathy.txt
└── streamlit_app.py   ← (추가)
```

## 메모
- 본 Streamlit 앱은 **기존 함수들을 그대로 호출**하며, `input()`을 사용하지 않도록 UI를 재구성했습니다.
- 원본 코드의 FAISS/SentenceTransformer 초기화 방식 그대로 동작합니다. 최초 실행 시 모델/의존성 설치로 시간이 걸릴 수 있습니다.
