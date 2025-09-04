# Dementia Chat (Modular)

VSCode에서 바로 실행 가능한 모듈형 구조입니다. `run.py`만 실행하면 대화를 시작합니다.

## 구조
```
dementia-chat-modular/
├─ run.py                # 엔트리포인트 (여기만 실행하면 됨)
├─ main.py               # 핵심 로직 (대부분의 기능 포함)
├─ model.py              # OpenAI 클라이언트, GPT 호출, FT 모델 해석기, 프롬프트 로더
├─ prompts/
│   └─ few_shot_empathy.txt
├─ requirements.txt
└─ README.md
```

## 설치 & 실행
1) 파이썬 3.10~3.12 권장, 가상환경(venv) 추천
2) 패키지 설치
```bash
pip install -r requirements.txt
```
3) OpenAI API 키 설정
- Windows (PowerShell)
```powershell
setx OPENAI_API_KEY "sk-..."
# 새 터미널을 열어 적용하거나, 현재 세션에만:
$env:OPENAI_API_KEY="sk-..."
```
- macOS / Linux (bash/zsh)
```bash
export OPENAI_API_KEY="sk-..."
```

4) 실행
```bash
python run.py
```

## 파인튜닝 모델
- 코드 상단의 `DIALECT_FT_JOB_ID`, `EMPATHY_FT_JOB_ID`를 사용하여 자동으로 최종 모델명을 조회합니다.
- 조회 실패 시 기본 `gpt-4o-mini`로 폴백합니다.

## 메모
- 일기장 체크리스트/주제 대화/요약/CSV 내보내기 포함
- 표준어/사투리 모두 로깅
- KST 기준 날짜 채점 안정화