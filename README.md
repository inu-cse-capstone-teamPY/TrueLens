# 🔎 TrueLens (SerpAPI + OpenAI Responses API)

LLM과 웹 검색을 결합한 **자동 사실 검증 파이프라인** 입니다.  
입력 텍스트에서 사실성 주장만 추출 → SerpAPI로 근거 수집 → LLM이 근거별 판정 →
출처 신뢰도와 모델 확신도를 반영해 **최종 점수(0–100)** 를 계산하고 **콘솔 리포트** 를 출력합니다.

> 본 README는 실제 구현(함수명/동작/콘솔 출력 형식)에 맞춰 정리되었습니다.

---

## 🚀 빠른 시작

### 1) 환경 변수(.env)
프로젝트 루트에 `.env` 파일 생성:

```bash
OPENAI_API_KEY=sk-...
SERPAPI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 선택(없으면 기본값)
FACTCHAIN_MODEL=gpt-4o-mini
FACTCHAIN_MAX_RESULTS=6
FACTCHAIN_TIMEOUT=20
FACTCHAIN_HL=ko   # Google UI 언어
FACTCHAIN_GL=kr   # Google 지역/국가
```

### 2) 실행
텍스트를 직접 넘겨 실행:

```bash
python main.py --text "HTTP/3는 QUIC 위에서 동작한다."
```

또는 소스 하단의 `DEMO_TEXT` 를 수정한 뒤 실행하세요.

---

## 🧱 파이프라인 개요

```
Step 1 → Step 2 → Step 3 → Step 4
┌──────────────┐ ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
│ step1_extract│ │ step2_collect  │ │ step3_evaluate│ │ step4_score  │
│ _claims      │ │ _evidence_serp │ │ _sources      │ │              │
└──────────────┘ └────────────────┘ └──────────────┘ └──────────────┘
  사실 주장 추출     웹 근거 수집         근거별 LLM 판정      점수 산출
```

| 단계 | 함수 | 역할 | 처리 방식 |
|---|---|---|---|
| **Step 1** | `step1_extract_claims` | 입력 텍스트에서 **사실 판단** 문장을 추출하고, 각 주장에 대해 `category ∈ {tech, science, policy, health, finance, general, community}` 를 예측 | **GPT Responses API** |
| **Step 2** | `step2_collect_evidence_serp` | SerpAPI로 **검색 버킷(categories)** 별 검색 후 **정규화/중복 제거/티어 부여**, 부족 시 **권위 도메인 폴백** | **규칙 기반 로직** |
| **Step 3** | `step3_evaluate_sources` | 근거 snippet들을 LLM에 전달, 근거별 `supports/refutes/irrelevant` 및 **overall verdict** 산출 | **GPT Responses API** |
| **Step 4** | `step4_score` | 휴리스틱 + LLM 판정/확신도 → **0–100 점수** 계산 | **규칙 기반 로직** |

---
## 🧾 Step 1: LLM 주장 추출 + 주제 분류 프로픔트

이 단계는 입력 문장 중 ‘사실판단’ 문장만 추출해
검증 가능한 간단한 주장 형태로 요약하는 역할을 합니다.
가치판단 (예: 좋다, 바람직하다) 이나 추측, 의견 등은 제외됩니다.

모델은 각 주장에 대해 검색용 핵심 쿼리(normalized_query) 와
주제(category) 를 함께 생성합니다.
카테고리는 아래 7개 중 하나로 분류됩니다:

["tech", "science", "policy", "health", "finance", "community", "general"]


프롬프트를 LLM에 아래 구조로 전달합니다.
$input_text 는 런타임 시 사용자가 입력한 전체 텍스트로 치환됩니다.

```text
당신은 사실검증 편집자입니다. 아래 입력 텍스트에서 "사실판단" 문장만 뽑아 간단한 주장 형태로 요약하세요.  
가치판단(좋다/나쁘다/바람직하다 등)이나 의견/추측은 제외합니다.  

또한 각 주장에 대해 주제 카테고리를 분류하세요.  
카테고리 후보: ["tech","science","policy","health","finance","community","general"]

[출력(JSON)]
{
  "claims": [
    {"id": "C1", "claim": "요약 주장 한 문장", "normalized_query": "웹검색용 핵심 키워드", "category": "tech"}
  ]
}

[입력]
$input_text
```
---

## 🔍 Step 2: 검색 버킷(categories)

Step 1에서 뽑힌 `category` 를 기반으로, 주제별 **검색 버킷 목록** 을 사용합니다.

```python
CATEGORY_PRESETS = {
  "tech":      ["scholarly", "government", "news", "general"],
  "science":   ["scholarly", "government", "news"],
  "policy":    ["government", "news", "general", "community"],
  "health":    ["government", "scholarly", "news"],
  "finance":   ["news", "general", "government"],
  "community": ["community", "news", "general"],
  "general":   ["news", "general", "government"],
}
```

각 버킷은 내부적으로 `search_serpapi(category, query, max_results)` 로 호출됩니다.

- **scholarly**: `engine=google_scholar` → 부족 시 `site:arxiv.org OR site:ieee.org ...`
- **government**: `site:.gov OR site:.go.kr OR site:.edu` 필터로 일반 웹 검색
- **news**: `engine=google`, `tbm=nws` (뉴스 탭)
- **blogs/community**: 대표 도메인 묶음에 `site:` 필터
- **general**: 일반 웹 검색
- 모든 경로에서 공통적으로 `-site:patents.google.com` 을 붙여 특허 노이즈 제거

결과는 URL 정규화(UTM 등 제거)와 **중복 제거** 후, 도메인별 **신뢰 티어** (1~3)를 부여합니다.

### 권위 도메인 폴백 (Authority Fallback)
- 일반 검색 결과에서 **서로 다른 도메인 기준** 고티어(티어 2~3) 출처가 **2개 미만** 일 때 자동 발동  
- 주제(`topic`) + 지역(`locale`)에 맞춘 **권위 도메인 세트** 를 `site:` 스윕으로 보강
- 과학/보건/기술 계열은 `edu`, `ac.kr` 등을 추가로 포함해 학술성을 강화

예시(일부):
- **science**: `nature.com`, `science.org`, `springer.com`, `nih.gov`
- **health**: `who.int`, `cdc.gov`, `fda.gov`, `thelancet.com`
- **policy**: `un.org`, `oecd.org`, `reuters.com`, `bbc.com`
- **KR locale**: `go.kr`, `korea.kr`, `yna.co.kr`, `kbs.co.kr` 등

---

## 🧾 Step 3: LLM 근거 평가 프롬프트

LLM에 아래 구조로 전달합니다. `$claim` 과 `$evidence_bullets` 는 런타임에 치환됩니다.

```text
[주장]
$claim

[근거 요약]
$evidence_bullets

[출력(JSON)]
{
  "per_evidence": [
    {"url": "...", "judgement": "supports|refutes|irrelevant", "rationale": "한 줄 근거 설명"}
  ],
  "overall_verdict": "supported|refuted|uncertain",
  "confidence": 0.0
}
```

---

## 🧮 Step 4: 점수 계산 규칙

```python
exists = len(evidences) > 0
score = 0.0
if exists: score += 10  # 근거 존재 보너스
score += tier_counts[3]*10 + tier_counts[2]*5 + tier_counts[1]*1  # 티어 가중
score += max(0, supports - refutes) * 3                           # 다수결 보너스
if verdict == "supported": score += 10                            # 판정 보정
elif verdict == "refuted": score -= 15
score += max(0.0, min(conf, 1.0)) * 40                            # 모델 확신도
score = max(0.0, min(score, 100.0))                               # 0~100 클리핑
```

- `supports/refutes` 는 **per_evidence** 판정의 개수
- **티어** 는 정규식 기반 도메인 매핑으로 산정 (3이 가장 신뢰도 높음)
- 점수는 상한/하한으로 **0–100** 사이로 고정

---

## 🖥️ 콘솔 출력 형식 (예시)

아래와 같은 형식으로 컬러/이모지와 함께 출력됩니다.

```
모델: gpt-4o-mini | 검색엔진: serpapi-google
실행시간: 21.271초 | 주장 수: 1개

프로세스별 소요시간:
STEP 1 (주장 추출): 2.417초
STEP 2 (근거 수집): 8.02초
STEP 3 (근거 평가): 10.834초
STEP 4 (점수화): 0.02초

🔹 [C1] 수은은 상온에서 액체 상태인 유일한 금속이다.
   → 판정: ✅ 사실로 판단 (90% 확신)
   → 신뢰점수: 98.0
   → 출처 신뢰도: 🟢3:3, 🟡2:0, 🔴1:3
      - oak.go.kr: ... (요약)
      - me.go.kr: ... (요약)
      - wwww.kca.go.kr: ... (요약)
───────────────────────────────
```

> 실제 실행 환경의 출력 색상/아이콘은 터미널 지원 여부에 따라 다를 수 있습니다.

---

## 📦 의존성

- Python 3.10+
- `openai`, `python-dotenv`, `requests`


