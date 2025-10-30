🧠 Truelens — AI Fact-Checking Pipeline

Truelens은 LLM과 웹 검색을 결합한 자동 사실검증(Automated Fact-Checking) 엔진입니다.
주어진 문장을 신뢰할 수 있는 웹 근거를 바탕으로 **“supported / refuted / uncertain”**으로 판정하고,
출처의 신뢰도와 모델 확신도를 기반으로 **최종 점수(0~100)**를 계산합니다.

⸻

⚙️ 전체 구조

Step 1 → Step 2 → Step 3 → Step 4
┌────────┐ ┌────────────┐ ┌────────────┐ ┌─────────────┐
│ Claim  │ │ Evidence   │ │ LLM Eval.  │ │ Scoring &   │
│ Parsing│ │ Collection │ │ (GPT)      │ │ Aggregation │
└────────┘ └────────────┘ └────────────┘ └─────────────┘
     ↓           ↓               ↓               ↓
   주제추출    SERP검색     근거별 판정     신뢰도 산출

단계	이름	주요 역할	처리 방식
Step 1	Claim 분석	주제 추출, 카테고리 예측 (tech, science, policy, health, finance, general)	GPT API
Step 2	근거 수집	SerpAPI를 이용한 웹 검색 + 권위 도메인 폴백	규칙 기반 로직
Step 3	근거 평가	각 snippet을 LLM이 판별 (supports / refutes / irrelevant)	GPT API
Step 4	신뢰도 산출	근거 개수, 티어 가중치, 모델 확신도로 점수 계산	규칙 기반 로직


⸻

🔍 Step 2: Evidence Collection (검색 버킷 구조)

각 주제별로 미리 정의된 **검색 버킷(category)**을 사용합니다.

주제(topic)	검색 버킷(categories)	예시
tech	scholarly, government, news, general	기술·표준 관련
science	scholarly, government, news	학술 연구 중심
policy	government, news, general, community	정책·사회
health	government, scholarly, news	보건·의학
finance	news, general, government	경제·금융
community	community, news, general	사용자 중심 논의
general	news, general, government	일반 주제

	•	각 버킷은 내부적으로 search_serpapi(category, query, max_results) 형태로 호출됩니다.
	•	결과 부족 시 general 검색 → 권위 도메인 폴백(authority fallback) 순으로 보강합니다.

⸻

🌐 권위 도메인(Authority Domains)

근거 신뢰도를 높이기 위해 주제별로 신뢰도 높은 도메인을 우선 검색합니다.

예시:
	•	science: nature.com, science.org, springer.com, nih.gov
	•	health: who.int, cdc.gov, fda.gov, thelancet.com
	•	policy: un.org, oecd.org, reuters.com, bbc.com
	•	finance: bloomberg.com, wsj.com, ft.com
	•	KR locale: go.kr, korea.kr, yna.co.kr

폴백(fallback)은 아래 조건에서만 작동합니다:

일반 검색 결과에서 고티어(2~3) 도메인이 2개 미만일 경우 자동 발동.

⸻

🧾 Step 3: LLM Evidence Evaluation

LLM에게 각 근거의 snippet과 claim을 함께 전달해 판정을 요청합니다.

Prompt 구조:

[주장]
$claim

[근거 요약]
$evidence_bullets

[출력(JSON)]
{
  "per_evidence": [
    {"url": "...", "judgement": "supports|refutes|irrelevant", "rationale": "..."}
  ],
  "overall_verdict": "supported|refuted|uncertain",
  "confidence": 0.0
}


⸻

🧮 Step 4: 신뢰도 평가 및 점수 계산

최종 신뢰도 점수(score)는 다음 요소로 계산됩니다:

항목	설명	가중치
근거 존재 보너스	evidence가 존재할 경우	+10
출처 티어 가중	Tier3 ×10 / Tier2 ×5 / Tier1 ×1	가변
다수결 보너스	(supports - refutes) × 3	±
판정 보정	supported +10 / refuted −15	±
모델 확신도	confidence × 40	+
총합	0~100으로 정규화	—


⸻

🧰 구성요소

모듈	역할
step1_identify_claim()	문장에서 주제 추출 및 카테고리 분류
step2_collect_evidence_serp()	SerpAPI 검색 및 authority fallback
step3_evaluate_sources()	LLM 기반 근거별 판정
step4_score_confidence()	신뢰도 점수 계산 및 최종 판단


⸻

⚡ 실행 예시

claim = "비트코인은 2009년에 처음 발행되었다."
result = factchain_pipeline(claim)
print(result)

출력 예시:

{
  "verdict": "supported",
  "confidence": 0.87,
  "score": 92.5,
  "evidences": [
    {"domain": "bbc.com", "trust_tier": 3, "judgement": "supports"},
    {"domain": "reddit.com", "trust_tier": 1, "judgement": "irrelevant"}
  ]
}


⸻

🧩 기술 스택
	•	Python 3.10+
	•	OpenAI GPT-4o-mini / GPT-5-mini API
	•	SerpAPI (Google Search API)
	•	urllib, dataclasses

⸻

