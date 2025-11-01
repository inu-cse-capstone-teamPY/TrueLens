#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
사실 검증 파이프라인 (SerpAPI + OpenAI Responses API)
=================================================================

이 모듈은 **입력 텍스트 → (사실 주장 추출) → (웹 근거 수집) → (LLM 근거판정)** 의
체인을 수행해, 각 주장에 대해 **신뢰도 점수**를 계산하고 콘솔용 리포트를 출력합니다.

1) 프로젝트 루트에 .env 파일 생성 (최소):

    OPENAI_API_KEY=sk-...
    SERPAPI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    선택 옵션(없으면 기본값 사용):

    FACTCHAIN_MODEL=gpt-5-mini
    FACTCHAIN_MAX_RESULTS=6
    FACTCHAIN_TIMEOUT=20
    FACTCHAIN_HL=ko
    FACTCHAIN_GL=kr

2) 실행:

    python main.py --text "HTTP/3는 QUIC 위에서 동작한다."

3) 혹은 파일 하단의 DEMO_TEXT를 수정해 바로 실행.

설계 개요
---------
파이프라인 단계:
1) step1_extract_claims  — 사실 주장 + (주제 카테고리) 추출
2) step2_collect_evidence — GOOGLE CSE API로 버킷 검색 후 병합/티어 정렬
3) step3_evaluate_sources — LLM으로 supports/refutes/irrelevant 판정
4) step4_score — 휴리스틱 + 판정/확신도 → 0~100 신뢰점수

"""
from __future__ import annotations

import os
import re
import json
import time
import argparse
import logging
import urllib.parse as urlparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from string import Template

from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


# ──────────────────────────────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger("factchain")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────
# 환경설정(.env) 기반 기본값 — 필요 시 CLI로 덮어쓰기 가능
# ──────────────────────────────────────────────────────────────────────
MODEL_DEFAULT = os.getenv("FACTCHAIN_MODEL", "gpt-4o-mini")
MAX_RESULTS = int(os.getenv("FACTCHAIN_MAX_RESULTS", "6"))
TIMEOUT_S = int(os.getenv("FACTCHAIN_TIMEOUT", "20"))
SERPAPI_ENDPOINT = os.getenv("SERPAPI_ENDPOINT", "https://serpapi.com/search.json")
SERP_HL = os.getenv("FACTCHAIN_HL", "ko")   # Google 언어(UI)
SERP_GL = os.getenv("FACTCHAIN_GL", "kr")   # Google 지역/국가

# step1에서 사용할 카테고리 목록
FACT_CATS = ["tech", "science", "policy", "health", "finance", "general", "community"]

# step2에서 사용할 각 주제별 검색 옵션
CATEGORY_PRESETS: Dict[str, List[str]] = {
    "tech":      ["scholarly", "government", "news", "general"],
    "science":   ["scholarly", "government", "news"],
    "policy":    ["government", "news", "general", "community"],
    "health":    ["government", "scholarly", "news"],
    "finance":   ["news", "general", "government"],
    "community": ["community", "news", "general"],
    "general":   ["news", "general", "government"],
}


DEFAULT_PRESET = "general"

# ──────────────────────────────────────────────────────────────────────
# 도메인 신뢰 티어(정규식) — 3: 최고(정부/국제/학술/표준) / 2: 언론/대기업 등 / 1: 기타
# 추가 시 보수적으로 관리, 리뷰 필수
# ──────────────────────────────────────────────────────────────────────
TRUST_TIER_PATTERNS: Dict[int, List[str]] = {
    # 3티어 — 정부/공공/학술/국제/핵심 표준
    3: [
        r"(^|\.)gov$", r"(^|\.)go\.kr$", r"(^|\.)g\.kr$", r"(^|\.)edu$",
        r"(^|\.)who\.int$", r"(^|\.)un\.org$", r"(^|\.)europe\.eu$",
        r"(^|\.)rfc-editor\.org$", r"(^|\.)ietf\.org$", r"(^|\.)iso\.org$", r"(^|\.)ieee\.org$",
    ],
    # 2티어 — 주요 언론/빅테크/표준 사이트
    2: [
        # 해외 언론
        r"(^|\.)reuters\.com$", r"(^|\.)apnews\.com$", r"(^|\.)bbc\.com$",
        r"(^|\.)nytimes\.com$", r"(^|\.)wsj\.com$", r"(^|\.)bloomberg\.com$",
        r"(^|\.)theguardian\.com$", r"(^|\.)cnn\.com$", r"(^|\.)cnbc\.com$",
        r"(^|\.)forbes\.com$", r"(^|\.)economist\.com$", r"(^|\.)washingtonpost\.com$",
        # 한국 언론
        r"(^|\.)hani\.co\.kr$", r"(^|\.)yna\.co\.kr$", r"(^|\.)yonhapnews\.co\.kr$",
        r"(^|\.)kbs\.co\.kr$", r"(^|\.)mbc\.co\.kr$", r"(^|\.)sbs\.co\.kr$",
        r"(^|\.)chosun\.com$", r"(^|\.)joongang\.co\.kr$", r"(^|\.)donga\.com$",
        r"(^|\.)jtbc\.co\.kr$", r"(^|\.)mk\.co\.kr$", r"(^|\.)edaily\.co\.kr$",
        r"(^|\.)koreatimes\.co\.kr$", r"(^|\.)koreaherald\.com$", r"(^|\.)asiatoday\.co\.kr$",
        r"(^|\.)newsis\.co\.kr$", r"(^|\.)heraldcorp\.com$",
        # 기술/표준/기업
        r"(^|\.)microsoft\.com$", r"(^|\.)apple\.com$", r"(^|\.)google\.com$",
        r"(^|\.)meta\.com$", r"(^|\.)cloudflare\.com$", r"(^|\.)mozilla\.org$",
        r"(^|\.)oracle\.com$", r"(^|\.)intel\.com$", r"(^|\.)nvidia\.com$",
    ],
}

COMPILED_PATTERNS = {tier: [re.compile(p) for p in patterns] for tier, patterns in TRUST_TIER_PATTERNS.items()}

def classify_domain(domain: str) -> int:
    """도메인 문자열을 1/2/3 티어로 분류. 일치 없으면 1.
    정규식은 도메인 끝 수준에서 일치(서브도메인 포함)하도록 설계됨."""
    d = domain.lower()
    for tier, pats in COMPILED_PATTERNS.items():
        for pat in pats:
            if re.search(pat, d):
                return tier
    return 1

# ──────────────────────────────────────────────────────────────────────
# 데이터 모델
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Evidence:
    title: str
    url: str
    snippet: str
    domain: str
    trust_tier: int

@dataclass
class ClaimAssessment:
    claim_id: str
    claim_text: str
    normalized_query: str
    evidence: List[Evidence]
    exists_evidence: bool
    source_trust_summary: Dict[str, Any]
    model_verdict: str  # supported|refuted|uncertain
    model_confidence: float  # 0~1
    credibility_score: float  # 0~100

# ──────────────────────────────────────────────────────────────────────
# 프롬프트 템플릿 (한국어)
# ──────────────────────────────────────────────────────────────────────

EXTRACT_PROMPT_TMPL = Template("""
당신은 사실검증 편집자입니다. 아래 입력 텍스트에서 "사실판단" 문장만 뽑아 간단한 주장 형태로 요약하세요.
가치판단(좋다/나쁘다/바람직하다 등)이나 의견/추측은 제외합니다.

또한 각 주장에 대해 주제 카테고리를 분류하세요.
카테고리 후보: ["tech","science","policy","health","finance","community","general"]

[출력 형식(JSON)]
{
    "claims": [
    {"id": "C1", "claim": "요약 주장 한 문장", "normalized_query": "웹검색용 핵심 키워드", "category": "tech"}
    ]
}

[입력]
$input_text
""")

EVIDENCE_EVAL_PROMPT_TMPL = Template("""
당신은 사실검증 전문가입니다. 다음 주장을, 아래 제공된 웹 자료 요약만 근거로 평가하세요.  
각 근거가 주장을 입증하는지, 반박하는지, 관련이 없는지 를 구분하고  
종합적으로 전체 판정을 내리세요.  
불확실하다면 "uncertain"으로 표시합니다. 과장이나 추측은 금지됩니다.

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
""")

# ──────────────────────────────────────────────────────────────────────
# LLM 호출 유틸
# ──────────────────────────────────────────────────────────────────────

def llm_json(client: OpenAI, prompt: str, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Responses API로 JSON 스키마 강제 출력.
    실패 시 빈 dict 반환(상위 단계에서 fail-safe 처리).
    """
    r = client.responses.create(
        model=MODEL_DEFAULT,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
        instructions="결과는 JSON만 출력.",
    )
    try:
        return json.loads(r.output_text)
    except Exception:
        logger.warning("JSON 파싱 실패 — 원시 출력 보관 필요 시 client 로그 확인")
        return {}

# ──────────────────────────────────────────────────────────────────────
# SerpAPI 검색 어댑터 (카테고리별 쿼리 구성)
# ──────────────────────────────────────────────────────────────────────

def search_cse(category: str, query: str, max_results: int = 6, *, time_window: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Google Custom Search JSON API 기반 검색.
    category ∈ {"scholarly","government","news","blogs","community","general"}

    - scholarly : 학술 도메인 site: 필터로 에뮬레이션
    - government: .gov/.go.kr/.edu 등 공공/교육 필터
    - news      : 주요 뉴스 도메인 묶음 site: 필터
    - blogs     : 블로그 도메인 묶음
    - community : 커뮤니티/Q&A 도메인 묶음
    - general   : 전체 웹 (특허 도메인 제외)

    time_window: {"d","w","m","y"} → CSE dateRestrict 로 매핑
    """
    import requests

    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_CX")
    if not api_key or not cx:
        return []

    # 특허 도메인 제외(쿼리 보강)
    common_exclude = " -site:patents.google.com"

    # 버킷별 도메인 그룹
    SCHOLAR_SITES = [
        "arxiv.org","acm.org","ieee.org","springer.com","sciencedirect.com",
        "nature.com","science.org","pnas.org","cell.com","cambridge.org",
    ]
    NEWS_SITES = [
        # 해외 주요
        "reuters.com","apnews.com","bbc.com","nytimes.com","wsj.com","bloomberg.com",
        "theguardian.com","cnn.com","cnbc.com","economist.com","washingtonpost.com",
        # 한국 주요
        "yna.co.kr","yonhapnews.co.kr","kbs.co.kr","mbc.co.kr","sbs.co.kr",
        "chosun.com","joongang.co.kr","donga.com","hani.co.kr","jtbc.co.kr","mk.co.kr","edaily.co.kr",
        "koreaherald.com","koreatimes.co.kr","asiatoday.co.kr","newsis.co.kr","heraldcorp.com",
    ]
    BLOG_SITES = [
        "medium.com","tistory.com","velog.io","dev.to","blogspot.com","hashnode.com","brunch.co.kr","naver.com/blog"
    ]
    COMMUNITY_SITES = [
        "reddit.com","stackoverflow.com","superuser.com","serverfault.com",
        "quora.com","news.ycombinator.com","okky.kr","discord.com/invite"
    ]

    # 카테고리별 쿼리 강화
    q = query
    if category == "scholarly":
        filt = " OR ".join(f"site:{d}" for d in SCHOLAR_SITES)
        q = f"{query} ({filt}){common_exclude}"
    elif category == "government":
        filt = "(site:.gov OR site:.go.kr OR site:.g.kr OR site:.edu)"
        q = f"{query} {filt}{common_exclude}"
    elif category == "news":
        filt = " OR ".join(f"site:{d}" for d in NEWS_SITES)
        q = f"{query} ({filt}){common_exclude}"
    elif category == "blogs":
        filt = " OR ".join(f"site:{d}" for d in BLOG_SITES)
        q = f"{query} ({filt}){common_exclude}"
    elif category == "community":
        filt = " OR ".join(f"site:{d}" for d in COMMUNITY_SITES)
        q = f"{query} ({filt}){common_exclude}"
    else:  # general
        q = f"{query}{common_exclude}"

    # CSE 파라미터 구성
    # 참고: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
    # gl/hl과 유사한 효과는 lr=lang_ko + 쿼리/엔진 설정으로 어느정도 유도 가능
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": q,
        "num": min(max_results, 10),   # CSE는 요청당 최대 10개
    }

    # time_window → dateRestrict 매핑 (d,w,m,y) → d1, w1, m1, y1
    if time_window in ("d","w","m","y"):
        params["dateRestrict"] = {"d":"d1","w":"w1","m":"m1","y":"y1"}[time_window]

    results: List[Dict[str, str]] = []
    start = 1
    remaining = max_results

    def _request(p) -> dict:
        try:
            r = requests.get(base_url, params=p, timeout=TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}

    while remaining > 0:
        params["start"] = start
        data = _request(params)
        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            break
        for it in items:
            results.append({
                "title": it.get("title", ""),
                "url": it.get("link", ""),
                "snippet": it.get("snippet", "") or it.get("htmlSnippet",""),
            })
            if len(results) >= max_results:
                break
        if len(items) < params["num"]:
            break
        start += params["num"]
        remaining = max_results - len(results)

    # 중복 URL 제거
    seen, out = set(), []
    for r in results:
        u = r.get("url", "")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
        if len(out) >= max_results:
            break
    return out

def search_serpapi(category: str, query: str, max_results: int = 6) -> List[Dict[str, str]]:
    """
    category ∈ {"scholarly","government","news","blogs","community","general"}
    - scholarly : google_scholar 우선, 부족 시 학술 도메인 site: 필터로 보강
    - government: .gov/.go.kr/.edu 등 공공/교육 도메인 우선
    - news      : 구글 뉴스 탭(tbmn=nws)
    - blogs     : 블로그 도메인 묶음 필터
    - community : 커뮤니티/Q&A 도메인 묶음 필터
    - general   : 일반 웹 검색(기본적으로 특허 도메인 제외)
    """
    import requests

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []

    def _request(params) -> dict:
        try:
            resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    # 특허 도메인은 공통적으로 배제 (노이즈 방지)
    common_exclude = " -site:patents.google.com"

    # 검색 도메인 그룹
    SCHOLAR_SITES = [
        "arxiv.org","acm.org","ieee.org","springer.com","sciencedirect.com",
        "nature.com","science.org","pnas.org","cell.com","cambridge.org",
    ]
    BLOG_SITES = [
        "medium.com","tistory.com","velog.io","dev.to","blogspot.com","hashnode.com","brunch.co.kr","naver.com/blog"
    ]
    COMMUNITY_SITES = [
        "reddit.com","stackoverflow.com","superuser.com","serverfault.com",
        "quora.com","news.ycombinator.com","okky.kr", "discord.com/invite"
    ]

    results: List[Dict[str, str]] = []

    # 	1.	scholarly → engine="google_scholar" (구글 스칼라 인덱스)
	#   2.	news → tbm="nws" (구글 뉴스 인덱스)
    if category == "scholarly":
        # 1) 구글 스칼라 우선
        params = {"engine": "google_scholar", "q": query, "api_key": api_key, "hl": SERP_HL}
        data = _request(params)
        for it in data.get("organic_results", [])[:max_results]:
            results.append({
                "title": it.get("title",""),
                "url": it.get("link",""),
                "snippet": it.get("snippet","") or it.get("publication_info",{}).get("summary",""),
            })
        # 2) 부족하면 일반 웹 + 학술 site 필터로 보강
        if len(results) < max_results:
            filt = " OR ".join(f"site:{d}" for d in SCHOLAR_SITES)
            params = {
                "engine": "google",
                "q": f"{query} ({filt}){common_exclude}",
                "api_key": api_key,
                "num": max_results - len(results),
                "hl": SERP_HL, "gl": SERP_GL,
            }
            data = _request(params)
            for it in data.get("organic_results", []):
                results.append({"title": it.get("title",""), "url": it.get("link",""), "snippet": it.get("snippet","")})

    elif category == "government":
        # 대표 접미사 기반 site 필터 (단순/안전)
        filt = "(site:.gov OR site:.go.kr OR site:.g.kr OR site:.edu)"
        params = {"engine": "google", "q": f"{query} {filt}{common_exclude}", "api_key": api_key, "num": max_results, "hl": SERP_HL, "gl": SERP_GL}
        data = _request(params)
        for it in data.get("organic_results", [])[:max_results]:
            results.append({"title": it.get("title",""), "url": it.get("link",""), "snippet": it.get("snippet","")})

    elif category == "news":
        # 구글 뉴스 탭
        params = {"engine": "google", "tbm": "nws", "q": f"{query}{common_exclude}", "api_key": api_key, "num": max_results, "hl": SERP_HL, "gl": SERP_GL}
        data = _request(params)
        for it in data.get("news_results", [])[:max_results]:
            results.append({"title": it.get("title",""), "url": it.get("link",""), "snippet": it.get("snippet","") or it.get("source","")})

    elif category == "blogs":
        filt = " OR ".join(f"site:{d}" for d in BLOG_SITES)
        params = {"engine": "google", "q": f"{query} ({filt}){common_exclude}", "api_key": api_key, "num": max_results, "hl": SERP_HL, "gl": SERP_GL}
        data = _request(params)
        for it in data.get("organic_results", [])[:max_results]:
            results.append({"title": it.get("title",""), "url": it.get("link",""), "snippet": it.get("snippet","")})

    elif category == "community":
        filt = " OR ".join(f"site:{d}" for d in COMMUNITY_SITES)
        params = {"engine": "google", "q": f"{query} ({filt}){common_exclude}", "api_key": api_key, "num": max_results, "hl": SERP_HL, "gl": SERP_GL}
        data = _request(params)
        for it in data.get("organic_results", [])[:max_results]:
            results.append({"title": it.get("title",""), "url": it.get("link",""), "snippet": it.get("snippet","")})

    else:  # general
        params = {"engine": "google", "q": f"{query}{common_exclude}", "api_key": api_key, "num": max_results, "hl": SERP_HL, "gl": SERP_GL}
        data = _request(params)
        for it in data.get("organic_results", [])[:max_results]:
            results.append({"title": it.get("title",""), "url": it.get("link",""), "snippet": it.get("snippet","")})

    # 중복 URL 제거(간단한 seen 세트)
    seen, out = set(), []
    for r in results:
        u = r.get("url", "")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
        if len(out) >= max_results:
            break
    return out

# ──────────────────────────────────────────────────────────────────────
# Step 1 — 주장 추출 (사실판단 + 카테고리)
# ──────────────────────────────────────────────────────────────────────

def step1_extract_claims(client: OpenAI, text: str) -> List[Dict[str, str]]:
    """입력 텍스트에서 사실 판단 문장만 추출 + 검색용 쿼리/카테고리 부여.
    실패 시 빈 리스트 반환.
    """
    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "claim": {"type": "string"},
                        "normalized_query": {"type": "string"},
                        "category": {"type": "string", "enum": FACT_CATS},
                    },
                    "required": ["id", "claim", "normalized_query", "category"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }
    prompt = EXTRACT_PROMPT_TMPL.substitute(input_text=text.strip())
    data = llm_json(client, prompt, "ClaimList", schema)
    claims = data.get("claims", []) if isinstance(data, dict) else []

    # 중복 제거 + 필드 보정
    out, seen_texts = [], set()
    for c in claims:
        t = (c.get("claim", "") or "").strip()
        if t and t not in seen_texts:
            seen_texts.add(t)
            out.append({
                "id": c.get("id", f"C{len(out)+1}"),
                "claim": t,
                "normalized_query": c.get("normalized_query", t),
                "category": c.get("category", "general"),
            })
    return out

# ──────────────────────────────────────────────────────────────────────
# Step 2 — 근거 수집(SerpAPI) + 정규화/티어/도메인 다양성/정렬
# ──────────────────────────────────────────────────────────────────────
def step2_collect_evidence(
    query: str,
    k: int = MAX_RESULTS,
    categories: Optional[List[str]] = None,   # 예: ["scholarly","government","news"]
    *,
    topic: str = "general",                   # 주제: tech/science/policy/health/finance/general
    locale: str = "KR",                       # 지역: "KR", "US" ...
    authority_policy: str = "auto",           # "auto" | "always" | "never"
    authority_extra: Optional[List[str]] = None,  # 추가로 강제 포함할 도메인들
    time_window: Optional[str] = None         # "d"=24h, "w"=7days, "m"=30days, "y"=year (간단 키워드 보강)
) -> List[Evidence]:
    
    """
    검색 전략 요약:
    1) categories가 있으면 버킷(학술/정부/뉴스/블로그/커뮤니티)별 검색
    2) 없으면 일반 검색 → (authority_policy에 따라) 권위 도메인 폴백 단계적 시도
    3) locale/주제에 따라 권위 도메인을 구성하고 필요 시 확장
    4) 특허 도메인 차단, URL 정규화/중복 제거, 도메인 다양성 유지, 티어 우선 정렬
    """

    # URL 정규화(utm 등 추적 파라미터 제거, fragment 제거)
    def _normalize_url(u: str) -> str:
        try:
            p = urlparse.urlparse(u)
            drop = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","mc_cid","mc_eid"}
            q = [(k,v) for (k,v) in urlparse.parse_qsl(p.query, keep_blank_values=True) if k.lower() not in drop]
            return urlparse.urlunparse((p.scheme, p.netloc.lower(), p.path, "", urlparse.urlencode(q), ""))
        except Exception:
            return u

    def _rows_by_category(cat: str, n: int) -> List[Dict[str, str]]:
        return search_cse(cat, query, max_results=n, time_window=time_window) or []

    def _rows_google(q: str, n: int) -> List[Dict[str, str]]:
        q2 = f"{q} -site:patents.google.com"
        if time_window in ("d","w","m","y"):
            q2 += " "
        return search_cse("general", q2, max_results=n, time_window=time_window) or []

    def _rows_site_sweep(domains: List[str], q: str, per_domain: int) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if not domains:
            return out
        chunk = 8
        for i in range(0, len(domains), chunk):
            group = domains[i:i+chunk]
            filt = " OR ".join(f"site:{d}" for d in group)
            q2 = f"{q} ({filt}) -site:patents.google.com"
            out += search_cse("general", q2, max_results=per_domain * len(group), time_window=time_window) or []
            if len(out) >= per_domain * len(domains):
                break
        return out

    # 권위 도메인(주제/지역 기반) 구성
    BASE_AUTHORITY: Dict[str, List[str]] = {
        "tech":    ["ietf.org","rfc-editor.org","w3.org","iana.org","developer.mozilla.org","learn.microsoft.com","cloudflare.com","google.com"],
        "science": ["arxiv.org","nature.com","science.org","springer.com","sciencedirect.com","pnas.org","cell.com","nih.gov"],
        "policy":  ["un.org","oecd.org","reuters.com","apnews.com","bbc.com","nytimes.com"],
        "health":  ["who.int","cdc.gov","fda.gov","ema.europa.eu","nih.gov","bmj.com","thelancet.com","nejm.org"],
        "finance": ["reuters.com","apnews.com","bloomberg.com","wsj.com","ft.com"],
        "general": ["reuters.com","apnews.com","bbc.com","nature.com"],
    }
    LOCALE_AUTHORITY: Dict[str, List[str]] = {
        "KR": ["korea.kr","go.kr","stat.go.kr","yna.co.kr","kbs.co.kr","mbc.co.kr","sbs.co.kr","chosun.com","joongang.co.kr","hani.co.kr"],
        "US": ["cdc.gov","fda.gov","nih.gov","nasa.gov","nytimes.com","wsj.com","apnews.com","reuters.com"],
        "EU": ["ec.europa.eu","ema.europa.eu","who.int","oecd.org","reuters.com","bbc.com"],
    }

    base = BASE_AUTHORITY.get(topic, BASE_AUTHORITY["general"])
    loc = LOCALE_AUTHORITY.get(locale.upper(), [])
    authority_domains = list(dict.fromkeys(base + loc + (authority_extra or [])))

    # 학술 강화 힌트(edu/ac.kr)
    scholarly_boost = ("scholarly" in (categories or [])) or (topic in ("science","health","tech"))
    if scholarly_boost:
        authority_domains = list(dict.fromkeys(authority_domains + ["edu","ac.kr"]))

    # 1) 검색 수행
    raw_rows: List[Dict[str, str]] = []

    if categories:  # 버킷별 검색 경로
        per_bucket = max(1, k // len(categories))
        for cat in categories:
            raw_rows += _rows_by_category(cat, per_bucket)
        if len(raw_rows) < k:
            raw_rows += _rows_by_category("general", k - len(raw_rows))
    else:           # 일반 검색 경로(+ 권위 폴백)
        raw_rows += _rows_google(query, k)

        def need_authority_fallback(rows: List[Dict[str,str]]) -> bool:
            # 티어2/3 도메인이 충분한지 평가(중복 도메인 제외)
            t23, seen_here = 0, set()
            for r in rows:
                u = r.get("url") or r.get("link") or ""
                if not u:
                    continue
                try:
                    d = urlparse.urlparse(u).netloc
                except Exception:
                    d = ""
                if d in seen_here:
                    continue
                seen_here.add(d)
                if classify_domain(d) >= 2:
                    t23 += 1
            return t23 < 2

        run_fallback = ((authority_policy == "always") or (authority_policy == "auto" and need_authority_fallback(raw_rows)))
        if run_fallback and authority_domains:
            per_dom = 1 if k <= 6 else 2
            raw_rows += _rows_site_sweep(authority_domains, query, per_dom)
            if scholarly_boost and len(raw_rows) < k:
                raw_rows += _rows_google(f'{query} (filetype:pdf OR "white paper")', max(2, k - len(raw_rows)))

    # 2) URL 정규화 + 중복 제거 + 특허 도메인 방화벽
    seen, rows = set(), []
    for r in raw_rows:
        u = r.get("url") or r.get("link") or ""
        if not u:
            continue
        nu = _normalize_url(u)
        if nu in seen:
            continue
        if "patents.google.com" in urlparse.urlparse(nu).netloc:
            continue
        seen.add(nu)
        rows.append({"title": r.get("title","") or r.get("name",""), "url": nu, "snippet": r.get("snippet","") or r.get("description","")})

    # 3) Evidence 변환 + 티어 부여
    evs_tmp: List[Evidence] = []
    for r in rows:
        u = r["url"]
        try:
            d = urlparse.urlparse(u).netloc
        except Exception:
            d = ""
        evs_tmp.append(Evidence(title=r["title"], url=u, snippet=r["snippet"], domain=d, trust_tier=classify_domain(d)))

    if not evs_tmp:
        return []

    # 4) 도메인 다양성 유지 + 정렬(티어 desc → 스니펫 길이 desc → 제목 길이 desc)
    def _key(ev: Evidence) -> Tuple[int,int,int]:
        return (ev.trust_tier, len(ev.snippet or ""), len(ev.title or ""))

    buckets: Dict[str, List[Evidence]] = {}
    for e in evs_tmp:
        buckets.setdefault(e.domain, []).append(e)
    for dmn in buckets:
        buckets[dmn].sort(key=_key, reverse=True)

    merged: List[Evidence] = []
    for tier in (3, 2, 1):
        # 1라운드: 도메인당 1개
        for dmn, lst in list(buckets.items()):
            if len(merged) >= k:
                break
            for i, e in enumerate(lst):
                if e.trust_tier == tier:
                    merged.append(lst.pop(i))
                    break
        # 2라운드: 남은 슬롯에서 도메인당 1개 더
        if len(merged) < k:
            for dmn, lst in list(buckets.items()):
                if len(merged) >= k:
                    break
                for i, e in enumerate(lst):
                    if e.trust_tier == tier:
                        merged.append(lst.pop(i))
                        break

    if len(merged) < k:  # 여전히 부족하면 나머지에서 채우기
        rest: List[Evidence] = []
        for lst in buckets.values():
            rest.extend(lst)
        rest.sort(key=_key, reverse=True)
        for e in rest:
            if len(merged) >= k:
                break
            merged.append(e)

    return merged[:k]

# ──────────────────────────────────────────────────────────────────────
# Step 3 — LLM으로 근거-주장 매핑 판정(supports/refutes/irrelevant)
# ──────────────────────────────────────────────────────────────────────

def step3_evaluate_sources(client: OpenAI, claim_text: str, evidences: List[Evidence]) -> Dict[str, Any]:
    bullets = [f"- [{ev.domain}] {ev.title} — {ev.snippet[:300]} (URL: {ev.url})" for ev in evidences]
    prompt = EVIDENCE_EVAL_PROMPT_TMPL.substitute(
        claim=claim_text,
        evidence_bullets="\n".join(bullets) or "(근거 없음)",
    )

    schema = {
        "type": "object",
        "properties": {
            "per_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "judgement": {"type": "string", "enum": ["supports", "refutes", "irrelevant"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["url", "judgement", "rationale"],
                    "additionalProperties": False,
                },
            },
            "overall_verdict": {"type": "string", "enum": ["supported", "refuted", "uncertain"]},
            "confidence": {"type": "number"},
        },
        "required": ["per_evidence", "overall_verdict", "confidence"],
        "additionalProperties": False,
    }

    out = llm_json(client, prompt, "EvidenceEval", schema)
    if not out:
        out = {"per_evidence": [], "overall_verdict": "uncertain", "confidence": 0.0}
    return out

# ──────────────────────────────────────────────────────────────────────
# Step 4 — 점수화(휴리스틱 + 판정/확신도 → 0~100)
# ──────────────────────────────────────────────────────────────────────

def step4_score(claim_text: str, evidences: List[Evidence], eval_out: Dict[str, Any]) -> ClaimAssessment:
    exists = len(evidences) > 0
    tiers = [ev.trust_tier for ev in evidences]
    tier_counts = {1: tiers.count(1), 2: tiers.count(2), 3: tiers.count(3)}

    per_evi = eval_out.get("per_evidence", [])
    supports = sum(1 for p in per_evi if p.get("judgement") == "supports")
    refutes  = sum(1 for p in per_evi if p.get("judgement") == "refutes")

    verdict = eval_out.get("overall_verdict", "uncertain")
    conf = float(eval_out.get("confidence", 0.0))

    # 점수 계산 규칙
    # 1. 근거 존재 기본 점수 + 10점
    # 2. 티어 가중치 : 3*10, 2*5, 1*1 
    # 3. 각 근거의 판정별 점수 합산 : (supports - refutes) * 3
    # 4. 최종 판정 점수 : supported +10 / refuted -15 (반박은 강하게)
    # 5. GPT가 반환해준 확신도(0~1) * 40
    score = 0.0
    if exists:
        score += 10
    score += tier_counts[3] * 10 + tier_counts[2] * 5 + tier_counts[1] * 1
    score += max(0, supports - refutes) * 3
    if verdict == "supported":
        score += 10
    elif verdict == "refuted":
        score -= 15
    score += max(0.0, min(conf, 1.0)) * 40
    score = max(0.0, min(score, 100.0))


    return ClaimAssessment(
        claim_id="",
        claim_text=claim_text,
        normalized_query=claim_text,
        evidence=evidences,
        exists_evidence=exists,
        source_trust_summary={"tier_counts": {str(k): v for k, v in tier_counts.items()}},
        model_verdict=verdict,
        model_confidence=round(conf, 3),
        credibility_score=round(score, 1),
    )

# ──────────────────────────────────────────────────────────────────────
# 실행 루틴(파이프라인) + 콘솔 리포트 출력
# ──────────────────────────────────────────────────────────────────────

def _process_one_claim(
    client: OpenAI,
    claim: Dict[str, Any],
    idx: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """단일 주장에 대해 Step2~4를 실행하고 (assessment, timings)를 반환"""
    timings: Dict[str, float] = {}

    claim_id = claim.get("id", f"C{idx}")
    ctext = claim.get("claim", "").strip()
    nquery = claim.get("normalized_query", ctext)
    claim_category = claim.get("category", "general")
    cats_for_search = CATEGORY_PRESETS.get(claim_category, CATEGORY_PRESETS["general"])

    # STEP 2 — 근거 수집
    t2 = time.perf_counter()
    ev = step2_collect_evidence(
        query=nquery,
        k=MAX_RESULTS,
        categories=cats_for_search,
        topic=claim_category,
        locale="KR",
        authority_policy="auto",
    )
    timings[f"{claim_id}_step2_collect"] = round(time.perf_counter() - t2, 3)

    # STEP 3 — 근거 판정 (GPT 호출)
    t3 = time.perf_counter()
    eval_out = step3_evaluate_sources(client, ctext, ev)
    timings[f"{claim_id}_step3_evaluate"] = round(time.perf_counter() - t3, 3)

    # STEP 4 — 점수화
    t4 = time.perf_counter()
    assess = step4_score(ctext, ev, eval_out)
    assess.claim_id = claim_id
    assess.normalized_query = nquery
    timings[f"{claim_id}_step4_score"] = round(time.perf_counter() - t4, 3)

    return asdict(assess), timings

def run_factchain(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """엔드투엔드 파이프라인 실행 → JSON 리포트 반환(주장 단위 병렬 처리)."""
    load_dotenv(override=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")

    global MODEL_DEFAULT
    if model:
        MODEL_DEFAULT = model

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # STEP 1 — 주장 추출 (단일 호출; 직렬로 OK)
    t1 = time.perf_counter()
    claims = step1_extract_claims(client, text)
    timings["step1_extract_claims"] = round(time.perf_counter() - t1, 3)

    # 주장별 Step2~4 병렬 처리
    assessments: List[Dict[str, Any]] = []

    # 병렬도는 환경/요금제(RPS, TPM)·SerpAPI 쿼터 고려해서 적절히 조절
    max_workers = min(6, max(1, os.cpu_count() or 4))  # 예: 4~6
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for idx, claim in enumerate(claims, start=1):
            futures.append(ex.submit(_process_one_claim, client, claim, idx))

        # 제출 순서가 아닌 **완료 순서**로 모으되,
        # 출력은 원래 claim_id 순서가 필요하면 나중에 정렬
        results = []
        for fut in as_completed(futures):
            try:
                assess, tdict = fut.result()
                results.append((assess["claim_id"], assess, tdict))
            except Exception as e:
                # 안전장치: 실패한 주장은 placeholder로 기록
                cid = f"CX_{int(time.time()*1000)}"
                results.append((cid, {
                    "claim_id": cid,
                    "claim_text": "<processing_failed>",
                    "normalized_query": "",
                    "evidence": [],
                    "exists_evidence": False,
                    "source_trust_summary": {"tier_counts": {"1":0,"2":0,"3":0}},
                    "model_verdict": "uncertain",
                    "model_confidence": 0.0,
                    "credibility_score": 0.0,
                }, {f"{cid}_error": 0.0}))
                logger.exception("claim 병렬 처리 중 오류: %s", e)

    # 원래 순서로 정렬(Claim ID가 C1, C2… 형태라는 가정)
    results.sort(key=lambda x: (
        int(x[0][1:]) if x[0].startswith("C") and x[0][1:].isdigit() else 10**9
    ))
    for _, assess, tdict in results:
        assessments.append(assess)
        timings.update(tdict)

    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "meta": {
            "model": MODEL_DEFAULT,
            "search_provider": "Google Custom Search API",
            "max_results": MAX_RESULTS,
            "elapsed_sec": elapsed,
            "hl": SERP_HL,
            "gl": SERP_GL,
            "timings": timings,
            "parallel": {"enabled": True, "max_workers": max_workers},
        },
        "claims": assessments,
    }

# ──────────────────────────────────────────────────────────────────────
# 콘솔 출력 유틸(색상/라벨)
# ──────────────────────────────────────────────────────────────────────

def paint(s: str, c: str) -> str:
    return (COLORS.get(c, "") + s + COLORS["reset"]) if USE_COLOR else s

def verdict_label(v: str) -> str:
    mapping = {
        "supported": ("✅ 사실로 판단", "green"),
        "refuted": ("❌ 사실 아님", "red"),
        "uncertain": ("⚠️ 불확실", "yellow"),
    }
    text, color = mapping.get(v, (v, "yellow"))
    return paint(text, color)

def tc_get(tc: dict, k: int) -> int:
    # int 키/문자열 키 모두 대응
    return tc.get(k, tc.get(str(k), 0))

def tier_icons(tc: dict) -> str:
    t3, t2, t1 = tc_get(tc, 3), tc_get(tc, 2), tc_get(tc, 1)
    parts = [paint(f"🟢3:{t3}", "green"), paint(f"🟡2:{t2}", "yellow"), paint(f"🔴1:{t1}", "red")]
    return "출처 신뢰도: " + ", ".join(parts)

# ──────────────────────────────────────────────────────────────────────
# 엔트리포인트 — CLI + 데모
# ──────────────────────────────────────────────────────────────────────

DEMO_TEXT = """
아이슈타인은 1905년 특수상대성이론을 발표하여 시간과 공간의 개념을 혁신적으로 바꾸었다.  
이 이론에 따르면 빛의 속도는 관성계에 관계없이 일정하며,  
시간은 절대적이 아니라 관측자에 따라 상대적으로 달라진다.  
이후 1915년 그는 일반상대성이론을 통해 중력이 시공간의 곡률로 설명될 수 있음을 제시했다.
"""

"""
1) HTTP/3는 TCP가 아니라 QUIC 위에서 동작한다.
2) 태양은 지구를 중심으로 돈다.
3) 비트코인은 2009년에 처음 발행되었다.
4) AI가 생성한 그림은 저작권 보호를 받을 수 있다.
5) 한국은 2022년 FIFA 월드컵에서 우승했다.
6) 수은은 상온에서 액체 상태인 유일한 금속이다.
"""
def _phase_time(timings: dict, suffix: str) -> float:
    vals = [v for k, v in timings.items() if k.endswith(suffix)]
    return round(max(vals) if vals else 0.0, 3)

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="사실 검증 파이프라인")
    parser.add_argument("--text", type=str, default=None, help="직접 입력 텍스트")
    parser.add_argument("--model", type=str, default=None, help="사용할 OpenAI 모델")
    args = parser.parse_args(argv)

    text = args.text or DEMO_TEXT
    report = run_factchain(text, model=args.model)

    # JSON 원본 저장(협업/디버깅용)
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 콘솔 요약 출력
    global USE_COLOR, COLORS
    USE_COLOR = True
    COLORS = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }

    meta = report.get("meta", {})
    claims = report.get("claims", [])
    timings = meta.get("timings", {})
    step1_time = timings.get("step1_extract_claims", 0.0)
    step2_time = _phase_time(timings, "_step2_collect") 
    step3_time = _phase_time(timings, "_step3_evaluate")  
    step4_time = _phase_time(timings, "_step4_score")

    print("\n[검증 결과]")
    print("───────────────────────────────")
    print(f"모델: {meta.get('model')} | 검색엔진: {meta.get('search_provider')}")
    print(f"실행시간: {meta.get('elapsed_sec')}초 | 주장 수: {len(claims)}개\n")
    print(f"프로세스별 소요시간:\nSTEP 1 (주장 추출): {step1_time:.2f}초\n"
            f"STEP 2 (근거 수집): {step2_time:.2f}초\n"
            f"STEP 3 (근거 평가): {step3_time:.2f}초\n"
            f"STEP 4 (점수화)  : {step4_time:.2f}초\n")

    for c in claims:
        evid = c.get("evidence", [])
        tc = c.get("source_trust_summary", {}).get("tier_counts", {})
        print(paint(f"🔹 [{c.get('claim_id')}] {c.get('claim_text')}", "cyan"))
        print(f"   → 판정: {verdict_label(c.get('model_verdict',''))} "
        f"({int(float(c.get('model_confidence', 0))*100)}% 확신)")
        print(f"   → 신뢰점수: {c.get('credibility_score')}")
        print(f"   → {tier_icons(tc)}")
        for ev in evid[:3]:
            dom = ev.get('domain') or ''
            title = (ev.get('title') or '')[:70]
            print(f"      - {dom}: {title}...")
        print("───────────────────────────────")

    print("\n*전체 JSON은 output.json 에 저장")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())