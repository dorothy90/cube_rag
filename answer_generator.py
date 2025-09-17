"""
Answer Generator
- Retrieval 결과 컨텍스트를 근거로 한국어 답변을 생성합니다.

Env keys:
- OPENAI_API_KEY (필수)
- OPENAI_MODEL_NAME (기본: gpt-4o-mini)
"""

import os
from typing import List, Dict, Optional, Set

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def _get_model_name() -> str:
    name = os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
    return name


SYSTEM_PROMPT = """
당신은 한국어 기술 Q&A 어시스턴트입니다.

원칙:
- 우선, 사용자의 질문에 대해 일반적/보편적 지식으로 최선의 답변을 제공합니다.
- 제공된 컨텍스트가 있다면 그것을 우선 인용/참조합니다.
- 컨텍스트가 불충분하더라도, 가능한 한 실용적인 가이드와 예시를 제시합니다.
- 다만 확실하지 않은 부분은 추측임을 명확히 표시합니다(예: "추정", "권장").
- 과도한 장황함을 피하고, 실용적인 단계/체크리스트/코드 예시를 우선합니다.
- 코드 예시는 최소한으로, 실행 가능하고 핵심만 보여줍니다.

출력 형식:
1) 요약: 한두 문장으로 결론 요약
2) 핵심 포인트: 최대 5개 불릿
3) 단계별 가이드: 필요 시 번호 목록. 각 단계는 1~2문장으로 구체적으로
4) 코드 예시: 필요 시 코드블록 사용 (언어 태그 명시)
5) 근거/출처: Q&A/웹검색 근거를 간단 요약(있으면 URL/메타 포함)
6) 주의사항/베스트 프랙티스: 2~4개 불릿
7) 다음에 할 일(선택): 사용자가 바로 실행할 수 있는 간단 액션 1~2개
""".strip()


USER_PROMPT_TEMPLATE = """
질문: {question}

최근 대화 이력({hist_count}개 이하):
{history}

컨텍스트({ctx_count}개 이하):
"""

# 컨텍스트/히스토리 블록은 모델이 인용하기 쉽도록 구분선을 추가합니다.
CONTEXTS_WRAPPER = """
----- 컨텍스트 시작 -----
{contexts}
----- 컨텍스트 끝 -----

지침:
- 반드시 위 컨텍스트와 최근 대화 이력 내 표현/사실에 근거해 작성하세요.
- 형식을 그대로 따르세요(요약/핵심 포인트/단계/코드/주의사항/다음에 할 일).
- 필요 시 컨텍스트의 핵심 문장을 짧게 인용("…")하세요.
""".strip()

HISTORY_WRAPPER = """
----- 최근 대화 이력 시작 -----
{history}
----- 최근 대화 이력 끝 -----
""".strip()


def generate_answer(question: str, contexts: List[str], preface: Optional[str] = None, sources: Optional[List[Dict]] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict:
    model = ChatOpenAI(model=_get_model_name(), temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE + "\n\n" + (preface + "\n\n" if preface else "") + HISTORY_WRAPPER + "\n\n" + CONTEXTS_WRAPPER),
    ])

    joined_contexts = "\n\n".join([c.strip() for c in contexts if c and c.strip()])
    # role 기반 히스토리 문자열 생성 (최대 10개로 제한)
    history = history or []
    trimmed_history = history[-10:]
    def _format_turn(turn: Dict[str, str]) -> str:
        r = (turn.get("role") or "").lower()
        c = (turn.get("content") or "").strip()
        if not c:
            return ""
        label = "사용자" if r == "user" else ("시스템" if r == "system" else "어시스턴트")
        return f"[{label}] {c}"
    joined_history = "\n".join([s for s in map(_format_turn, trimmed_history) if s])
    formatted = prompt.format_messages(
        question=question.strip(),
        ctx_count=len(contexts),
        contexts=joined_contexts,
        hist_count=len(trimmed_history),
        history=joined_history,
    )

    resp = model.invoke(formatted)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    if preface:
        # 프리페이스를 본문 상단에 명시적으로 포함
        answer = f"알림: {preface}\n\n" + answer

    # 항상 출처 섹션을 하단에 추가
    def _truncate(text: str, n: int = 160) -> str:
        t = (text or "").strip()
        return t if len(t) <= n else (t[: n - 1] + "…")

    def _format_sources(srcs: Optional[List[Dict]]) -> str:
        items: List[str] = []
        if not srcs:
            return ""

        seen: Set[str] = set()
        count = 0
        for s in srcs:
            if count >= 5:
                break
            # 웹 결과 우선 처리 (url 존재)
            url = (s.get("url") if isinstance(s, dict) else None) or ""
            title = (s.get("title") if isinstance(s, dict) else None) or ""
            snippet = (s.get("snippet") if isinstance(s, dict) else None) or ""
            if url:
                key = f"web::{url}"
                if key in seen:
                    continue
                seen.add(key)
                label = "웹"
                ti = title or url
                items.append(f"- {label}: {ti} ({url})")
                count += 1
                continue

            # Q&A 메타 처리
            meta = s.get("metadata") if isinstance(s, dict) else None
            if isinstance(meta, dict):
                ts = (meta.get("timestamp") or "").strip()
                q = _truncate((meta.get("question") or "").strip(), 80)
                qa = _truncate(((meta.get("question_author") or meta.get("q_author") or "").strip()), 60)
                a = _truncate((meta.get("answer") or "").strip(), 80)
                aa = _truncate(((meta.get("answer_author") or meta.get("a_author") or "").strip()), 60)
                key = f"qa::{ts}::{q}"
                if key in seen:
                    continue
                seen.add(key)
                label = "Q&A"
                lines: List[str] = []
                if ts:
                    lines.append(f"[{ts}]")
                if q:
                    lines.append(f"Q: {q}")
                if qa:
                    lines.append(f"Q_Author: {qa}")
                if a:
                    lines.append(f"A: {a}")
                if aa:
                    lines.append(f"A_Author: {aa}")
                if lines:
                    items.append("- " + label + ":\n" + "\n".join(lines))
                count += 1

        if not items:
            return ""

        return "출처:\n" + "\n".join(items)

    sources_section = _format_sources(sources)
    if sources_section:
        answer = f"{answer}\n\n{sources_section}"

    return {"answer": answer, "sources": sources or []}


__all__ = ["generate_answer"]


