"""
Agent Orchestrator
- QueryAnalyzerAgent -> ContextHandlerAgent 순으로 실행하여 다음 액션을 결정합니다.
- 추후 Retrieval/Answer/Memory Agent로 확장 가능하도록 최소한의 인터페이스 제공.
"""

import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
from query_analyzer_agent import QueryAnalyzerAgent
from context_handler_agent import ContextHandlerAgent
from retrieval_agent import retrieve
from answer_generator import generate_answer
import os
from web_searcher import search_web

# .env 로드
load_dotenv()

class AgentOrchestrator:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.query_agent = QueryAnalyzerAgent(api_key=self.api_key)
        self.context_agent = ContextHandlerAgent(api_key=self.api_key)

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None):
        # 1) 질문 분석
        analysis = self.query_agent.analyze_query(question)
        # 2) 추가 맥락 필요 여부에 따라 분기
        if analysis.context_needed:
            decision = self.context_agent.handle_context_needed(analysis, question)
            return {
                "stage": "context_handling",
                "analysis": analysis,
                "decision": decision,
            }
        else:
            # 3) 리트리벌 실행
            hits = retrieve(question)
            # 질문 전용 컬렉션을 사용하는 경우, 메타의 question/answer로 컨텍스트를 재구성
            contexts = []
            sources = []
            for h in hits:
                meta = h.get("metadata") or {}
                q_meta = (meta.get("question") or "").strip()
                a_meta = (meta.get("answer") or "").strip()
                content = h.get("content")
                if q_meta or a_meta:
                    contexts.append(f"Q: {q_meta}\nA: {a_meta}".strip())
                else:
                    contexts.append(content)
                sources.append({"metadata": meta, "score": h.get("score")})

            # 단일 리트리벌 기반 컨센서스 규칙
            # has_direct_match = (top1 >= T_high) OR (상위 K 중 T_mid 이상 C개)
            try:
                t_high = float(os.getenv("DIRECT_MATCH_HIGH", "0.5"))
            except Exception:
                t_high = 0.5
            try:
                t_mid = float(os.getenv("DIRECT_MATCH_SCORE_THRESHOLD", "0.45"))
            except Exception:
                t_mid = 0.45
            try:
                top_k_cons = int(os.getenv("DIRECT_MATCH_TOPK", "3"))
            except Exception:
                top_k_cons = 3
            try:
                count_cons = int(os.getenv("DIRECT_MATCH_COUNT", "2"))
            except Exception:
                count_cons = 2

            def _as_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            raw_scores = [
                _as_float(s.get("score")) if isinstance(s, dict) else None
                for s in sources
            ]
            valid_scores = [v for v in raw_scores if v is not None]
            valid_scores.sort(reverse=True)
            top1 = valid_scores[0] if valid_scores else 0.0
            topk_scores = valid_scores[: max(0, top_k_cons)] if valid_scores else []
            consensus = sum(1 for v in topk_scores if v >= t_mid)
            has_direct_match = (top1 >= t_high) or (consensus >= count_cons)

            if contexts and has_direct_match:
                # Q&A에 있는 내용: 출처 표시 포함 답변
                preface = "요청하신 질문은 내부 Q&A 데이터에서 근거를 찾았습니다. 아래 출처를 참고하세요."
                gen = generate_answer(question, contexts, preface=preface, sources=sources, history=history)
            else:
                # Q&A에 없음: 환경변수로 웹검색 사용 여부 제어
                use_search = os.getenv("USE_WEB_SEARCH", "false").lower() in ("1", "true", "yes", "on")
                if use_search:
                    # 공급자 선택은 web_searcher 내부에서 처리
                    web_results = search_web(question, max_results=8)
                    web_contexts = []
                    for w in web_results:
                        title = (w.get("title") or "").strip()
                        url = (w.get("url") or "").strip()
                        snippet = (w.get("snippet") or "").strip()
                        if title or snippet:
                            web_contexts.append(f"{title}\n{snippet}\n출처: {url}")
                    preface = (
                        "내부 Q&A에 해당 내용이 없어, LLM 일반 지식과 웹 검색 결과를 근거로 답변합니다.\n"
                        "가능한 경우 출처(URL/메타)를 함께 표기합니다."
                    )
                    combined_contexts = contexts + web_contexts
                    # 내부 Q&A 직접 매칭이 없을 때는 벡터 DB 출처는 표기하지 않고,
                    # 웹 검색 출처만 전달한다.
                    gen = generate_answer(question, combined_contexts, preface=preface, sources=web_results, history=history)
                else:
                    preface = (
                        "내부 Q&A에 해당 내용이 없어, 웹 검색 없이 LLM 일반 지식만으로 답변합니다."
                    )
                    # 내부 Q&A 직접 매칭이 없는 경우 벡터 DB 출처는 표기하지 않는다.
                    gen = generate_answer(question, contexts, preface=preface, sources=None, history=history)
            return {
                "stage": "generation",
                "analysis": analysis,
                "decision": None,
                # 내부 Q&A 직접 매칭이 없는 경우, 벡터 DB 기반 Retrieval 결과는 표시하지 않음
                "contexts": contexts if has_direct_match else [],
                "sources": sources if has_direct_match else [],
                "answer": gen.get("answer"),
                "web": web_results if (not has_direct_match and use_search) else [],
            }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Agent Orchestrator")
    parser.add_argument("question", type=str, help="사용자 질문 문장")
    args = parser.parse_args()

    orchestrator = AgentOrchestrator()
    result = orchestrator.run(args.question)

    stage = result["stage"]
    print(f"단계: {stage}")

    analysis = result["analysis"]
    print(f"질문 유형: {analysis.question_type}")
    print(f"기술 스택: {', '.join(analysis.technical_stack)}")
    print(f"추가 맥락 필요: {'예' if analysis.context_needed else '아니오'}")

    if result["decision"]:
        decision = result["decision"]
        print(f"결정 액션: {decision.action}")
        print(f"신뢰도: {decision.confidence:.2f}")
        if decision.missing_context:
            print(f"부족한 맥락: {', '.join(decision.missing_context)}")
        if decision.suggested_questions:
            print(f"제안 질문: {', '.join(decision.suggested_questions[:3])}")
    elif stage == "generation":
        contexts = result.get("contexts", [])
        sources = result.get("sources", [])
        if contexts and sources:
            print("\n🔎 Retrieval Results:")
            for i, (c, s) in enumerate(zip(contexts, sources), 1):
                meta = s.get("metadata", {}) if isinstance(s, dict) else {}
                score = s.get("score") if isinstance(s, dict) else None
                ts = meta.get("timestamp", "")
                print(f"\n[{i}] score={score if score is not None else 'NA'} {ts}")
                preview = (c or "")[:400]
                print(preview)

        print("\n🧠 Answer:\n")
        print(result.get("answer", "(no answer)"))
