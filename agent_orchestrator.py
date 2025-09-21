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

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None, last_domain: Optional[str] = None):
        # 1) 질문 분석
        analysis = self.query_agent.analyze_query(question)
        # 도메인 결정 로직
        try:
            dom_thr = float(os.getenv("DOMAIN_CONFIDENCE_THRESHOLD", "0.6"))
        except Exception:
            dom_thr = 0.6
        domain = getattr(analysis, "domain", "unknown") or "unknown"
        dom_conf = float(getattr(analysis, "domain_confidence", 0.0) or 0.0)

        def _map_collection(d: str) -> Optional[str]:
            mapping = {
                "python": "qa_questions_python",
                "sql": "qa_questions_sql",
                "semiconductor": "qa_questions_semiconductor",
            }
            return mapping.get((d or "").lower())

        # 새 채팅(히스토리 없음)에서 애매하면 도메인 확인 질문 반환
        is_new_chat = not history or len(history) == 0
        if is_new_chat:
            if domain not in ("python", "sql", "semiconductor") or dom_conf < dom_thr:
                clarify_msg = "이 질문은 Python, SQL, 반도체 중 어떤 도메인과 가장 관련이 있나요? (예: '파이썬' 또는 'SQL' 또는 '반도체')"
                return {
                    "stage": "clarify_domain",
                    "analysis": analysis,
                    "decision": None,
                    "answer": clarify_msg,
                    "selected_domain": None,
                }
            selected_domain = domain
        else:
            # 후속 질문: 애매하면 이전 도메인 사용, 없으면 확인 질문
            if domain in ("python", "sql", "semiconductor") and dom_conf >= dom_thr:
                selected_domain = domain
            elif last_domain in ("python", "sql", "semiconductor"):
                selected_domain = last_domain
            else:
                clarify_msg = "이 질문은 Python, SQL, 반도체 중 어떤 도메인과 가장 관련이 있나요?"
                return {
                    "stage": "clarify_domain",
                    "analysis": analysis,
                    "decision": None,
                    "answer": clarify_msg,
                    "selected_domain": None,
                }

        collection_name = _map_collection(selected_domain)
        # 2) 추가 맥락 필요 여부에 따라 분기
        if analysis.context_needed:
            decision = self.context_agent.handle_context_needed(analysis, question)
            return {
                "stage": "context_handling",
                "analysis": analysis,
                "decision": decision,
                "selected_domain": selected_domain,
            }
        else:
            # 3) 리트리벌 실행
            hits = retrieve(question, collection_name=collection_name)
            # 질문 전용 컬렉션을 사용하는 경우, 메타의 question/answer로 컨텍스트를 재구성
            contexts = []
            sources = []
            for h in hits:
                meta = h.get("metadata") or {}
                q_meta = (meta.get("question") or "").strip()
                # '||' 분리 (서버 규칙과 동일)
                ans_csv = meta.get("answers")
                tokens = [t.strip() for t in str(ans_csv or '').split('||') if t and t.strip()]
                a_meta = " ".join(tokens)
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

            web_results = []
            if contexts and has_direct_match:
                # Q&A에 있는 내용: 출처 표시 포함 답변
                preface = f"요청하신 질문은 내부 Q&A 데이터(도메인: {selected_domain})에서 근거를 찾았습니다. 아래 출처를 참고하세요."
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
                        f"내부 Q&A(도메인: {selected_domain})에 직접 일치하는 근거가 없어, LLM 일반 지식과 웹 검색 결과를 근거로 답변합니다.\n"
                        "가능한 경우 출처(URL/메타)를 함께 표기합니다."
                    )
                    combined_contexts = contexts + web_contexts
                    # 내부 Q&A 직접 매칭이 없을 때는 벡터 DB 출처는 표기하지 않고,
                    # 웹 검색 출처만 전달한다.
                    gen = generate_answer(question, combined_contexts, preface=preface, sources=web_results, history=history)
                else:
                    preface = f"내부 Q&A(도메인: {selected_domain})에 해당 내용이 없어, 웹 검색 없이 LLM 일반 지식만으로 답변합니다."
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
                "selected_domain": selected_domain,
            }

def main(question: str, history: Optional[List[Dict[str, str]]] = None, last_domain: Optional[str] = None) -> Dict:
    orchestrator = AgentOrchestrator()
    return orchestrator.run(question, history=history, last_domain=last_domain)


if __name__ == "__main__":
    q = os.getenv("QUESTION", "Django 기본값 설정 방법?")
    out = main(q)
    print(out.get("answer") or "(no answer)")
