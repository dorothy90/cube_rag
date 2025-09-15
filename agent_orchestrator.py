"""
Agent Orchestrator
- QueryAnalyzerAgent -> ContextHandlerAgent 순으로 실행하여 다음 액션을 결정합니다.
- 추후 Retrieval/Answer/Memory Agent로 확장 가능하도록 최소한의 인터페이스 제공.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from query_analyzer_agent import QueryAnalyzerAgent
from context_handler_agent import ContextHandlerAgent
from retrieval_agent import retrieve
from answer_generator import generate_answer

# .env 로드
load_dotenv()

class AgentOrchestrator:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.query_agent = QueryAnalyzerAgent(api_key=self.api_key)
        self.context_agent = ContextHandlerAgent(api_key=self.api_key)

    def run(self, question: str):
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
            contexts = [h.get("content") for h in hits]
            sources = [
                {
                    "metadata": h.get("metadata"),
                    "score": h.get("score"),
                }
                for h in hits
            ]
            # 4) 생성 단계: 컨텍스트 기반 답변 생성
            gen = generate_answer(question, contexts)
            return {
                "stage": "generation",
                "analysis": analysis,
                "decision": None,
                "contexts": contexts,
                "sources": sources,
                "answer": gen.get("answer"),
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
