"""
Context Handler Agent - 추가 맥락 처리 Agent
추가 맥락이 필요한 질문을 처리하는 전략을 결정합니다.
"""
#%%
import os
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from query_analyzer_agent import QueryAnalysis

# .env 파일 로드
load_dotenv()

@dataclass
class ContextDecision:
    """맥락 처리 결정 결과"""
    action: str                    # "generate_answer" | "request_context" | "search_memory"
    confidence: float             # 결정 신뢰도 (0.0 ~ 1.0)
    missing_context: List[str]    # 부족한 맥락 정보
    suggested_questions: List[str] # 제안 질문들
    memory_context: Optional[Dict] # 메모리에서 찾은 관련 정보

class ContextHandlerAgent:
    """맥락 처리 Agent"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Context Handler Agent 초기화"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        
        # 간단한 메모리 저장소 (실제로는 Redis 사용)
        self.memory_store = {}
        
    def handle_context_needed(self, query_analysis: QueryAnalysis, 
                            user_question: str) -> ContextDecision:
        """
        추가 맥락이 필요한 질문을 처리합니다.
        
        Args:
            query_analysis (QueryAnalysis): 질문 분석 결과
            user_question (str): 사용자 질문
            
        Returns:
            ContextDecision: 맥락 처리 결정
        """
        
        # 1. 메모리에서 관련 정보 검색
        memory_context = self._search_memory(query_analysis)
        
        # 2. 맥락 충분성 판단
        context_decision = self._decide_context_action(
            query_analysis, user_question, memory_context
        )
        
        return context_decision
    
    def _search_memory(self, query_analysis: QueryAnalysis) -> Optional[Dict]:
        """메모리에서 관련 정보를 검색합니다."""
        # 실제로는 Redis나 벡터 DB에서 검색
        # 여기서는 간단한 키워드 매칭으로 시뮬레이션
        
        relevant_memories = []
        for memory_key, memory_data in self.memory_store.items():
            # 키워드 기반 매칭
            if any(keyword in memory_key.lower() for keyword in query_analysis.keywords):
                relevant_memories.append(memory_data)
        
        if relevant_memories:
            return {
                "found": True,
                "count": len(relevant_memories),
                "memories": relevant_memories[:3]  # 최대 3개
            }
        
        return {"found": False, "count": 0, "memories": []}
    
    def _decide_context_action(self, query_analysis: QueryAnalysis, 
                              user_question: str, memory_context: Optional[Dict]) -> ContextDecision:
        """맥락 처리 액션을 결정합니다."""
        

        # 분석 프롬프트 생성
        prompt = f"""
다음은 추가 맥락이 필요한 한국어 개발자 질문입니다. 
이 질문에 대해 어떤 액션을 취해야 할지 결정해주세요.

질문: {user_question}

질문 분석 결과:
- 질문 유형: {query_analysis.question_type}
- 기술 스택: {', '.join(query_analysis.technical_stack)}
- 키워드: {', '.join(query_analysis.keywords)}
- 의도: {query_analysis.intent}
- 추가 맥락 필요: {query_analysis.context_needed}

메모리 검색 결과:
- 관련 정보 발견: {memory_context['found'] if memory_context else False}
- 관련 대화 수: {memory_context['count'] if memory_context else 0}

다음 JSON 형태로 결정 결과를 출력해주세요:
{{
    "action": "액션 (generate_answer/request_context/search_memory)",
    "confidence": 0.0~1.0,
    "missing_context": ["부족한 맥락 정보 리스트"],
    "suggested_questions": ["제안 질문 리스트"]
}}

액션 설명:
- generate_answer: 메모리 정보로 충분히 답변 가능
- request_context: 사용자에게 추가 정보 요청 필요
- search_memory: 더 많은 메모리 검색 필요

판단 기준:
1. 질문이 개념 정의/설명이며 추가 맥락이 불필요하면 generate_answer
2. 메모리에 관련 정보가 충분히 있으면 generate_answer
3. 질문이 너무 모호하면 request_context
4. 에러 메시지나 코드가 없으면 request_context
5. 이전 대화와 연관되지만 정보가 부족하면 search_memory
"""
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 맥락 처리 전략을 결정하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 응답 파싱
            result = response.choices[0].message.content
            
            # JSON 파싱 (마크다운 코드 블록 제거)
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            decision_data = json.loads(result)
            
            return ContextDecision(
                action=decision_data.get("action", "request_context"),
                confidence=decision_data.get("confidence", 0.5),
                missing_context=decision_data.get("missing_context", []),
                suggested_questions=decision_data.get("suggested_questions", []),
                memory_context=memory_context
            )
            
        except Exception as e:
            print(f"❌ 맥락 처리 결정 실패: {e}")
            # 기본값 반환
            return ContextDecision(
                action="request_context",
                confidence=0.3,
                missing_context=["구체적인 상황 설명"],
                suggested_questions=["어떤 부분에서 문제가 발생했나요?"],
                memory_context=memory_context
            )
    
    def add_memory(self, key: str, data: Dict):
        """메모리에 정보를 추가합니다."""
        self.memory_store[key] = data
    
    def generate_context_request_message(self, context_decision: ContextDecision, 
                                       query_analysis: QueryAnalysis) -> str:
        """사용자에게 보낼 맥락 요청 메시지를 생성합니다."""
        
        # 질문 유형별 맞춤 메시지
        if query_analysis.question_type == "에러해결":
            base_message = "에러 해결을 위해 추가 정보가 필요합니다."
            suggestions = [
                "에러 메시지 전체를 공유해주세요",
                "관련 코드를 보여주세요",
                "언제 에러가 발생하는지 설명해주세요"
            ]
        elif query_analysis.question_type == "코드질문":
            base_message = "코드 관련 질문에 답하기 위해 추가 정보가 필요합니다."
            suggestions = [
                "현재 코드를 공유해주세요",
                "사용 중인 라이브러리 버전을 알려주세요",
                "원하는 결과를 설명해주세요"
            ]
        elif query_analysis.question_type == "개념질문":
            base_message = "개념 질문에 더 정확한 답변을 위해 추가 정보가 필요합니다."
            suggestions = [
                "구체적인 사용 사례를 설명해주세요",
                "현재 프로젝트 상황을 알려주세요",
                "어떤 부분이 궁금한지 구체적으로 말해주세요"
            ]
        else:
            base_message = "더 정확한 답변을 위해 추가 정보가 필요합니다."
            suggestions = context_decision.suggested_questions
        
        # 메시지 구성
        message = f"🤔 {base_message}\n\n"
        message += "다음 정보를 공유해주시면 더 정확한 답변을 드릴 수 있습니다:\n"
        
        for i, suggestion in enumerate(suggestions[:3], 1):
            message += f"{i}. {suggestion}\n"
        
        if context_decision.memory_context and context_decision.memory_context['found']:
            message += f"\n💡 이전 대화에서 관련 정보를 {context_decision.memory_context['count']}개 찾았습니다."
        
        return message

if __name__ == "__main__":
    # 간단한 사용 예시
    agent = ContextHandlerAgent()
    
    # 테스트용 메모리 추가
    agent.add_memory("django 모델 필드 기본값", {
        "question": "Django 모델 필드 기본값 설정",
        "answer": "default 매개변수 사용",
        "timestamp": "2024-01-01"
    })
    
    msg = "파이썬에서 uv 라이브러리는 뭔가요"
    # Query Analyzer로 분석
    from query_analyzer_agent import QueryAnalyzerAgent
    query_analyzer = QueryAnalyzerAgent()
    query_analysis = query_analyzer.analyze_query(msg)
    
    # Context Handler로 처리
    context_decision = agent.handle_context_needed(query_analysis,msg)
    print(f"결정된 액션: {context_decision.action}")
    print(f"신뢰도: {context_decision.confidence:.2f}")


# %%
