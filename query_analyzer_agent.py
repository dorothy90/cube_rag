"""
Query Analyzer Agent - 첫 번째 Agent 구현
한국어 개발자 질문을 분석하여 구조화된 정보를 추출합니다.
"""

import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

@dataclass
class QueryAnalysis:
    """질문 분석 결과를 담는 데이터 클래스"""
    question_type: str          # 질문 유형 (코드질문, 개념질문, 에러해결, 일반대화 등)
    technical_stack: List[str] # 관련 기술 스택 (Python, Django, FastAPI 등)
    difficulty_level: str      # 난이도 (초급, 중급, 고급)
    keywords: List[str]        # 핵심 키워드
    intent: str                # 질문 의도 (학습, 문제해결, 토론 등)
    priority: str             # 우선순위 (높음, 보통, 낮음)
    context_needed: bool      # 추가 맥락 정보 필요 여부
    domain: str               # 분류 도메인: python | sql | semiconductor | unknown
    domain_confidence: float  # 0.0 ~ 1.0

class QueryAnalyzerAgent:
    """질문 분석 Agent"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Query Analyzer Agent 초기화"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        
    def analyze_query(self, question: str) -> QueryAnalysis:
        """
        질문을 분석하여 구조화된 정보를 추출합니다.
        
        Args:
            question (str): 분석할 질문
            
        Returns:
            QueryAnalysis: 분석 결과
        """
        
        # 분석 프롬프트 생성
        prompt = f"""
다음은 한국어 개발자 채팅에서 나온 질문입니다. 이 질문을 분석하여 구조화된 정보를 추출해주세요.

질문: {question}

다음 JSON 형태로 분석 결과를 출력해주세요:
{{
    "question_type": "질문 유형 (코드질문/개념질문/에러해결/일반대화/토론)",
    "technical_stack": ["관련 기술 스택 리스트"],
    "difficulty_level": "난이도 (초급/중급/고급)",
    "keywords": ["핵심 키워드 리스트"],
    "intent": "질문 의도 (학습/문제해결/토론/정보요청)",
    "priority": "우선순위 (높음/보통/낮음)",
    "context_needed": true/false,
    "domain": "python|sql|semiconductor|unknown",
    "domain_confidence": 0.0
}}

분석 기준:
- question_type: 질문의 성격을 파악
- technical_stack: 언급된 기술이나 도구들
- difficulty_level: 질문의 복잡도
- keywords: 핵심 개념이나 용어들
- intent: 질문자가 원하는 것
- priority: 답변의 긴급도
- context_needed: 추가 정보가 필요한지 여부
- domain: 질문이 가장 관련 있는 도메인 분류
- domain_confidence: 0.0~1.0 사이 신뢰도 (모호하면 낮게)

추가 규칙(도메인 매핑, 강한 힌트):
- 질문에 'agile ipa' 또는 'datalake'가 포함되면 domain은 'sql'로 설정하고 domain_confidence는 0.9 이상으로 설정하세요.
- 질문에 'syld' 또는 'nand' 또는 'dram'이 포함되면 domain은 'semiconductor'로 설정하고 domain_confidence는 0.9 이상으로 설정하세요.
- 위 규칙은 다른 신호보다 우선 적용하며, 키워드는 대소문자 구분 없이 포함 여부(부분 일치)로 판단하세요.
"""
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 한국어 개발자 질문을 분석하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 응답 파싱
            result = response.choices[0].message.content
            
            # JSON 파싱 (마크다운 코드 블록 제거)
            try:
                # 마크다운 코드 블록 제거
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()
                
                analysis_data = json.loads(result)
                dom = (analysis_data.get("domain") or "unknown").strip().lower()
                if dom not in ("python", "sql", "semiconductor", "unknown"):
                    dom = "unknown"
                try:
                    dom_conf = float(analysis_data.get("domain_confidence", 0.0))
                except Exception:
                    dom_conf = 0.0
                
                # QueryAnalysis 객체 생성
                return QueryAnalysis(
                    question_type=analysis_data.get("question_type", "일반대화"),
                    technical_stack=analysis_data.get("technical_stack", []),
                    difficulty_level=analysis_data.get("difficulty_level", "중급"),
                    keywords=analysis_data.get("keywords", []),
                    intent=analysis_data.get("intent", "정보요청"),
                    priority=analysis_data.get("priority", "보통"),
                    context_needed=analysis_data.get("context_needed", False),
                    domain=dom,
                    domain_confidence=dom_conf,
                )
                
            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 실패: {result}")
                # 기본값 반환
                return QueryAnalysis(
                    question_type="일반대화",
                    technical_stack=[],
                    difficulty_level="중급",
                    keywords=[],
                    intent="정보요청",
                    priority="보통",
                    context_needed=False,
                    domain="unknown",
                    domain_confidence=0.0,
                )
                
        except Exception as e:
            print(f"❌ 질문 분석 실패: {e}")
            # 기본값 반환
            return QueryAnalysis(
                question_type="일반대화",
                technical_stack=[],
                difficulty_level="중급",
                keywords=[],
                intent="정보요청",
                priority="보통",
                context_needed=False,
                domain="unknown",
                domain_confidence=0.0,
            )
    
    def batch_analyze(self, questions: List[str]) -> List[QueryAnalysis]:
        """
        여러 질문을 배치로 분석합니다.
        
        Args:
            questions (List[str]): 분석할 질문 리스트
            
        Returns:
            List[QueryAnalysis]: 분석 결과 리스트
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"🔄 질문 {i+1}/{len(questions)} 분석 중...")
            result = self.analyze_query(question)
            results.append(result)
            
        return results

if __name__ == "__main__":
    # 간단한 사용 예시
    agent = QueryAnalyzerAgent()
    analysis = agent.analyze_query("Django에서 모델 필드에 기본값을 설정하는 방법이 궁금합니다.")
    print(f"질문 유형: {analysis.question_type}")
    print(f"기술 스택: {', '.join(analysis.technical_stack)}")
    print(f"추가 맥락 필요: {'예' if analysis.context_needed else '아니오'}")