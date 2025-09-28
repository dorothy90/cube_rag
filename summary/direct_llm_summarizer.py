# │ │ # LLM이 분석 결과를 보고 자동으로 판단                                                          │ │
# │ │ # - 토픽이 많고 중요하면 → 보고서 생성 + 이메일 전송                                            │ │
# │ │ # - 토픽이 적거나 품질이 낮으면 → 보고서만 생성                                                 │ │
# │ │ # - 오류가 많으면 → 아무것도 하지 않음                                                          │ │
# │ ╰───────────────────────────────────────────────

"""
직접 LLM 요약기
웹스크래핑 후 임베딩/클러스터링 없이 바로 LLM으로 주제별 요약 생성
"""

import logging
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import time

# LLM APIs
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ OpenAI API: pip install openai")

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("⚠️ Anthropic API: pip install anthropic")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectLLMSummarizer:
    """직접 LLM 요약기 - 임베딩/클러스터링 우회"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_provider: str = "openai",
        max_retries: int = 3,
        request_delay: float = 1.0,
        max_tokens_per_request: int = 4000
    ):
        """
        초기화
        
        Args:
            openai_api_key: OpenAI API 키
            anthropic_api_key: Anthropic API 키  
            default_provider: 기본 LLM 제공자 ("openai" 또는 "anthropic")
            max_retries: 최대 재시도 횟수
            request_delay: 요청 간 대기 시간 (초)
            max_tokens_per_request: 요청당 최대 토큰 수
        """
        self.openai_client = None
        self.anthropic_client = None
        self.default_provider = default_provider
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.max_tokens_per_request = max_tokens_per_request
        self.embedding_cache = {}  # 임베딩 캐시
        self.batch_embeddings = {}  # 배치 임베딩 저장소
        
        # 환경변수로부터 키 로드 (명시 인자 우선)
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # OpenAI 클라이언트 초기화
        if openai_api_key and HAS_OPENAI:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI 클라이언트 초기화 완료")
            except Exception as e:
                logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        
        # Anthropic 클라이언트 초기화  
        if anthropic_api_key and HAS_ANTHROPIC:
            try:
                self.anthropic_client = Anthropic(api_key=anthropic_api_key)
                logger.info("✅ Anthropic 클라이언트 초기화 완료")
            except Exception as e:
                logger.error(f"❌ Anthropic 클라이언트 초기화 실패: {e}")
        
        # 사용 가능한 클라이언트 확인
        if not self.openai_client and not self.anthropic_client:
            logger.warning("⚠️ LLM 클라이언트가 없습니다.")
    
    def summarize_messages_directly(
        self,
        messages: List[Dict[str, Any]],
        llm_provider: str = None,
        model: str = None,
        language: str = "korean",
        max_topics: int = 10
    ) -> Dict[str, Any]:
        """
        메시지들을 직접 LLM으로 주제별 요약
        
        Args:
            messages: 요약할 메시지 리스트 (각 메시지는 {'content': str, 'reaction_count': int, 'message_id': str} 형태)
            llm_provider: LLM 제공자 ("openai", "anthropic", "auto")
            model: 사용할 모델명
            language: 결과 언어 ("korean", "english")
            max_topics: 최대 주제 수
            
        Returns:
            주제별 요약 결과 (각 주제에 top_messages 필드 포함)
        """
        if not messages:
            logger.warning("요약할 메시지가 없습니다")
            return {}
        
        logger.info(f"🤖 직접 LLM 요약 시작 - {len(messages)}개 메시지")
        
        # LLM 제공자 선택
        if llm_provider is None:
            llm_provider = self._select_best_provider()
        
        # 메시지 전처리 및 청킹
        processed_chunks = self._prepare_message_chunks(messages)
        
        all_topics = []
        raw_chunk_topics: List[Dict[str, Any]] = []
        
        # 청크별로 요약 처리
        for i, chunk in enumerate(processed_chunks, 1):
            logger.info(f"청크 {i}/{len(processed_chunks)} 처리 중...")
            
            try:
                chunk_topics = self._summarize_chunk(
                    chunk=chunk,
                    llm_provider=llm_provider,
                    model=model,
                    language=language,
                    max_topics=max_topics
                )
                
                if chunk_topics:
                    # 청크 내에서 관련 메시지 부착 및 통계 계산
                    enriched_chunk_topics, _ = self._ensure_top_messages(chunk_topics, chunk)
                    for t in enriched_chunk_topics:
                        t['source_chunk_index'] = i - 1

                    all_topics.extend(enriched_chunk_topics)
                    # 청크별 주제 보존 (소주제 용도, 연관 메시지 포함)
                    raw_chunk_topics.append({
                        "chunk_index": i - 1,
                        "topics": enriched_chunk_topics,
                    })
                
                # Rate limiting
                if i < len(processed_chunks) and self.request_delay > 0:
                    time.sleep(self.request_delay)
                    
            except Exception as e:
                logger.error(f"청크 {i} 처리 실패: {e}")
                continue
        
        # 최종 통합 및 중복 제거
        final_summary = self._consolidate_topics(
            topics=all_topics,
            max_topics=max_topics,
            llm_provider=llm_provider,
            model=model,
            language=language
        )
        
        # 백업: 상위 메시지가 없는 주제에 대해 직접 추출 + 연관 메시지 부착
        final_summary, unassigned_messages = self._ensure_top_messages(final_summary, messages)

        # 최종 주제에 병합된 소주제 정보 부착
        final_summary = self._attach_merged_subtopics(final_summary, all_topics)
        
        # reaction_count > 2인 미배정 메시지들 필터링
        high_reaction_unassigned = [
            msg for msg in unassigned_messages 
            if msg.get('reaction_count', 0) > 2
        ]
        
        # 결과 구성
        result = {
            'topics': final_summary,
            'raw_chunk_topics': raw_chunk_topics,
            'statistics': {
                'total_messages': len(messages),
                'processed_chunks': len(processed_chunks),
                'total_topics': len(final_summary),
                'assigned_messages': len(messages) - len(unassigned_messages),
                'unassigned_messages': len(unassigned_messages),
                'high_reaction_unassigned': len(high_reaction_unassigned),
                'llm_provider': llm_provider,
                'processing_time': datetime.now().isoformat(),
                'processing_method': 'direct_llm_summary'
            },
            'unassigned_high_reaction_messages': [
                {
                    'content': msg.get('content', ''),
                    'reaction_count': msg.get('reaction_count', 0),
                    'message_id': msg.get('message_id', ''),
                    'content_length': len(msg.get('content', ''))
                }
                for msg in sorted(high_reaction_unassigned, 
                                key=lambda x: x.get('reaction_count', 0), 
                                reverse=True)
            ]
        }
        
        logger.info(f"🎉 직접 LLM 요약 완료: {len(final_summary)}개 주제")
        if high_reaction_unassigned:
            logger.warning(f"⚠️ reaction_count > 2인 미배정 메시지 {len(high_reaction_unassigned)}개 발견")
            for msg in high_reaction_unassigned[:5]:  # 상위 5개만 로그 출력
                logger.warning(f"  - (👍 {msg.get('reaction_count', 0)}) {msg.get('content', '')[:50]}...")
        
        return result
    
    def _prepare_message_chunks(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """메시지들을 처리 가능한 청크로 분할"""
        
        # 메시지 전처리 (중복 제거, 빈 메시지 제거)
        cleaned_messages = []
        seen = set()
        
        for msg in messages:
            content = msg.get('content', '').strip()
            if content and len(content) > 3 and content not in seen:
                cleaned_messages.append(msg)
                seen.add(content)
        
        logger.info(f"메시지 전처리: {len(messages)} → {len(cleaned_messages)}개")
        
        # 토큰 수 기준으로 청킹
        chunks = []
        current_chunk = []
        current_length = 0
        
        for msg in cleaned_messages:
            content = msg.get('content', '')
            # 대략적인 토큰 수 계산 (한글은 약 0.5토큰/글자)
            msg_tokens = len(content) * 0.7  # 여유를 두고 계산
            
            if current_length + msg_tokens > self.max_tokens_per_request and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [msg]
                current_length = msg_tokens
            else:
                current_chunk.append(msg)
                current_length += msg_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"메시지 청킹: {len(chunks)}개 청크 생성")
        return chunks
    
    def _summarize_chunk(
        self,
        chunk: List[Dict[str, Any]],
        llm_provider: str,
        model: str = None,
        language: str = "korean",
        max_topics: int = 10
    ) -> List[Dict[str, Any]]:
        """단일 청크를 주제별로 요약"""
        
        # 프롬프트 생성
        prompt = self._build_direct_summary_prompt(chunk, language, max_topics)
        
        # LLM 호출
        if llm_provider == "openai" and self.openai_client:
            return self._call_openai_for_summary(prompt, model, chunk)
        elif llm_provider == "anthropic" and self.anthropic_client:
            return self._call_anthropic_for_summary(prompt, model, chunk)
        else:
            logger.error(f"LLM 제공자 '{llm_provider}'를 사용할 수 없습니다")
            return []
    
    def _build_direct_summary_prompt(
        self,
        messages: List[Dict[str, Any]],
        language: str,
        max_topics: int
    ) -> str:
        """직접 요약용 프롬프트 구성"""
        
        # 메시지 리스트 구성 (실제 정수형 message_id 사용)
        messages_str = "\n".join([
            f"ID:{msg.get('message_id', i)} (반응수: {msg.get('reaction_count', 0)}) {msg.get('content', '')}"
            for i, msg in enumerate(messages)
        ])
        
        # 🔍 디버깅: 프롬프트에 포함된 실제 message_id 확인
        message_ids_in_prompt = [str(msg.get('message_id', i)) for i, msg in enumerate(messages)]
        logger.info(f"🔍 프롬프트에 포함된 message_id들: {message_ids_in_prompt[:10]}... (총 {len(message_ids_in_prompt)}개)")
        logger.info(f"🔍 첫 번째 메시지 예시: ID:{message_ids_in_prompt[0]} (반응수: {messages[0].get('reaction_count', 0)}) {messages[0].get('content', '')[:50]}...")
        
        if language == "korean":
            prompt = f"""다음은 채팅방에서 수집된 {len(messages)}개의 메시지입니다. 이 메시지들을 분석하여 주제별로 분류하고 각 주제마다 핵심 내용을 한 문장으로 요약해주세요.

메시지 목록:
{messages_str}

요구사항:
1. 최대 {max_topics}개의 주요 주제로 분류
2. 각 주제마다 **한 문장으로 핵심 요약**
3. 주제명은 **구체적이고 설명적**으로 작성 (30-50자)
4. 관련성 낮거나 중복되는 내용은 통합 또는 제외
5. **반드시** 각 주제에 관련된 메시지 ID들을 포함 (ID: 뒤에 제시된 실제 정수형 message_id 사용)
6. JSON 형식으로 응답하며, **related_message_ids 필드는 필수입니다**

응답 형식 (모든 필드가 필수):
```json
[
  {{
    "topic_name": "구체적인 주제명",
    "summary": "해당 주제의 핵심 내용을 한 문장으로 요약",
    "message_count": 관련_메시지_개수,
    "keywords": ["주요", "키워드", "리스트"],
    "related_message_ids": ["0", "5", "12"]
  }}
]
```

**중요: related_message_ids는 프롬프트의 ID: 뒤에 표시된 실제 정수형 message_id만 사용하세요. 임의 인덱스나 접두어를 만들지 마세요.**

예시:
```json
[
  {{
    "topic_name": "팀 회의 일정 조율 및 회의실 예약",
    "summary": "다음 주 화요일 오후 2시 팀 회의를 위해 A동 3층 회의실을 예약하기로 결정했습니다.",
    "message_count": 3,
    "keywords": ["회의", "일정", "예약", "화요일"],
    "related_message_ids": ["15", "23", "47"]
  }}
]
```

주제별 요약:"""

        else:  # English
            prompt = f"""The following are {len(messages)} messages collected from a chat room. Please analyze these messages, classify them by topic, and provide a one-sentence summary for each topic.

Message list:
{messages_str}

Requirements:
1. Classify into maximum {max_topics} major topics
2. **One sentence core summary** for each topic
3. Topic names should be **specific and descriptive** (30-50 characters)
4. Integrate or exclude irrelevant or duplicate content
5. Include related message IDs (integer indices provided in ID: format) for each topic
6. Respond in JSON format

Response format:
```json
[
  {{
    "topic_name": "Specific topic name",
    "summary": "One sentence summary of the topic's core content",
    "message_count": number_of_related_messages,
    "keywords": ["key", "word", "list"],
    "related_message_ids": ["0", "5", "12"]
  }}
]
```

**Important: Use only integer indices from ID:0, ID:1, ID:2 format. Do not use prefixes like "msg_".**

Topic summaries:"""
        
        return prompt
    
    def _call_openai_for_summary(self, prompt: str, model: str = None, chunk: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """OpenAI API 호출하여 요약 생성"""
        try:
            model = model or "gpt-4o-mini"  # 비용 효율적인 모델
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 채팅 메시지 분석 전문가입니다. 주어진 메시지들을 주제별로 분류하고 각 주제의 핵심을 정확히 요약합니다. 항상 유효한 JSON 형식으로 응답하세요."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3,
                top_p=0.9
            )
            
            content = response.choices[0].message.content.strip()
            
            # 🔍 디버깅: LLM 원본 응답 로깅
            logger.info(f"🔍 OpenAI 원본 응답: {content[:500]}...")
            
            # JSON 파싱
            try:
                # ```json 제거
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1]
                
                topics = json.loads(content.strip())
                
                # 🔍 디버깅: 파싱된 topics의 related_message_ids 확인
                logger.info(f"🔍 파싱된 주제 개수: {len(topics) if isinstance(topics, list) else 'N/A'}")
                if isinstance(topics, list):
                    for i, topic in enumerate(topics):
                        topic_name = topic.get('topic_name', 'Unknown')
                        related_ids = topic.get('related_message_ids', [])
                        logger.info(f"🔍 주제 {i+1} '{topic_name}': related_message_ids = {related_ids}")
                
                if isinstance(topics, list):
                    # related_message_ids 검증 및 보완
                    validated_topics = self._validate_and_fix_topics(topics, chunk)
                    return validated_topics
                else:
                    logger.error("응답이 리스트 형태가 아닙니다")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {e}")
                logger.error(f"원본 응답: {content}")
                return []
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            return []
    
    def _call_anthropic_for_summary(self, prompt: str, model: str = None, chunk: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Anthropic API 호출하여 요약 생성"""
        try:
            model = model or "claude-3-haiku-20240307"  # 빠르고 저렴한 모델
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            
            # 🔍 디버깅: LLM 원본 응답 로깅
            logger.info(f"🔍 Anthropic 원본 응답: {content[:500]}...")
            
            # JSON 파싱
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1]
                
                topics = json.loads(content.strip())
                
                # 🔍 디버깅: 파싱된 topics의 related_message_ids 확인
                logger.info(f"🔍 파싱된 주제 개수: {len(topics) if isinstance(topics, list) else 'N/A'}")
                if isinstance(topics, list):
                    for i, topic in enumerate(topics):
                        topic_name = topic.get('topic_name', 'Unknown')
                        related_ids = topic.get('related_message_ids', [])
                        logger.info(f"🔍 주제 {i+1} '{topic_name}': related_message_ids = {related_ids}")
                
                if isinstance(topics, list):
                    # related_message_ids 검증 및 보완
                    validated_topics = self._validate_and_fix_topics(topics, chunk)
                    return validated_topics
                else:
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 실패: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Anthropic API 호출 실패: {e}")
            return []
    
    def _consolidate_topics(
        self,
        topics: List[Dict[str, str]],
        max_topics: int,
        llm_provider: str,
        model: str = None,
        language: str = "korean"
    ) -> List[Dict[str, str]]:
        """여러 청크의 주제들을 통합하고 중복 제거"""
        
        if not topics:
            return []
        
        # 단일 청크인 경우 그대로 반환
        if len(topics) <= max_topics:
            return topics[:max_topics]
        
        logger.info(f"주제 통합 시작: {len(topics)}개 → 최대 {max_topics}개")
        
        # 중복 제거 및 통합을 위한 LLM 호출
        consolidation_prompt = self._build_consolidation_prompt(
            topics, max_topics, language
        )
        
        if llm_provider == "openai" and self.openai_client:
            consolidated = self._call_openai_for_summary(consolidation_prompt, model)
        elif llm_provider == "anthropic" and self.anthropic_client:
            consolidated = self._call_anthropic_for_summary(consolidation_prompt, model)
        else:
            # Fallback: 단순 중복 제거
            consolidated = self._simple_deduplication(topics, max_topics)
        
        return consolidated[:max_topics]
    
    def _build_consolidation_prompt(
        self,
        topics: List[Dict[str, str]],
        max_topics: int,
        language: str
    ) -> str:
        """주제 통합용 프롬프트 구성"""
        
        topics_str = ""
        for i, topic in enumerate(topics, 1):
            topics_str += f"{i}. 주제: {topic.get('topic_name', 'Unknown')}\n"
            topics_str += f"   요약: {topic.get('summary', 'No summary')}\n"
            topics_str += f"   키워드: {', '.join(topic.get('keywords', []))}\n"
            topics_str += f"   관련 메시지 ID: {topic.get('related_message_ids', [])}\n\n"
        
        if language == "korean":
            prompt = f"""다음은 여러 청크에서 추출된 {len(topics)}개의 주제들입니다. 이들을 분석하여 중복되거나 유사한 주제들을 통합하고, 최종적으로 **가장 중요한 {max_topics}개 주제**로 정리해주세요.

추출된 주제들:
{topics_str}

요구사항:
1. 유사하거나 중복되는 주제들을 통합
2. 가장 중요하고 의미있는 {max_topics}개 주제만 선별
3. 각 주제의 요약문을 더욱 포괄적이고 정확하게 개선
4. **통합 시 관련 메시지 ID들을 모두 합쳐서 보존** (중복 제거). 관련 메시지 ID는 원본 데이터의 message_id 문자열입니다.
5. JSON 형식으로 응답

응답 형식:
```json
[
  {{
    "topic_name": "통합된 주제명",
    "summary": "개선된 한 문장 요약",
    "message_count": 예상_메시지_개수,
    "keywords": ["통합된", "키워드", "리스트"],
    "related_message_ids": ["msg_xxx", "msg_yyy", "msg_zzz"]
  }}
]
```

**중요: 통합되는 주제들의 모든 related_message_ids를 합치되, 중복은 제거하고 원본 message_id 문자열로 반환하세요.**

최종 {max_topics}개 주제:"""

        else:
            prompt = f"""The following are {len(topics)} topics extracted from multiple chunks. Please analyze them, consolidate duplicate or similar topics, and finalize the **most important {max_topics} topics**.

Extracted topics:
{topics_str}

Requirements:
1. Consolidate similar or duplicate topics
2. Select only the most important and meaningful {max_topics} topics
3. Improve each topic's summary to be more comprehensive and accurate
4. **Merge and preserve all related_message_ids during consolidation** (remove duplicates). Related IDs are original message_id strings.
5. Respond in JSON format

Response format:
```json
[
  {{
    "topic_name": "Consolidated topic name",
    "summary": "Improved one sentence summary",
    "message_count": estimated_message_count,
    "keywords": ["consolidated", "keyword", "list"],
    "related_message_ids": ["msg_xxx", "msg_yyy", "msg_zzz"]
  }}
]
```

**Important: Merge all related_message_ids from consolidated topics, remove duplicates, and return as original message_id strings.**

Final {max_topics} topics:"""
        
        return prompt
    
    def _validate_and_fix_topics(self, topics: List[Dict[str, Any]], chunk_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """주제 데이터 검증 및 related_message_ids 보완 (개선된 버전)
        - 프롬프트에서는 정수 인덱스를 사용하지만, 이 단계에서 실제 message_id로 변환한다.
        """
        
        if not chunk_messages:
            return topics
            
        # 청크 내 인덱스→메시지, 인덱스→실제 message_id 맵 생성
        index_to_message = {}
        index_to_real_id = {}
        for i, msg in enumerate(chunk_messages):
            idx = str(i)
            index_to_message[idx] = msg
            index_to_real_id[idx] = msg.get('message_id', idx)
        
        fixed_topics = []
        total_valid_assignments = 0
        
        for topic in topics:
            # 기본 필드 확인
            if not isinstance(topic, dict):
                logger.warning("주제가 딕셔너리가 아닙니다")
                continue
            
            topic_name = topic.get('topic_name', 'Unknown')
                
            # related_message_ids 검증 및 보완 (정수 인덱스 기준)
            original_ids = [str(x) for x in topic.get('related_message_ids', [])]
            valid_index_ids = []
            invalid_ids = []
            
            # 기존 인덱스 검증
            for idx in original_ids:
                if idx in index_to_message:
                    valid_index_ids.append(idx)
                else:
                    invalid_ids.append(idx)
            
            if invalid_ids:
                logger.warning(f"주제 '{topic_name}': {len(invalid_ids)}개 유효하지 않은 인덱스 제거")
            
            # related_message_ids가 없거나 부족한 경우 키워드 기반 보완 (인덱스 반환)
            if len(valid_index_ids) == 0:
                logger.info(f"주제 '{topic_name}': related_message_ids가 없어 키워드 기반 보완 시도")
                supplementary_index_ids = self._find_related_messages_by_keywords(
                    topic, index_to_message, max_supplements=3
                )
                if supplementary_index_ids:
                    valid_index_ids.extend([str(x) for x in supplementary_index_ids])
                    logger.info(f"주제 '{topic_name}': 키워드 매칭으로 {len(supplementary_index_ids)}개 인덱스 보완")
                else:
                    logger.warning(f"주제 '{topic_name}': 키워드 매칭으로도 관련 메시지를 찾을 수 없음")
            
            # 정수 인덱스를 실제 message_id로 변환
            resolved_real_ids = [index_to_real_id[idx] for idx in valid_index_ids]
            topic['related_message_ids'] = resolved_real_ids
            total_valid_assignments += len(resolved_real_ids)
            
            # 필수 필드 확인 및 기본값 설정
            if not topic.get('topic_name'):
                topic['topic_name'] = f'주제_{len(fixed_topics) + 1}'
            if not topic.get('summary'):
                topic['summary'] = '요약 정보가 없습니다.'
            if not topic.get('keywords'):
                topic['keywords'] = []
            # message_count를 실제 배정된 메시지 개수로 업데이트
            topic['message_count'] = len(resolved_real_ids)
            
            # 최종 검증
            if all(key in topic for key in ['topic_name', 'summary', 'keywords', 'related_message_ids']):
                fixed_topics.append(topic)
                logger.debug(f"주제 '{topic_name}': 검증 완료 ({len(resolved_real_ids)}개 관련 메시지)")
            else:
                logger.error(f"주제 '{topic_name}': 필수 필드 누락으로 제외")
        
        logger.info(f"주제 검증 완료: {len(fixed_topics)}개 주제, 총 {total_valid_assignments}개 메시지 배정")
        return fixed_topics
    
    def _find_related_messages_by_keywords(self, topic: Dict[str, Any], message_map: Dict[str, Dict[str, Any]], 
                                         max_supplements: int = 3) -> List[str]:
        """키워드 기반으로 관련 메시지 ID 찾기"""
        
        keywords = topic.get('keywords', [])
        topic_name = topic.get('topic_name', '')
        
        if not keywords and not topic_name:
            return []
        
        # 주제명에서 의미있는 단어 추출
        topic_words = []
        for word in topic_name.lower().split():
            if any('\uac00' <= char <= '\ud7a3' for char in word):  # 한글
                if len(word) >= 2:
                    topic_words.append(word)
            else:  # 영문/숫자
                if len(word) >= 3:
                    topic_words.append(word)
        
        # 메시지별 매칭 점수 계산
        scored_messages = []
        for msg_id, msg in message_map.items():
            content = msg.get('content', '').lower()
            score = 0
            
            # 키워드 매칭
            for keyword in keywords:
                keyword_lower = keyword.lower().strip()
                if keyword_lower and keyword_lower in content:
                    score += 3
            
            # 주제명 단어 매칭
            for word in topic_words:
                if word in content:
                    score += 2
            
            if score > 0:
                scored_messages.append((msg_id, score))
        
        # 점수 순으로 정렬하여 상위 메시지 선택
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        selected_ids = [msg_id for msg_id, _ in scored_messages[:max_supplements]]
        
        return selected_ids

    def _simple_deduplication(self, topics: List[Dict[str, str]], max_topics: int) -> List[Dict[str, str]]:
        """간단한 중복 제거 (LLM 호출 실패시 사용)"""
        
        unique_topics = []
        seen_keywords = set()
        
        for topic in topics:
            keywords = set(topic.get('keywords', []))
            
            # 키워드 유사도 체크
            if not any(len(keywords & seen) >= 2 for seen in seen_keywords):
                unique_topics.append(topic)
                seen_keywords.add(frozenset(keywords))
            
            if len(unique_topics) >= max_topics:
                break
        
        return unique_topics
    
    def _simple_keyword_matching(self, topic: Dict[str, Any], unassigned_messages: List[Dict[str, Any]], 
                                needed_count: int) -> List[Dict[str, Any]]:
        """간단한 키워드 매칭으로 백업 메시지 배정"""
        if not unassigned_messages or needed_count <= 0:
            return []
        
        topic_name = topic.get('topic_name', '').lower()
        keywords = [kw.lower().strip() for kw in topic.get('keywords', [])]
        
        # 주제명에서 의미있는 단어 추출 (2글자 이상 한글, 3글자 이상 영문)
        topic_words = []
        for word in topic_name.split():
            if any('\uac00' <= char <= '\ud7a3' for char in word):  # 한글
                if len(word) >= 2:
                    topic_words.append(word)
            else:  # 영문/숫자
                if len(word) >= 3:
                    topic_words.append(word)
        
        # 매칭 점수 계산
        scored_messages = []
        for msg in unassigned_messages:
            content = msg.get('content', '').lower()
            score = 0
            
            # 키워드 매칭 (높은 가중치)
            for keyword in keywords:
                if keyword and keyword in content:
                    score += 5
            
            # 주제명 단어 매칭 (중간 가중치)
            for word in topic_words:
                if word in content:
                    score += 3
            
            # reaction_count 보너스 (낮은 가중치)
            reaction_bonus = min(msg.get('reaction_count', 0) * 0.5, 2.0)
            score += reaction_bonus
            
            if score > 0:
                scored_messages.append((msg, score))
        
        # 점수 순으로 정렬하여 필요한 만큼 선택
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        selected_messages = [msg for msg, _ in scored_messages[:needed_count]]
        
        topic_name_display = topic.get('topic_name', 'Unknown')
        if selected_messages:
            avg_score = sum(score for _, score in scored_messages[:len(selected_messages)]) / len(selected_messages)
            logger.info(f"주제 '{topic_name_display}': 키워드 매칭으로 {len(selected_messages)}개 선택 (평균 점수: {avg_score:.1f})")
        
        return selected_messages
    
    
    def _get_top_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """reaction_count 기준으로 상위 메시지 선별"""
        
        if not messages:
            return []
        
        # reaction_count로 정렬 (내림차순)
        sorted_by_reaction = sorted(
            messages, 
            key=lambda x: x.get('reaction_count', 0), 
            reverse=True
        )
        
        # 모든 메시지의 reaction_count가 0인지 확인
        max_reaction = max(msg.get('reaction_count', 0) for msg in messages)
        
        if max_reaction == 0:
            # 모두 0이면 길이 순으로 정렬
            sorted_messages = sorted(
                messages,
                key=lambda x: len(x.get('content', '')),
                reverse=True
            )
        else:
            # reaction_count가 있으면 그것을 우선
            sorted_messages = sorted_by_reaction
        
        # 상위 3개 메시지 선택
        top_3 = sorted_messages[:3]
        
        # 반환할 형태로 변환
        result = []
        for msg in top_3:
            result.append({
                'content': msg.get('content', ''),
                'reaction_count': msg.get('reaction_count', 0),
                'message_id': msg.get('message_id', '')
            })
        
        return result
    
    def _calculate_keyword_score(self, message_content: str, keywords: List[str], topic_name: str) -> float:
        """메시지와 주제 간의 키워드 매칭 점수를 계산 (개선된 버전)"""
        content_lower = message_content.lower()
        score = 0.0
        matched_keywords = []
        
        # 키워드 매칭 점수 (완전 일치 우선)
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue
                
            if keyword_lower == content_lower:
                score += 15.0  # 완전 일치 (가중치 증가)
                matched_keywords.append(keyword)
            elif f' {keyword_lower} ' in f' {content_lower} ':
                score += 8.0   # 단어 단위 일치 (가중치 증가)
                matched_keywords.append(keyword)
            elif keyword_lower in content_lower:
                score += 3.0   # 부분 일치
                matched_keywords.append(keyword)
        
        # 주제명 단어 매칭 점수 (한글 2글자, 영문 3글자 이상)
        topic_words = []
        for word in topic_name.lower().split():
            # 한글인지 영문인지 판단
            if any('\uac00' <= char <= '\ud7a3' for char in word):
                # 한글: 2글자 이상
                if len(word) >= 2:
                    topic_words.append(word)
            else:
                # 영문/숫자: 3글자 이상
                if len(word) >= 3:
                    topic_words.append(word)
        
        for word in topic_words:
            if f' {word} ' in f' {content_lower} ':
                score += 4.0   # 주제명 단어 단위 일치 (가중치 증가)
            elif word in content_lower:
                score += 1.5   # 주제명 부분 일치
        
        # 키워드 매칭 개수에 따른 보너스 점수
        if len(matched_keywords) > 1:
            score += len(matched_keywords) * 0.5
        
        return score
    
    
    def _assign_messages_by_ids(self, topics: List[Dict[str, Any]], message_id_map: Dict[str, Dict[str, Any]]) -> tuple[Dict[int, List[Dict[str, Any]]], set]:
        """LLM이 제공한 related_message_ids를 사용하여 메시지 배정"""
        
        # 🔍 디버깅: 메시지 ID 맵 정보
        logger.info(f"🔍 사용 가능한 메시지 ID들: {list(message_id_map.keys())[:10]}... (총 {len(message_id_map)}개)")
        
        topic_messages = {i: [] for i in range(len(topics))}
        assigned_message_ids = set()
        
        for topic_idx, topic in enumerate(topics):
            topic_name = topic.get('topic_name', f'주제_{topic_idx}')
            related_ids = topic.get('related_message_ids', [])
            
            # 🔍 디버깅: 주제별 LLM 제공 ID 확인
            logger.info(f"🔍 주제 '{topic_name}': LLM이 제공한 related_message_ids = {related_ids}")
            
            valid_messages = []
            invalid_ids = []
            duplicate_ids = []
            
            for msg_id in related_ids:
                # 정수형 인덱스를 문자열로 변환
                msg_id_str = str(msg_id)
                
                if msg_id_str in message_id_map and msg_id_str not in assigned_message_ids:
                    valid_messages.append(message_id_map[msg_id_str])
                    assigned_message_ids.add(msg_id_str)
                    logger.debug(f"🔍 ID {msg_id_str} 매칭 성공")
                elif msg_id_str in assigned_message_ids:
                    duplicate_ids.append(msg_id_str)
                    logger.debug(f"🔍 ID {msg_id_str}는 이미 다른 주제에 배정됨")
                else:
                    invalid_ids.append(msg_id_str)
                    logger.warning(f"🔍 ID {msg_id_str} 매칭 실패 - message_id_map에 존재하지 않음")
            
            topic_messages[topic_idx] = valid_messages
            
            # 🔍 디버깅: 결과 요약
            logger.info(f"🔍 주제 '{topic_name}' 매칭 결과:")
            logger.info(f"  - 성공: {len(valid_messages)}개")
            logger.info(f"  - 실패: {len(invalid_ids)}개 {invalid_ids}")
            logger.info(f"  - 중복: {len(duplicate_ids)}개 {duplicate_ids}")
            
            logger.info(f"주제 '{topic_name}': LLM 제공 ID로 {len(valid_messages)}개 메시지 배정")
        
        return topic_messages, assigned_message_ids
    
    def _ensure_top_messages(self, topics: List[Dict[str, Any]], all_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """LLM의 related_message_ids를 우선 사용하여 메시지 배정 및 상위 메시지 추출"""
        
        logger.info("🎯 LLM 기반 메시지 배정 시작")
        
        # 메시지 ID -> 메시지 객체 맵핑 생성
        message_id_map = {}
        for msg in all_messages:
            msg_id = msg.get('message_id', '')
            if msg_id:
                message_id_map[msg_id] = msg
        
        # 1. LLM이 제공한 related_message_ids를 우선적으로 사용
        topic_messages, assigned_message_ids = self._assign_messages_by_ids(topics, message_id_map)
        
        # 2. 미배정 메시지 수집
        unassigned_messages = []
        for msg in all_messages:
            msg_id = msg.get('message_id', '')
            if msg_id not in assigned_message_ids:
                unassigned_messages.append(msg)
        
        logger.info(f"LLM 기반 1차 배정 완료: {len(assigned_message_ids)}개 배정, {len(unassigned_messages)}개 미배정")
        
        # 3. related_message_ids가 없거나 부족한 주제에 대해 간단한 백업 배정
        min_messages_per_topic = 2  # 최소 보장 메시지 수 감소
        for topic_idx, topic in enumerate(topics):
            current_messages = topic_messages.get(topic_idx, [])
            topic_name = topic.get('topic_name', f'주제_{topic_idx}')
            
            if len(current_messages) < min_messages_per_topic and unassigned_messages:
                needed_count = min_messages_per_topic - len(current_messages)
                
                # 간단한 키워드 매칭으로 백업 배정
                backup_messages = self._simple_keyword_matching(
                    topic, unassigned_messages, needed_count
                )
                
                if backup_messages:
                    current_messages.extend(backup_messages)
                    topic_messages[topic_idx] = current_messages
                    
                    # 배정된 메시지들을 미배정 목록에서 제거
                    for backup_msg in backup_messages:
                        if backup_msg in unassigned_messages:
                            unassigned_messages.remove(backup_msg)
                    
                    logger.info(f"주제 '{topic_name}': 백업 배정으로 {len(backup_messages)}개 메시지 추가")
        
        logger.info("✅ LLM 기반 메시지 배정 완료")
        
        # 3. 모든 주제에 대해 연관 메시지 정보 추가
        for topic_idx, topic in enumerate(topics):
            topic_name = topic.get('topic_name', '')
            related_candidates = topic_messages.get(topic_idx, [])
            
            logger.info(f"주제 '{topic_name}': 배정된 관련 메시지 {len(related_candidates)}개")
            
            # 모든 관련 메시지들을 결과에 포함
            all_related = self._format_all_messages(related_candidates)
            topic['all_related_messages'] = all_related
            topic['all_related_messages_count'] = len(related_candidates)
            
            # message_count를 실제 배정된 메시지 개수로 최종 업데이트 (LLM 원본 + 백업 배정 포함)
            topic['message_count'] = len(related_candidates)
            
            # 연관 메시지 통계 계산
            if related_candidates:
                total_reactions = sum(msg.get('reaction_count', 0) for msg in related_candidates)
                avg_reactions = total_reactions / len(related_candidates)
                avg_length = sum(len(msg.get('content', '')) for msg in related_candidates) / len(related_candidates)
                
                topic['related_messages_stats'] = {
                    'total_reactions': total_reactions,
                    'average_reactions': round(avg_reactions, 2),
                    'average_length': round(avg_length, 1),
                    'max_reactions': max(msg.get('reaction_count', 0) for msg in related_candidates)
                }
                
                # all_related_messages에서 직접 top_messages 선별
                max_reaction = max(msg.get('reaction_count', 0) for msg in related_candidates)
                
                if max_reaction > 0:
                    # reaction_count 기준 정렬
                    sorted_messages = sorted(
                        related_candidates, 
                        key=lambda x: (x.get('reaction_count', 0), len(x.get('content', ''))), 
                        reverse=True
                    )
                    logger.info(f"주제 '{topic_name}': reaction_count 기준 정렬 (최고: {max_reaction})")
                else:
                    # 텍스트 길이 기준 정렬
                    sorted_messages = sorted(
                        related_candidates,
                        key=lambda x: len(x.get('content', '')),
                        reverse=True
                    )
                    logger.info(f"주제 '{topic_name}': 텍스트 길이 기준 정렬")
                
                # 상위 3개 선별
                top_messages = self._format_top_messages(sorted_messages[:3])
                topic['top_messages'] = top_messages
                
            else:
                topic['related_messages_stats'] = {
                    'total_reactions': 0,
                    'average_reactions': 0.0,
                    'average_length': 0.0,
                    'max_reactions': 0
                }
                topic['top_messages'] = []
                logger.warning(f"주제 '{topic_name}': 매칭된 관련 메시지가 없습니다")
            
            # 로그 출력
            logger.info(f"주제 '{topic_name}' 완료:")
            logger.info(f"  - 전체 관련 메시지: {len(related_candidates)}개")
            logger.info(f"  - 상위 메시지: {len(topic.get('top_messages', []))}개") 
            logger.info(f"  - 평균 reaction_count: {topic['related_messages_stats']['average_reactions']}")
        
        return topics, unassigned_messages
    
    def _format_top_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """메시지 리스트를 상위 메시지 형태로 포맷팅"""
        result = []
        for msg in messages:
            result.append({
                'content': msg.get('content', ''),
                'reaction_count': msg.get('reaction_count', 0),
                'message_id': msg.get('message_id', '')
            })
        return result

    def _format_all_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모든 연관 메시지를 포맷팅 (reaction_count 순으로 정렬)"""
        if not messages:
            return []
        
        # reaction_count 내림차순, 텍스트 길이 내림차순으로 정렬
        sorted_messages = sorted(
            messages,
            key=lambda x: (x.get('reaction_count', 0), len(x.get('content', ''))),
            reverse=True
        )
        
        result = []
        for msg in sorted_messages:
            result.append({
                'content': msg.get('content', ''),
                'reaction_count': msg.get('reaction_count', 0),
                'message_id': msg.get('message_id', ''),
                'content_length': len(msg.get('content', ''))
            })
        return result

    def _attach_merged_subtopics(self, final_topics: List[Dict[str, Any]], subtopics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """최종 통합 주제에 병합된 소주제 정보를 부착
        - 기준: related_message_ids 겹침이 1개 이상이면 해당 최종 주제에 귀속
        - 다수의 최종 주제와 겹치면 겹침 수가 가장 큰 주제에 할당
        - 각 항목에 overlap_count, overlap_rate_subtopic, jaccard를 기록
        """
        if not final_topics or not subtopics:
            return final_topics

        # 최종 주제별 ID 집합 미리 구성
        final_id_sets: List[set] = []
        for ft in final_topics:
            ids = set(str(x) for x in ft.get('related_message_ids', []) if str(x))
            final_id_sets.append(ids)

        # 소주제 엔트리 정규화
        normalized_subtopics = []
        for idx, st in enumerate(subtopics):
            st_ids = set(str(x) for x in st.get('related_message_ids', []) if str(x))
            normalized_subtopics.append({
                'subtopic_index': idx,
                'source_chunk_index': st.get('source_chunk_index'),
                'topic_name': st.get('topic_name', ''),
                'summary': st.get('summary', ''),
                'keywords': st.get('keywords', []),
                'related_message_ids': list(st_ids),
                'message_count': st.get('message_count', len(st_ids))
            })

        # 소주제 → 최종 주제 매핑 (최대 겹침 기준)
        assignments: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(final_topics))}
        for st in normalized_subtopics:
            st_ids = set(st['related_message_ids'])
            if not st_ids:
                continue

            best_ft_idx = None
            best_overlap = 0
            best_jaccard = 0.0
            for ft_idx, ft_ids in enumerate(final_id_sets):
                if not ft_ids:
                    continue
                overlap = len(st_ids & ft_ids)
                if overlap == 0:
                    continue
                union = len(st_ids | ft_ids)
                jaccard = overlap / union if union else 0.0
                if overlap > best_overlap or (overlap == best_overlap and jaccard > best_jaccard):
                    best_overlap = overlap
                    best_ft_idx = ft_idx
                    best_jaccard = jaccard

            if best_ft_idx is not None and best_overlap > 0:
                overlap_rate_sub = best_overlap / max(len(st_ids), 1)
                assignments[best_ft_idx].append({
                    **st,
                    'overlap_count': best_overlap,
                    'overlap_rate_subtopic': round(overlap_rate_sub, 3),
                    'jaccard': round(best_jaccard, 3)
                })

        # 최종 토픽에 merged_subtopics 부착
        for ft_idx, ft in enumerate(final_topics):
            merged_list = assignments.get(ft_idx, [])
            # 정렬: overlap_count desc, jaccard desc
            merged_list.sort(key=lambda x: (x.get('overlap_count', 0), x.get('jaccard', 0.0)), reverse=True)
            ft['merged_subtopics'] = merged_list
            ft['merged_subtopic_count'] = len(merged_list)

        return final_topics
    
    def _select_best_provider(self) -> str:
        """최적 LLM 제공자 자동 선택"""
        if self.openai_client and self.default_provider == "openai":
            return "openai"
        elif self.anthropic_client and self.default_provider == "anthropic":
            return "anthropic"
        elif self.openai_client:
            return "openai"
        elif self.anthropic_client:
            return "anthropic"
        else:
            return "fallback"
    
    def save_summary_to_file(self, summary_result: Dict[str, Any], output_path: str):
        """요약 결과를 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 직접 LLM 요약 결과 저장: {output_path}")
            
        except Exception as e:
            logger.error(f"요약 결과 저장 실패: {e}")


if __name__ == "__main__":
    print("DirectLLMSummarizer 클래스는 다른 파일에서 import하여 사용하세요.")
    print("실제 데이터 분석을 위해서는 다음 파일들을 사용하세요:")
    print("  - analyze_data.py: 메인 분석 실행")
    print("  - analyze_with_improved_logic.py: 개선된 로직 테스트")