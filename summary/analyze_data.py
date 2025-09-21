#!/usr/bin/env python3
"""
CSV 데이터 분석 스크립트
DirectLLMSummarizer를 사용하여 채팅 메시지 주제별 요약
"""

import pandas as pd
import os
import sys
from datetime import datetime
import logging
import argparse

# Backend 모듈 경로 추가
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.append(backend_path)

try:
    from direct_llm_summarizer import DirectLLMSummarizer
except ImportError as e:
    print(f"❌ DirectLLMSummarizer 모듈을 찾을 수 없습니다: {e}")
    print("백엔드 경로를 확인해주세요.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """CSV 파일에서 데이터 로드"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 CSV 데이터 로드 완료: {len(df)}행")
        return df
    except Exception as e:
        logger.error(f"❌ CSV 파일 로드 실패: {e}")
        return None

def convert_to_message_format(df: pd.DataFrame) -> list:
    """DataFrame을 DirectLLMSummarizer에서 사용하는 메시지 형태로 변환"""
    messages = []
    
    for _, row in df.iterrows():
        # content 필드에서 실제 메시지 내용 추출
        content = str(row.get('content', ''))
        
        # 타임스탬프가 포함된 경우 메시지 내용만 추출
        if ' ' in content and any(char.isdigit() for char in content[:20]):
            # 일반적으로 "24. 9. 2. 오후 12:27 실제메시지내용" 형태
            parts = content.split(' ', 4)  # 날짜+시간 부분을 건너뛰기
            if len(parts) > 4:
                content = parts[4]
            elif len(parts) > 3:
                content = parts[3]
        
        # 빈 메시지나 너무 짧은 메시지 필터링
        content = content.strip()
        if len(content) < 3:
            continue
            
        message = {
            'content': content,
            'reaction_count': int(row.get('reaction_count', 0)),
            'message_id': str(row.get('message_id', '')),
            'timestamp': str(row.get('timestamp', '')),
            'nickname': str(row.get('nickname', ''))
        }
        messages.append(message)
    
    logger.info(f"📝 메시지 변환 완료: {len(messages)}개")
    return messages

def analyze_data(csv_path: str, output_dir: str = "analysis_results", from_date: str = None, to_date: str = None, email: str = None, print_only: bool = False):
    """CSV 데이터 분석 실행"""
    if print_only:
        # 전처리 없이 인자만 출력하고 즉시 종료
        print("[PRINT-ONLY] analyze_data.py invoked")
        print(f"from_date={from_date}")
        print(f"to_date={to_date}")
        print(f"email={email}")
        return

    # CSV 데이터 로드
    df = load_csv_data(csv_path)
    if df is None:
        return
    
    print(f"\n📊 데이터 개요:")
    print(f"  - 전체 메시지: {len(df)}개")
    print(f"  - 컬럼: {list(df.columns)}")
    print(f"  - 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 메시지 형태로 변환
    messages = convert_to_message_format(df)
    
    if not messages:
        print("❌ 분석할 메시지가 없습니다.")
        return
    
    print(f"\n🔍 필터링된 메시지: {len(messages)}개")
    print(f"예시 메시지:")
    for i, msg in enumerate(messages[:3], 1):
        print(f"  {i}. (👍{msg['reaction_count']}) {msg['content'][:50]}...")
    
    # DirectLLMSummarizer 초기화
    try:
        # 환경변수에서 API 키 가져오기 (기본값 없음: 하드코딩 금지)
        openai_key = os.getenv('OPENAI_API_KEY')
        
        summarizer = DirectLLMSummarizer(
            openai_api_key=openai_key,
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            request_delay=1.0
        )
        
        if not summarizer.openai_client and not summarizer.anthropic_client:
            print("⚠️ API 키가 설정되지 않았습니다. 환경변수를 확인해주세요.")
            print("OPENAI_API_KEY 또는 ANTHROPIC_API_KEY를 설정하세요.")
            return
        
    except Exception as e:
        print(f"❌ DirectLLMSummarizer 초기화 실패: {e}")
        return
    
    # 주제별 요약 실행
    print(f"\n🤖 LLM 주제별 요약 시작...")
    
    try:
        result = summarizer.summarize_messages_directly(
            messages=messages,
            max_topics=8,
            language="korean"
        )
        
        # 결과 출력
        print("\n" + "="*60)
        print("📋 주제별 요약 결과")
        print("="*60)
        
        stats = result['statistics']
        print(f"\n📊 처리 통계:")
        print(f"  - 전체 메시지: {stats['total_messages']}개")
        print(f"  - 처리 청크: {stats['processed_chunks']}개") 
        print(f"  - 추출된 주제: {stats['total_topics']}개")
        print(f"  - 배정된 메시지: {stats.get('assigned_messages', 'N/A')}개")
        print(f"  - 미배정 메시지: {stats.get('unassigned_messages', 'N/A')}개")
        print(f"  - 고반응 미배정: {stats.get('high_reaction_unassigned', 'N/A')}개")
        print(f"  - 사용된 LLM: {stats['llm_provider']}")
        print(f"  - 처리 시간: {stats['processing_time']}")
        
        print(f"\n📌 주제별 상세 분석:")
        for i, topic in enumerate(result['topics'], 1):
            print(f"\n{i}. 📂 {topic['topic_name']}")
            print(f"   💬 {topic['summary']}")
            print(f"   📊 관련 메시지: 약 {topic.get('message_count', 0)}개")
            
            if topic.get('keywords'):
                print(f"   🏷️  키워드: {', '.join(topic['keywords'])}")
            
            # 상위 메시지 출력
            if topic.get('top_messages'):
                print(f"   🔥 상위 메시지 (반응 수 기준):")
                for j, msg in enumerate(topic['top_messages'], 1):
                    content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                    print(f"      {j}. (👍 {msg['reaction_count']}) {content}")
        
        # 미배정된 고반응 메시지 출력
        if result.get('unassigned_high_reaction_messages'):
            print(f"\n🚨 미배정된 고반응 메시지 (reaction_count > 2):")
            print(f"총 {len(result['unassigned_high_reaction_messages'])}개 메시지")
            for i, msg in enumerate(result['unassigned_high_reaction_messages'][:10], 1):  # 상위 10개만 출력
                content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                print(f"  {i}. (👍 {msg['reaction_count']}) {content}")
            
            if len(result['unassigned_high_reaction_messages']) > 10:
                print(f"  ... 및 {len(result['unassigned_high_reaction_messages']) - 10}개 더")
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"analysis_result_{timestamp}.json")
        
        summarizer.save_summary_to_file(result, output_file)
        print(f"\n💾 분석 결과 저장: {output_file}")
        
        # 간단한 통계 정보도 저장
        stats_file = os.path.join(output_dir, f"analysis_stats_{timestamp}.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("채팅 메시지 분석 결과 통계\n")
            f.write("="*50 + "\n\n")
            f.write(f"분석 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"원본 CSV 파일: {csv_path}\n")
            f.write(f"전체 메시지 수: {stats['total_messages']}개\n")
            f.write(f"처리 청크 수: {stats['processed_chunks']}개\n")
            f.write(f"추출된 주제 수: {stats['total_topics']}개\n")
            f.write(f"배정된 메시지: {stats.get('assigned_messages', 'N/A')}개\n")
            f.write(f"미배정 메시지: {stats.get('unassigned_messages', 'N/A')}개\n")
            f.write(f"고반응 미배정: {stats.get('high_reaction_unassigned', 'N/A')}개\n")
            f.write(f"사용된 LLM: {stats['llm_provider']}\n\n")
            
            f.write("주제 목록:\n")
            f.write("-" * 30 + "\n")
            for i, topic in enumerate(result['topics'], 1):
                f.write(f"{i}. {topic['topic_name']}\n")
                f.write(f"   요약: {topic['summary']}\n")
                f.write(f"   키워드: {', '.join(topic.get('keywords', []))}\n\n")
            
            # 미배정된 고반응 메시지 정보 추가
            if result.get('unassigned_high_reaction_messages'):
                f.write("미배정된 고반응 메시지 (reaction_count > 2):\n")
                f.write("-" * 40 + "\n")
                for i, msg in enumerate(result['unassigned_high_reaction_messages'][:5], 1):  # 상위 5개만 저장
                    f.write(f"{i}. (👍 {msg['reaction_count']}) {msg['content'][:100]}...\n")
                f.write(f"\n총 {len(result['unassigned_high_reaction_messages'])}개의 미배정 고반응 메시지\n")
        
        print(f"📄 통계 정보 저장: {stats_file}")
        print("\n✅ 데이터 분석 완료!")
        
    except Exception as e:
        logger.error(f"❌ 요약 처리 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="채팅 메시지 데이터 분석 및 요약")
    parser.add_argument('--csv', dest='csv_path', default="/Users/daehwankim/cube_rag/summary/scraped_messages_293msgs_20250831_152924.csv", help='분석할 CSV 파일 경로')
    # 예약어 회피를 위해 dest 지정
    parser.add_argument('--from', dest='from_date', default=None, help='시작 날짜 YYYY-MM-DD')
    parser.add_argument('--to', dest='to_date', default=None, help='종료 날짜 YYYY-MM-DD')
    parser.add_argument('--email', dest='email', default=None, help='결과 수신 이메일 주소')
    parser.add_argument('--print-only', dest='print_only', action='store_true', help='전처리 없이 인자만 출력하고 종료')
    args = parser.parse_args()

    if args.print_only:
        analyze_data(args.csv_path, from_date=args.from_date, to_date=args.to_date, email=args.email, print_only=True)
        return

    print("🚀 채팅 메시지 데이터 분석 시작")
    print("DirectLLMSummarizer를 사용하여 주제별 요약을 생성합니다.\n")

    csv_file = args.csv_path
    if os.path.exists(csv_file):
        print(f"📁 분석 파일: {os.path.basename(csv_file)}")
        analyze_data(csv_file, from_date=args.from_date, to_date=args.to_date, email=args.email)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file}")

if __name__ == "__main__":
    main()