"""
한국어 개발자 채팅 데이터 전처리 - Step별 진행
"""

import pandas as pd
import os

def step1_load_csv():
    """Step 1: CSV 파일 로드 및 기본 정보 확인"""
    
    # CSV 파일 경로 (프로젝트 루트 기준 상대 경로)
    csv_path = 'data/korean_dev_chat_sample.csv'
    # 파일 존재 확인
    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return None
    
    print(f"📁 파일 경로: {csv_path}")
    
    # CSV 파일 로드
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        return None
 
    return df

def step3_llm_extract_qa_pairs(df, batch_size=200, api_key=None):
    """Step 3: 시간순 정렬 후 LLM으로 Q&A 쌍 추출"""
 
    # OpenAI API 키 확인
    if not api_key:
        # 환경변수에서 로드 (.env 사용 권장)
        from dotenv import load_dotenv
        load_dotenv()
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OpenAI API 키가 설정되지 않았습니다.")
            print("환경변수 OPENAI_API_KEY를 설정하거나 api_key 매개변수를 전달하세요.")
            return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI 클라이언트 초기화 완료")
    except ImportError:
        print("❌ OpenAI 라이브러리가 설치되지 않았습니다.")
        return None
    
    # 시간순으로 정렬
    df_sorted = df.copy()
    df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
    df_sorted = df_sorted.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ 시간순 정렬 완료")
    
    # 메시지들을 배치로 나누기
    total_messages = len(df_sorted)
    batches = []
    
    for i in range(0, total_messages, batch_size):
        batch_df = df_sorted.iloc[i:i + batch_size]
        batches.append(batch_df)
    
    print(f"📊 총 {total_messages}개 메시지를 {len(batches)}개 배치로 나누어 처리합니다.")
    print(f"배치 크기: {batch_size}개")
    
    # 각 배치를 LLM에 전송하여 Q&A 쌍 추출
    all_qa_pairs = []
    
    for batch_idx, batch_df in enumerate(batches):
        print(f"\n🔄 배치 {batch_idx + 1}/{len(batches)} 처리 중...")
        
        # 배치 메시지들을 시간순으로 텍스트로 결합
        batch_text = ""
        for idx, row in batch_df.iterrows():
            timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            author = row['name']
            message = row['message']
            batch_text += f"[{timestamp}] {author}: {message}\n"
        
        # LLM 프롬프트 생성
        prompt = f"""
다음은 시간순으로 정렬된 한국어 개발자 채팅 메시지들입니다. 
이 대화에서 질문과 답변이 있는 Q&A 쌍을 추출해주세요.

대화 내용:
{batch_text}

위 대화에서 질문과 답변이 있는 Q&A 쌍을 찾아서 다음 JSON 형태로 출력해주세요:
{{
  "qa_pairs": [
    {{
      "question": "질문 내용",
      "answer": "답변 내용",
      "question_author": "질문자",
      "answer_author": "답변자",
      "timestamp": "시간"
    }}
  ]
}}

만약 명확한 Q&A 쌍이 없다면 빈 배열 []로 출력해주세요.
"""
        
        try:
            # OpenAI API 호출
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 한국어 개발자 채팅에서 Q&A 쌍을 추출하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # 응답 파싱
            result = response.choices[0].message.content
            print(f"✅ 배치 {batch_idx + 1} 처리 완료")
            
            # JSON 파싱 시도
            import json
            try:
                batch_result = json.loads(result)
                if 'qa_pairs' in batch_result:
                    all_qa_pairs.extend(batch_result['qa_pairs'])
                    print(f"  - 추출된 Q&A 쌍: {len(batch_result['qa_pairs'])}개")
                else:
                    print(f"  - 추출된 Q&A 쌍: 0개")
            except json.JSONDecodeError:
                print(f"⚠️ 배치 {batch_idx + 1} JSON 파싱 실패")
                
        except Exception as e:
            print(f"❌ 배치 {batch_idx + 1} 처리 실패: {e}")
    
    print(f"\n📈 전체 Q&A 쌍 추출 결과:")
    print(f"  - 총 추출된 Q&A 쌍: {len(all_qa_pairs)}개")
    
    # Q&A 쌍들을 DataFrame으로 변환
    if all_qa_pairs:
        qa_df = pd.DataFrame(all_qa_pairs)
        print(f"✅ Q&A 쌍 DataFrame 생성 완료")
        
        # 샘플 출력
        print(f"\n📋 추출된 Q&A 쌍 샘플:")
        for i, qa in enumerate(all_qa_pairs[:3], 1):
            print(f"\n{i}. 질문: {qa['question']}")
            print(f"   답변: {qa['answer']}")
            print(f"   질문자: {qa['question_author']} → 답변자: {qa['answer_author']}")
    else:
        print(f"⚠️ 추출된 Q&A 쌍이 없습니다.")
        qa_df = pd.DataFrame()
    
    print(f"\n🎉 Step 3 완료!")
    return qa_df

def main():
    """메인 실행 함수"""
    print("🚀 한국어 개발자 채팅 데이터 전처리 시작")
    print("=" * 60)
    
    # Step 1 실행
    df = step1_load_csv()
    
    if df is not None:
        print(f"\n✅ Step 1 성공!")
        
        # Step 3 실행 (Step 2 건너뛰기)
        print(f"\n다음 단계: Step 3 - LLM으로 Q&A 쌍 추출")
        
        # 사용자 설정 가능한 배치 크기
        batch_size = 200  # 기본값, 필요시 변경 가능
        
        qa_df = step3_llm_extract_qa_pairs(df, batch_size=batch_size)
        
        if qa_df is not None and not qa_df.empty:
            print(f"\n✅ 모든 단계 완료!")
            print(f"추출된 Q&A 쌍을 확인하려면 qa_df 변수를 사용하세요.")
            
            # 결과 저장 (프로젝트 루트 기준 상대 경로)
            output_path = 'data/extracted_qa_pairs.json'
            qa_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            print(f"💾 Q&A 쌍이 {output_path}에 저장되었습니다.")
        else:
            print(f"\n❌ Step 3 실패 또는 Q&A 쌍이 추출되지 않았습니다!")
    else:
        print(f"\n❌ Step 1 실패! 문제를 해결한 후 다시 시도하세요.")

if __name__ == "__main__":
    main()
