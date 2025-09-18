## Domain RAG 사용 가이드

- **도메인**: `python`, `sql`, `semiconductor`
- **동작 요약**
  - 새 채팅의 첫 질문이 분류 애매하면: 어시스턴트가 도메인 확인 질문을 반환
  - 이후 질문이 애매하면: 직전 턴의 도메인으로 폴백
  - 응답에 `selected_domain` 필드가 포함됨(API)

### 환경 변수
- **OPENAI_API_KEY**: OpenAI 키 (필수)
- **OPENAI_MODEL_NAME**: 답변 생성 모델 (기본: `gpt-4o-mini`)
- **EMBEDDING_MODEL_NAME**: 임베딩 모델 (기본: `text-embedding-3-small`)
- **CHROMA_PERSIST_DIR**: Chroma 저장 경로 (기본: `./chroma_db`)
- **DOMAIN_CONFIDENCE_THRESHOLD**: 도메인 분류 애매 임계값 (기본: `0.6`)
- **USE_WEB_SEARCH**: 내부 매칭 실패 시 웹검색 사용 여부 (`true|false`, 기본: `false`)

### 데이터 & 컬렉션
- 데이터 파일(한국어 템플릿):
  - `data/extracted_qa_pairs_python.json`
  - `data/extracted_qa_pairs_sql.json`
  - `data/extracted_qa_pairs_semiconductor.json`
- 컬렉션명(질문 전용 인덱스):
  - `qa_questions_python`
  - `qa_questions_sql`
  - `qa_questions_semiconductor`

### 인덱싱 명령 예시
```bash
python chroma_setup.py --mode index_questions --json data/extracted_qa_pairs_python.json --persist chroma_db --collection qa_questions_python
python chroma_setup.py --mode index_questions --json data/extracted_qa_pairs_sql.json --persist chroma_db --collection qa_questions_sql
python chroma_setup.py --mode index_questions --json data/extracted_qa_pairs_semiconductor.json --persist chroma_db --collection qa_questions_semiconductor
```

### API 사용
- 엔드포인트: `POST /ask`
- 요청 바디:
```json
{
  "question": "INNER JOIN과 LEFT JOIN 차이는?",
  "conversation_id": "선택 사항"
}
```
- 응답 필드(발췌):
```json
{
  "answer": "...",
  "sources": [ ... ],
  "web": [ ... ],
  "analysis": { ... },
  "conversation_id": "cid_xxx",
  "selected_domain": "sql"
}
```
- 동작 규칙:
  - 첫 질문 분류가 애매하면 `answer`에 도메인 확인 질문이 반환됨
  - 이후 애매한 질문은 직전 도메인으로 폴백

### 테스트 빠른 확인
```bash
python tests/regression_domain_rag.py
```
출력에 각 케이스의 `selected_domain`과 PASS/FAIL 요약이 표시됩니다.


test
python - << 'PY'
import os
from retrieval_agent import retrieve
q = '정규화와 비정규화의 차이점을 알려주세요'
res = retrieve(q, collection_name='qa_questions_sql')
scores = [r.get('score') for r in res]
valid = sorted([s for s in scores if isinstance(s,(int,float))], reverse=True)
t_high=float(os.getenv('DIRECT_MATCH_HIGH','0.5')); t_mid=float(os.getenv('DIRECT_MATCH_SCORE_THRESHOLD','0.45'))
K=int(os.getenv('DIRECT_MATCH_TOPK','3')); C=int(os.getenv('DIRECT_MATCH_COUNT','2'))
top1 = valid[0] if valid else 0.0
consensus = sum(1 for v in valid[:K] if v>=t_mid)
print({'scores':scores,'top1':top1,'consensus':consensus,'has_direct_match':(top1>=t_high) or (consensus>=C)})
PY