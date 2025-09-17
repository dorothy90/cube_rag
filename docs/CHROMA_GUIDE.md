## Chroma 사용 가이드 (Cube RAG)

이 문서는 `chroma_setup.py`와 연동 도구들의 사용법, 전체 흐름, 임베딩(인덱싱) 이후 데이터를 읽는 방법을 간단히 정리합니다.

### 1) 개요
- 이 프로젝트는 Chroma 벡터DB를 사용해 Q/A 데이터를 임베딩하고 검색합니다.
- 운영 추천: 질문 전용 컬렉션(`qa_questions`)으로 “질문↔질문” 매칭을 한 뒤, 히트의 메타데이터에서 `answer`를 가져와 RAG 컨텍스트로 주입합니다.

### 2) 준비사항
- Python 패키지 설치
```bash
pip install -r requirements.txt
```
- .env 설정(필수)
```bash
OPENAI_API_KEY=sk-...
# 선택값
EMBEDDING_MODEL_NAME=text-embedding-3-small
CHROMA_COLLECTION=qa_questions   # 기본 컬렉션. 미설정 시 코드 기본값은 qa_questions
```

### 3) 데이터 파일
- `data/extracted_qa_pairs.json`: 상위 키(`question`, `answer`, ...)
- `data/chunked_qa_pairs.json`: `content`와 `metadata.question/answer`를 보유한 청크 포맷

### 4) 주요 스크립트/함수
- `chroma_setup.py`
  - `index_qa(...)`: Q/A 페어를 `Q: ...\nA: ...`로 합쳐 인덱싱
  - `index_chunks(...)`: 청크(`content`, `metadata`) 인덱싱
  - `index_questions(...)`: 질문 텍스트만 인덱싱(질문 전용 컬렉션)
  - `delete_collection(...)`: 컬렉션 삭제
  - `query(...)`: 간단 질의(테스트용)
- `retrieval_agent.py`
  - `retrieve(query, ...)`: 현재 기본 컬렉션(`qa_questions`)에서 상위 k를 코사인 유사도(0..1)로 검색하여 `[ {content, metadata, score} ]` 반환

### 5) 빠른 시작(질문 전용 컬렉션 권장)
1. 질문 전용 인덱싱(임베딩 포함)
```bash
python3 chroma_setup.py --mode index_questions --json data/chunked_qa_pairs.json --persist chroma_db --collection qa_questions
# 또는 원본 페어에서 바로
python3 chroma_setup.py --mode index_questions --json data/extracted_qa_pairs.json --persist chroma_db --collection qa_questions
```
2. 간단 검색 테스트
```bash
python3 - << 'PY'
from chroma_setup import query
out = query(
    persist_dir="chroma_db",
    q="컨테이너 오케스트레이션이 뭐야",
    k=3,
    collection_name="qa_questions",
)
for i, r in enumerate(out, 1):
    print(i, r.get("metadata", {}).get("question"), r.get("score"))
    print(r.get("content")[:120])
    print()
PY
```

### 6) 전체 흐름(서비스 관점)
- 입력 질문 → `retrieval_agent.retrieve()` → `qa_questions`에서 상위 K 질의
- 히트의 `metadata.question/answer`로 컨텍스트를 `Q: ...\nA: ...` 형태로 재구성(이미 코드 반영)
- `answer_generator`가 컨텍스트를 포함해 응답 생성
- 직접매칭 판정은 “리트리벌 점수 컨센서스”로 단순화됨
  - env: `DIRECT_MATCH_HIGH`(기본 0.65), `DIRECT_MATCH_SCORE_THRESHOLD`(0.55), `DIRECT_MATCH_TOPK`(5), `DIRECT_MATCH_COUNT`(2)

### 7) 임베딩(인덱싱) 이후 ‘파일 읽는 법’
임베딩이 끝나면 Chroma에 벡터와 메타데이터가 저장됩니다. 읽는 방법은 두 가지가 있습니다.

- A. 테스트 유틸 사용(`chroma_setup.query`)
```python
from chroma_setup import query
results = query(
    persist_dir="chroma_db",
    q="컨테이너 오케스트레이션이 뭐야",
    k=3,
    collection_name="qa_questions",
)
for r in results:
    meta = r["metadata"]
    print(meta.get("question"), "=>", meta.get("answer"))
```

- B. 리트리벌 API 사용(`retrieval_agent.retrieve`)
```python
from retrieval_agent import retrieve
hits = retrieve("컨테이너 오케스트레이션이 뭐야", top_k=3)
for h in hits:
    meta = h.get("metadata", {})
    print(h.get("score"), meta.get("question"), "=>", meta.get("answer"))
```

### 8) 컬렉션 관리
- 컬렉션 삭제(초기화)
```bash
python3 chroma_setup.py --mode delete_collection --persist chroma_db --collection qa_questions
```
- 재인덱싱
```bash
python3 chroma_setup.py --mode index_questions --json data/chunked_qa_pairs.json --persist chroma_db --collection qa_questions
```

### 9) 자주 묻는 질문(FAQ)
- Q: 인덱싱과 임베딩 차이?
  - A: 이 프로젝트에서는 인덱싱 호출 시 내부에서 임베딩을 계산한 뒤, 벡터DB에 저장까지 진행합니다.
- Q: 질문 전용 컬렉션이 필요한 이유?
  - A: `Q:...\nA:...` 통짜 텍스트는 질문 단문과의 매칭이 희석될 수 있어, “질문↔질문” 매칭이 더 안정적입니다.
- Q: 점수 기준 튜닝은?
  - A: `.env`에서 `DIRECT_MATCH_*` 값을 조정하세요. 데이터가 조밀하면 `DIRECT_MATCH_COUNT`를 2→3으로 올리거나 `SCORE_THRESHOLD`를 0.58 정도로 상향하세요.

### 10) 용어 정리
- 임베딩(Embedding): 텍스트→벡터 변환
- 인덱싱(Indexing): 임베딩 결과를 벡터DB에 저장하고 검색 가능하게 만드는 과정

필요 시 본 문서(`docs/CHROMA_GUIDE.md`)를 업데이트해 운영 기준과 환경 변수를 팀에 공유하세요.


