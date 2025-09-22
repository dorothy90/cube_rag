## 목적

- **목표**: `analysis_results/analysis_result_*.json`의 토픽/소주제 관계를 이용해, 노트북LM 스타일의 마인드맵과 우측 채팅형 메시지 뷰어를 직접 구현합니다.
- **완성 형태**:
  - 좌측: D3 트리 기반 마인드맵
  - 우측: 소주제 클릭 시 관련 메시지를 시간순으로 표시하는 채팅 UI
  - 경로: `/mindmap`

## 사전 준비

- Python 3.9+
- 필수 라이브러리 (이미 프로젝트에 포함): FastAPI, Jinja2, Starlette StaticFiles
- 실행 방법(예시):
```bash
uvicorn api_server:app --reload --port 8000
```
- 브라우저에서 `http://localhost:8000/mindmap` 접속

## 데이터 이해 (핵심 필드)

분석 결과 JSON은 대략 아래 구조를 가집니다. 이 중 다음을 활용합니다.
- `topics[*].topic_name`: 부모 주제
- `topics[*].merged_subtopics[*].subtopic_index`: 소주제 인덱스
- `topics[*].merged_subtopics[*].topic_name`: 소주제 이름
- `topics[*].merged_subtopics[*].related_message_ids`: 관련 메시지 ID 목록
- `topics[*].all_related_messages[*]`: 실제 메시지 본문/메타

```json
{
  "topics": [
    {
      "topic_name": "AI 도구 및 서비스 활용",
      "merged_subtopics": [
        {
          "subtopic_index": 0,
          "topic_name": "AI 도구 및 유튜브 요약 서비스 소개",
          "related_message_ids": ["0"]
        }
      ],
      "all_related_messages": [
        {
          "message_id": "0",
          "content": "10:13 ...",  
          "reaction_count": 0
        }
      ]
    }
  ]
}
```

## 구현 큰 흐름

1) 백엔드 라우팅/정적 파일/템플릿 설정
2) 마인드맵 페이지 라우트(`/mindmap`)
3) 마인드맵 데이터 API(`/api/mindmap-data`): JSON → 트리 변환
4) 소주제 메시지 API(`/api/mindmap-messages`): 소주제별 메시지 반환(시간순)
5) 템플릿(`templates/mindmap.html`): 좌우 2컬럼 레이아웃
6) 정적 JS(`static/mindmap.js`): D3 트리 렌더/노드 클릭 시 메시지 로드
7) 스타일(`static/styles.css`): 마인드맵/채팅 패널 기본 스타일

---

## 1) 백엔드 기본 설정

다음이 갖춰져 있어야 합니다.
- 정적 파일 마운트: `/static` → `static/`
- 템플릿: Jinja2 `templates/`

관련 코드 위치: [api_server.py](mdc:api_server.py)

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
```

## 2) 마인드맵 페이지 라우트(`/mindmap`)

템플릿을 반환하는 GET 라우트를 추가합니다.

```python
from fastapi import Request
from fastapi.responses import HTMLResponse

@app.get("/mindmap", response_class=HTMLResponse)
def mindmap_page(request: Request):
    return templates.TemplateResponse("mindmap.html", {"request": request})
```

## 3) 마인드맵 데이터 API(`/api/mindmap-data`)

역할: 최신 분석 결과 파일(또는 쿼리로 지정한 파일)을 읽어 D3 트리 형식으로 변환하여 반환합니다.

- 최신 파일 찾기(관례: `analysis_results/analysis_result_*.json`):
```python
def _find_latest_analysis_file() -> Optional[str]:
    base = os.path.abspath("analysis_results")
    # base에서 최신 수정시간 파일을 선택
```

- 트리 변환 함수 포맷(예시):
```python
def _build_mindmap_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    # 반환 예: { name: "주제 맵", children: [ { name, meta, children: [...] }, ... ] }
    # meta.topic_index, meta.subtopic_index 등을 포함해 프론트에서 클릭 시 쿼리로 사용
```

- API 라우트: 파일 경로 파라미터 `?file=`(선택)
```python
@app.get("/api/mindmap-data")
def mindmap_data(file: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    # 파일 결정 → JSON 로드 → _build_mindmap_from_analysis → 반환
```

## 4) 소주제 메시지 API(`/api/mindmap-messages`)

역할: `topic_index`와 `subtopic_index`를 받아 해당 소주제의 메시지들을 정렬해 반환합니다.

- 정렬 기준: 메시지 `content` 앞에 `HH:MM` 형태가 있으면 이를 시간으로 해석해 오름차순 정렬. 없으면 원 순서 유지.

핵심 로직(요지):
```python
def _extract_time_minutes_prefix(text: Optional[str]) -> Optional[int]:
    # "10:13 내용..." → 10*60 + 13

def _resolve_subtopic_messages(topic: Dict[str, Any], subtopic_index: int):
    # merged_subtopics에서 subtopic_index 찾기 → related_message_ids
    # all_related_messages에서 message_id 매칭
    # 시간 프리픽스 기준 정렬

@app.get("/api/mindmap-messages")
def mindmap_messages(topic_index: int, subtopic_index: int, file: Optional[str] = None):
    # 파일 결정 → topics[topic_index] → _resolve_subtopic_messages
    # { topic, subtopic, messages: [...] } 반환
```

## 5) 템플릿(`templates/mindmap.html`)

- 상단 툴바(파일 경로 입력/불러오기)
- 2컬럼 레이아웃: 좌측 SVG, 우측 채팅 패널
- D3 스크립트 CDN + 전용 JS 로드

관련 파일: [mindmap.html](mdc:templates/mindmap.html)

요지:
```html
<div class="mindmap-grid">
  <section class="mindmap-left">
    <div class="mindmap-wrap">
      <svg id="mindmap" class="mindmap-svg"></svg>
      <div id="tooltip" class="mindmap-tooltip"></div>
    </div>
  </section>
  <aside class="mindmap-right">
    <section id="messages-panel" class="messages-panel" style="display:none;">
      <header class="messages-header">
        <h3 id="messages-title">관련 메시지</h3>
      </header>
      <div id="messages-list" class="chat-messages"></div>
    </section>
  </aside>
  
  <script src="https://unpkg.com/d3@7"></script>
  <script src="/static/mindmap.js"></script>
</div>
```

## 6) 정적 JS(`static/mindmap.js`)

역할: D3 트리 렌더링, 노드 클릭 시 메시지 패널에 채팅 버블로 표시

- 트리 데이터 로드: `GET /api/mindmap-data[?file=...]`
- D3 트리 레이아웃: `d3.tree().nodeSize([rowHeight, colWidth])`
- 노드 텍스트에 툴팁/클릭 핸들러 연결
- 메시지 로드: `GET /api/mindmap-messages?topic_index=..&subtopic_index=..[&file=..]`

핵심 흐름(요지):
```javascript
async function fetchData() { /* mindmap-data 불러오기 */ }
function render(tree) {
  const root = d3.hierarchy(tree);
  d3.tree().nodeSize([32, 200])(root);
  // links, nodes 생성
  // node text에 .on('click', ...) → loadMessages
}
async function loadMessages(topicIndex, subtopicIndex) {
  // mindmap-messages 불러와 오른쪽 패널에 채팅 버블 렌더
}
```

관련 파일: [mindmap.js](mdc:static/mindmap.js)

## 7) 스타일(`static/styles.css`)

- 2컬럼 레이아웃: `.mindmap-grid { display: grid; grid-template-columns: 1.6fr 1fr; }`
- 마인드맵 SVG/노드/링크 색상
- 우측 채팅 패널: `.messages-panel`, `.chat-messages`, `.chat-row .bubble`

관련 파일: [styles.css](mdc:static/styles.css)

```css
.mindmap-grid { display: grid; grid-template-columns: 1.6fr 1fr; gap: 14px; }
.chat-messages { overflow: auto; display: flex; flex-direction: column; gap: 12px; }
.chat-row .bubble { padding: 10px 12px; border-radius: 12px; }
```

---

## 실행/테스트 체크리스트

- 서버 실행: `uvicorn api_server:app --reload`
- 브라우저 접속: `http://localhost:8000/mindmap`
- 트리 노드 확인: 부모/자식이 올바른지 확인
- 소주제(leaf) 클릭: 우측 채팅 패널에 메시지들이 시간순으로 출력되는지 확인
- 특정 파일 테스트: 상단 입력에 JSON 경로를 넣고 “불러오기”

## 트러블슈팅

- 트리가 비어있음: `analysis_results` 폴더에 파일이 없는지 확인하거나 `?file=`로 명시
- 소주제 클릭해도 메시지 없음: 해당 소주제의 `related_message_ids`와 `all_related_messages.message_id`가 매칭되는지 확인
- 정렬이 이상함: 메시지 앞 `HH:MM` 프리픽스가 없으면 기존 순서로 보이므로, 정렬 규칙을 필요에 맞게 변경

## 확장 아이디어

- 노드 검색/필터(키워드, 메시지 수 범위)
- 링크 클릭 시 원본 메시지로 이동(별도 URL/앱 연계)
- 그래프 레이아웃(포스 다이어그램) 전환 옵션
- PNG/SVG 내보내기 버튼 추가

## 파일 체크리스트

- 라우트/API: [api_server.py](mdc:api_server.py)
- 템플릿: [templates/mindmap.html](mdc:templates/mindmap.html)
- 스크립트: [static/mindmap.js](mdc:static/mindmap.js)
- 스타일: [static/styles.css](mdc:static/styles.css)
- 데이터 샘플: [analysis_results/analysis_result_20250922_231029.json](mdc:analysis_results/analysis_result_20250922_231029.json)

---

## 정리

- 핵심은 “JSON → 트리 변환”과 “소주제 → 메시지 매핑” 두 단계입니다.
- 트리 렌더는 D3 기본 트리 레이아웃을 사용하여 복잡도를 낮추고, 메시지 패널은 단순한 채팅 버블 UI로 구현합니다.
- 위 순서를 그대로 따라가면 초보자도 충분히 구현 가능합니다. 작은 단위로 나눠서 하나씩 확인하면서 진행하세요.


---

## 빠르게 따라하기: 순서와 코드 + 라인별 설명

아래는 “맨 처음부터” 차례대로 추가할 코드와, 각 줄이 무슨 역할을 하는지 주석으로 설명한 가이드입니다. 이미 같은 기능이 있다면, 비교하면서 읽고 이해용으로 활용하세요.

### 1) 백엔드 기본 세팅 (정적/템플릿)

파일: `api_server.py`

```python
from fastapi import FastAPI                           # FastAPI 웹 프레임워크 임포트
from fastapi.staticfiles import StaticFiles            # 정적 파일 제공을 위한 헬퍼
from fastapi.templating import Jinja2Templates        # Jinja2 템플릿 엔진 연결

app = FastAPI(title="Cube RAG API", version="0.1.0") # FastAPI 앱 생성

app.mount("/static", StaticFiles(directory="static"), name="static") # /static 경로로 정적 파일 서빙
templates = Jinja2Templates(directory="templates")                   # templates 디렉토리를 템플릿 루트로 등록
```

핵심 포인트
- 정적 리소스(CSS/JS)는 `/static`으로 접근합니다.
- HTML은 `templates/` 아래에서 Jinja2로 렌더링합니다.

### 2) 마인드맵 페이지 라우트 추가

파일: `api_server.py`

```python
from fastapi import Request                         # 요청 객체 타입
from fastapi.responses import HTMLResponse          # HTML 반환 타입

@app.get("/mindmap", response_class=HTMLResponse)  # /mindmap GET 엔드포인트
def mindmap_page(request: Request):                 # request는 템플릿에 전달할 컨텍스트 포함
    return templates.TemplateResponse(              # mindmap.html을 렌더링해 반환
        "mindmap.html", {"request": request}
    )
```

핵심 포인트
- 템플릿 렌더 시 Jinja2는 `{"request": request}` 컨텍스트가 필요합니다.

### 3) 최신 분석 파일 찾기 + 트리 변환 함수

파일: `api_server.py`

```python
import os, json                                        # 파일/JSON 처리
from typing import Optional, Dict, Any, List            # 타입 힌트

def _find_latest_analysis_file() -> Optional[str]:     # 최신 분석 결과 파일 경로 반환
    base = os.path.abspath("analysis_results")        # 분석 결과 폴더
    if not os.path.isdir(base):                        # 폴더 없으면 None
        return None
    files = [
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.startswith("analysis_result_") and f.endswith(".json")
    ]
    files = [p for p in files if os.path.isfile(p)]    # 파일만 필터
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True) # 최신 수정순 정렬
    return files[0]                                              # 가장 최신 파일 경로

def _build_mindmap_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    topics = analysis.get("topics") or []               # 주제 배열
    children = []                                        # D3 트리 children 배열
    for t_idx, t in enumerate(topics):                   # 각 주제를 순회 (인덱스 포함)
        parent_name = (t or {}).get("topic_name") or "(무제)" # 부모 이름
        sub_children = []                                # 소주제들 담을 리스트
        merged = (t or {}).get("merged_subtopics") or []       # 소주제 집합
        for m in merged:                                 # 각 소주제를 순회
            s_idx = (m or {}).get("subtopic_index")            # 소주제 인덱스
            s_name = (m or {}).get("topic_name") or "(하위주제)" # 소주제 이름
            label = f"(#" + str(s_idx) + ") " + s_name if s_idx is not None else s_name
            sub_children.append({                        # D3 노드로 변환
                "name": label,
                "meta": {
                    "topic_index": t_idx,              # 부모 인덱스(메시지 조회에 필요)
                    "subtopic_index": s_idx,          # 소주제 인덱스
                    "summary": (m or {}).get("summary"),
                    "message_count": (m or {}).get("message_count"),
                    "keywords": (m or {}).get("keywords") or [],
                },
            })
        children.append({
            "name": parent_name,                        # 부모 노드
            "meta": {
                "topic_index": t_idx,
                "summary": (t or {}).get("summary"),
                "message_count": (t or {}).get("message_count"),
                "keywords": (t or {}).get("keywords") or [],
            },
            "children": sub_children,                   # 소주제들 연결
        })
    return {"name": "주제 맵", "children": children}   # 루트 노드 반환
```

핵심 포인트
- 프론트에서 클릭 시 어떤 소주제인지 식별하려면 `topic_index`와 `subtopic_index`가 메타에 필요합니다.

### 4) 마인드맵 데이터 API 만들기

파일: `api_server.py`

```python
from fastapi import Query, HTTPException                     # 쿼리/에러

@app.get("/api/mindmap-data")                               # 트리 데이터 제공
def mindmap_data(file: Optional[str] = Query(default=None)):
    try:
        # 1) 파일 결정 (사용자가 ?file=로 지정하지 않으면 최신 파일)
        if file:
            target = os.path.abspath(file)
            if not os.path.isfile(target):
                raise HTTPException(status_code=404, detail="파일 없음")
        else:
            target = _find_latest_analysis_file()
            if not target:
                raise HTTPException(status_code=404, detail="분석 파일 없음")
        # 2) JSON 로드
        with open(target, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        # 3) 트리 변환
        tree = _build_mindmap_from_analysis(analysis)
        # 4) 응답
        return {"ok": True, "file": target, "tree": tree}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

핵심 포인트
- 파일 경로를 실수로 잘못 넘겨도 친절한 에러를 주는 편이 디버깅에 유리합니다.

### 5) 소주제 메시지 정렬 유틸 + 메시지 API

파일: `api_server.py`

```python
import re                                              # 정규식: 시간 프리픽스 파싱용
from typing import List, Tuple                         # 타입 힌트

def _extract_time_minutes_prefix(text: Optional[str]) -> Optional[int]:
    s = (text or "").strip()                          # 공백 제거
    m = re.match(r"^(\d{1,2}):(\d{2})\b", s)        # 앞부분에서 HH:MM 매칭
    if not m:
        return None
    h, mm = int(m.group(1)), int(m.group(2))          # 시/분 정수화
    if h < 0 or h > 23 or mm < 0 or mm > 59:          # 값 검증
        return None
    return h * 60 + mm                                 # 분 단위로 환산

def _resolve_subtopic_messages(topic: Dict[str, Any], sub_idx: int) -> Tuple[str, List[Dict[str, Any]]]:
    merged = (topic or {}).get("merged_subtopics") or []       # 소주제 집합
    target = next((m for m in merged if (m or {}).get("subtopic_index") == sub_idx), None)
    if target is None:
        return ("", [])
    sub_name = (target or {}).get("topic_name") or ""           # 소주제 이름
    ids = ((target or {}).get("related_message_ids") or [])     # 관련 메시지 ID들
    pool = (topic or {}).get("all_related_messages") or []      # 상세 메시지 풀
    by_id = {str((it or {}).get("message_id")): {
        "message_id": str((it or {}).get("message_id")),
        "content": (it or {}).get("content"),
        "reaction_count": (it or {}).get("reaction_count"),
        "content_length": (it or {}).get("content_length"),
    } for it in pool}
    items = [by_id[str(mid)] for mid in ids if str(mid) in by_id] # 대상 메시지 모으기

    def sort_key(x):                                            # 정렬 키: 시간 프리픽스
        t = _extract_time_minutes_prefix(x.get("content"))
        return (9999 if t is None else t, x.get("message_id"))

    return (sub_name, sorted(items, key=sort_key))               # 시간순 정렬 결과 반환

@app.get("/api/mindmap-messages")
def mindmap_messages(topic_index: int, subtopic_index: int, file: Optional[str] = None):
    try:
        # 1) 파일 결정
        target = os.path.abspath(file) if file else _find_latest_analysis_file()
        if not target or not os.path.isfile(target):
            raise HTTPException(status_code=404, detail="분석 파일 없음")
        # 2) JSON 로드
        with open(target, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        topics = analysis.get("topics") or []
        if topic_index < 0 or topic_index >= len(topics):
            raise HTTPException(status_code=400, detail="topic_index 범위 오류")
        # 3) 해당 토픽에서 소주제 메시지 해석
        topic = topics[topic_index]
        sub_name, msgs = _resolve_subtopic_messages(topic, subtopic_index)
        # 4) 응답
        return {
            "ok": True,
            "file": target,
            "topic": {"index": topic_index, "name": (topic or {}).get("topic_name") or ""},
            "subtopic": {"index": subtopic_index, "name": sub_name},
            "messages": msgs,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

핵심 포인트
- 정렬 규칙은 프로젝트에 맞게 바꿀 수 있습니다(예: 반응 수, 길이 등).

### 6) 템플릿(좌/우 레이아웃 + SVG + 패널)

파일: `templates/mindmap.html`

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mindmap</title>
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <main class="container">
    <section class="mindmap-toolbar">
      <input id="file-input" type="text" placeholder="특정 JSON 경로 (선택)" style="flex:1;" />
      <button id="btn-load" class="btn primary" type="button">불러오기</button>
      <button id="btn-reset" class="btn" type="button">초기화</button>
    </section>
    <div class="mindmap-grid">
      <section class="mindmap-left">
        <div class="mindmap-wrap">
          <svg id="mindmap" class="mindmap-svg"></svg>     <!-- 마인드맵 SVG 루트 -->
          <div id="tooltip" class="mindmap-tooltip"></div>  <!-- 노드 툴팁 -->
        </div>
      </section>
      <aside class="mindmap-right">
        <section id="messages-panel" class="messages-panel" style="display:none;">
          <header class="messages-header">
            <h3 id="messages-title">관련 메시지</h3>
          </header>
          <div id="messages-list" class="chat-messages"></div>
        </section>
      </aside>
    </div>
  </main>
  <script src="https://unpkg.com/d3@7"></script>  <!-- D3 로더 -->
  <script src="/static/mindmap.js"></script>      <!-- 전용 스크립트 -->
  </body>
</html>
```

핵심 포인트
- `mindmap-grid`로 좌(트리)/우(채팅) 영역을 나눕니다.

### 7) CSS(2컬럼 + 채팅 스타일)

파일: `static/styles.css`

```css
.mindmap-grid { display: grid; grid-template-columns: 1.6fr 1fr; gap: 14px; } /* 좌우 비율 */
.mindmap-wrap { background: var(--panel); height: 80vh; border: 1px solid var(--border); border-radius: 8px; }
.mindmap-svg { width: 100%; height: 100%; }

.messages-panel { display: flex; flex-direction: column; height: 80vh; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; }
.messages-header { padding: 10px 12px; border-bottom: 1px solid var(--border); }
.chat-messages { flex: 1; padding: 12px; overflow: auto; display: flex; flex-direction: column; gap: 12px; }
.chat-row { display: flex; }
.chat-row .bubble { max-width: 86%; padding: 10px 12px; border-radius: 12px; border: 1px solid var(--border); background: var(--assistant-bubble); white-space: pre-wrap; font-size: 0.92rem; line-height: 1.5; }
```

핵심 포인트
- 높이는 프로젝트에 맞춰 조정하세요(예: 70vh/80vh).

### 8) JS(D3 트리 + 클릭 → 메시지 로드)

파일: `static/mindmap.js`

```javascript
(function() {
  const svg = document.getElementById('mindmap');                 // SVG 루트
  const tooltip = document.getElementById('tooltip');             // 노드 툴팁
  const fileInput = document.getElementById('file-input');        // 파일 경로 입력
  const btnLoad = document.getElementById('btn-load');            // 불러오기 버튼
  const btnReset = document.getElementById('btn-reset');          // 초기화 버튼

  const d3sel = d3.select(svg);                                   // D3 셀렉션
  let gRoot = d3sel.append('g');                                   // 전체 그룹 (줌 대상)
  let zoom = d3.zoom().scaleExtent([0.3, 2]).on('zoom', (ev) => { // 줌/이동 설정
    gRoot.attr('transform', ev.transform);
  });
  d3sel.call(zoom);                                               // SVG에 줌 바인딩

  function clear() { gRoot.selectAll('*').remove(); }             // 기존 렌더링 초기화

  async function fetchData() {                                     // 트리 데이터 로드
    const file = (fileInput && fileInput.value.trim()) || null;
    const url = new URL('/api/mindmap-data', window.location.origin);
    if (file) url.searchParams.set('file', file);
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    return data.tree;
  }

  function render(treeData) {                                      // D3 트리 렌더
    clear();
    const root = d3.hierarchy(treeData);                           // D3 계층 데이터 생성
    const layout = d3.tree().nodeSize([32, 200]);                  // 노드 간격(행, 열)
    layout(root);                                                  // 좌표 계산

    const x0 = d3.min(root.descendants(), d => d.x) || 0;         // 수직 범위 최소/최대
    const x1 = d3.max(root.descendants(), d => d.x) || 0;
    const offX = ((svg.clientHeight || 600) - (x1 - x0)) / 2;      // 중앙 정렬 오프셋
    const offY = 40;                                               // 좌측 여백
    const container = gRoot.append('g').attr('transform', `translate(${offY},${offX - x0})`);

    // 링크 렌더
    container.selectAll('path.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('fill', 'none')
      .attr('stroke', '#aaa')
      .attr('stroke-width', 1.5)
      .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));

    // 노드 렌더
    const node = container.selectAll('g.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', d => 'node' + (d.depth === 0 ? ' root' : (d.children ? ' parent' : ' leaf')))
      .attr('transform', d => `translate(${d.y},${d.x})`);

    node.append('circle')
      .attr('r', d => d.depth === 0 ? 8 : (d.children ? 6 : 4))       // 루트/부모/리프 반지름
      .attr('fill', d => d.depth === 0 ? '#0ea5e9' : (d.children ? '#22c55e' : '#6366f1'));

    node.append('text')
      .attr('dy', '0.32em')
      .attr('x', d => d.children ? -10 : 10)                           // 부모 방향에 따라 위치
      .attr('text-anchor', d => d.children ? 'end' : 'start')
      .style('font-size', '12px')
      .text(d => (d.data && d.data.name) ? d.data.name : '(이름없음)') // 라벨 표시
      .style('cursor', d => (!d.children && d.depth > 0) ? 'pointer' : 'default')
      .on('click', async (event, d) => {                                // 리프 노드 클릭 시
        if (d.children || d.depth === 0) return;                        // 부모/루트는 무시
        const meta = (d && d.data && d.data.meta) ? d.data.meta : null; // 메타 추출
        if (!meta || typeof meta.subtopic_index === 'undefined') return;
        await loadMessages(meta.topic_index, meta.subtopic_index);       // 메시지 로드
      })
      .on('mousemove', (event, d) => {                                  // 툴팁 표시
        const meta = (d && d.data && d.data.meta) ? d.data.meta : null;
        const lines = [];
        if (meta) {
          if (typeof meta.subtopic_index !== 'undefined') lines.push(`#${meta.subtopic_index}`);
          if (meta.summary) lines.push(meta.summary);
          if (Array.isArray(meta.keywords) && meta.keywords.length) lines.push(`키워드: ${meta.keywords.join(', ')}`);
          if (typeof meta.message_count === 'number') lines.push(`관련 메시지 수: ${meta.message_count}`);
        }
        if (lines.length) {
          tooltip.style.display = 'block';
          tooltip.textContent = lines.join('\n');
          const rect = svg.getBoundingClientRect();
          tooltip.style.left = (event.pageX - rect.left + 12) + 'px';
          tooltip.style.top = (event.pageY - rect.top + 12) + 'px';
        }
      })
      .on('mouseleave', () => { tooltip.style.display = 'none'; });     // 툴팁 숨김

  }

  async function loadMessages(topicIndex, subtopicIndex) {               // 메시지 API 호출
    const listEl = document.getElementById('messages-list');
    const panel = document.getElementById('messages-panel');
    const titleEl = document.getElementById('messages-title');
    if (!listEl || !panel || !titleEl) return;
    const file = (fileInput && fileInput.value.trim()) || null;
    const url = new URL('/api/mindmap-messages', window.location.origin);
    url.searchParams.set('topic_index', String(topicIndex));
    url.searchParams.set('subtopic_index', String(subtopicIndex));
    if (file) url.searchParams.set('file', file);
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(await res.text());
    const payload = await res.json();
    listEl.innerHTML = '';
    titleEl.textContent = (payload.subtopic && payload.subtopic.name) ? `관련 메시지 · ${payload.subtopic.name}` : '관련 메시지';
    for (const m of (payload.messages || [])) {
      const row = document.createElement('div'); row.className = 'chat-row';
      const col = document.createElement('div');
      const meta = document.createElement('div'); meta.className = 'meta';
      meta.textContent = `${m.message_id ? '#' + m.message_id : ''}${typeof m.reaction_count === 'number' ? ' · 반응 ' + m.reaction_count : ''}`;
      const bubble = document.createElement('div'); bubble.className = 'bubble';
      bubble.textContent = m.content || '';
      col.appendChild(meta); col.appendChild(bubble); row.appendChild(col); listEl.appendChild(row);
    }
    panel.style.display = '';
  }

  async function load() { try { render(await fetchData()); } catch (e) { alert(e.message || e); } }
  btnLoad && btnLoad.addEventListener('click', load);                    // 불러오기 버튼
  btnReset && btnReset.addEventListener('click', () => { if (fileInput) fileInput.value = ''; load(); });
  load();                                                                // 초기 렌더
})();
```

핵심 포인트
- 최소 기능부터 구현하고, 이후 확대(검색/필터, 애니메이션 등)하세요.

### 9) 실행

```bash
uvicorn api_server:app --reload --port 8000
```

- 브라우저에서 `/mindmap` 접속 → 트리 확인 → 소주제 클릭 → 우측 메시지 노출.


