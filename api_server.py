import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from agent_orchestrator import AgentOrchestrator
from retrieval_agent import retrieve
from web_searcher import search_web

# Streaming(SSE)
try:
    from sse_starlette.sse import EventSourceResponse  # type: ignore
except Exception:  # pragma: no cover
    EventSourceResponse = None  # type: ignore

# For streaming generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from answer_generator import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, CONTEXTS_WRAPPER
import json


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None  # 클라이언트가 세션 키를 주면 이어서 답변


class AskResponse(BaseModel):
    answer: str
    sources: list
    web: list
    analysis: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None


app = FastAPI(title="Cube RAG API", version="0.1.0")

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
"""
간단한 서버 메모리 저장소. 실서비스라면 Redis/DB 또는 사용자 인증 세션 사용 권장.
구조: { conversation_id: [ {"role": "user"|"assistant", "content": str}, ... ] }
"""
conversation_store: Dict[str, list] = {}

def _get_history(cid: Optional[str]) -> list:
    if not cid:
        return []
    return conversation_store.get(cid, [])

def _append_turn(cid: Optional[str], role: str, content: str) -> Optional[str]:
    if not cid:
        # cid 미제공 시 간단히 생성 (주의: 충돌 가능성 낮추기 위해 접두사 사용)
        import uuid
        cid = f"cid_{uuid.uuid4().hex[:12]}"
    turns = conversation_store.setdefault(cid, [])
    turns.append({"role": role, "content": content})
    # 메모리 크기 제한 (최근 50턴만 유지)
    if len(turns) > 50:
        conversation_store[cid] = turns[-50:]
    return cid


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        orchestrator = AgentOrchestrator()
        history = _get_history(req.conversation_id)
        result = orchestrator.run(req.question, history=history)
        answer = result.get("answer") or ""
        sources = result.get("sources") or []
        web = result.get("web") or []
        analysis_obj = result.get("analysis")

        analysis_payload: Optional[Dict[str, Any]] = None
        if analysis_obj is not None:
            if hasattr(analysis_obj, "model_dump"):
                try:
                    analysis_payload = analysis_obj.model_dump()  # pydantic v2
                except Exception:
                    analysis_payload = None
            elif hasattr(analysis_obj, "dict"):
                try:
                    analysis_payload = analysis_obj.dict()  # pydantic v1/dataclass-like
                except Exception:
                    analysis_payload = None
            elif hasattr(analysis_obj, "__dict__"):
                try:
                    analysis_payload = dict(vars(analysis_obj))
                except Exception:
                    analysis_payload = None

        # 히스토리 업데이트
        cid = _append_turn(req.conversation_id, "user", req.question)
        _append_turn(cid, "assistant", answer)

        return AskResponse(answer=answer, sources=sources, web=web, analysis=analysis_payload, conversation_id=cid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask/stream")
def ask_stream(q: str, cid: Optional[str] = None):
    if EventSourceResponse is None:
        raise HTTPException(status_code=500, detail="SSE dependency not installed")

    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="q is required")

    # Retrieval (vector DB)
    hits = retrieve(question)
    # 질문 전용 컬렉션 고려: 메타 question/answer로 컨텍스트 재구성
    contexts = []
    sources_vec = []
    for h in hits:
        meta = h.get("metadata") or {}
        q_meta = (meta.get("question") or "").strip()
        a_meta = (meta.get("answer") or "").strip()
        content = h.get("content")
        if q_meta or a_meta:
            contexts.append(f"Q: {q_meta}\nA: {a_meta}".strip())
        else:
            contexts.append(content)
        sources_vec.append({"metadata": meta, "score": h.get("score")})

    # 단일 리트리벌 기반 컨센서스 규칙(오케스트레이터와 동일)
    try:
        t_high = float(os.getenv("DIRECT_MATCH_HIGH", "0.65"))
    except Exception:
        t_high = 0.65
    try:
        t_mid = float(os.getenv("DIRECT_MATCH_SCORE_THRESHOLD", "0.55"))
    except Exception:
        t_mid = 0.55
    try:
        top_k_cons = int(os.getenv("DIRECT_MATCH_TOPK", "5"))
    except Exception:
        top_k_cons = 5
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
        for s in sources_vec
    ]
    valid_scores = [v for v in raw_scores if v is not None]
    valid_scores.sort(reverse=True)
    top1 = valid_scores[0] if valid_scores else 0.0
    topk_scores = valid_scores[: max(0, top_k_cons)] if valid_scores else []
    consensus = sum(1 for v in topk_scores if v >= t_mid)
    has_direct_match = (top1 >= t_high) or (consensus >= count_cons)

    # Web search toggle
    use_search = os.getenv("USE_WEB_SEARCH", "false").lower() in ("1", "true", "yes", "on")
    web_results = []
    web_contexts = []
    if (not has_direct_match) and use_search:
        web_results = search_web(question, max_results=8)
        for w in web_results:
            title = (w.get("title") or "").strip()
            url = (w.get("url") or "").strip()
            snippet = (w.get("snippet") or "").strip()
            if title or snippet:
                web_contexts.append(f"{title}\n{snippet}\n출처: {url}")

    if has_direct_match:
        preface = "요청하신 질문은 내부 Q&A 데이터에서 근거를 찾았습니다. 아래 출처를 참고하세요."
        effective_sources = sources_vec
        combined_contexts = contexts
    else:
        if use_search:
            preface = (
                "내부 Q&A에 해당 내용이 없어, LLM 일반 지식과 웹 검색 결과를 근거로 답변합니다.\n"
                "가능한 경우 출처(URL/메타)를 함께 표기합니다."
            )
            # 벡터 DB 출처는 숨기고 웹 출처만 표기
            effective_sources = web_results
            combined_contexts = contexts + web_contexts
        else:
            preface = "내부 Q&A에 해당 내용이 없어, 웹 검색 없이 LLM 일반 지식만으로 답변합니다."
            # 출처는 비표기
            effective_sources = None
            combined_contexts = contexts

    model_name = os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
    model = ChatOpenAI(model=model_name, temperature=0.2)

    # 대화 이력 주입
    history = _get_history(cid)
    # history는 answer_generator의 포맷과 동일한 형태
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "user",
            USER_PROMPT_TEMPLATE + "\n\n" + (preface + "\n\n" if preface else "") + CONTEXTS_WRAPPER,
        ),
    ])
    joined_contexts = "\n\n".join([c.strip() for c in combined_contexts if c and c.strip()])
    # history용 문자열 구성
    def _format_turn(turn: Dict[str, str]) -> str:
        r = (turn.get("role") or "").lower()
        c = (turn.get("content") or "").strip()
        if not c:
            return ""
        label = "사용자" if r == "user" else ("시스템" if r == "system" else "어시스턴트")
        return f"[{label}] {c}"
    joined_history = "\n".join([s for s in map(_format_turn, history[-10:]) if s])

    formatted = prompt.format_messages(
        question=question,
        ctx_count=len(combined_contexts),
        contexts=joined_contexts,
        hist_count=len(history[-10:]),
        history=joined_history,
    )

    def event_iterator():
        # Stream tokens
        # Send preface first if exists so users can see source type immediately
        if preface:
            yield {"event": "token", "data": f"알림: {preface}\n\n"}
        try:
            for chunk in model.stream(formatted):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    yield {"event": "token", "data": piece}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

        # Send sources at the end
        try:
            payload = effective_sources if effective_sources is not None else []
            yield {"event": "sources", "data": json.dumps(payload, ensure_ascii=False)}
        except Exception:
            pass

        # 마지막에 히스토리 저장 (user 질문 + assistant 응답 전문은 클라이언트에서 재조합 필요)
        try:
            saved_cid = _append_turn(cid, "user", question)
            # 스트리밍 시 전체 답변은 클라이언트가 조합하므로, 여기서는 간단 표기
            _append_turn(saved_cid, "assistant", "(streamed)")
            yield {"event": "cid", "data": saved_cid}
        except Exception:
            pass

        yield {"event": "done", "data": "end"}

    return EventSourceResponse(event_iterator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api_server:app", host=host, port=port, reload=False)


