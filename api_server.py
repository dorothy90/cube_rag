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


class AskResponse(BaseModel):
    answer: str
    sources: list
    web: list
    analysis: Optional[Dict[str, Any]] = None


app = FastAPI(title="Cube RAG API", version="0.1.0")

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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
        result = orchestrator.run(req.question)
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

        return AskResponse(answer=answer, sources=sources, web=web, analysis=analysis_payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask/stream")
def ask_stream(q: str):
    if EventSourceResponse is None:
        raise HTTPException(status_code=500, detail="SSE dependency not installed")

    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="q is required")

    # Retrieval (vector DB)
    hits = retrieve(question)
    contexts = [h.get("content") for h in hits]
    sources_vec = [
        {
            "metadata": h.get("metadata"),
            "score": h.get("score"),
        }
        for h in hits
    ]

    # Direct match heuristic (same as orchestrator)
    normalized_q = question.lower()
    has_direct_match = any(
        (c or "").lower().find(normalized_q[: min(len(normalized_q), 20)]) >= 0 for c in contexts
    )

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE + "\n\n" + (preface + "\n\n" if preface else "") + CONTEXTS_WRAPPER),
    ])
    joined_contexts = "\n\n".join([c.strip() for c in combined_contexts if c and c.strip()])
    formatted = prompt.format_messages(
        question=question,
        ctx_count=len(combined_contexts),
        contexts=joined_contexts,
    )

    def event_iterator():
        # Stream tokens
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

        yield {"event": "done", "data": "end"}

    return EventSourceResponse(event_iterator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api_server:app", host=host, port=port, reload=False)


