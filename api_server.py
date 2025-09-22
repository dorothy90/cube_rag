import os
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Query
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
from query_analyzer_agent import QueryAnalyzerAgent
import json
from urllib.parse import urlparse
import subprocess
import shlex
import time
from pathlib import Path


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None  # 클라이언트가 세션 키를 주면 이어서 답변


class AskResponse(BaseModel):
    answer: str
    sources: list
    web: list
    analysis: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    selected_domain: Optional[str] = None


class SummaryRequest(BaseModel):
    from_date: str
    to_date: str
    email: str
    mode: Optional[str] = None
    conversation_id: Optional[str] = None


class SummaryResponse(BaseModel):
    ok: bool
    message: str
    conversation_id: Optional[str] = None


class SendEmailRequest(BaseModel):
    from_date: str
    to_date: str
    email: str


class SendEmailResponse(BaseModel):
    ok: bool
    message: str


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


# -----------------------------
# Sources normalization helpers
# -----------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _canonical_key_for_url(u: str) -> str:
    try:
        p = urlparse(u or "")
        return f"{(p.hostname or '').lower()}{p.path or ''}"
    except Exception:
        return (u or "").strip().lower()


def normalize_sources(
    internal: Optional[list],
    web: Optional[list],
    only: str,
) -> list:
    """Return unified sources schema.

    only: "internal" | "web" (우선 노출 대상을 선택)
    """
    max_internal = _env_int("SOURCES_MAX_INTERNAL", 3)
    max_web = _env_int("SOURCES_MAX_WEB", 5)

    result: list = []

    if only == "internal" and internal:
        # internal item shape: {metadata: {...}, score: float}
        dedupe = set()
        items = []
        for it in internal:
            meta = (it or {}).get("metadata") or {}
            q = (meta.get("question") or "").strip()
            ts = (meta.get("timestamp") or "").strip()
            q_author = (meta.get("question_author") or "").strip()
            # '||' 조인 문자열 정규화 (간결화)
            def _split_pipe(s):
                return [p.strip() for p in str(s or "").split("||") if p and p.strip()]
            answers = _split_pipe(meta.get("answers"))
            answer_authors = _split_pipe(meta.get("answer_author") or meta.get("answer_authors"))
            if answers and len(answer_authors) < len(answers):
                answer_authors = answer_authors + ["알 수 없음"] * (len(answers) - len(answer_authors))
            key = f"qa::{ts}::{q}"
            if key in dedupe:
                continue
            dedupe.add(key)
            items.append({
                "type": "internal",
                "score": (it or {}).get("score"),
                "question": q,
                "answers": answers,
                "timestamp": ts,
                "question_author": q_author,
                "answer_authors": answer_authors,
            })
        # sort by score desc then timestamp desc
        items.sort(key=lambda x: (x.get("score") or 0.0, x.get("timestamp") or ""), reverse=True)
        result.extend(items[:max_internal])

    if only == "web" and web:
        dedupe = set()
        items = []
        for w in web:
            url = (w or {}).get("url") or ""
            title = (w or {}).get("title") or ""
            key = _canonical_key_for_url(url)
            if not key or key in dedupe:
                continue
            dedupe.add(key)
            items.append({
                "type": "web",
                "score": (w or {}).get("score"),
                "title": title.strip(),
                "url": url.strip(),
            })
        result.extend(items[:max_web])

    return result


def _normalize_domain_text(text: Optional[str]) -> Optional[str]:
    """주어진 텍스트에서 도메인 선택 의도를 정규화하여 반환.

    반환값: "python" | "sql" | "semiconductor" | None
    """
    if not text:
        return None
    s = (text or "").strip().lower()
    # 간단한 정규화: 포함어 기반 매칭 (도메인 선택 답변 문맥에서만 사용)
    if ("python" in s) or ("파이썬" in s) or ("py" == s):
        return "python"
    if ("sql" in s) or ("에스큐엘" in s) or ("씨큐엘" in s):
        return "sql"
    if ("semiconductor" in s) or ("반도체" in s) or ("semi" in s):
        return "semiconductor"
    return None


def _find_latest_analysis_file() -> Optional[str]:
    """analysis_results 디렉토리에서 가장 최신 분석 결과 파일 경로를 반환한다.

    파일 네이밍 규칙: analysis_result_*.json
    """
    try:
        base = os.path.abspath("analysis_results")
        if not os.path.isdir(base):
            return None
        files = [
            os.path.join(base, f)
            for f in os.listdir(base)
            if f.startswith("analysis_result_") and f.endswith(".json")
        ]
        files = [p for p in files if os.path.isfile(p)]
        if not files:
            return None
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]
    except Exception:
        return None


def _build_mindmap_from_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """analysis JSON에서 mindmap용 트리 데이터를 생성한다.

    출력 스키마 (D3 tree 호환):
    { name: str, children: [ { name: str, children: [ { name: str }, ... ] }, ... ] }
    """
    topics = analysis.get("topics") or []
    children = []
    for t_idx, t in enumerate(topics):
        t_name = (t or {}).get("topic_name") or "(무제)"
        sub_children = []
        merged = (t or {}).get("merged_subtopics") or []
        for m in merged:
            idx = (m or {}).get("subtopic_index")
            st_name = (m or {}).get("topic_name") or "(하위주제)"
            # 노드 이름에 인덱스 표시
            label = f"(#" + str(idx) + ") " + st_name if idx is not None else st_name
            sub_children.append({
                "name": label,
                "meta": {
                    "topic_index": t_idx,
                    "subtopic_index": idx,
                    "summary": (m or {}).get("summary"),
                    "message_count": (m or {}).get("message_count"),
                    "keywords": (m or {}).get("keywords") or [],
                },
            })
        children.append({
            "name": t_name,
            "meta": {
                "topic_index": t_idx,
                "summary": (t or {}).get("summary"),
                "message_count": (t or {}).get("message_count"),
                "keywords": (t or {}).get("keywords") or [],
            },
            "children": sub_children,
        })
    return {"name": "주제 맵", "children": children}


def _extract_time_minutes_prefix(text: Optional[str]) -> Optional[int]:
    """문장 앞의 'H:MM' 또는 'HH:MM' 형태 시간을 분 단위로 변환.
    발견 못하면 None.
    """
    try:
        s = (text or "").strip()
        if not s:
            return None
        # 앞쪽 토큰에서 시:분 패턴 추출
        import re
        m = re.match(r"^(\d{1,2}):(\d{2})\b", s)
        if not m:
            return None
        h = int(m.group(1))
        mm = int(m.group(2))
        if h < 0 or h > 23 or mm < 0 or mm > 59:
            return None
        return h * 60 + mm
    except Exception:
        return None


def _resolve_subtopic_messages(topic: Dict[str, Any], subtopic_index: int) -> Tuple[str, List[Dict[str, Any]]]:
    """토픽 객체에서 특정 subtopic_index에 해당하는 메시지들을 찾아 시간순으로 정렬해 반환한다.

    반환: (subtopic_name, messages[...])
    메시지 원소 스키마: { message_id, content, reaction_count, content_length }
    """
    merged = (topic or {}).get("merged_subtopics") or []
    target = None
    for m in merged:
        if (m or {}).get("subtopic_index") == subtopic_index:
            target = m
            break
    if target is None:
        return ("", [])
    subtopic_name = (target or {}).get("topic_name") or ""
    ids = ((target or {}).get("related_message_ids") or [])
    ids_set = set([str(x) for x in ids])

    # 토픽 레벨의 all_related_messages에서 상세 메시지 찾기
    pool = (topic or {}).get("all_related_messages") or []
    by_id = {}
    for it in pool:
        mid = str((it or {}).get("message_id"))
        by_id[mid] = {
            "message_id": mid,
            "content": (it or {}).get("content"),
            "reaction_count": (it or {}).get("reaction_count"),
            "content_length": (it or {}).get("content_length"),
        }

    items = [by_id[mid] for mid in ids if str(mid) in by_id]

    # 시간 파싱 정렬 (실패 시 기존 순서 유지)
    def sort_key(x):
        t = _extract_time_minutes_prefix(x.get("content"))
        return (9999 if t is None else t, x.get("message_id"))

    items_sorted = sorted(items, key=sort_key)
    return (subtopic_name, items_sorted)


@app.get("/mindmap", response_class=HTMLResponse)
def mindmap_page(request: Request):
    return templates.TemplateResponse("mindmap.html", {"request": request})


@app.get("/api/mindmap-data")
def mindmap_data(file: Optional[str] = Query(default=None, description="분석 결과 JSON 파일 경로(선택)")) -> Dict[str, Any]:
    try:
        target_path = None
        if file:
            abs_path = os.path.abspath(file)
            if not os.path.isfile(abs_path):
                raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {abs_path}")
            target_path = abs_path
        else:
            latest = _find_latest_analysis_file()
            if not latest:
                raise HTTPException(status_code=404, detail="analysis_results 폴더에서 분석 파일을 찾을 수 없습니다.")
            target_path = latest

        with open(target_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        tree = _build_mindmap_from_analysis(analysis)
        return {"ok": True, "file": target_path, "tree": tree}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mindmap-messages")
def mindmap_messages(
    topic_index: int = Query(..., description="부모 토픽 인덱스 (0-base)"),
    subtopic_index: int = Query(..., description="소주제 subtopic_index"),
    file: Optional[str] = Query(default=None, description="분석 결과 JSON 파일 경로(선택)"),
) -> Dict[str, Any]:
    try:
        # 파일 결정
        if file:
            target_path = os.path.abspath(file)
            if not os.path.isfile(target_path):
                raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {target_path}")
        else:
            target_path = _find_latest_analysis_file()
            if not target_path:
                raise HTTPException(status_code=404, detail="analysis_results 폴더에서 분석 파일을 찾을 수 없습니다.")

        with open(target_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        topics = analysis.get("topics") or []
        if topic_index < 0 or topic_index >= len(topics):
            raise HTTPException(status_code=400, detail="topic_index 범위 오류")
        topic = topics[topic_index]
        subtopic_name, msgs = _resolve_subtopic_messages(topic, subtopic_index)
        return {
            "ok": True,
            "file": target_path,
            "topic": {
                "index": topic_index,
                "name": (topic or {}).get("topic_name") or "",
            },
            "subtopic": {
                "index": subtopic_index,
                "name": subtopic_name,
            },
            "messages": msgs,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
        # 마지막 도메인 기억값 로드
        last_domain: Optional[str] = None
        if history:
            # 직전 assistant 메타에 도메인 라벨을 저장했다면 활용 (간단 구현: 별도 맵)
            # 여기서는 conversation_store에 별도 키로 저장
            pass
        # 대화별 마지막 도메인 저장소 (간단 구현)
        domain_memory = conversation_store.setdefault("__domain_memory__", {})
        last_domain = domain_memory.get(req.conversation_id or "")
        # 도메인 확인 대기 중인 원 질문 저장소
        pending_map: Dict[str, str] = conversation_store.setdefault("__pending_domain__", {})  # cid -> original question
        pending_q = pending_map.get(req.conversation_id or "")

        # 0) 직전 턴이 도메인 확인이었고, 이번 입력이 도메인 선택 응답이면 원 질문으로 이어서 답변
        if pending_q:
            forced_domain = _normalize_domain_text(req.question)
            if forced_domain in ("python", "sql", "semiconductor"):
                # 선택된 도메인을 메모리에 먼저 기록 (오케스트레이터는 last_domain을 활용)
                result = orchestrator.run(pending_q, history=history, last_domain=forced_domain)
                answer = result.get("answer") or ""
                raw_internal = result.get("sources") or []
                raw_web = result.get("web") or []
                analysis_obj = result.get("analysis")
                selected_domain = result.get("selected_domain") or forced_domain

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

                # 히스토리 업데이트: 도메인 선택 입력은 질문으로 기록하지 않음 (원 질문은 이미 기록됨)
                cid = req.conversation_id
                if not cid:
                    # 예외적으로 cid가 없으면 생성 후 어시스턴트만 기록
                    cid = _append_turn(None, "assistant", answer)
                else:
                    _append_turn(cid, "assistant", answer)

                # 도메인 메모리 업데이트 및 보류 상태 해제
                if selected_domain in ("python", "sql", "semiconductor"):
                    domain_memory[cid] = selected_domain
                # 보류 해제
                try:
                    pending_map.pop(cid, None)
                except Exception:
                    pass

                # Normalize sources: 내부가 있으면 내부만, 없으면 웹만 노출
                only = "internal" if raw_internal else ("web" if raw_web else "internal")
                normalized = normalize_sources(raw_internal, raw_web, only=only)
                return AskResponse(
                    answer=answer,
                    sources=normalized,
                    web=raw_web,
                    analysis=analysis_payload,
                    conversation_id=cid,
                    selected_domain=selected_domain,
                )

        result = orchestrator.run(req.question, history=history, last_domain=last_domain)
        answer = result.get("answer") or ""
        raw_internal = result.get("sources") or []
        raw_web = result.get("web") or []
        stage = result.get("stage")
        analysis_obj = result.get("analysis")
        selected_domain = result.get("selected_domain")

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
        # 도메인 확인 요청을 보낸 경우, 다음 턴의 선택을 대기하기 위해 원 질문을 보관
        if (stage == "clarify_domain") and req.question:
            pending_map[cid] = req.question
        # 도메인 메모리 업데이트
        if selected_domain in ("python", "sql", "semiconductor"):
            domain_memory[cid] = selected_domain

        # Normalize sources: 내부가 있으면 내부만, 없으면 웹만 노출
        only = "internal" if raw_internal else ("web" if raw_web else "internal")
        normalized = normalize_sources(raw_internal, raw_web, only=only)
        return AskResponse(answer=answer, sources=normalized, web=raw_web, analysis=analysis_payload, conversation_id=cid, selected_domain=selected_domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=SummaryResponse)
def summarize(req: SummaryRequest) -> SummaryResponse:
    try:
        # 간단한 유효성 검사
        try:
            d_from = datetime.strptime((req.from_date or "").strip(), "%Y-%m-%d")
            d_to = datetime.strptime((req.to_date or "").strip(), "%Y-%m-%d")
        except Exception:
            raise HTTPException(status_code=400, detail="from_date/to_date는 YYYY-MM-DD 형식이어야 합니다.")
        if d_from > d_to:
            raise HTTPException(status_code=400, detail="from_date는 to_date보다 이후일 수 없습니다.")
        email = (req.email or "").strip()
        if ("@" not in email) or (len(email) < 5):
            raise HTTPException(status_code=400, detail="유효한 이메일 주소를 입력해주세요.")

        # 대화 히스토리에는 요약 요청을 간단히 남김
        msg_user = f"[요약요청] {req.from_date} ~ {req.to_date} -> {email}"
        cid = _append_turn(req.conversation_id, "user", msg_user)
        ack = f"요약 요청이 접수되었습니다. 처리 후 {email} 로 결과를 전송합니다. (기간: {req.from_date} ~ {req.to_date})"
        _append_turn(cid, "assistant", ack)

        # TODO: 여기서 실제 요약 파이프라인을 비동기로 실행하도록 연계 가능합니다.
        return SummaryResponse(ok=True, message=ack, conversation_id=cid)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send-summary-email", response_model=SendEmailResponse)
def send_summary_email(req: SendEmailRequest) -> SendEmailResponse:
    """요약 결과를 이메일로 보내는 워크플로를 트리거한다.
    내부적으로 summary/analyze_data.py를 호출하여 from/to 범위에 해당하는 데이터 요약을 실행하도록 한다.
    (현 구현은 analyze_data.py의 메인 로직 실행을 트리거하는 형태이며, 필요 시 인자로 구체 범위를 추가 연계 가능)
    """
    try:
        # 입력 유효성 검사
        try:
            d_from = datetime.strptime((req.from_date or "").strip(), "%Y-%m-%d")
            d_to = datetime.strptime((req.to_date or "").strip(), "%Y-%m-%d")
        except Exception:
            raise HTTPException(status_code=400, detail="from_date/to_date는 YYYY-MM-DD 형식이어야 합니다.")
        if d_from > d_to:
            raise HTTPException(status_code=400, detail="from_date는 to_date보다 이후일 수 없습니다.")
        email = (req.email or "").strip()
        if ("@" not in email) or (len(email) < 5):
            raise HTTPException(status_code=400, detail="유효한 이메일 주소를 입력해주세요.")

        # analyze_data.py 실행: 로그 스트리밍 + 타임아웃 설정
        # - PYTHONUNBUFFERED 또는 -u 옵션으로 버퍼링 비활성화하여 실시간 출력
        # - SUMMARY_SCRIPT_TIMEOUT_SECS 환경변수로 타임아웃(초) 조정 가능 (기본 600초)
        script_path = "/Users/daehwankim/cube_rag/summary/analyze_data.py"
        timeout_secs = int(os.getenv("SUMMARY_SCRIPT_TIMEOUT_SECS", "600"))
        cmd_list = [
            "python3",
            "-u",
            script_path,
            "--from",
            req.from_date,
            "--to",
            req.to_date,
            "--email",
            email,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"스크립트 시작 실패: {e}")

        start_ts = time.monotonic()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(f"[analyze_data] {line.rstrip()}")
                if (time.monotonic() - start_ts) > timeout_secs:
                    proc.kill()
                    raise HTTPException(status_code=500, detail=f"스크립트 실행 타임아웃({timeout_secs}s)")
        finally:
            try:
                rc = proc.wait(timeout=5)
            except Exception:
                proc.kill()
                rc = -1
        if rc != 0:
            raise HTTPException(status_code=500, detail=f"스크립트 종료 코드: {rc}")

        return SendEmailResponse(ok=True, message="메일 전송 스크립트가 실행되었습니다. (콘솔 출력 확인)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/send-summary-email/stream")
def send_summary_email_stream(from_date: str, to_date: str, email: str):
    """요약 스크립트 표준출력을 SSE로 스트리밍한다.

    프론트에서는 EventSource로 연결하여 'log' 이벤트를 수신해 UI에 실시간 출력한다.
    """
    if EventSourceResponse is None:
        raise HTTPException(status_code=500, detail="SSE dependency not installed")

    # 입력 유효성 검사
    try:
        d_from = datetime.strptime((from_date or "").strip(), "%Y-%m-%d")
        d_to = datetime.strptime((to_date or "").strip(), "%Y-%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="from_date/to_date는 YYYY-MM-DD 형식이어야 합니다.")
    if d_from > d_to:
        raise HTTPException(status_code=400, detail="from_date는 to_date보다 이후일 수 없습니다.")
    email_v = (email or "").strip()
    if ("@" not in email_v) or (len(email_v) < 5):
        raise HTTPException(status_code=400, detail="유효한 이메일 주소를 입력해주세요.")

    script_path = "/Users/daehwankim/cube_rag/summary/analyze_data.py"
    timeout_secs = int(os.getenv("SUMMARY_SCRIPT_TIMEOUT_SECS", "600"))
    cmd_list = [
        "python3",
        "-u",
        script_path,
        "--from",
        from_date,
        "--to",
        to_date,
        "--email",
        email_v,
    ]

    def event_iterator():
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        except Exception as e:
            yield {"event": "error", "data": f"스크립트 시작 실패: {e}"}
            yield {"event": "done", "data": "end"}
            return

        start_ts = time.monotonic()
        try:
            assert proc.stdout is not None
            # 시작 알림 한 줄
            yield {"event": "log", "data": "요약 스크립트를 시작합니다…"}
            for line in proc.stdout:
                msg = (line or "").rstrip("\n")
                if msg:
                    yield {"event": "log", "data": msg}
                # 타임아웃 체크
                if (time.monotonic() - start_ts) > timeout_secs:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    yield {"event": "error", "data": f"스크립트 실행 타임아웃({timeout_secs}s)"}
                    yield {"event": "done", "data": "end"}
                    return
        finally:
            try:
                rc = proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                rc = -1

        if rc != 0:
            yield {"event": "error", "data": f"스크립트 종료 코드: {rc}"}
        yield {"event": "done", "data": "end"}

    return EventSourceResponse(event_iterator(), media_type="text/event-stream")

@app.get("/ask/stream")
def ask_stream(q: str, cid: Optional[str] = None):
    if EventSourceResponse is None:
        raise HTTPException(status_code=500, detail="SSE dependency not installed")

    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="q is required")

    # Domain selection (same logic as orchestrator)
    history = _get_history(cid)
    analyzer = QueryAnalyzerAgent()
    analysis = analyzer.analyze_query(question)
    try:
        dom_thr = float(os.getenv("DOMAIN_CONFIDENCE_THRESHOLD", "0.6"))
    except Exception:
        dom_thr = 0.6
    domain = getattr(analysis, "domain", "unknown") or "unknown"
    dom_conf = float(getattr(analysis, "domain_confidence", 0.0) or 0.0)

    def _map_collection(d: str) -> Optional[str]:
        mapping = {
            "python": "qa_questions_python",
            "sql": "qa_questions_sql",
            "semiconductor": "qa_questions_semiconductor",
        }
        return mapping.get((d or "").lower())

    # domain memory store (shared with /ask)
    domain_memory = conversation_store.setdefault("__domain_memory__", {})
    last_domain = domain_memory.get(cid or "")
    # pending domain selection store
    pending_map: Dict[str, str] = conversation_store.setdefault("__pending_domain__", {})
    pending_q = pending_map.get(cid or "")

    # 사용자가 직전 턴의 도메인 확인에 응답한 경우 처리
    forced_domain = None
    skip_user_append = False
    if pending_q:
        nd = _normalize_domain_text(question)
        if nd in ("python", "sql", "semiconductor"):
            forced_domain = nd
            # 원 질문으로 교체하여 이어서 처리
            question = pending_q
            skip_user_append = True
            try:
                pending_map.pop(cid or "", None)
            except Exception:
                pass

    is_new_chat = not history or len(history) == 0
    if forced_domain:
        selected_domain = forced_domain
    else:
        if is_new_chat:
            if domain not in ("python", "sql", "semiconductor") or dom_conf < dom_thr:
                clarify_msg = "이 질문은 Python, SQL, 반도체 중 어떤 도메인과 가장 관련이 있나요? (예: '파이썬' 또는 'SQL' 또는 '반도체')"

                def event_iterator_clarify():
                    yield {"event": "token", "data": clarify_msg}
                    # propagate cid if given or allocate new
                    saved_cid = _append_turn(cid, "user", question)
                    _append_turn(saved_cid, "assistant", clarify_msg)
                    # 다음 턴에서 도메인 선택을 받기 위해 원 질문 보관
                    pending_map[saved_cid] = question
                    yield {"event": "cid", "data": saved_cid}
                    yield {"event": "done", "data": "end"}

                return EventSourceResponse(event_iterator_clarify(), media_type="text/event-stream")
            selected_domain = domain
        else:
            if domain in ("python", "sql", "semiconductor") and dom_conf >= dom_thr:
                selected_domain = domain
            elif last_domain in ("python", "sql", "semiconductor"):
                selected_domain = last_domain
            else:
                clarify_msg = "이 질문은 Python, SQL, 반도체 중 어떤 도메인과 가장 관련이 있나요?"

                def event_iterator_clarify2():
                    yield {"event": "token", "data": clarify_msg}
                    saved_cid = _append_turn(cid, "user", question)
                    _append_turn(saved_cid, "assistant", clarify_msg)
                    # 다음 턴에서 도메인 선택을 받기 위해 원 질문 보관
                    pending_map[saved_cid] = question
                    yield {"event": "cid", "data": saved_cid}
                    yield {"event": "done", "data": "end"}

                return EventSourceResponse(event_iterator_clarify2(), media_type="text/event-stream")

    collection_name = _map_collection(selected_domain)

    # Retrieval (vector DB) with selected collection
    hits = retrieve(question, collection_name=collection_name)
    # 질문 전용 컬렉션 고려: 메타 question/answer로 컨텍스트 재구성
    contexts = []
    sources_vec = []
    for h in hits:
        meta = h.get("metadata") or {}
        q_meta = (meta.get("question") or "").strip()
        # '||' 분리
        tokens = [t.strip() for t in str(meta.get("answers") or "").split("||") if t and t.strip()]
        a_meta = " ".join(tokens)
        content = h.get("content")
        if q_meta or a_meta:
            contexts.append(f"Q: {q_meta}\nA: {a_meta}".strip())
        else:
            contexts.append(content)
        sources_vec.append({"metadata": meta, "score": h.get("score")})

    # 단일 리트리벌 기반 컨센서스 규칙(오케스트레이터와 동일 기본값)
    try:
        t_high = float(os.getenv("DIRECT_MATCH_HIGH", "0.5"))
    except Exception:
        t_high = 0.5
    try:
        t_mid = float(os.getenv("DIRECT_MATCH_SCORE_THRESHOLD", "0.45"))
    except Exception:
        t_mid = 0.45
    try:
        top_k_cons = int(os.getenv("DIRECT_MATCH_TOPK", "3"))
    except Exception:
        top_k_cons = 3
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
        preface = f"요청하신 질문은 내부 Q&A 데이터(도메인: {selected_domain})에서 근거를 찾았습니다. 아래 출처를 참고하세요."
        effective_sources = normalize_sources(sources_vec, None, only="internal")
        combined_contexts = contexts
    else:
        if use_search:
            preface = (
                f"내부 Q&A(도메인: {selected_domain})에 직접 일치하는 근거가 없어, LLM 일반 지식과 웹 검색 결과를 근거로 답변합니다.\n"
                "가능한 경우 출처(URL/메타)를 함께 표기합니다."
            )
            # 내부 미매칭인 경우 웹 출처만 표기
            effective_sources = normalize_sources(None, web_results, only="web")
            combined_contexts = contexts + web_contexts
        else:
            preface = f"내부 Q&A(도메인: {selected_domain})에 해당 내용이 없어, 웹 검색 없이 LLM 일반 지식만으로 답변합니다."
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
            if skip_user_append:
                # 도메인 선택 응답은 사용자 질문으로 기록하지 않음
                saved_cid = cid
                if not saved_cid:
                    # cid가 없으면 어시스턴트만 기록하면서 cid 생성
                    saved_cid = _append_turn(None, "assistant", "(streamed)")
                else:
                    _append_turn(saved_cid, "assistant", "(streamed)")
            else:
                saved_cid = _append_turn(cid, "user", question)
                # 스트리밍 시 전체 답변은 클라이언트가 조합하므로, 여기서는 간단 표기
                _append_turn(saved_cid, "assistant", "(streamed)")
            # 도메인 메모리 업데이트
            if selected_domain in ("python", "sql", "semiconductor"):
                domain_memory[saved_cid] = selected_domain
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


