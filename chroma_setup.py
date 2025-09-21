"""
Minimal Chroma setup: index extracted QA JSON -> vector DB, and simple query.
- Dependencies: langchain-openai, langchain-chroma, chromadb, python-dotenv
- Env: OPENAI_API_KEY
"""

import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb


def get_default_paths() -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 기본 입력은 chunked 포맷
    data_path = os.path.join(base_dir, "data", "chunked_qa_pairs.json")
    persist_dir = os.path.join(base_dir, "chroma_db")
    return {"data_path": data_path, "persist_dir": persist_dir}


def load_chunks_json(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Chunked JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Chunked JSON must be a list of objects")
    return data


# -----------------------------
# Internal utilities
# -----------------------------


def _sanitize_metadata(meta: Dict) -> Dict:
    """Convert metadata values to Chroma-compatible primitives.

    - Allowed types: str, int, float, bool
    - Drop None values
    - list/tuple -> comma-joined string
    - dict -> JSON string (ensure_ascii=False to preserve Korean)
    - other -> str(value)
    """
    sanitized: Dict[str, object] = {}
    for key, value in (meta or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
        elif isinstance(value, (list, tuple)):
            try:
                sanitized[str(key)] = "||".join(str(v) for v in value)
            except Exception:
                sanitized[str(key)] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, dict):
            sanitized[str(key)] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[str(key)] = str(value)
    return sanitized


def _add_texts_batched(vectorstore: Chroma, texts: List[str], metadatas: List[Dict], batch: int) -> int:
    total = len(texts)
    if total == 0:
        return 0
    added = 0
    for i in range(0, total, batch):
        j = min(i + batch, total)
        # Sanitize metadata to ensure only primitive types are passed to Chroma
        sanitized_metadatas = [_sanitize_metadata(m) for m in metadatas[i:j]]
        try:
            vectorstore.add_texts(texts=texts[i:j], metadatas=sanitized_metadatas)
        except Exception as e:
            msg = str(e).lower()
            if "readonly" in msg or "read-only" in msg:
                raise RuntimeError(
                    "Chroma DB가 읽기 전용으로 열려 있습니다. 다른 프로세스/커널에서 DB를 점유하고 있지 않은지 확인하고, "
                    "필요 시 커널 재시작 후 'chroma_db' 디렉터리를 삭제한 뒤 다시 인덱싱하세요."
                ) from e
            raise
        added += (j - i)
        print(f"Indexed {j}/{total}")
    return added


def _extract_question_from_content(content: str) -> Optional[str]:
    try:
        first_line = (content or "").splitlines()[0]
        if first_line.strip().lower().startswith("q:"):
            return first_line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def get_embeddings(model: Optional[str] = None) -> OpenAIEmbeddings:
    # Force a safe OpenAI embedding model unless explicitly provided and valid
    env_model = os.getenv("EMBEDDING_MODEL_NAME")
    chosen = model or (env_model if env_model and env_model.startswith("text-embedding") else "text-embedding-3-small")
    return OpenAIEmbeddings(model=chosen)


def get_vectorstore(persist_dir: str, embedding: OpenAIEmbeddings, collection_name: str = "qa_pairs") -> Chroma:
    # 절대 경로 고정 및 쓰기 가능 여부 사전 점검
    persist_dir = os.path.abspath(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    try:
        test_path = os.path.join(persist_dir, ".write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception as e:
        raise RuntimeError(f"Persist directory not writable: {persist_dir} ({e})")

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
    )


def index_questions(
    json_path: str,
    persist_dir: str,
    collection_name: str = "qa_questions",
    batch: int = 100,
    model: Optional[str] = None,
) -> int:
    """chunked 포맷에서 question 텍스트만 인덱싱합니다.

    - 입력: chunked JSON (각 item에 content, metadata.question 존재)
    - 보강: metadata.question이 없으면 content 첫 줄의 "Q:"에서 추출 시도
    - 메타데이터는 그대로 저장(리스트는 _sanitize_metadata에서 '||'로 조인)
    """
    items: List[Dict] = load_chunks_json(json_path)

    embedding = get_embeddings(model=model)
    vectorstore = get_vectorstore(persist_dir, embedding, collection_name=collection_name)

    texts: List[str] = []
    metadatas: List[Dict] = []

    for item in items:
        meta = (item or {}).get("metadata") or {}
        question = str(meta.get("question", "")).strip()
        if not question:
            question = _extract_question_from_content(str((item or {}).get("content", ""))) or ""
        if not question:
            continue
        texts.append(question)
        metadatas.append(meta)

    total = len(texts)
    if total == 0:
        print("⚠️ No questions to index.")
        return 0

    return _add_texts_batched(vectorstore, texts, metadatas, batch)


# (위에서 chunked 전용 index_questions로 대체)

def query(persist_dir: str, q: str, k: int = 3, collection_name: str = "qa_questions", model: Optional[str] = None) -> List[Dict]:
    embedding = get_embeddings(model=model)
    vectorstore = get_vectorstore(persist_dir, embedding, collection_name=collection_name)
    docs = vectorstore.similarity_search(q, k=k)
    results = []
    for d in docs:
        results.append({
            "content": d.page_content,
            "metadata": d.metadata,
            "score": getattr(d, "distance", None),
        })
    return results


def delete_collection(persist_dir: str, collection_name: str) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(name=collection_name)
        print(f"🗑️ Deleted collection: {collection_name}")
    except Exception as e:
        print(f"⚠️ Delete failed or collection not found: {collection_name} ({e})")


def main(json_path: Optional[str] = None, persist_dir: Optional[str] = None, collection_name: str = "qa_questions", model: Optional[str] = None, batch: int = 100) -> int:
    """Notebook/Script-friendly entry point to index questions from chunked JSON.

    Returns: number of indexed items.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. Please configure your .env.")

    defaults = get_default_paths()
    json_path = os.path.abspath(json_path or defaults["data_path"])
    persist_dir = os.path.abspath(persist_dir or defaults["persist_dir"])
    count = index_questions(json_path, persist_dir, collection_name=collection_name, batch=batch, model=model)
    print(f"\n✅ Indexed {count} questions into Chroma at: {persist_dir} (collection={collection_name})")
    return count


if __name__ == "__main__":
    main()
