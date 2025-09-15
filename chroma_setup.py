"""
Minimal Chroma setup: index extracted QA JSON -> vector DB, and simple query.
- Dependencies: langchain-openai, langchain-chroma, chromadb, python-dotenv
- Env: OPENAI_API_KEY
"""

import os
import json
import argparse
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def get_default_paths() -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "extracted_qa_pairs.json")
    persist_dir = os.path.join(base_dir, "chroma_db")
    return {"data_path": data_path, "persist_dir": persist_dir}


def load_qa_json(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"QA JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("QA JSON must be a list of objects")
    return data


def build_text_and_metadata(qa_item: Dict) -> (str, Dict):
    question = str(qa_item.get("question", "")).strip()
    answer = str(qa_item.get("answer", "")).strip()
    q_author = qa_item.get("question_author") or qa_item.get("q_author")
    a_author = qa_item.get("answer_author") or qa_item.get("a_author")
    timestamp = qa_item.get("timestamp")

    text = f"Q: {question}\nA: {answer}".strip()
    metadata = {
        "question": question,
        "answer": answer,
        "question_author": q_author,
        "answer_author": a_author,
        "timestamp": timestamp,
    }
    return text, metadata


def get_embeddings(model: Optional[str] = None) -> OpenAIEmbeddings:
    # Force a safe OpenAI embedding model unless explicitly provided and valid
    env_model = os.getenv("EMBEDDING_MODEL_NAME")
    chosen = model or (env_model if env_model and env_model.startswith("text-embedding") else "text-embedding-3-small")
    return OpenAIEmbeddings(model=chosen)


def get_vectorstore(persist_dir: str, embedding: OpenAIEmbeddings, collection_name: str = "qa_pairs") -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
    )


def index_qa(json_path: str, persist_dir: str, batch: int = 100) -> int:
    qa_list = load_qa_json(json_path)
    embedding = get_embeddings()
    vectorstore = get_vectorstore(persist_dir, embedding)

    texts: List[str] = []
    metadatas: List[Dict] = []

    for item in qa_list:
        text, meta = build_text_and_metadata(item)
        if text:
            texts.append(text)
            metadatas.append(meta)

    total = len(texts)
    if total == 0:
        print("⚠️ No QA pairs to index.")
        return 0

    # Simple batching to avoid large single-call payloads
    added = 0
    for i in range(0, total, batch):
        j = min(i + batch, total)
        vectorstore.add_texts(texts=texts[i:j], metadatas=metadatas[i:j])
        added += (j - i)
        print(f"Indexed {j}/{total}")

    # Persistence is handled automatically via persist_directory
    return added


def query(persist_dir: str, q: str, k: int = 3) -> List[Dict]:
    embedding = get_embeddings()
    vectorstore = get_vectorstore(persist_dir, embedding)
    docs = vectorstore.similarity_search(q, k=k)
    results = []
    for d in docs:
        results.append({
            "content": d.page_content,
            "metadata": d.metadata,
            "score": getattr(d, "distance", None),  # may be None depending on backend
        })
    return results


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. Please configure your .env.")

    defaults = get_default_paths()

    parser = argparse.ArgumentParser(description="Chroma minimal setup")
    parser.add_argument("--mode", choices=["index", "query"], default="index")
    parser.add_argument("--json", default=defaults["data_path"], help="Path to QA JSON")
    parser.add_argument("--persist", default=defaults["persist_dir"], help="Chroma persist dir")
    parser.add_argument("--q", default="Django 기본값 설정 방법?", help="Query text for query mode")
    parser.add_argument("--k", type=int, default=3, help="top_k for retrieval")
    args = parser.parse_args()

    if args.mode == "index":
        count = index_qa(args.json, args.persist)
        print(f"\n✅ Indexed {count} QA items into Chroma at: {args.persist}")
    else:
        results = query(args.persist, args.q, args.k)
        print("\n🔎 Query Results:")
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            print(f"\n[{i}] {meta.get('timestamp', '')}")
            print(r["content"][:400])


if __name__ == "__main__":
    main()
