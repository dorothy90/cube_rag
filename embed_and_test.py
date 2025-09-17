"""
Embed and Test Script
- .env를 로드한 뒤, qa_chunks 컬렉션으로 인덱싱하고 질의 테스트를 수행합니다.

사용 예:
  python3 embed_and_test.py --reset
  python3 embed_and_test.py --question "정규화와 비정규화의 차이점을 알려주세요."
"""

import os
import json
import argparse
import shutil
from typing import Optional, List, Dict

from dotenv import load_dotenv

from chroma_setup import (
    index_chunks,
    query,
)


def _default_paths() -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        "json": os.path.join(base_dir, "data", "chunked_qa_pairs.json"),
        "persist": os.path.join(base_dir, "chroma_db"),
    }


def run_embedding_and_test(
    json_path: str,
    persist_dir: str,
    collection: str,
    question: str,
    top_k: int = 3,
    reset: bool = False,
) -> Dict:
    load_dotenv()

    if reset and os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    added = index_chunks(
        json_path=json_path,
        persist_dir=persist_dir,
        collection_name=collection,
        batch=100,
        model=None,
    )

    results: List[Dict] = query(
        persist_dir=persist_dir,
        q=question,
        k=top_k,
        collection_name=collection,
        model=None,
    )

    return {
        "indexed": added,
        "results": results,
    }


def main() -> None:
    defaults = _default_paths()

    parser = argparse.ArgumentParser(description="Embed and test retrieval against qa_chunks collection")
    parser.add_argument("--json", default=defaults["json"], help="Path to chunked QA JSON")
    parser.add_argument("--persist", default=defaults["persist"], help="Chroma persist directory")
    parser.add_argument("--collection", default="qa_chunks", help="Chroma collection name")
    parser.add_argument("--question", default="react 상태관리 방식 알려줘?", help="Question to test")
    parser.add_argument("--k", type=int, default=3, help="Top K for query")
    parser.add_argument("--reset", action="store_true", help="Reset (delete) the persist directory before indexing")
    args = parser.parse_args()

    out = run_embedding_and_test(
        json_path=args.json,
        persist_dir=args.persist,
        collection=args.collection,
        question=args.question,
        top_k=args.k,
        reset=args.reset,
    )

    indexed = out.get("indexed", 0)
    results: List[Dict] = out.get("results", [])

    print(f"\n✅ Indexed: {indexed}")
    print("\n🔎 Query Results:")
    if not results:
        print("(no results)")
    else:
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            ts = meta.get("timestamp", "")
            content = (r.get("content") or "").strip()
            print(f"\n[{i}] {ts}")
            print(content[:400])


if __name__ == "__main__":
    main()


