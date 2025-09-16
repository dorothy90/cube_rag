"""
임베딩 생성을 위한 텍스트 청킹 유틸리티 및 CLI.

기능:
- Q&A JSON(`data/extracted_qa_pairs.json`) 로드
- LangChain 텍스트 스플리터로 겹침(overlap)을 두고 텍스트를 청크로 분할
- 메타데이터 보존 및 청크 단위 메타데이터(source_id, position 등) 추가
- 결과를 `data/chunked_qa_pairs.json`에 저장

사용 예시:
  python chunking.py --input data/extracted_qa_pairs.json --output data/chunked_qa_pairs.json \
                     --size 1000 --overlap 200 --splitter recursive
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

from dotenv import load_dotenv

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        SentenceTransformersTokenTextSplitter,
    )
except Exception:  # pragma: no cover
    # langchain_text_splitters 미설치 시를 대비해 최소 기능 스플리터를 사용
    RecursiveCharacterTextSplitter = None
    SentenceTransformersTokenTextSplitter = None


def read_json_list(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")
    return data


def write_json_list(path: str, data: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_source_text(item: Dict) -> Tuple[str, Dict]:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    text = f"Q: {question}\nA: {answer}".strip()
    metadata = {
        "question": question,
        "answer": answer,
        "question_author": item.get("question_author") or item.get("q_author"),
        "answer_author": item.get("answer_author") or item.get("a_author"),
        "timestamp": item.get("timestamp"),
    }
    return text, metadata


def get_text_splitter(
    method: str,
    chunk_size: int,
    chunk_overlap: int,
):
    method = (method or "recursive").lower()

    if method == "recursive":
        if RecursiveCharacterTextSplitter is None:
            # 간단한 대체 스플리터 사용
            return _SimpleSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif method == "sentence-token":
        # 토큰 기반 스플리터(문장 임베딩 토크나이저 필요); 선택적
        if SentenceTransformersTokenTextSplitter is None:
            return _SimpleSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return SentenceTransformersTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        return _SimpleSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class _SimpleSplitter:
    """겹침을 지원하는 단순 문자 기반 스플리터.

    초기 개발 단계에서 외부 의존성 문제를 피하기 위한 대체 구현입니다.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))

    def split_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        if not text:
            return chunks
        n = len(text)
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks


def chunk_items(
    items: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    method: str = "recursive",
) -> List[Dict]:
    splitter = get_text_splitter(method, chunk_size, chunk_overlap)

    chunked: List[Dict] = []
    for idx, item in enumerate(items):
        text, base_meta = build_source_text(item)
        # 가능하면 LangChain 스플리터 인터페이스(split_text)에 맞춤
        if hasattr(splitter, "split_text"):
            parts = splitter.split_text(text)
        else:
            # 일부 스플리터는 split_documents만 제공하므로, 여기서는 단일 텍스트로 처리
            parts = [text]

        for pos, chunk in enumerate(parts):
            meta = dict(base_meta)
            meta.update({
                "source_id": idx,
                "chunk_index": pos,
                "total_chunks": len(parts),
            })
            chunked.append({
                "content": chunk,
                "metadata": meta,
            })

    return chunked


def main():
    load_dotenv()

    defaults_in = os.path.join(os.path.dirname(__file__), "data", "extracted_qa_pairs.json")
    defaults_out = os.path.join(os.path.dirname(__file__), "data", "chunked_qa_pairs.json")

    parser = argparse.ArgumentParser(description="임베딩을 위한 Q&A 텍스트 청킹")
    parser.add_argument("--input", default=defaults_in, help="입력 Q&A JSON 경로")
    parser.add_argument("--output", default=defaults_out, help="출력 청킹 JSON 경로")
    parser.add_argument("--size", type=int, default=1000, help="청크 크기(문자/토큰)")
    parser.add_argument("--overlap", type=int, default=200, help="청크 겹침 크기(문자/토큰)")
    parser.add_argument("--splitter", default="recursive", choices=["recursive", "sentence-token"], help="스플리터 방식")
    args = parser.parse_args()

    items = read_json_list(args.input)
    chunks = chunk_items(items, chunk_size=args.size, chunk_overlap=args.overlap, method=args.splitter)
    write_json_list(args.output, chunks)
    print(f"✅ 총 {len(chunks)}개 청크를 생성하여 저장했습니다 -> {args.output}")


if __name__ == "__main__":
    main()


