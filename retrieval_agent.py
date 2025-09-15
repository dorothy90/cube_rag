"""
Retrieval Agent
- Chroma 기반 유사도 검색을 수행하여 상위 컨텍스트를 반환합니다.
- 기본 파라미터는 .env 로 오버라이드 가능합니다.

Env keys (optional):
- OPENAI_API_KEY (필수)
- EMBEDDING_MODEL_NAME (기본: text-embedding-3-small)
- CHROMA_PERSIST_DIR (기본: ./chroma_db)
- CHROMA_COLLECTION (기본: qa_pairs)
- RETRIEVER_TOP_K (기본: 5)
- RETRIEVER_SCORE_THRESHOLD (기본: 0.2)
- RETRIEVER_TIMEOUT_SEC (기본: 2.5)
- RETRIEVER_SEARCH_TYPE (기본: similarity_score_threshold)
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from langchain_chroma import Chroma

# 내부 유틸 재사용 (중복 방지)
from chroma_setup import (
    get_embeddings,
    get_vectorstore,
    get_default_paths,
)


load_dotenv()
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _read_config() -> Dict:
    defaults = get_default_paths()
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", defaults["persist_dir"])
    collection = os.getenv("CHROMA_COLLECTION", "qa_pairs")
    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    threshold = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.2"))
    timeout_sec = float(os.getenv("RETRIEVER_TIMEOUT_SEC", "2.5"))
    search_type = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity_score_threshold")
    return {
        "persist_dir": persist_dir,
        "collection": collection,
        "top_k": top_k,
        "threshold": threshold,
        "timeout_sec": timeout_sec,
        "search_type": search_type,
    }


def _get_vectorstore_by_config(cfg: Dict) -> Chroma:
    embedding = get_embeddings()
    vectorstore = get_vectorstore(
        persist_dir=cfg["persist_dir"],
        embedding=embedding,
        collection_name=cfg["collection"],
    )
    return vectorstore


def get_retriever(
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    search_type: Optional[str] = None,
) -> Tuple[Chroma, Dict]:
    """Chroma VectorStore와 설정값을 반환합니다. as_retriever 사용은 선택 사항입니다.

    반환: (vectorstore, config)
    """
    cfg = _read_config()
    if top_k is not None:
        cfg["top_k"] = top_k
    if score_threshold is not None:
        cfg["threshold"] = score_threshold
    if search_type is not None:
        cfg["search_type"] = search_type

    vs = _get_vectorstore_by_config(cfg)
    return vs, cfg


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    search_type: Optional[str] = None,
) -> List[Dict]:
    """질의어로 유사 문서를 검색하여 [{content, metadata, score}] 리스트를 반환합니다.

    - score는 0~1 범위를 선호하며, 백엔드 지원에 따라 근사값일 수 있습니다.
    - 빈 결과는 빈 리스트로 반환합니다.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query는 비어있지 않은 문자열이어야 합니다.")

    vs, cfg = get_retriever(top_k=top_k, score_threshold=score_threshold, search_type=search_type)

    effective_k = cfg["top_k"]
    threshold = cfg["threshold"]
    timeout_sec = cfg["timeout_sec"]

    start = time.time()
    results: List[Dict] = []

    try:
        # 1) 거리 기반 점수 우선: 안정적으로 수치가 제공됨
        try:
            docs_with_scores = vs.similarity_search_with_score(query, k=effective_k)
            for doc, distance in docs_with_scores:
                score = None
                try:
                    d = float(distance)
                    # 일반 거리값(>=0)에 대해 안정적인 유사도 근사: 1/(1+d) ∈ (0,1]
                    score = 1.0 / (1.0 + max(0.0, d))
                except Exception:
                    score = None

                if (threshold is None) or (score is None) or (score >= threshold):
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    })
        except Exception:
            # 2) 대체: relevance score API (버전에 따라 범위가 다를 수 있음)
            try:
                docs_with_scores = vs.similarity_search_with_relevance_scores(query, k=effective_k)
                for doc, rel in docs_with_scores:
                    score = None
                    try:
                        s = float(rel)
                        # 만약 0..1 범위가 아니면 휴리스틱 변환 없이 임계값 체크 생략
                        if 0.0 <= s <= 1.0:
                            score = s
                    except Exception:
                        score = None

                    # score가 None이면 임계값 체크를 생략하고 포함시켜 관측 가능하게 함
                    if (score is None) or (threshold is None) or (score >= threshold):
                        results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": score,
                        })
            except Exception:
                # 3) 최후: 점수 없이 문서만 반환
                docs = vs.similarity_search(query, k=effective_k)
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None,
                    })

        elapsed = time.time() - start
        if elapsed > timeout_sec:
            logger.warning(f"retrieval timeout: {elapsed:.3f}s > {timeout_sec:.3f}s")
        else:
            logger.info(f"retrieval ok: {len(results)} hits in {elapsed:.3f}s (k={effective_k}, thr={threshold})")

        return results

    except Exception as e:
        logger.exception(f"retrieval error: {e}")
        return []


__all__ = [
    "get_retriever",
    "retrieve",
]


