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
    collection = os.getenv("CHROMA_COLLECTION", "qa_questions")
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

    - score는 0..1 범위의 "코사인 유사도"를 우선 사용합니다.
      1) 가능하면 vectorstore의 relevance score(API가 0..1 유사도)를 그대로 사용
      2) 그렇지 않으면 distance에서 cos_sim = clamp01(1 - distance)로 변환
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
        # 1) relevance score가 제공되면 그대로 사용 (0..1 유사도)
        used_relevance = False
        try:
            docs_with_scores = vs.similarity_search_with_relevance_scores(query, k=effective_k)
            for doc, rel in docs_with_scores:
                score = None
                try:
                    s = float(rel)
                    if 0.0 <= s <= 1.0:
                        score = s
                except Exception:
                    score = None

                if (threshold is None) or (score is None) or (score >= threshold):
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    })
            used_relevance = True
        except Exception:
            used_relevance = False

        # 2) fallback: distance → cosine similarity 변환 (cos_sim = clamp01(1 - distance))
        if not used_relevance:
            try:
                docs_with_scores = vs.similarity_search_with_score(query, k=effective_k)
                for doc, distance in docs_with_scores:
                    score = None
                    try:
                        d = float(distance)
                        if d < 0:
                            d = 0.0
                        # distance가 [0,1] 범위인 cosine distance라고 가정하고 1-d로 변환
                        # (범위를 벗어나면 0..1로 클램프)
                        cos_sim = 1.0 - d
                        if cos_sim < 0.0:
                            cos_sim = 0.0
                        if cos_sim > 1.0:
                            cos_sim = 1.0
                        score = cos_sim
                    except Exception:
                        score = None

                    if (threshold is None) or (score is None) or (score >= threshold):
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


