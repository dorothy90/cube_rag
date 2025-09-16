"""
간단한 웹 검색 유틸리티.

지원 공급자:
- Tavily(권장, ENV: TAVILY_API_KEY 필요)

ENV 제어:
- USE_WEB_SEARCH=true | false (기본 false)
- WEB_SEARCH_PROVIDER=tavily (명시 시 우선)
- TAVILY_API_KEY(필요)
"""

import os
from typing import List, Dict, Optional

try:
    from tavily import TavilyClient  # type: ignore
except Exception:  # pragma: no cover
    TavilyClient = None


# DuckDuckGo 지원 제거됨


def _search_tavily(query: str, max_results: int) -> List[Dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or TavilyClient is None:
        return []
    try:
        client = TavilyClient(api_key=api_key)  # type: ignore
        resp = client.search(query=query, max_results=max_results)  # type: ignore
        items = resp.get("results") if isinstance(resp, dict) else resp
        results: List[Dict] = []
        for it in (items or []):
            results.append({
                "title": it.get("title"),
                "url": it.get("url"),
                "snippet": it.get("content") or it.get("snippet"),
            })
        return results
    except Exception:
        return []


def search_web(query: str, max_results: int = 5, provider: Optional[str] = None) -> List[Dict]:
    if not isinstance(query, str) or not query.strip():
        return []

    # 전역 토글 확인
    use_search = os.getenv("USE_WEB_SEARCH", "false").lower() in ("1", "true", "yes", "on")
    if not use_search:
        return []

    # 공급자 선택 우선순위: 명시된 provider > ENV(WEB_SEARCH_PROVIDER) > tavily
    provider = (provider or os.getenv("WEB_SEARCH_PROVIDER", "")).strip().lower() or "tavily"

    if provider != "tavily":
        provider = "tavily"

    # Tavily만 지원. 키 없으면 빈 결과
    return _search_tavily(query, max_results)


__all__ = ["search_web"]


