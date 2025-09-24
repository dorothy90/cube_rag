"""
수동 질의/맥락 테스트 스크립트

사용법:
  - 그냥 실행: 기본 샘플 질문(etch tilt)을 분석하고 결과를 콘솔 출력 + JSON 파일 저장
      python3 tests/manual_context_test.py

  - 질문 지정 실행:
      python3 tests/manual_context_test.py --q "etch tilt 에 대해 알려줘" --q "uv 라이브러리 뭔가요" --pretty

  - 파일 입력(줄 단위 질문 목록):
      python3 tests/manual_context_test.py --file questions.txt --pretty

출력:
  - 콘솔 JSON
  - analysis_results/manual_context_<timestamp>.json 저장
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import List, Dict, Any

# 프로젝트 루트 경로를 import 경로에 추가
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from query_analyzer_agent import QueryAnalyzerAgent
from context_handler_agent import ContextHandlerAgent


def _ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("[경고] OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.", file=sys.stderr)


def load_questions(args: argparse.Namespace) -> List[str]:
    if args.q:
        return args.q
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[오류] 파일을 읽을 수 없습니다: {e}", file=sys.stderr)
            sys.exit(1)
    # 기본 샘플
    return ["마진성 불량에 대해 알려주세요"]


def run_tests(questions: List[str]) -> List[Dict[str, Any]]:
    qa_agent = QueryAnalyzerAgent()
    ctx_agent = ContextHandlerAgent()

    results: List[Dict[str, Any]] = []
    for q in questions:
        analysis = qa_agent.analyze_query(q)
        decision = ctx_agent.handle_context_needed(analysis, q)
        results.append(
            {
                "question": q,
                "analysis": asdict(analysis),
                "decision": asdict(decision),
            }
        )
    return results


def save_results(results: List[Dict[str, Any]]) -> str:
    try:
        os.makedirs("analysis_results", exist_ok=True)
    except Exception:
        pass
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.abspath(os.path.join("analysis_results", f"manual_context_{ts}.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    _ensure_api_key()
    p = argparse.ArgumentParser(description="Query/Context 수동 테스트")
    p.add_argument("--q", action="append", help="질문(여러 번 지정 가능)", default=[])
    p.add_argument("--file", help="질문 목록 파일(줄 단위)")
    p.add_argument("--pretty", action="store_true", help="pretty JSON 출력")
    args = p.parse_args()

    questions = load_questions(args)
    results = run_tests(questions)

    # 콘솔 출력
    print(json.dumps(results, ensure_ascii=False, indent=2 if args.pretty else None))

    # 파일 저장
    out_path = save_results(results)
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()


