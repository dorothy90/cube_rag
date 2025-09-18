import json
import os
import sys

# Ensure project root on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from agent_orchestrator import AgentOrchestrator


def expect(label: str, cond: bool):
    print(f"[{label}] {'PASS' if cond else 'FAIL'}")


def main():
    o = AgentOrchestrator()

    # T1: SQL 분류
    r1 = o.run("INNER JOIN과 LEFT JOIN 차이는?", history=[], last_domain=None)
    print("T1:", r1.get("stage"), r1.get("selected_domain"))
    expect("T1 selected_domain==sql", r1.get("selected_domain") == "sql")

    # T2: 후속 모호 질문 → 직전 도메인 폴백(sql)
    hist = [{"role": "user", "content": "INNER JOIN과 LEFT JOIN 차이는?"}, {"role": "assistant", "content": "..."}]
    r2 = o.run("성능을 올리려면?", history=hist, last_domain="sql")
    print("T2:", r2.get("stage"), r2.get("selected_domain"))
    expect("T2 selected_domain==sql", r2.get("selected_domain") == "sql")

    # T3: Python 분류
    r3 = o.run("파이썬에서 리스트와 튜플 차이는?", history=[], last_domain=None)
    print("T3:", r3.get("stage"), r3.get("selected_domain"))
    expect("T3 selected_domain==python", r3.get("selected_domain") == "python")


if __name__ == "__main__":
    main()


