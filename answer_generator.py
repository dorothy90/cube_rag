"""
Answer Generator
- Retrieval 결과 컨텍스트를 근거로 한국어 답변을 생성합니다.

Env keys:
- OPENAI_API_KEY (필수)
- OPENAI_MODEL_NAME (기본: gpt-4o-mini)
"""

import os
from typing import List, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def _get_model_name() -> str:
    name = os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
    return name


SYSTEM_PROMPT = """
당신은 한국어 기술 Q&A 어시스턴트입니다.
- 제공된 컨텍스트에 근거하여 간결하고 정확한 답변을 작성하세요.
- 컨텍스트에 근거가 부족하면, 추측 대신 "주어진 컨텍스트만으로는 충분하지 않습니다"라고 명시하세요.
- 단계/명령/예시가 필요하면 코드블록을 사용하세요.
""".strip()


USER_PROMPT_TEMPLATE = """
질문: {question}

컨텍스트(최대 {ctx_count}개):
{contexts}

지침:
- 한국어로 답변하세요.
- 과도한 추측을 피하고, 가능하면 컨텍스트의 표현을 인용하세요.
""".strip()


def generate_answer(question: str, contexts: List[str]) -> Dict:
    model = ChatOpenAI(model=_get_model_name(), temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])

    joined_contexts = "\n\n".join([c.strip() for c in contexts if c and c.strip()])
    formatted = prompt.format_messages(
        question=question.strip(),
        ctx_count=len(contexts),
        contexts=joined_contexts,
    )

    resp = model.invoke(formatted)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {"answer": answer}


__all__ = ["generate_answer"]


