"""
LLM 호출 모듈 — OpenAI API 래퍼 + 캐싱
"""

import json
import hashlib
from pathlib import Path

from openai import OpenAI

from .config import OPENAI_API_KEY, LLM_MODEL, CACHE_DIR


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY가 설정되지 않았습니다.\n"
            "1) .env 파일에 OPENAI_API_KEY=sk-... 를 추가하거나\n"
            "2) export OPENAI_API_KEY=sk-... 를 실행하세요."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def llm_call(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    use_cache: bool = True,
) -> str:
    """
    OpenAI LLM을 호출합니다. 동일한 입력에 대해 캐시를 사용합니다.

    Args:
        prompt: 사용자 프롬프트
        system: 시스템 프롬프트
        model: 모델명 (기본: gpt-4o-mini)
        max_tokens: 최대 출력 토큰
        temperature: 0.0 = 결정적, 1.0 = 창의적
        use_cache: True면 동일 입력에 캐시된 결과 반환

    Returns:
        str: LLM 응답 텍스트

    Example:
        >>> answer = llm_call("한국의 수도는?", system="짧게 답하세요")
        >>> print(answer)
    """
    # 캐시 확인
    if use_cache:
        cache_key = hashlib.md5(
            f"{model}:{system}:{prompt}".encode()
        ).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["response"]

    # API 호출
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = resp.choices[0].message.content

    # 캐시 저장
    if use_cache:
        cache_file.write_text(json.dumps({"response": result}))

    return result


def llm_call_json(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    max_tokens: int = 2000,
) -> dict | list:
    """
    LLM을 호출하고 JSON으로 파싱합니다.

    LLM 응답에서 ```json 블록을 자동 추출합니다.

    Returns:
        dict | list: 파싱된 JSON

    Raises:
        json.JSONDecodeError: JSON 파싱 실패 시
    """
    raw = llm_call(prompt, system, model, max_tokens)
    text = raw.strip()

    # ```json ... ``` 블록 추출
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:  # 홀수 인덱스 = 코드 블록 내부
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # 직접 파싱 시도
    return json.loads(text)
