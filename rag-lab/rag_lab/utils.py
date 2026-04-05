"""
유틸리티 함수 — 논문 로딩, 텍스트 청킹
"""

from pathlib import Path
from .config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_papers(papers_dir: Path = PAPERS_DIR) -> dict[str, str]:
    """
    papers/ 디렉토리에서 모든 .txt 파일을 로드합니다.

    Returns:
        dict: {파일이름: 텍스트내용}

    Example:
        >>> papers = load_papers()
        >>> print(f"논문 {len(papers)}편 로드됨")
    """
    papers = {}
    for ext in ["*.txt", "*.md"]:
        for f in sorted(papers_dir.glob(ext)):
            papers[f.stem] = f.read_text(encoding="utf-8", errors="replace")
    if not papers:
        raise FileNotFoundError(
            f"papers/ 디렉토리에 논문이 없습니다.\n"
            f"경로: {papers_dir}\n"
            f"먼저 `python scripts/download_papers.py` 를 실행하세요."
        )
    return papers


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_length: int = 50,
) -> list[str]:
    """
    텍스트를 겹치는 청크(chunk)로 분할합니다.

    ┌─────────────────────┐
    │     Chunk 1         │
    │              ┌──────┼──────────────┐
    │   overlap ──►│      │   Chunk 2    │
    └──────────────┼──────┘              │
                   │              ┌──────┼──────────────┐
                   │   overlap ──►│      │   Chunk 3    │
                   └──────────────┼──────┘              │
                                  └─────────────────────┘

    Args:
        text: 원본 텍스트
        chunk_size: 각 청크의 최대 문자 수
        overlap: 인접 청크 간 겹치는 문자 수
        min_length: 이보다 짧은 청크는 버림

    Returns:
        list[str]: 청크 리스트
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= min_length:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def count_tokens_approx(text: str) -> int:
    """대략적인 토큰 수 추정 (영어 기준 1토큰 ≈ 4글자)"""
    return len(text) // 4
