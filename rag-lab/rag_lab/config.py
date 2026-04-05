"""
설정 모듈 — 모든 설정값을 한 곳에서 관리
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

# ── 경로 설정 ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
WIKI_DIR = PROJECT_ROOT / "wiki"
CACHE_DIR = PROJECT_ROOT / ".cache"

# 디렉토리 자동 생성
for d in [PAPERS_DIR, WIKI_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM 설정 ────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ── 임베딩 설정 ──────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # MiniLM-L6-v2 output dimension

# ── 청킹 설정 ────────────────────────────────────────────────────────
CHUNK_SIZE = 2000       # 문자 수 기준 (≈500 토큰)
CHUNK_OVERLAP = 200     # 겹치는 문자 수

# ── 검색 설정 ────────────────────────────────────────────────────────
TOP_K = 5               # 검색 시 반환할 청크 수
