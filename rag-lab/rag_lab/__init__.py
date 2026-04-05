"""
RAG Lab — 세 가지 RAG 아키텍처 비교 실습
========================================
Standard RAG · LightRAG · Karpathy LLM-Wiki

서울대학교 경제학과 수업용 실습 패키지
"""

__version__ = "1.0.0"

from .embeddings import EmbeddingEngine
from .llm import llm_call, llm_call_json
from .utils import load_papers, chunk_text
from .standard_rag import StandardRAG
from .lightrag import LightRAG
from .karpathy_wiki import KarpathyWiki
from .evaluate import Evaluator
