"""
Method 1: Standard RAG
======================
가장 기본적인 RAG 파이프라인

    문서 → 청킹 → 임베딩 → FAISS 인덱스 → 쿼리 시 검색 → LLM 생성

장점: 빠른 구축, 대규모 확장 가능
단점: 청크가 문맥을 잃음, 매 쿼리마다 지식을 재발견
"""

import time
from dataclasses import dataclass

from .embeddings import EmbeddingEngine
from .llm import llm_call
from .utils import chunk_text
from .config import TOP_K


@dataclass
class RAGResult:
    """RAG 검색+생성 결과"""
    answer: str
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_retrieved: int
    sources: list[str]
    context_length: int


class StandardRAG:
    """
    Standard RAG: Chunk → Embed → Retrieve → Generate

    동작 원리:
    ┌─────────┐    ┌──────────┐    ┌───────────┐
    │  문서들  │───►│  청킹    │───►│  임베딩   │
    └─────────┘    └──────────┘    └─────┬─────┘
                                         │
                                   ┌─────▼─────┐
                                   │ FAISS 인덱스│
                                   └─────┬─────┘
                                         │
    ┌─────────┐    ┌──────────┐    ┌─────▼─────┐
    │  답변   │◄───│  LLM    │◄───│  검색     │◄── 질문
    └─────────┘    └──────────┘    └───────────┘

    사용법:
        >>> engine = EmbeddingEngine()
        >>> rag = StandardRAG(papers, engine)
        >>> result = rag.query("이 논문의 주요 발견은?")
        >>> print(result.answer)
    """

    def __init__(
        self,
        papers: dict[str, str],
        engine: EmbeddingEngine,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        self.name = "Standard RAG"
        self.engine = engine
        self.chunks: list[str] = []
        self.chunk_sources: list[str] = []
        self.index = None
        self._build(papers, chunk_size, chunk_overlap)

    def _build(self, papers: dict[str, str], chunk_size: int, overlap: int):
        """인덱스 구축: 청킹 → 임베딩 → FAISS"""
        print(f"[{self.name}] {len(papers)}편 논문 청킹 중...")
        for name, text in papers.items():
            chunks = chunk_text(text, chunk_size, overlap)
            self.chunks.extend(chunks)
            self.chunk_sources.extend([name] * len(chunks))
        print(f"  → {len(self.chunks)}개 청크 생성")

        print(f"[{self.name}] 임베딩 생성 중...")
        embeddings = self.engine.encode(self.chunks)
        self.index = self.engine.build_index(embeddings)
        print(f"  → FAISS 인덱스 구축 완료 ({self.index.ntotal} 벡터)")

    def query(self, question: str, top_k: int = TOP_K) -> RAGResult:
        """
        질문에 대해 관련 청크를 검색하고 LLM으로 답변을 생성합니다.

        Args:
            question: 사용자 질문
            top_k: 검색할 청크 수

        Returns:
            RAGResult: 답변과 메타데이터
        """
        # 1. 검색 (Retrieve)
        t0 = time.time()
        scores, indices = self.engine.search(self.index, question, top_k)

        context_parts = []
        sources = set()
        for score, idx in zip(scores, indices):
            chunk = self.chunks[idx]
            src = self.chunk_sources[idx]
            sources.add(src)
            context_parts.append(
                f"[출처: {src}, 유사도: {score:.3f}]\n{chunk}"
            )

        context = "\n\n---\n\n".join(context_parts)
        retrieval_time = time.time() - t0

        # 2. 생성 (Generate)
        t1 = time.time()
        answer = llm_call(
            prompt=(
                f"다음 검색된 문단들을 참고하여 질문에 답하세요.\n\n"
                f"질문: {question}\n\n"
                f"검색된 문단:\n{context}\n\n"
                f"구체적인 내용과 출처를 인용하여 답변하세요."
            ),
            system=(
                "You are a research assistant. Answer based ONLY on the "
                "provided passages. Cite sources when possible."
            ),
        )
        generation_time = time.time() - t1

        return RAGResult(
            answer=answer,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=retrieval_time + generation_time,
            chunks_retrieved=top_k,
            sources=list(sources),
            context_length=len(context),
        )

    def retrieve_only(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """
        LLM 생성 없이 검색 결과만 반환합니다 (디버깅/학습용).

        Returns:
            list[dict]: [{chunk, source, score}, ...]
        """
        scores, indices = self.engine.search(self.index, question, top_k)
        results = []
        for score, idx in zip(scores, indices):
            results.append({
                "chunk": self.chunks[idx],
                "source": self.chunk_sources[idx],
                "score": float(score),
            })
        return results
