"""
임베딩 엔진 — 텍스트를 벡터로 변환
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


class EmbeddingEngine:
    """
    문장 임베딩 생성기 + FAISS 인덱스 빌더

    사용법:
        >>> engine = EmbeddingEngine()
        >>> vectors = engine.encode(["Hello world", "안녕하세요"])
        >>> print(vectors.shape)  # (2, 384)

    내부 모델: all-MiniLM-L6-v2 (384차원, 영어 최적화)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"  임베딩 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  → 차원 수: {self.dim}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 리스트를 정규화된 벡터로 변환합니다.

        Args:
            texts: 인코딩할 텍스트 리스트

        Returns:
            np.ndarray: (N, dim) 크기의 float32 벡터 배열
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        ).astype("float32")

    def build_index(self, vectors: np.ndarray) -> faiss.IndexFlatIP:
        """
        FAISS Inner-Product (코사인 유사도) 인덱스를 생성합니다.

        정규화된 벡터에서 Inner Product = Cosine Similarity 입니다.

        Args:
            vectors: (N, dim) 크기의 정규화된 벡터

        Returns:
            faiss.IndexFlatIP: 검색 가능한 FAISS 인덱스
        """
        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors)
        return index

    def search(
        self,
        index: faiss.IndexFlatIP,
        query: str,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        쿼리와 가장 유사한 벡터 top_k개를 검색합니다.

        Returns:
            (scores, indices): 유사도 점수와 인덱스
        """
        q_vec = self.encode([query])
        scores, indices = index.search(q_vec, top_k)
        return scores[0], indices[0]
