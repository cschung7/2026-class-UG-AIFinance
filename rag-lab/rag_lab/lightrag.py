"""
Method 2: LightRAG
==================
그래프 기반 검색 + 벡터 검색 결합

    문서 → 엔티티 추출 → 관계 그래프 구축
                        ↕
    쿼리 → 그래프 탐색 + 벡터 검색 → 결합 → LLM 생성

장점: 엔티티 간 관계를 활용, 크로스 문서 질문에 강함
단점: 엔티티 추출에 LLM 비용, 그래프 품질에 의존
"""

import time
import json
from dataclasses import dataclass

from .embeddings import EmbeddingEngine
from .llm import llm_call, llm_call_json
from .utils import chunk_text
from .config import TOP_K


@dataclass
class LightRAGResult:
    """LightRAG 검색+생성 결과"""
    answer: str
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_retrieved: int
    entities_count: int
    relations_count: int
    graph_context_length: int
    sources: list[str]
    context_length: int


class LightRAG:
    """
    LightRAG: Entity Graph + Vector Retrieval

    동작 원리:
    ┌─────────┐    ┌──────────────┐    ┌───────────┐
    │  문서들  │───►│ LLM 엔티티   │───►│ 지식 그래프 │
    └────┬────┘    │  추출        │    └─────┬─────┘
         │         └──────────────┘          │
         │                                   │
    ┌────▼────┐                        ┌─────▼─────┐
    │  청킹   │───► FAISS 인덱스        │ 그래프 탐색 │
    └─────────┘         │              └─────┬─────┘
                        │                    │
                  ┌─────▼─────┐        ┌─────▼─────┐
        질문 ───► │ 벡터 검색  │        │ 그래프 검색 │
                  └─────┬─────┘        └─────┬─────┘
                        │                    │
                        └────────┬───────────┘
                           ┌─────▼─────┐
                           │ LLM 생성  │ ───► 답변
                           └───────────┘

    사용법:
        >>> engine = EmbeddingEngine()
        >>> lrag = LightRAG(papers, engine)
        >>> result = lrag.query("엔티티 간 관계는?")
        >>> print(lrag.get_graph_stats())
    """

    def __init__(
        self,
        papers: dict[str, str],
        engine: EmbeddingEngine,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        self.name = "LightRAG"
        self.engine = engine
        self.entities: dict[str, dict] = {}
        self.relations: list[dict] = []
        self.chunks: list[str] = []
        self.chunk_sources: list[str] = []
        self.all_texts: list[str] = []
        self.index = None
        self._build(papers, chunk_size, chunk_overlap)

    def _extract_entities(self, paper_name: str, text: str):
        """LLM으로 논문에서 엔티티와 관계를 추출합니다."""
        sample = text[:8000]
        try:
            data = llm_call_json(
                prompt=(
                    f"Extract key entities and relationships from this "
                    f"economics paper.\n\nText:\n{sample}\n\n"
                    f"Return JSON:\n"
                    f'{{"entities": [{{"name": "...", "type": '
                    f'"person|concept|method|dataset|institution", '
                    f'"description": "one line"}}], '
                    f'"relations": [{{"source": "...", "target": "...", '
                    f'"relation": "studies|uses|finds|proposes|extends"}}]}}'
                ),
                system="Extract structured knowledge. Return ONLY valid JSON.",
            )
            for ent in data.get("entities", []):
                name = ent["name"]
                if name not in self.entities:
                    self.entities[name] = {
                        "type": ent.get("type", "concept"),
                        "description": ent.get("description", ""),
                        "sources": [paper_name],
                    }
                else:
                    if paper_name not in self.entities[name]["sources"]:
                        self.entities[name]["sources"].append(paper_name)

            for rel in data.get("relations", []):
                self.relations.append({
                    "source": rel["source"],
                    "target": rel["target"],
                    "relation": rel.get("relation", "related_to"),
                    "paper": paper_name,
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠ 엔티티 추출 실패 ({paper_name}): {e}")

    def _build(self, papers: dict[str, str], chunk_size: int, overlap: int):
        """그래프 구축 + 벡터 인덱스 구축"""
        print(f"[{self.name}] 엔티티 & 관계 추출 중...")
        for name, text in papers.items():
            print(f"  {name} 처리 중...")
            self._extract_entities(name, text)
            chunks = chunk_text(text, chunk_size, overlap)
            self.chunks.extend(chunks)
            self.chunk_sources.extend([name] * len(chunks))
        print(
            f"  → 엔티티 {len(self.entities)}개, "
            f"관계 {len(self.relations)}개"
        )

        # 청크 + 엔티티 설명을 함께 인덱싱
        print(f"[{self.name}] 벡터 인덱스 구축 중...")
        self.all_texts = self.chunks.copy()
        for name, info in self.entities.items():
            self.all_texts.append(
                f"{name} ({info['type']}): {info['description']}"
            )
        embeddings = self.engine.encode(self.all_texts)
        self.index = self.engine.build_index(embeddings)
        print(f"  → FAISS 인덱스 구축 완료 ({self.index.ntotal} 벡터)")

    def _graph_traverse(self, question: str) -> str:
        """질문과 관련된 엔티티를 그래프에서 탐색합니다."""
        q_lower = question.lower()
        relevant = []
        for name in self.entities:
            if name.lower() in q_lower or any(
                w in name.lower()
                for w in q_lower.split() if len(w) > 3
            ):
                relevant.append(name)

        # 1-hop 확장: 관련 엔티티의 이웃도 포함
        expanded = set(relevant)
        for rel in self.relations:
            if rel["source"] in relevant:
                expanded.add(rel["target"])
            if rel["target"] in relevant:
                expanded.add(rel["source"])

        # 그래프 컨텍스트 구성
        parts = []
        for name in expanded:
            if name in self.entities:
                info = self.entities[name]
                parts.append(
                    f"Entity: {name} ({info['type']}): "
                    f"{info['description']}"
                )
        for rel in self.relations:
            if rel["source"] in expanded or rel["target"] in expanded:
                parts.append(
                    f"Relation: {rel['source']} "
                    f"--{rel['relation']}--> {rel['target']} "
                    f"[{rel['paper']}]"
                )
        return "\n".join(parts)

    def query(self, question: str, top_k: int = TOP_K) -> LightRAGResult:
        """
        그래프 탐색 + 벡터 검색을 결합하여 답변을 생성합니다.
        """
        t0 = time.time()

        # 벡터 검색
        scores, indices = self.engine.search(self.index, question, top_k)
        vector_parts = []
        sources = set()
        for score, idx in zip(scores, indices):
            text = self.all_texts[idx]
            if idx < len(self.chunks):
                src = self.chunk_sources[idx]
                sources.add(src)
            else:
                src = "entity_graph"
            vector_parts.append(f"[{src}, {score:.3f}]\n{text}")

        # 그래프 검색
        graph_ctx = self._graph_traverse(question)
        retrieval_time = time.time() - t0

        combined = (
            "=== Knowledge Graph ===\n" + graph_ctx +
            "\n\n=== Retrieved Passages ===\n" +
            "\n\n---\n\n".join(vector_parts)
        )

        t1 = time.time()
        answer = llm_call(
            prompt=(
                f"Using both the knowledge graph and retrieved passages, "
                f"answer:\n\nQuestion: {question}\n\n"
                f"{combined}\n\n"
                f"Synthesize from both sources. Cite specific papers."
            ),
            system=(
                "You are a research assistant with access to a knowledge "
                "graph and document passages."
            ),
        )
        generation_time = time.time() - t1

        return LightRAGResult(
            answer=answer,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=retrieval_time + generation_time,
            chunks_retrieved=top_k,
            entities_count=len(self.entities),
            relations_count=len(self.relations),
            graph_context_length=len(graph_ctx),
            sources=list(sources),
            context_length=len(combined),
        )

    def get_graph_stats(self) -> dict:
        """그래프 통계를 반환합니다 (학습/디버깅용)."""
        entity_types = {}
        for info in self.entities.values():
            t = info["type"]
            entity_types[t] = entity_types.get(t, 0) + 1

        relation_types = {}
        for rel in self.relations:
            r = rel["relation"]
            relation_types[r] = relation_types.get(r, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "papers_covered": list({
                s for info in self.entities.values()
                for s in info["sources"]
            }),
        }

    def get_entity_neighborhood(self, entity_name: str) -> dict:
        """특정 엔티티의 이웃 관계를 반환합니다 (학습용)."""
        neighbors = {"outgoing": [], "incoming": []}
        for rel in self.relations:
            if rel["source"] == entity_name:
                neighbors["outgoing"].append(rel)
            if rel["target"] == entity_name:
                neighbors["incoming"].append(rel)
        return {
            "entity": self.entities.get(entity_name, {}),
            "neighbors": neighbors,
        }
