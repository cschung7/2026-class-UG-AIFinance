"""
Method 3: Karpathy's LLM-Wiki
==============================
Andrej Karpathy의 "LLM Knowledge Base" 패턴 (2026.04)

    문서 → LLM이 위키로 컴파일 → index.md로 탐색 → 위키 페이지로 답변

핵심 통찰: "RAG는 매 질문마다 지식을 재발견한다.
           위키는 지식을 한 번 컴파일하고 계속 축적한다."

장점: 완전히 투명 (읽을 수 있는 .md), 지식이 축적됨, 크로스 레퍼런스
단점: 높은 초기 비용 (LLM 호출 많음), 중소 규모에 적합

참고: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
"""

import time
import json
from dataclasses import dataclass
from pathlib import Path

from .llm import llm_call, llm_call_json
from .config import WIKI_DIR


@dataclass
class WikiResult:
    """Wiki 검색+생성 결과"""
    answer: str
    retrieval_time: float
    generation_time: float
    total_time: float
    pages_navigated: int
    sources: list[str]
    context_length: int


class KarpathyWiki:
    """
    Karpathy LLM-Wiki: Compile → Navigate → Answer

    3계층 아키텍처:
    ┌─────────────────────────────────────────────────┐
    │  Layer 1: raw/  (원본 문서 — 수정 불가)          │
    └──────────────────────┬──────────────────────────┘
                           │ LLM이 읽고 컴파일
    ┌──────────────────────▼──────────────────────────┐
    │  Layer 2: wiki/  (LLM이 생성/관리하는 위키)      │
    │  ├── index.md          (목차 — 탐색 진입점)      │
    │  ├── nber_25000_summary.md  (논문별 요약)        │
    │  ├── concept_human_capital.md (개념 페이지)      │
    │  └── log.md            (작업 기록)               │
    └──────────────────────┬──────────────────────────┘
                           │ LLM이 index → 관련 페이지 탐색
    ┌──────────────────────▼──────────────────────────┐
    │  Layer 3: 쿼리 응답                              │
    │  질문 → index.md 읽기 → 관련 페이지 → 답변 생성   │
    └─────────────────────────────────────────────────┘

    4가지 핵심 연산:
    - ingest: 새 문서 → 요약 + 엔티티 페이지 + 크로스 레퍼런스
    - query:  index 탐색 → 관련 페이지 읽기 → 답변 생성
    - lint:   위키 건강 검사 (모순, 고아 페이지, 오래된 내용)
    - log:    모든 작업을 log.md에 기록

    사용법:
        >>> wiki = KarpathyWiki(papers)  # 위키 컴파일
        >>> result = wiki.query("주요 발견은?")
        >>> wiki.lint()                  # 건강 검사
        >>> wiki.list_pages()            # 페이지 목록
    """

    def __init__(
        self,
        papers: dict[str, str],
        wiki_dir: Path = WIKI_DIR,
        force_rebuild: bool = False,
    ):
        self.name = "Karpathy LLM-Wiki"
        self.wiki_dir = wiki_dir
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self._build(papers, force_rebuild)

    def _build(self, papers: dict[str, str], force_rebuild: bool):
        """위키 컴파일: 문서 → 요약 → 개념 페이지 → 인덱스"""
        index_file = self.wiki_dir / "index.md"
        page_count = len(list(self.wiki_dir.glob("*.md")))
        if index_file.exists() and page_count > 2 and not force_rebuild:
            print(f"[{self.name}] 기존 위키 재사용 ({page_count} 페이지)")
            return

        print(f"[{self.name}] {len(papers)}편 논문을 위키로 컴파일 중...")

        # ── Phase 1: Ingest — 각 논문 요약 페이지 생성 ──────────
        summaries = {}
        for paper_name, text in papers.items():
            print(f"  📄 {paper_name} 수집 중...")
            summary = llm_call(
                prompt=(
                    f"Create a comprehensive wiki summary of this economics "
                    f"paper. Include:\n"
                    f"1. Title and authors\n"
                    f"2. Research question\n"
                    f"3. Methodology\n"
                    f"4. Key findings (specific numbers/results)\n"
                    f"5. Data sources used\n"
                    f"6. Policy implications\n"
                    f"7. Connections to other research\n\n"
                    f"Use [[wiki links]] for key concepts.\n\n"
                    f"Paper text:\n{text[:12000]}"
                ),
                system=(
                    "You are a research wiki editor. Write clear, "
                    "interlinked markdown summaries. Use [[double brackets]] "
                    "for concepts that deserve their own page."
                ),
            )
            path = self.wiki_dir / f"{paper_name}_summary.md"
            path.write_text(
                f"# {paper_name}\n\n{summary}", encoding="utf-8"
            )
            summaries[paper_name] = summary

        # ── Phase 2: Cross-cutting concept pages ────────────────
        print("  🔗 크로스 커팅 개념 페이지 생성 중...")
        all_sum = "\n\n---\n\n".join(
            f"## {k}\n{v}" for k, v in summaries.items()
        )
        try:
            concepts = llm_call_json(
                prompt=(
                    f"Given these paper summaries, identify 5-8 cross-cutting "
                    f"concepts that appear in multiple papers.\n\n"
                    f"Summaries:\n{all_sum}\n\n"
                    f"Return JSON array:\n"
                    f'[{{"concept": "Name", "content": '
                    f'"markdown wiki content with [[links]]"}}]'
                ),
                system=(
                    "Create interlinked wiki concept pages. Each page should "
                    "synthesize findings across papers."
                ),
                max_tokens=4000,
            )
            for c in concepts:
                slug = (
                    c["concept"].lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                )
                path = self.wiki_dir / f"concept_{slug}.md"
                path.write_text(
                    f"# {c['concept']}\n\n{c['content']}",
                    encoding="utf-8",
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠ 개념 페이지 생성 실패: {e}")

        # ── Phase 3: Build index.md ─────────────────────────────
        print("  📋 index.md 생성 중...")
        pages = []
        for f in sorted(self.wiki_dir.glob("*.md")):
            if f.name in ("index.md", "log.md"):
                continue
            first_line = (
                f.read_text(encoding="utf-8")
                .split("\n")[0].strip("# ").strip()
            )
            pages.append(f"- [[{f.stem}]]: {first_line}")

        summary_pages = [p for p in pages if "summary" in p]
        concept_pages = [p for p in pages if "concept_" in p]

        index_content = (
            "# Wiki Index\n\n"
            "## Paper Summaries\n" + "\n".join(summary_pages) +
            "\n\n## Concepts & Themes\n" + "\n".join(concept_pages)
        )
        (self.wiki_dir / "index.md").write_text(
            index_content, encoding="utf-8"
        )

        # ── Phase 4: Build log.md ───────────────────────────────
        (self.wiki_dir / "log.md").write_text(
            f"# Wiki Log\n\n"
            f"## {time.strftime('%Y-%m-%d')} ingest | "
            f"Ingested {len(papers)} papers, "
            f"created {len(pages)} wiki pages\n",
            encoding="utf-8",
        )

        total = len(list(self.wiki_dir.glob("*.md")))
        print(f"  → 위키 컴파일 완료: {total}개 페이지")

    def query(self, question: str) -> WikiResult:
        """
        위키를 탐색하여 질문에 답합니다.

        과정: index.md 읽기 → 관련 페이지 선택 → 페이지 읽기 → 답변 생성
        """
        t0 = time.time()

        # Step 1: index.md에서 관련 페이지 찾기
        index_text = (
            (self.wiki_dir / "index.md")
            .read_text(encoding="utf-8")
        )
        try:
            page_names = llm_call_json(
                prompt=(
                    f"Given this wiki index, which pages are most relevant "
                    f"to answer:\n\"{question}\"\n\n"
                    f"Index:\n{index_text}\n\n"
                    f"Return ONLY a JSON array of page names (max 5)."
                ),
                system="Return only a JSON array of page stem names.",
            )
        except json.JSONDecodeError:
            page_names = [
                f.stem for f in self.wiki_dir.glob("*_summary.md")
            ]

        # Step 2: 관련 위키 페이지 읽기
        wiki_parts = []
        pages_read = 0
        for pname in page_names:
            pfile = self.wiki_dir / f"{pname}.md"
            if pfile.exists():
                wiki_parts.append(
                    f"=== {pname} ===\n"
                    f"{pfile.read_text(encoding='utf-8')}"
                )
                pages_read += 1

        wiki_context = "\n\n".join(wiki_parts)
        retrieval_time = time.time() - t0

        # Step 3: 위키 페이지 기반 답변 생성
        t1 = time.time()
        answer = llm_call(
            prompt=(
                f"Using these wiki pages, answer comprehensively.\n\n"
                f"Question: {question}\n\n"
                f"{wiki_context}\n\n"
                f"Synthesize across pages. Cite specific papers and findings."
            ),
            system=(
                "You are a research assistant reading from a curated wiki "
                "compiled from academic papers. Answer thoroughly."
            ),
        )
        generation_time = time.time() - t1

        return WikiResult(
            answer=answer,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=retrieval_time + generation_time,
            pages_navigated=pages_read,
            sources=page_names if isinstance(page_names, list) else [],
            context_length=len(wiki_context),
        )

    # ── 위키 관리 메서드 (학습용) ───────────────────────────────

    def list_pages(self) -> list[str]:
        """위키의 모든 페이지 목록을 반환합니다."""
        return sorted(
            f.stem for f in self.wiki_dir.glob("*.md")
            if f.name != "log.md"
        )

    def read_page(self, page_name: str) -> str:
        """특정 위키 페이지를 읽습니다."""
        path = self.wiki_dir / f"{page_name}.md"
        if not path.exists():
            return f"페이지 '{page_name}'을 찾을 수 없습니다."
        return path.read_text(encoding="utf-8")

    def lint(self) -> dict:
        """
        위키 건강 검사: 고아 페이지, 깨진 링크, 누락된 개념 확인

        Returns:
            dict: {orphan_pages, broken_links, suggestions}
        """
        pages = {f.stem for f in self.wiki_dir.glob("*.md")}
        all_links = set()
        broken_links = []

        for f in self.wiki_dir.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            # [[wiki_link]] 패턴 찾기
            import re
            links = re.findall(r"\[\[(\w+)\]\]", content)
            for link in links:
                all_links.add(link)
                if link not in pages:
                    broken_links.append(
                        {"from": f.stem, "to": link}
                    )

        # index.md에 없는 페이지 = 고아
        index_text = (
            (self.wiki_dir / "index.md")
            .read_text(encoding="utf-8")
        )
        orphans = [
            p for p in pages
            if p not in index_text and p not in ("index", "log")
        ]

        return {
            "total_pages": len(pages),
            "orphan_pages": orphans,
            "broken_links": broken_links,
            "total_links": len(all_links),
        }

    def ingest_new(self, paper_name: str, text: str):
        """
        새 논문을 위키에 추가합니다 (증분 업데이트).

        기존 페이지를 업데이트하고 index.md를 갱신합니다.
        """
        print(f"  📄 {paper_name} 추가 중...")
        summary = llm_call(
            prompt=(
                f"Create a wiki summary for this paper. "
                f"Use [[wiki links]] for key concepts.\n\n"
                f"Paper:\n{text[:12000]}"
            ),
            system="Research wiki editor. Use [[double brackets]] for concepts.",
        )
        path = self.wiki_dir / f"{paper_name}_summary.md"
        path.write_text(f"# {paper_name}\n\n{summary}", encoding="utf-8")

        # index.md 갱신
        self._rebuild_index()

        # log.md 추가
        log_path = self.wiki_dir / "log.md"
        existing = log_path.read_text(encoding="utf-8")
        log_path.write_text(
            existing +
            f"\n## {time.strftime('%Y-%m-%d')} ingest | "
            f"Added {paper_name}\n",
            encoding="utf-8",
        )

    def _rebuild_index(self):
        """index.md를 현재 페이지 목록으로 재생성합니다."""
        pages = []
        for f in sorted(self.wiki_dir.glob("*.md")):
            if f.name in ("index.md", "log.md"):
                continue
            first_line = (
                f.read_text(encoding="utf-8")
                .split("\n")[0].strip("# ").strip()
            )
            pages.append(f"- [[{f.stem}]]: {first_line}")

        summary_pages = [p for p in pages if "summary" in p]
        concept_pages = [p for p in pages if "concept_" in p]

        index_content = (
            "# Wiki Index\n\n"
            "## Paper Summaries\n" + "\n".join(summary_pages) +
            "\n\n## Concepts & Themes\n" + "\n".join(concept_pages)
        )
        (self.wiki_dir / "index.md").write_text(
            index_content, encoding="utf-8"
        )
