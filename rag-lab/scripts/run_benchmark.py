#!/usr/bin/env python3
"""
벤치마크 실행 스크립트

3가지 RAG 방법을 5개 질문으로 비교합니다.
결과를 benchmark/ 폴더에 저장합니다.

실행:
    python scripts/run_benchmark.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_lab import (
    EmbeddingEngine, StandardRAG, LightRAG, KarpathyWiki, Evaluator,
    load_papers,
)

# ── 벤치마크 질문 ───────────────────────────────────────────────────
QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "What is the main finding about intergenerational mobility "
            "in the paper by Card, Domnisoru, and Taylor?"
        ),
        "type": "single_paper_factual",
        "difficulty": "easy",
    },
    {
        "id": "Q2",
        "question": (
            "How do negative nominal interest rates affect bank "
            "performance according to the evidence?"
        ),
        "type": "single_paper_factual",
        "difficulty": "easy",
    },
    {
        "id": "Q3",
        "question": (
            "Compare the methodological approaches used in studying "
            "human capital transmission versus auction market dynamics."
        ),
        "type": "cross_paper_synthesis",
        "difficulty": "hard",
    },
    {
        "id": "Q4",
        "question": (
            "What role does employer credit checking play in labor "
            "markets, and how does it relate to information asymmetry "
            "themes across these papers?"
        ),
        "type": "cross_paper_synthesis",
        "difficulty": "hard",
    },
    {
        "id": "Q5",
        "question": (
            "Across all papers, what datasets are used? Which rely "
            "on natural experiments vs structural models?"
        ),
        "type": "multi_paper_aggregation",
        "difficulty": "medium",
    },
]


def main():
    print("=" * 65)
    print("  RAG vs LightRAG vs Karpathy LLM-Wiki 벤치마크")
    print("=" * 65)

    papers = load_papers()
    total_chars = sum(len(v) for v in papers.values())
    print(f"\n논문 {len(papers)}편 로드 ({total_chars:,} 글자)\n")

    # ── 시스템 구축 ─────────────────────────────────────────────
    print("─" * 65)
    print("시스템 구축")
    print("─" * 65)

    engine = EmbeddingEngine()

    t0 = time.time()
    rag = StandardRAG(papers, engine)
    rag_build = time.time() - t0

    t0 = time.time()
    lrag = LightRAG(papers, engine)
    lrag_build = time.time() - t0

    t0 = time.time()
    wiki = KarpathyWiki(papers)
    wiki_build = time.time() - t0

    print(
        f"\n구축 시간: RAG={rag_build:.1f}s, "
        f"LightRAG={lrag_build:.1f}s, Wiki={wiki_build:.1f}s"
    )

    # ── 벤치마크 실행 ───────────────────────────────────────────
    evaluator = Evaluator()
    methods = {
        "Standard RAG": rag,
        "LightRAG": lrag,
        "Karpathy Wiki": wiki,
    }
    all_results = []

    print(f"\n{'─' * 65}")
    print("벤치마크 질문 실행")
    print(f"{'─' * 65}")

    for q in QUESTIONS:
        print(f"\n{'━' * 55}")
        print(f"[{q['id']}] {q['question'][:70]}...")
        print(f"  유형: {q['type']} | 난이도: {q['difficulty']}")

        q_result = {"question": q, "methods": {}}

        for name, method in methods.items():
            print(f"\n  ▸ {name}...", end=" ", flush=True)
            result = method.query(q["question"])
            score = evaluator.score(
                q["question"], result.answer, q["type"]
            )
            print(
                f"시간={result.total_time:.1f}s | "
                f"점수={score.average:.1f} "
                f"(A={score.accuracy} C={score.completeness} "
                f"S={score.specificity} Sy={score.synthesis})"
            )
            q_result["methods"][name] = {
                "answer": result.answer,
                "time": result.total_time,
                "context_length": result.context_length,
                "score": {
                    "accuracy": score.accuracy,
                    "completeness": score.completeness,
                    "specificity": score.specificity,
                    "synthesis": score.synthesis,
                    "average": score.average,
                    "reasoning": score.reasoning,
                },
            }
        all_results.append(q_result)

    # ── 결과 요약 ───────────────────────────────────────────────
    print(f"\n\n{'═' * 65}")
    print("  벤치마크 결과 요약")
    print(f"{'═' * 65}")

    method_names = list(methods.keys())
    agg = {
        m: {"accuracy": [], "completeness": [], "specificity": [],
            "synthesis": [], "time": []}
        for m in method_names
    }

    for r in all_results:
        for m in method_names:
            s = r["methods"][m]["score"]
            for dim in ["accuracy", "completeness", "specificity", "synthesis"]:
                agg[m][dim].append(s[dim])
            agg[m]["time"].append(r["methods"][m]["time"])

    header = (
        f"{'지표':<14} | {'Standard RAG':>14} | "
        f"{'LightRAG':>14} | {'Karpathy Wiki':>14}"
    )
    print(header)
    print("─" * len(header))

    for dim in ["accuracy", "completeness", "specificity", "synthesis"]:
        label = {
            "accuracy": "정확성",
            "completeness": "완전성",
            "specificity": "구체성",
            "synthesis": "통합성",
        }[dim]
        vals = [f"{np.mean(agg[m][dim]):.1f}" for m in method_names]
        print(f"{label:<14} | {vals[0]:>14} | {vals[1]:>14} | {vals[2]:>14}")

    overalls = []
    for m in method_names:
        all_s = []
        for dim in ["accuracy", "completeness", "specificity", "synthesis"]:
            all_s.extend(agg[m][dim])
        overalls.append(np.mean(all_s))

    print("─" * len(header))
    print(
        f"{'전체 평균':<14} | {overalls[0]:>14.1f} | "
        f"{overalls[1]:>14.1f} | {overalls[2]:>14.1f}"
    )
    print()
    vals = [f"{np.mean(agg[m]['time']):.1f}s" for m in method_names]
    print(
        f"{'평균 쿼리 시간':<14} | {vals[0]:>14} | "
        f"{vals[1]:>14} | {vals[2]:>14}"
    )
    print(
        f"{'구축 시간':<14} | {rag_build:>13.1f}s | "
        f"{lrag_build:>13.1f}s | {wiki_build:>13.1f}s"
    )

    # ── 결과 저장 ───────────────────────────────────────────────
    bench_dir = Path(__file__).parent.parent / "benchmark"
    bench_dir.mkdir(exist_ok=True)

    (bench_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {bench_dir / 'results.json'}")


if __name__ == "__main__":
    main()
