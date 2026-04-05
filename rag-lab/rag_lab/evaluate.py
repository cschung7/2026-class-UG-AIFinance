"""
평가 모듈 — LLM-as-Judge로 답변 품질 평가
"""

import json
import numpy as np
from dataclasses import dataclass

from .llm import llm_call_json


@dataclass
class EvalScore:
    """평가 점수 (각 1-10)"""
    accuracy: int       # 정확성: 사실에 부합하는가
    completeness: int   # 완전성: 질문의 모든 측면을 다루는가
    specificity: int    # 구체성: 논문명, 수치, 방법론을 인용하는가
    synthesis: int      # 통합성: 여러 출처를 연결하는가
    reasoning: str      # 평가 근거

    @property
    def average(self) -> float:
        return np.mean([
            self.accuracy, self.completeness,
            self.specificity, self.synthesis,
        ])


class Evaluator:
    """
    LLM-as-Judge 평가기

    4가지 차원으로 답변 품질을 평가합니다:
    - Accuracy (정확성): 사실적으로 올바른가?
    - Completeness (완전성): 빠진 부분이 없는가?
    - Specificity (구체성): 구체적 인용이 있는가?
    - Synthesis (통합성): 여러 출처를 연결하는가?

    사용법:
        >>> evaluator = Evaluator()
        >>> score = evaluator.score(question, answer, "cross_paper")
        >>> print(f"평균 점수: {score.average:.1f}")
    """

    def score(
        self,
        question: str,
        answer: str,
        question_type: str = "general",
    ) -> EvalScore:
        """
        답변의 품질을 평가합니다.

        Args:
            question: 원래 질문
            answer: 평가할 답변
            question_type: 질문 유형 (factual, synthesis, aggregation)

        Returns:
            EvalScore: 4차원 점수 + 평가 근거
        """
        try:
            data = llm_call_json(
                prompt=(
                    f"Rate this answer on 4 dimensions (1-10 each):\n\n"
                    f"Question: {question}\n"
                    f"Type: {question_type}\n\n"
                    f"Answer:\n{answer}\n\n"
                    f"Criteria:\n"
                    f"1. Accuracy: factual correctness, no hallucinations\n"
                    f"2. Completeness: covers all aspects\n"
                    f"3. Specificity: cites papers, numbers, methods\n"
                    f"4. Synthesis: integrates across sources\n\n"
                    f"Return JSON:\n"
                    f'{{"accuracy": N, "completeness": N, '
                    f'"specificity": N, "synthesis": N, '
                    f'"reasoning": "brief explanation"}}'
                ),
                system=(
                    "Expert evaluator. Be strict but fair. "
                    "Score based on what a domain expert would expect."
                ),
            )
            return EvalScore(
                accuracy=data.get("accuracy", 5),
                completeness=data.get("completeness", 5),
                specificity=data.get("specificity", 5),
                synthesis=data.get("synthesis", 5),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return EvalScore(5, 5, 5, 5, "평가 파싱 실패")

    def compare(
        self,
        question: str,
        answers: dict[str, str],
        question_type: str = "general",
    ) -> dict[str, EvalScore]:
        """
        여러 방법의 답변을 비교 평가합니다.

        Args:
            question: 원래 질문
            answers: {"method_name": "answer_text"}
            question_type: 질문 유형

        Returns:
            dict[str, EvalScore]: 방법별 점수
        """
        scores = {}
        for method, answer in answers.items():
            scores[method] = self.score(question, answer, question_type)
        return scores

    @staticmethod
    def print_comparison(scores: dict[str, EvalScore]):
        """비교 결과를 테이블로 출력합니다."""
        header = (
            f"{'Method':<20} | {'Acc':>4} | {'Comp':>4} | "
            f"{'Spec':>4} | {'Syn':>4} | {'Avg':>5}"
        )
        print(header)
        print("─" * len(header))
        for method, s in scores.items():
            print(
                f"{method:<20} | {s.accuracy:>4} | {s.completeness:>4} | "
                f"{s.specificity:>4} | {s.synthesis:>4} | "
                f"{s.average:>5.1f}"
            )
