"""RAGAS evaluation harness.

Runs the graph over a labelled dataset and scores the answered cases with RAGAS.
Requires the `eval` extra, a built index and paid access to the judge model.

RAGAS scores are LLM judgements: they are noisy, run-dependent and are not proof
of factual correctness.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    question: str
    reference: str = ""
    answerable: bool = True


@dataclass(frozen=True)
class Outcome:
    case: Case
    answer: str
    contexts: list[str]
    is_answered: bool
    trace: list[str]


def load_cases(path: Path) -> list[Case]:
    """Read a JSONL dataset: question, reference, answerable."""
    cases: list[Case] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: JSON object expected")
        question = str(row.get("question", "")).strip()
        reference = str(row.get("reference", "")).strip()
        answerable = bool(row.get("answerable", True))
        if not question:
            raise ValueError(f"Line {number}: 'question' is required")
        if answerable and not reference:
            raise ValueError(f"Line {number}: an answerable case requires 'reference'")
        cases.append(Case(question, reference, answerable))
    if not cases:
        raise ValueError("Dataset is empty")
    return cases


def collect(cases: Sequence[Case], answer_fn: Callable[[str], dict]) -> list[Outcome]:
    """Run the pipeline once per case. Every call is independent and billable."""
    outcomes: list[Outcome] = []
    for case in cases:
        state = answer_fn(case.question)
        documents = state.get("docs") or []
        outcomes.append(
            Outcome(
                case=case,
                answer=str(state.get("answer", "")),
                contexts=[document["text"] for document in documents],
                is_answered=bool(state.get("is_answered")),
                trace=list(state.get("trace", [])),
            )
        )
    return outcomes


def scorable(outcomes: Sequence[Outcome]) -> list[Outcome]:
    """RAGAS needs a response and contexts, so refusals cannot be scored."""
    return [o for o in outcomes if o.case.answerable and o.is_answered and o.contexts]


def routing_report(outcomes: Sequence[Outcome]) -> dict[str, int]:
    """Counts that RAGAS does not cover: answers produced and refusals."""
    answerable = [o for o in outcomes if o.case.answerable]
    unanswerable = [o for o in outcomes if not o.case.answerable]
    return {
        "cases": len(outcomes),
        "answerable": len(answerable),
        "answered": sum(o.is_answered for o in answerable),
        "unanswerable": len(unanswerable),
        "refused": sum(not o.is_answered for o in unanswerable),
        "scored": len(scorable(outcomes)),
    }


def build_dataset(outcomes: Sequence[Outcome]) -> Any:
    try:
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError:  # older layout
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=o.case.question,
                response=o.answer,
                retrieved_contexts=o.contexts,
                reference=o.case.reference,
            )
            for o in outcomes
        ]
    )


def judge_model(name: str, settings: Any) -> Any:
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    if not name:
        raise ValueError("Set --judge-model or GENERATION_MODEL")
    return LangchainLLMWrapper(
        ChatAnthropic(
            model_name=name,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=1500,
            timeout=90,
            max_retries=2,
        )
    )


def judge_embeddings(settings: Any) -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))


def build_metrics(llm: Any, embeddings: Any = None) -> list:
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    metrics = [
        Faithfulness(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    if embeddings is not None:
        metrics.append(ResponseRelevancy(llm=llm, embeddings=embeddings))
    return metrics


def average_scores(result: Any) -> dict[str, float]:
    frame = result.to_pandas()
    numeric = frame.select_dtypes("number")
    return {column: round(float(numeric[column].mean()), 3) for column in numeric.columns}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the pipeline with RAGAS.")
    parser.add_argument("--dataset", type=Path, default=Path("examples/ragas_golden.jsonl"))
    parser.add_argument("--judge-model", default="", help="Defaults to GENERATION_MODEL.")
    parser.add_argument(
        "--with-relevancy",
        action="store_true",
        help="Add ResponseRelevancy; downloads a local embedding model.",
    )
    parser.add_argument("--report", type=Path, default=Path("reports/ragas.json"))
    args = parser.parse_args(argv)

    from .config import Settings
    from .runtime import answer

    settings = Settings()
    outcomes = collect(load_cases(args.dataset), answer)
    routing = routing_report(outcomes)
    for key, value in routing.items():
        print(f"{key:<13} {value}")

    scored = scorable(outcomes)
    if not scored:
        print("\nNo answered case with context: RAGAS metrics were not computed.")
        return 1

    from ragas import evaluate

    llm = judge_model(args.judge_model or settings.generation_model, settings)
    embeddings = judge_embeddings(settings) if args.with_relevancy else None
    result = evaluate(dataset=build_dataset(scored), metrics=build_metrics(llm, embeddings))
    scores = average_scores(result)

    print(f"\nRAGAS over {len(scored)} answered case(s):")
    for name, value in scores.items():
        print(f"{name:<26} {value}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "judge_model": args.judge_model or settings.generation_model,
                "retrieval_mode": settings.retrieval_mode,
                "top_k": settings.top_k,
                "routing": routing,
                "scores": scores,
                "cases": [
                    {
                        "question": o.case.question,
                        "answerable": o.case.answerable,
                        "is_answered": o.is_answered,
                        "contexts": len(o.contexts),
                        "answer": o.answer,
                        "trace": o.trace,
                    }
                    for o in outcomes
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport: {args.report}")
    return 0
