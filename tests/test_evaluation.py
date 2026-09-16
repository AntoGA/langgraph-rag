import json

import pytest

from rag.evaluation import Case, collect, load_cases, routing_report, scorable


def write_dataset(tmp_path, rows):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    return path


def test_load_cases_reads_labels(tmp_path):
    path = write_dataset(
        tmp_path,
        [
            {"question": "q1", "reference": "r1"},
            {"question": "q2", "answerable": False},
        ],
    )
    cases = load_cases(path)
    assert [case.answerable for case in cases] == [True, False]
    assert cases[0].reference == "r1"


def test_answerable_case_requires_reference(tmp_path):
    with pytest.raises(ValueError):
        load_cases(write_dataset(tmp_path, [{"question": "q1"}]))


def test_empty_dataset_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"question": "q1"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Line 1"):
        load_cases(path)


def test_collect_maps_documents_and_excludes_refusals():
    cases = [Case("q1", "r1", True), Case("q2", "", False)]
    states = {
        "q1": {
            "answer": "answer [1]",
            "docs": [{"id": "a", "source": "demo.md", "text": "evidence"}],
            "is_answered": True,
            "trace": ["retrieve", "grade", "generate", "verify"],
        },
        "q2": {"answer": "refusal", "docs": [], "is_answered": False, "trace": ["fallback"]},
    }
    outcomes = collect(cases, lambda question: states[question])

    assert [outcome.contexts for outcome in outcomes] == [["evidence"], []]
    assert [outcome.case.question for outcome in scorable(outcomes)] == ["q1"]
    assert routing_report(outcomes) == {
        "cases": 2,
        "answerable": 1,
        "answered": 1,
        "unanswerable": 1,
        "refused": 1,
        "scored": 1,
    }


def test_answered_case_without_context_is_not_scored():
    cases = [Case("q1", "r1", True)]
    outcomes = collect(cases, lambda _: {"answer": "a", "docs": [], "is_answered": True})
    assert scorable(outcomes) == []
