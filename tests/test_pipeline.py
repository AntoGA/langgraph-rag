import pytest

from rag.config import Settings
from rag.pipeline import Verdict, build_graph, run

DOC = {"id": "test", "source": "demo.md", "text": "Two attempts."}


class FakeModels:
    def __init__(self, grounded=True, on_topic=True, text="Two attempts [1]."):
        self.grounded = grounded
        self.on_topic = on_topic
        self.text = text

    def relevant(self, question, doc):
        return True

    def generate(self, question, docs, critique):
        return self.text

    def rewrite(self, question, previous, critique):
        return question + " synonyms"

    def verify(self, question, docs, answer):
        return Verdict(is_grounded=self.grounded, answers_question=self.on_topic,
                       critique="Try again")


def settings():
    return Settings(_env_file=None, max_retrieval_attempts=2, max_generation_attempts=2)


def test_success_and_isolation():
    config = settings()
    graph = build_graph(FakeModels(), lambda q: [DOC], config)
    first = run(graph, "question", config)
    second = run(graph, "another question", config)
    for result in (first, second):
        assert result["is_answered"]
        assert result["retrievals"] == 1
        assert result["generations"] == 1
        assert result["trace"] == ["retrieve", "grade", "generate", "verify"]
        assert result["citations"][0]["marker"] == "[1]"


def test_empty_retrieval():
    config = settings()
    result = run(build_graph(FakeModels(), lambda q: [], config), "question", config)
    assert not result["is_answered"]
    assert result["retrievals"] == 2
    assert "generate" not in result["trace"]
    assert result["trace"][-1] == "fallback"


@pytest.mark.parametrize("grounded,on_topic,expected", [(False, True, 4), (True, False, 2)])
def test_bounded_routes(grounded, on_topic, expected):
    config = settings()
    graph = build_graph(FakeModels(grounded, on_topic), lambda q: [DOC], config)
    result = run(graph, "question", config)
    assert not result["is_answered"]
    assert result["trace"].count("generate") == expected
    assert result["citations"] == []


@pytest.mark.parametrize("text", ["No citation", "Invalid [0]", "Invalid [2]", "Mix [1] [9]"])
def test_invalid_citations(text):
    config = settings()
    graph = build_graph(FakeModels(text=text), lambda q: [DOC], config)
    assert not run(graph, "question", config)["is_answered"]


def test_empty_question():
    config = settings()
    graph = build_graph(FakeModels(), lambda q: [DOC], config)
    with pytest.raises(ValueError):
        run(graph, "   ", config)
