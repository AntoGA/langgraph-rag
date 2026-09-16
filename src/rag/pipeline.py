import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .config import Settings
from .index import Chunk


class Verdict(BaseModel):
    is_grounded: bool
    answers_question: bool
    critique: str = ""


class State(TypedDict, total=False):
    question: str
    query: str
    docs: list[Chunk]
    answer: str
    citations: list[dict]
    retrievals: int
    generations: int
    critique: str
    grounded: bool
    on_topic: bool
    is_answered: bool
    trace: list[str]


def citation_numbers(answer: str, count: int) -> list[int]:
    numbers = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not numbers or any(n < 1 or n > count for n in numbers):
        return []
    return sorted(set(numbers))


def build_graph(models, search, settings: Settings):
    def trace(state, name):
        return [*state.get("trace", []), name]

    def retrieve(state):
        return {"docs": search(state["query"])[:settings.top_k],
                "retrievals": state["retrievals"] + 1, "generations": 0,
                "trace": trace(state, "retrieve")}

    def grade(state):
        docs = [d for d in state["docs"] if models.relevant(state["question"], d)]
        return {"docs": docs, "critique": "" if docs else "No relevant evidence; try other terms.",
                "trace": trace(state, "grade")}

    def generate(state):
        return {"answer": models.generate(state["question"], state["docs"], state["critique"]),
                "generations": state["generations"] + 1, "trace": trace(state, "generate")}

    def verify(state):
        verdict = models.verify(state["question"], state["docs"], state["answer"])
        numbers = citation_numbers(state["answer"], len(state["docs"]))
        grounded = verdict.is_grounded and bool(numbers)
        valid = grounded and verdict.answers_question
        citations = [{"marker": f"[{n}]", **state["docs"][n - 1]} for n in numbers] if valid else []
        return {"grounded": grounded, "on_topic": verdict.answers_question,
                "is_answered": valid, "citations": citations,
                "critique": verdict.critique if numbers else "Use valid numbered citations.",
                "trace": trace(state, "verify")}

    def retry(state):
        return {"trace": trace(state, "retry")}

    def rewrite(state):
        query = models.rewrite(state["question"], state["query"], state["critique"])
        return {"query": query.strip() or state["question"], "trace": trace(state, "rewrite")}

    def fallback(state):
        return {"answer": "Не удалось получить подтверждённый ответ в рамках лимита попыток. "
                          "Уточните вопрос или добавьте документы.",
                "citations": [], "is_answered": False, "trace": trace(state, "fallback")}

    def after_verify(state):
        if state["is_answered"]:
            return END
        if (not state["grounded"] and state["on_topic"]
                and state["generations"] < settings.max_generation_attempts):
            return "generate"
        return "retry"

    graph = StateGraph(State)
    for name, node in (("retrieve", retrieve), ("grade", grade), ("generate", generate),
                       ("verify", verify), ("retry", retry), ("rewrite", rewrite),
                       ("fallback", fallback)):
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", lambda s: "generate" if s["docs"] else "retry",
                                ["generate", "retry"])
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify, [END, "generate", "retry"])
    graph.add_conditional_edges("retry", lambda s: "rewrite"
                                if s["retrievals"] < settings.max_retrieval_attempts else "fallback",
                                ["rewrite", "fallback"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("fallback", END)
    return graph.compile()


def run(graph, question: str, settings: Settings) -> State:
    question = question.strip()
    if not question or len(question) > 2000:
        raise ValueError("Question must contain 1–2000 characters")
    limit = settings.max_retrieval_attempts * (2 * settings.max_generation_attempts + 5) + 5
    return graph.invoke({"question": question, "query": question, "retrievals": 0,
                         "generations": 0, "critique": "", "trace": [],
                         "answer": "", "citations": [], "is_answered": False},
                        config={"recursion_limit": limit})
