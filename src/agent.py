"""LangGraph workflow with a dependency-light sequential fallback."""
from __future__ import annotations

import re
from typing import TypedDict
from .guardrail import validate_answer
from .retriever import HybridRetriever


class State(TypedDict, total=False):
    query: str
    evidence: list[dict]
    answer: str
    result: dict


def _make_graph(retriever: HybridRetriever):
    def planner(state: State) -> State:
        # Keep query intact; no external LLM or remote service is involved.
        return {"query": state["query"].strip()}

    def retrieve(state: State) -> State:
        return {"evidence": retriever.retrieve(state["query"], top_k=5)}

    def generate(state: State) -> State:
        # Extract a complete sentence directly from retrieved passages.
        query_terms = {t.casefold() for t in re.findall(r"[\w'-]+", state["query"]) if len(t) > 2}
        candidates = []
        for item in state.get("evidence", []):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", item.get("text", "")):
                sentence = sentence.strip()
                if not sentence:
                    continue
                terms = {t.casefold() for t in re.findall(r"[\w'-]+", sentence) if len(t) > 2}
                coverage = len(query_terms & terms) / max(1, len(query_terms))
                candidates.append((coverage + item.get("score", 0) * 0.2, sentence))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        answer = candidates[0][1] if candidates and candidates[0][0] >= 0.20 else ""
        return {"answer": answer}

    def guardrail(state: State) -> State:
        return {"result": validate_answer(state.get("answer", ""), state.get("evidence", []))}

    try:
        from langgraph.graph import END, START, StateGraph
        graph = StateGraph(State)
        graph.add_node("planner", planner)
        graph.add_node("hybrid_retriever", retrieve)
        graph.add_node("generator", generate)
        graph.add_node("guardrail", guardrail)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "hybrid_retriever")
        graph.add_edge("hybrid_retriever", "generator")
        graph.add_edge("generator", "guardrail")
        graph.add_edge("guardrail", END)
        return graph.compile()
    except ImportError:
        # Useful for minimal local installs; install requirements for LangGraph mode.
        return None


class RAGAgent:
    def __init__(self, index_dir: str):
        self.retriever = HybridRetriever(index_dir)
        self.graph = _make_graph(self.retriever)

    def answer(self, query: str) -> dict:
        if self.graph is not None:
            state = self.graph.invoke({"query": query})
            return state.get("result", {"answer": "", "citations": []})
        evidence = self.retriever.retrieve(query, top_k=5)
        terms = {t.casefold() for t in re.findall(r"[\w'-]+", query) if len(t) > 2}
        sentences = []
        for item in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", item["text"]):
                coverage = len(terms & {t.casefold() for t in re.findall(r"[\w'-]+", sentence) if len(t) > 2}) / max(1, len(terms))
                if sentence.strip():
                    sentences.append((coverage + item.get("score", 0) * 0.2, sentence.strip()))
        sentences.sort(key=lambda x: (-x[0], x[1]))
        answer = sentences[0][1] if sentences and sentences[0][0] >= 0.20 else ""
        return validate_answer(answer, evidence)
