"""Evidence checks and fail-closed answer validation."""
from __future__ import annotations

import re


def _terms(text: str) -> set[str]:
    return {x.casefold() for x in re.findall(r"[\w'-]+", text, re.UNICODE) if len(x) > 1}


def validate_answer(answer: str, evidence: list[dict], min_coverage: float = 0.45) -> dict:
    """Return contract-safe answer/citations; refuse if text is not supported.

    The generator is extractive. This lexical check is intentionally conservative,
    but is not a formal entailment proof; deployment evaluation should include
    domain-specific adversarial questions.
    """
    if not answer.strip() or not evidence:
        return {"answer": "", "citations": []}
    answer_terms = _terms(answer)
    if not answer_terms:
        return {"answer": "", "citations": []}
    supporting = []
    covered: set[str] = set()
    for item in evidence:
        overlap = answer_terms & _terms(item.get("text", ""))
        if overlap:
            supporting.append(item)
            covered.update(overlap)
    if len(covered) / len(answer_terms) < min_coverage or not supporting:
        return {"answer": "", "citations": []}
    citations = sorted({str(item["source"]) for item in supporting if item.get("source")})
    if not citations:
        return {"answer": "", "citations": []}
    return {"answer": answer.strip(), "citations": citations}
