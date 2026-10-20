"""Persistent deterministic local vector and BM25 indexes (no model downloads)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from .chunker import Chunk

TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.casefold() for t in TOKEN_RE.findall(text)]


def _vector(tokens: list[str], dims: int = 2048) -> dict[str, float]:
    # Signed feature hashing gives a compact deterministic vector representation.
    counts = Counter(tokens)
    raw: dict[str, float] = {}
    for token, count in counts.items():
        slot = hash_token(token, dims)
        raw[str(slot)] = raw.get(str(slot), 0.0) + math.copysign(1.0, hash_token(token, 2 * dims) - dims) * (1 + math.log(count))
    norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
    return {key: value / norm for key, value in raw.items()}


def hash_token(token: str, mod: int) -> int:
    # Python's built-in hash is salted per process; use a stable FNV-1a hash.
    value = 2166136261
    for byte in token.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value % mod


def build_index(chunks: list[Chunk], index_dir: str | Path) -> None:
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    docs = []
    for chunk in chunks:
        tokens = tokenize(chunk.text)
        docs.append({"chunk_id": chunk.chunk_id, "source": chunk.source, "text": chunk.text,
                     "tokens": tokens, "vector": _vector(tokens)})
    payload = {"version": 1, "documents": docs}
    tmp = directory / "index.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(directory / "index.json")


def load_index(index_dir: str | Path) -> list[dict]:
    path = Path(index_dir) / "index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("documents"), list):
        raise ValueError("unsupported or invalid index format")
    return data["documents"]


def bm25_scores(query_tokens: list[str], docs: list[dict], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(d.get("tokens", [])) for d in docs) / n or 1.0
    df = Counter(term for term in set(query_tokens) for d in docs if term in set(d.get("tokens", [])))
    scores = []
    for doc in docs:
        counts = Counter(doc.get("tokens", [])); dl = len(doc.get("tokens", [])); score = 0.0
        for term in query_tokens:
            tf = counts[term]
            if tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def _cosine_sparse(left: dict, right: dict) -> float:
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def hybrid_search(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    tokens = tokenize(query)
    if not tokens or not docs:
        return []
    lexical = bm25_scores(tokens, docs)
    query_vector = _vector(tokens)
    max_lex = max(lexical, default=0.0) or 1.0
    ranked = []
    for i, doc in enumerate(docs):
        dense = max(0.0, _cosine_sparse(query_vector, doc.get("vector", {})))
        lex = lexical[i] / max_lex
        score = 0.65 * lex + 0.35 * dense
        ranked.append((score, doc))
    ranked.sort(key=lambda item: (-item[0], item[1].get("source", ""), item[1].get("chunk_id", "")))
    # Local reranking boosts direct query-token coverage and short focused passages.
    qset = set(tokens)
    out = []
    for score, doc in ranked[:max(top_k * 3, top_k)]:
        terms = set(doc.get("tokens", []))
        coverage = len(qset & terms) / max(1, len(qset))
        focused = 1 / (1 + len(doc.get("tokens", [])) / 500)
        reranked = score + 0.10 * coverage + 0.02 * focused
        out.append({**doc, "score": reranked, "coverage": coverage})
    out.sort(key=lambda d: (-d["score"], d.get("source", ""), d.get("chunk_id", "")))
    return out[:top_k]
