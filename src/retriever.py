"""Hybrid retrieval facade."""
from .indexer import hybrid_search, load_index


class HybridRetriever:
    def __init__(self, index_dir: str):
        self.documents = load_index(index_dir)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return hybrid_search(query, self.documents, top_k=top_k)
