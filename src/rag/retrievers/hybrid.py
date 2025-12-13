from src.rag.retrievers.base import BaseRetriever
from src.rag.retrievers.bm25 import BM25Retriever
from src.rag.retrievers.semantic import SemanticRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines BM25 and semantic search."""

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        semantic: SemanticRetriever | None = None,
        *,
        use_bm25: bool = True,
        use_semantic: bool = True,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            bm25: BM25 retriever
            semantic: Semantic retriever
            use_bm25: Whether to use BM25
            use_semantic: Whether to use semantic search

        """
        self.bm25 = bm25
        self.semantic = semantic
        self.use_bm25 = use_bm25
        self.use_semantic = use_semantic

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the hybrid retriever.

        Args:
            query: query to search
            top_k: number of chunks to return

        """
        results = []

        if self.use_bm25 and self.bm25:
            results.extend(self.bm25.search(query, top_k))

        if self.use_semantic and self.semantic:
            results.extend(self.semantic.search(query, top_k))

        seen = set()
        unique = []
        for r in results:
            key = (r['doc_id'], r['chunk_id'])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:top_k]
