from rank_bm25 import BM25Okapi

from src.rag.retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25 retriever."""

    def __init__(self, chunks: list[dict]) -> None:
        """Initialize the BM25 retriever.

        Args:
            chunks: list of chunks

        """
        self.chunks = chunks
        self.tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the BM25 retriever.

        Args:
            query: query to search
            top_k: number of chunks to return

        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(zip(self.chunks, scores, strict=True), key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in ranked[:top_k]]
