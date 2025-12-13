from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Reranker using CrossEncoder."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> None:
        """Initialize the reranker.

        Args:
            model_name: name of the model to use

        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        """Rerank the chunks.

        Args:
            query: query to rerank
            chunks: list of chunks to rerank
            top_k: number of chunks to return

        """
        if not chunks:
            return []

        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]
