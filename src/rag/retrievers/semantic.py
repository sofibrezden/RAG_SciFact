import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.rag.retrievers.base import BaseRetriever


class SemanticRetriever(BaseRetriever):
    """Semantic retriever."""

    def __init__(self, chunks: list[dict], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> None:
        """Initialize the semantic retriever.

        Args:
            chunks: list of chunks
            model_name: name of the model to use

        """
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        self.embeddings = self.model.encode([chunk['text'] for chunk in chunks], convert_to_tensor=True, show_progress_bar=True)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the semantic retriever.

        Args:
            query: query to search
            top_k: number of chunks to return

        """
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]

        top_results = np.argsort(-scores.cpu().numpy())[:top_k]
        return [self.chunks[i] for i in top_results]
