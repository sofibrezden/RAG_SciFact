from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Base class for retrievers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the retriever.

        Args:
            query: query to search
            top_k: number of chunks to return

        Returns:
            list of chunks

        """
        ...
