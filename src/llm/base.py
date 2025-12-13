from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, Field

F = TypeVar('F', bound=Callable[..., Any])


class LLMParams(BaseModel):
    """Base LLM parameters."""

    api_key: str = Field(..., description='API key for LLM provider')
    model_name: str = Field(..., description='Name of the LLM model to use')
    temperature: float = Field(default=0.0, description='Temperature for the LLM')


class BaseLLM(ABC):
    """Base class for LLM implementations."""

    def __init__(self, params: LLMParams) -> None:
        self.params = params
        self.api_key = params.api_key
        self.model_name = params.model_name
        self.temperature = params.temperature

    @abstractmethod
    def send_message(self, messages: Sequence[Any]) -> str:
        """Send a message to the LLM.

        Args:
            messages (Sequence[Any]): The messages to send to the LLM.

        Returns:
            str: The response from the LLM.

        """
        ...

    @abstractmethod
    def process_query(self, query: str, retrieved_docs: list[dict]) -> dict:
        """Process a query with retrieved documents to generate an answer.

        Args:
            query: User's question
            retrieved_docs: List of retrieved document chunks

        Returns:
            dict: Contains 'answer', 'context', 'sources' keys

        """
        ...

    @abstractmethod
    def process_with_followup(self, query: str, retrieved_docs: list[dict], conversation_history: list[dict] | None = None) -> dict:
        """Process a query with conversation history for follow-up questions.

        Args:
            query: User's question
            retrieved_docs: List of retrieved document chunks
            conversation_history: Previous conversation messages (list of dicts with 'role' and 'content' keys)

        Returns:
            dict: Contains 'answer', 'context', 'sources', 'num_sources', 'conversation_history' keys

        """
        ...
