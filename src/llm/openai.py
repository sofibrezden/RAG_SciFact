from collections.abc import Sequence

from openai import OpenAI

from src.llm.base import BaseLLM, LLMParams
from src.utils import format_context, load_prompt


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation using the OpenAI API."""

    def __init__(self, params: LLMParams) -> None:
        super().__init__(params)
        self.client = OpenAI(api_key=self.api_key)

    def send_message(self, messages: Sequence[dict[str, str]]) -> str:
        """Send messages to OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            str: The response content from the LLM

        """
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=self.temperature)
        return response.choices[0].message.content

    def process_query(self, query: str, retrieved_docs: list[dict]) -> dict:
        """Process a query with retrieved documents to generate an answer.

        Args:
            query: User's question
            retrieved_docs: List of retrieved document chunks

        Returns:
            dict: Contains 'answer', 'context', 'sources' keys

        """
        context = format_context(retrieved_docs)

        system_prompt = load_prompt('src/rag/prompts/system_prompt.txt')
        user_prompt = load_prompt('src/rag/prompts/user_prompt.txt').format(context=context, query=query)

        answer = self.send_message([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}])

        sources = []
        for doc in retrieved_docs:
            source_info = {'doc_id': doc.get('doc_id'), 'title': doc.get('title', 'Untitled'), 'chunk_id': doc.get('chunk_id')}
            if source_info not in sources:
                sources.append(source_info)

        return {'answer': answer, 'context': context, 'sources': sources, 'num_sources': len(retrieved_docs)}

    def process_with_followup(self, query: str, retrieved_docs: list[dict], conversation_history: list[dict] | None = None) -> dict:
        """Process a query with conversation history for follow-up questions.

        Args:
            query: User's question
            retrieved_docs: List of retrieved document chunks
            conversation_history: Previous conversation messages (list of dicts with 'role' and 'content' keys)

        Returns:
            dict: Contains 'answer', 'context', 'sources', 'num_sources', 'conversation_history' keys

        """
        context = format_context(retrieved_docs)
        messages = []

        system_prompt = load_prompt('src/rag/prompts/system_prompt.txt')
        messages.append({'role': 'system', 'content': system_prompt})

        if conversation_history:
            messages.extend(conversation_history)

        user_prompt = load_prompt('src/rag/prompts/user_prompt.txt').format(context=context, query=query)

        messages.append({'role': 'user', 'content': user_prompt})
        answer = self.send_message(messages)

        sources = []
        for doc in retrieved_docs:
            source_info = {'doc_id': doc.get('doc_id'), 'title': doc.get('title', 'Untitled'), 'chunk_id': doc.get('chunk_id')}
            if source_info not in sources:
                sources.append(source_info)

        return {
            'answer': answer,
            'context': context,
            'sources': sources,
            'num_sources': len(retrieved_docs),
            'conversation_history': messages,
        }
