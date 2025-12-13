from pathlib import Path
from typing import Any

import ir_datasets

DATASET_NAME = 'beir/scifact'


def load_prompt(path: str) -> str:
    """Load prompt from a file.

    Args:
        path: Path to the file

    Returns:
        str: The prompt from the file

    """
    with Path(path).open(encoding='utf-8') as file:
        return file.read()


def load_docs(dataset_name: str) -> list[dict]:
    """Load documents from the dataset.

    Args:
        dataset_name: name of the dataset to load

    Returns:
        list of documents

    """
    ds = ir_datasets.load(dataset_name)
    docs_iter = ds.docs_iter()
    return [{'doc_id': doc.doc_id, 'text': doc.text, 'title': doc.title} for doc in docs_iter]


def format_context(retrieved_docs: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context string.

    Args:
        retrieved_docs: List of document chunks with 'text', 'doc_id', etc.

    Returns:
        str: Formatted context string

    """
    if not retrieved_docs:
        return 'No relevant documents found.'

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        title = doc.get('title', f'Document {doc.get("doc_id", i)}')
        text = doc['text']
        context_parts.append(f'[{i}] {title}\n{text}')

    return '\n\n'.join(context_parts)
