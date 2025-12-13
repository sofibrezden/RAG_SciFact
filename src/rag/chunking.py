from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def build_semantic_chunker() -> SemanticChunker:
    """Build a semantic chunker."""
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    return SemanticChunker(embeddings=embeddings, breakpoint_threshold_type='percentile', breakpoint_threshold_amount=95)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents."""
    chunker = build_semantic_chunker()
    all_chunks = []

    for doc in documents:
        chunks = chunker.split_text(doc['text'])

        for idx, chunk in enumerate(chunks):
            all_chunks.append({'doc_id': doc['doc_id'], 'title': doc.get('title', ''), 'chunk_id': idx, 'text': chunk})

    return all_chunks
