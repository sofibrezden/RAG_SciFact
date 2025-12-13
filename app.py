import streamlit as st
from dotenv import load_dotenv

from src.llm.base import BaseLLM, LLMParams
from src.llm.openai import OpenAILLM
from src.rag.chunking import chunk_documents
from src.rag.reranker import CrossEncoderReranker
from src.rag.retrievers.base import BaseRetriever
from src.rag.retrievers.bm25 import BM25Retriever
from src.rag.retrievers.hybrid import HybridRetriever
from src.rag.retrievers.semantic import SemanticRetriever
from src.utils import load_docs


def setup_page() -> None:
    """Set up the page."""
    st.set_page_config(page_title='RAG Demo', page_icon='🔍', layout='wide')
    st.title('🔍 RAG Search Demo')
    load_dotenv()


def init_session_state() -> None:
    """Initialize the session state."""
    st.session_state.setdefault('conversation_history', [])
    st.session_state.setdefault('last_query', '')
    st.session_state.setdefault('api_key', '')


def render_sidebar() -> tuple[str, bool, bool, int, int, str]:
    """Render the sidebar."""
    with st.sidebar:
        st.header('⚙ Search settings')

        api_key = st.text_input(
            'OpenAI API Key', value=st.session_state.api_key, type='password', help='Enter your OpenAI API key to enable LLM features'
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key

        retriever_name = st.selectbox('Retriever', ['BM25', 'Semantic', 'Hybrid'])
        use_reranker = st.checkbox('Use reranker', value=True)
        use_llm = st.checkbox('Use LLM for answer generation', value=True, help='Generate answers using retrieved documents')
        top_k = st.slider('Top-k results', 1, 20, 5)
        num_docs_to_use = st.slider('Number of docs to use', 1, 5000, 10)

        if use_llm and st.button('🗑 Clear conversation'):
            st.session_state.conversation_history = []
            st.session_state.last_query = ''
            st.rerun()

    return retriever_name, use_reranker, use_llm, top_k, num_docs_to_use, api_key


def render_conversation_history() -> None:
    """Render the conversation history."""
    history = st.session_state.conversation_history
    if not history:
        return

    with st.expander('💬 Conversation History', expanded=False):
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                st.write(f'**Q:** {history[i]["content"]}')
                st.write(f'**A:** {history[i + 1]["content"]}')
                st.write('---')


@st.cache_resource
def prepare_rag(num_docs: int) -> tuple[list[dict], dict[str, BaseRetriever], CrossEncoderReranker]:
    """Prepare the RAG."""
    docs = load_docs('beir/scifact')[:num_docs]
    chunks = chunk_documents(docs)

    bm25 = BM25Retriever(chunks)
    semantic = SemanticRetriever(chunks)
    hybrid = HybridRetriever(bm25=bm25, semantic=semantic)

    retrievers = {'BM25': bm25, 'Semantic': semantic, 'Hybrid': hybrid}

    return chunks, retrievers, CrossEncoderReranker()


@st.cache_resource
def prepare_llm(*, use_llm: bool, api_key: str) -> BaseLLM | None:
    """Prepare the LLM."""
    if not use_llm or not api_key:
        return None

    return OpenAILLM(LLMParams(api_key=api_key, model_name='gpt-4.1', temperature=0.0))


def render_project_description() -> None:
    """Render project description."""
    st.markdown(
        """
        This project is a **Retrieval-Augmented Question Answering (RAG-QA) demo** built with Streamlit.
        It uses the **BEIR SciFact (`beir/scifact`) dataset**, which contains biomedical research paper
        abstracts commonly used for scientific fact verification. The system retrieves relevant
        document chunks using lexical (BM25), semantic, or hybrid retrieval strategies, optionally
        applies cross-encoder reranking, and then generates **evidence-grounded answers** using a
        large language model.
        """
    )
    st.divider()


def run_search(*, query: str, retriever: BaseRetriever, reranker: CrossEncoderReranker, top_k: int, use_reranker: bool) -> list[dict]:
    """Run the search."""
    if use_reranker:
        candidates = retriever.search(query, top_k=top_k)
        return reranker.rerank(query=query, chunks=candidates, top_k=top_k)
    return retriever.search(query, top_k=top_k)


def generate_answer(*, query: str, results: list[dict], llm: BaseLLM) -> dict:
    """Generate the answer."""
    history = st.session_state.conversation_history
    rag_result = llm.process_with_followup(query, results, history) if history else llm.process_query(query, results)
    history.extend([{'role': 'user', 'content': query}, {'role': 'assistant', 'content': rag_result['answer']}])

    return rag_result


def render_results(results: list[dict]) -> None:
    """Render the results."""
    st.subheader(f'📄 Retrieved Documents ({len(results)})')

    for i, r in enumerate(results, 1):
        title = r.get('title', 'Untitled')
        label = f'### {i}. {title}'

        with st.expander(label):
            st.write(r['text'])
            st.caption(f'doc_id_from_dataset={r["doc_id"]} | chunk_id={r.get("chunk_id")}')


def main() -> None:
    """Run the app."""
    setup_page()
    init_session_state()

    retriever_name, use_reranker, use_llm, top_k, num_docs_to_use, api_key = render_sidebar()

    render_project_description()

    render_conversation_history()

    _, retrievers, reranker = prepare_rag(num_docs_to_use)
    llm = prepare_llm(use_llm=use_llm, api_key=api_key)

    query = st.text_input('Enter your question', placeholder='e.g. What is DNA methylation?')

    if not st.button('🔍 Search', type='primary'):
        return

    if not query:
        st.warning('Please enter a question.')
        return

    if use_llm and not api_key:
        st.error('Please enter your OpenAI API key in the sidebar to use LLM features.')
        return

    if use_llm and not llm:
        st.error('Failed to initialize LLM. Please check your API key.')
        return

    retriever = retrievers[retriever_name]

    with st.spinner('Searching...'):
        results = run_search(query=query, retriever=retriever, reranker=reranker, top_k=top_k, use_reranker=use_reranker)

    if use_llm and results:
        with st.spinner('Generating answer...'):
            rag_result = generate_answer(query=query, results=results, llm=llm)

        st.subheader('🤖 Generated Answer')
        st.write(rag_result['answer'])
        st.caption(f'Based on {rag_result["num_sources"]} documents')
        st.divider()

    render_results(results)


if __name__ == '__main__':
    main()
