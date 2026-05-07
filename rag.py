import os
import pickle
from typing import Any

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

from config import (
    FAISS_INDEX_PATH, CHUNKS_META_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    STREAMLIT_PAGE_TITLE, STREAMLIT_LAYOUT,
    validate_paths, validate_env,
)


@st.cache_resource
def load_models() -> tuple[SentenceTransformer, Groq]:
    """Load embedding model and Groq client with error handling."""
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        client = Groq(api_key=groq_key)
        return embedding_model, client

    except ValueError as e:
        st.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise


@st.cache_resource
def load_index() -> tuple[Any, list[dict[str, Any]]]:
    """Load FAISS index and metadata with error handling."""
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(CHUNKS_META_PATH, "rb") as f:
            chunks_meta = pickle.load(f)

        return index, chunks_meta

    except FileNotFoundError as e:
        st.error(f"Index files not found. Run 'python indexation.py' first: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        raise


def embed_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """Embed a text query."""
    embedding = model.encode(text, convert_to_numpy=True).astype(np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
    return embedding.reshape(1, -1)


def search_chunks(
    question: str,
    model: SentenceTransformer,
    index: Any,
    chunks_meta: list[dict[str, Any]],
    k: int = 4,
) -> list[dict[str, Any]]:
    """Search for most relevant chunks."""
    query_embedding = embed_text(question, model)
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        if idx < len(chunks_meta):
            results.append(chunks_meta[idx])

    return results


def construct_prompt(question: str, relevant_chunks: list[dict[str, Any]]) -> str:
    """Build the prompt with context."""
    context = "\n\n".join([f"Title: {chunk['title']}\nRating: {chunk['rating']}\n{chunk['content']}"
                            for chunk in relevant_chunks])

    prompt = f"""You are a helpful film recommendation assistant. Use the provided film information to answer the user's question accurately.

Film Context:
{context}

User Question: {question}

Provide a helpful, conversational answer based on the films provided. If the information is not in the context, say so."""

    return prompt


def generate_response(
    question: str,
    relevant_chunks: list[dict[str, Any]],
    client: Groq,
    model: str = LLM_MODEL,
) -> str:
    """Generate response using Groq with error handling."""
    try:
        prompt = construct_prompt(question, relevant_chunks)

        message = client.chat.completions.create(
            model=model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.choices[0].message.content

    except Exception as e:
        return f"Erreur lors de la génération de la réponse: {str(e)}"


def display_sources(relevant_chunks: list[dict[str, Any]]) -> None:
    """Display relevant films as sources."""
    st.subheader("Films Sources")
    for i, chunk in enumerate(relevant_chunks, 1):
        with st.expander(f"{i}. {chunk['title']} (Rating: {chunk['rating']}/10)"):
            st.write(chunk["content"])


def main() -> None:
    st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, layout=STREAMLIT_LAYOUT)

    st.title("🎬 RAG Films TMDB")
    st.markdown("Ask questions about films! Powered by Groq + FAISS")

    missing_files = validate_paths()
    missing_vars = validate_env()
    errors = (
        [f"{p.name} not found. Run: python indexation.py" for p in missing_files]
        + [f"{v} not set. Create a .env file with: {v}=your_key" for v in missing_vars]
    )

    if errors:
        st.error("Configuration errors detected:")
        for error in errors:
            st.error(f"- {error}")
        st.stop()

    with st.spinner("Initializing RAG system..."):
        try:
            embedding_model, groq_client = load_models()
            index, chunks_meta = load_index()
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            st.stop()

    with st.sidebar:
        st.header("Settings")
        k = st.slider("Number of relevant films", min_value=1, max_value=10, value=4)
        show_sources = st.checkbox("Show film sources", value=True)

    question = st.text_input("Ask a question about films...")

    if question and question.strip():
        with st.spinner("Searching films..."):
            relevant_chunks = search_chunks(question, embedding_model, index, chunks_meta, k=k)

        if not relevant_chunks:
            st.warning("No relevant films found for your query.")
            return

        with st.spinner("Generating response..."):
            response = generate_response(question, relevant_chunks, groq_client, model="llama-3.3-70b-versatile")

        st.subheader("Answer")
        st.write(response)

        if show_sources:
            display_sources(relevant_chunks)


if __name__ == "__main__":
    main()
