import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # reads .env on your PC; does nothing on Streamlit Cloud

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"


def _get_setting(name: str, default=None):
    """Look in environment variables / .env first, then in Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # not running inside Streamlit, or no secrets file
    return default


GOOGLE_API_KEY = _get_setting("GOOGLE_API_KEY") or _get_setting("GEMINI_API_KEY")

# Model names get retired often. Change them in .env or in Streamlit secrets:
#   LLM_MODEL=gemini-3.1-flash-lite
LLM_MODEL = _get_setting("LLM_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = _get_setting("EMBED_MODEL", "models/gemini-embedding-001")


def get_llm():
    # temperature=0 -> factual, less creative answers (good for QA)
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL, temperature=0, google_api_key=GOOGLE_API_KEY
    )


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY
    )


def get_vectorstore():
    return Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())


def get_retriever(k: int = 4):
    # k = how many chunks to retrieve per question
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    # Turn retrieved chunks into one text block, tagged with source and page
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', '?')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )