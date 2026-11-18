import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

# Model names get retired often. Override them in .env without touching code:
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")


def get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

def get_vectorstore():
    return Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())

def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', '?')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )