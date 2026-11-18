import time
import uuid
from typing import List, TypedDict

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from config import get_llm, get_embeddings

# Used by the LangGraph engine. Higher = stricter.
# Tune it by watching the "RETRIEVAL SCORES" line in the terminal.
SCORE_THRESHOLD = 0.55


def format_context(docs: List[Document]) -> str:
    # Plain text only, with no source or page labels
    return "\n\n".join(d.page_content for d in docs)


# ---------------------------------------------------------------
# PART 1: turn uploaded files into a searchable retriever
# ---------------------------------------------------------------
def load_uploaded_files(files) -> List[Document]:
    docs = []
    for f in files:
        name = f.name
        if name.lower().endswith(".pdf"):
            reader = PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": name, "page": i + 1}))
        elif name.lower().endswith(".txt"):
            text = f.getvalue().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
    return docs


def _add_with_retry(store, batch_docs, max_tries: int = 5):
    # Embedding happens before anything is stored, so a failed batch
    # leaves nothing behind and is safe to retry.
    for attempt in range(max_tries):
        try:
            store.add_documents(batch_docs)
            return
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 30 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s before retrying...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        "Still rate limited after several retries. Wait a few minutes and try again."
    )


def build_retriever(files, k: int = 4, progress_callback=None):
    docs = load_uploaded_files(files)
    if not docs:
        raise ValueError(
            "No readable text found. Scanned PDFs (images) aren't supported. "
            "Try a text-based PDF."
        )

    # Bigger chunks -> fewer embedding requests
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # No persist_directory -> in-memory store, unique name per upload.
    # Cosine distance makes the similarity scores easy to interpret.
    store = Chroma(
        collection_name=f"upload_{uuid.uuid4().hex}",
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 30 chunks every ~20s stays under the 100 requests/minute free limit
    batch = 30
    for i in range(0, len(chunks), batch):
        _add_with_retry(store, chunks[i:i + batch])
        done = min(i + batch, len(chunks))
        if progress_callback:
            progress_callback(done, len(chunks))
        if done < len(chunks):
            time.sleep(20)

    return store.as_retriever(search_kwargs={"k": k}), len(docs), len(chunks)

# ---------------------------------------------------------------
# PART 2: prompt
# ---------------------------------------------------------------
QA_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the documents."
Do not mention sources, file names or page numbers.

Context:
{context}

Question: {question}

Answer:"""
)

# ---------------------------------------------------------------
# PART 3: LangChain engine  (1 LLM call per question)
# ---------------------------------------------------------------
def build_chain_engine(retriever):
    chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | QA_PROMPT
        | get_llm()
        | StrOutputParser()
    )

    def ask(question: str) -> str:
        return chain.invoke(question)

    return ask

# ---------------------------------------------------------------
# PART 4: LangGraph engine  (1 LLM call per question)
#   retrieve (with score check) -> generate | fallback
# ---------------------------------------------------------------
class RAGState(TypedDict):
    question: str
    docs: List[Document]
    relevant: bool
    answer: str

def build_graph_engine(retriever):
    llm = get_llm()

    def retrieve(state: RAGState):
        results = retriever.vectorstore.similarity_search_with_relevance_scores(
            state["question"], k=4
        )
        scores = [round(s, 3) for _, s in results]
        print("RETRIEVAL SCORES:", scores)  # watch these in the terminal
        best = max(scores, default=0)
        return {
            "docs": [d for d, _ in results],
            "relevant": best >= SCORE_THRESHOLD,
        }

    def generate(state: RAGState):
        answer = (QA_PROMPT | llm | StrOutputParser()).invoke(
            {"context": format_context(state["docs"]), "question": state["question"]}
        )
        return {"answer": answer}

    def fallback(state: RAGState):
        return {"answer": "I couldn't find that in the documents."}

    def route(state: RAGState):
        return "generate" if state["relevant"] else "fallback"

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("fallback", fallback)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route, {"generate": "generate", "fallback": "fallback"})
    g.add_edge("generate", END)
    g.add_edge("fallback", END)
    app = g.compile()

    def ask(question: str) -> str:
        return app.invoke({"question": question})["answer"]

    return ask