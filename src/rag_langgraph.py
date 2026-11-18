from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from config import get_llm, get_retriever, format_docs

retriever = get_retriever()
llm = get_llm()


# 1. STATE: shared data every node can read and update
class RAGState(TypedDict):
    question: str
    docs: List[Document]
    relevant: bool
    answer: str


grade_prompt = ChatPromptTemplate.from_template(
    """Does the context contain information that helps answer the question?
Reply with only "yes" or "no".

Context:
{context}

Question: {question}"""
)

gen_prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
Mention the source file and page when useful.

Context:
{context}

Question: {question}

Answer:"""
)


# 2. NODES: each returns the part of the state it updates
def retrieve(state: RAGState):
    return {"docs": retriever.invoke(state["question"])}


def grade(state: RAGState):
    verdict = (grade_prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(state["docs"]), "question": state["question"]}
    )
    return {"relevant": verdict.strip().lower().startswith("yes")}


def generate(state: RAGState):
    answer = (gen_prompt | llm | StrOutputParser()).invoke(
        {"context": format_docs(state["docs"]), "question": state["question"]}
    )
    return {"answer": answer}


def fallback(state: RAGState):
    return {"answer": "I couldn't find that in the documents."}


# 3. CONDITIONAL EDGE: picks the next node based on state
def route_after_grade(state: RAGState):
    return "generate" if state["relevant"] else "fallback"


# 4. BUILD THE GRAPH:  START -> retrieve -> grade -> (generate | fallback) -> END
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("fallback", fallback)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade", route_after_grade, {"generate": "generate", "fallback": "fallback"}
)
graph.add_edge("generate", END)
graph.add_edge("fallback", END)

app = graph.compile()


if __name__ == "__main__":
    print("LangGraph RAG ready. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            result = app.invoke({"question": q})
            print("\nAnswer:", result["answer"])