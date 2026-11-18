from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import DATA_DIR, DB_DIR, get_embeddings

def load_documents():
    docs = []
    for pdf in DATA_DIR.glob("**/*.pdf"):
        reader = PdfReader(str(pdf))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():  
                docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf.name, "page": i},
                ))
    for txt in DATA_DIR.glob("**/*.txt"):
        docs.append(Document(
            page_content=txt.read_text(encoding="utf-8"),
            metadata={"source": txt.name},
        ))
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    
        chunk_overlap=200,  
    )
    return splitter.split_documents(docs)

def build_index():
    docs = load_documents()
    if not docs:
        raise SystemExit("No readable files found in data/. Add text-based PDFs or .txt files first.")
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks")
    Chroma.from_documents(
        chunks, embedding=get_embeddings(), persist_directory=str(DB_DIR)
    )
    print("Vector store saved to chroma_db/")

if __name__ == "__main__":
    build_index()