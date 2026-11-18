
 DocuMind – Chat with Your Documents

 Project Overview
- DocuMind is an AI-powered document question-answering chatbot.
- Built using Retrieval-Augmented Generation (RAG).
- Allows users to upload PDF and TXT documents.
- Answers user questions based on uploaded documents.
- Implements RAG using both LangChain and LangGraph.

 Features
- Upload multiple PDF and TXT files.
- Supports up to 100 pages.
- Ask questions in natural language.
- Get answers based on document content.
- Two selectable AI engines: LangChain and LangGraph.
- LangGraph checks whether relevant information exists before answering.
- Progress bar during document processing.
- Automatic retry for API rate limits.
- Simple and responsive Streamlit UI.
- Session-based document storage.

 How It Works
1. Upload PDF or TXT documents.
2. Extract text from uploaded files.
3. Split text into smaller chunks.
4. Convert chunks into embeddings using Gemini.
5. Store embeddings in ChromaDB.
6. Convert the user question into an embedding.
7. Retrieve the top 4 relevant chunks.
8. Send the retrieved context to Google Gemini.
9. Generate an answer based on the document.

AI Engines
- **LangChain:** Uses a simple retrieve-and-generate pipeline.
- **LangGraph:** Uses a graph-based workflow to check document relevance before generating answers.
- Returns a "Not Found" response when relevant information is unavailable.

 Tech Stack
- Python
- Streamlit
- LangChain
- LangGraph
- Google Gemini
- Gemini Embeddings
- ChromaDB
- PyPDF

 Project Structure
```text
DocuMind/
├── app.py
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── ingest.py
│   ├── rag_langchain.py
│   └── rag_langgraph.py
├── data/
├── docs/
├── requirements.txt
└── README.md
```
 Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/docmind-rag-chatbot.git
   ```
2. Navigate to the project folder:
   ```bash
   cd docmind-rag-chatbot
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the environment:
   ```bash
   venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Add your Gemini API key in a `.env` file.
7. Run the application:
   ```bash
   streamlit run app.py
   ```

 Deployment
- Deployed using Streamlit Community Cloud.
- Connected to GitHub repository.
- Configured Gemini API key through Streamlit Secrets.

