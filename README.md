 DocuMind – Chat with Your Documents
An end-to-end RAG (Retrieval-Augmented Generation) application that allows users to upload PDF or TXT documents and ask questions in natural language.
DocuMind retrieves relevant information from uploaded documents and generates answers using Google Gemini.
The project implements two RAG approaches using LangChain and LangGraph.

 Features
Upload multiple PDF and TXT files.
Supports up to 100 pages.
Ask questions and get document-based answers.
Two selectable engines: LangChain and LangGraph.
LangGraph checks document relevance before generating answers.
Progress bar during document processing.
Automatic retry for API rate limits.
Simple and responsive Streamlit interface.
Session-based in-memory document storage.

⚙️ How It Works

PDF / TXT Upload
       ↓
Text Extraction
       ↓
Text Splitting
       ↓
Gemini Embeddings
       ↓
ChromaDB Vector Store
       ↓
User Question
       ↓
Retrieve Top 4 Relevant Chunks
       ↓
Google Gemini
       ↓
Final Answer

LangChain: Follows a simple retrieve-and-generate process.
LangGraph: Checks whether relevant information exists before generating an answer. If information is unavailable, it returns a "Not Found" response instead of guessing.

🛠️ Tech Stack

Python
Streamlit
AI Frameworks
LangChain, LangGraph
LLM & Embeddings
Google Gemini
Vector Database
ChromaDB
PDF Processing

 Project Structure

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

 Run Locally

1. Clone the repository

git clone https://github.com/<your-username>/docmind-rag-chatbot.git
cd docmind-rag-chatbot

2. Create and activate virtual environment

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Create a .env file

GOOGLE_API_KEY=your_key_here
LLM_MODEL=gemini-3.1-flash-lite

Get your API key from Google AI Studio.

5. Run the application

streamlit run app.py

☁️ Deployment

Deployed using Streamlit Community Cloud.

Add your GOOGLE_API_KEY and LLM_MODEL in the application's Secrets settings.

