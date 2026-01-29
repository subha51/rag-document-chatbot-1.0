
# RAG Document Question Answering Chatbot

A conversational Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask context-aware questions strictly grounded in the uploaded content.  
The system supports multi-turn conversations with chat history and avoids hallucinations by refusing to answer questions not present in the document.

---
## Features
- Upload one or multiple PDF documents
- Conversational question answering with memory
- Context-aware follow-up questions
- Strict document-grounded responses (hallucination controlled)
- Clean modular architecture
- Interactive Streamlit UI

---
## 🧠 How It Works (RAG Pipeline)
1. Document Ingestion – PDFs are loaded and split into chunks  
2. Embedding Generation – Text chunks are converted into vector embeddings  
3. Vector Search – FAISS retrieves relevant document chunks  
4. Contextual Question Rewriting – Follow-up questions are rewritten using chat history  
5. Answer Generation – LLM answers using *only* retrieved context  
6. Chat Memory – Session-based conversational history is maintained  

If the answer is not found in the document, the system responds:
> *“I do not know based on the provided document.”*

---
## Project Structure
```
rag-document-chatbot/
│
├── app.py                 # Streamlit UI for user interaction
│
├── rag/
│   ├── init.py        # Marks rag as a Python package
│   ├── pipeline.py        # Core RAG pipeline (retrieval + generation)
│   ├── utils.py           # PDF loading and text chunking utilities
│   └── llm_utils.py       # LLM and embedding configuration
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---
## How to Run
```
pip install -r requirements.txt
streamlit run app.py
```
🔐 Environment Variables
-Create a .env file in the project root:
```
HF_API_KEY=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
```


## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM

---
## Architecture Overview
- Streamlit UI for user interaction
- Modular RAG pipeline handling document ingestion, retrieval, and generation
- FAISS vector store for similarity search
- Conversational memory using session-based chat history
- LLM-powered contextual question answering

---
## Key Highlights
- Supports multi-document PDF uploads
- Maintains conversational context across queries
- Prevents hallucinations by restricting answers to retrieved context
- Clean separation of UI and core RAG logic

---
## Use Cases
- Ask questions from research papers and academic PDFs
- Explore and understand technical documentation
- Perform context-aware search over uploaded documents
- Conduct multi-turn conversations with document memory
- Prevent hallucinations by restricting answers to document content