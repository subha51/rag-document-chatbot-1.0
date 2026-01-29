
# Embeddings and LLM set up

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def get_embeddings():
    return HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant")
