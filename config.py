import os
from dotenv import load_dotenv

load_dotenv()

# Try to import streamlit for secrets, fallback to environment variables
try:
    import streamlit as st
    def get_api_token():
        try:
            return st.secrets["REPLICATE_API_TOKEN"]
        except:
            return os.getenv("REPLICATE_API_TOKEN")
except ImportError:
    def get_api_token():
        return os.getenv("REPLICATE_API_TOKEN")

class Config:
    # Replicate Configuration
    REPLICATE_API_TOKEN = get_api_token()
    
    # Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using sentence-transformers model
    CHAT_MODEL = "deepseek-ai/deepseek-r1"
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "accessibility_docs"
    
    # PDF Processing Configuration
    PDF_DIRECTORY = "./data/pdfs"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Configuration
    TOP_K_DOCUMENTS = 5
    
    # UI Configuration
    APP_TITLE = "Web Accessibility Q&A Chatbot"
    APP_DESCRIPTION = "Ask questions about web accessibility using our PDF knowledge base"
    
    @classmethod
    def validate(cls):
        if not cls.REPLICATE_API_TOKEN:
            raise ValueError("REPLICATE_API_TOKEN environment variable is required")
        return True