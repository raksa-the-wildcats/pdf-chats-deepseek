import os
from langchain.prompts import PromptTemplate
from typing import List, Dict
from config import Config
from .vector_store import VectorStore

# Try to import replicate, but don't fail if it's not available
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("Warning: replicate module not available. Install with: pip install replicate")

class QAChain:
    def __init__(self):
        # Set Replicate API token
        if Config.REPLICATE_API_TOKEN:
            os.environ["REPLICATE_API_TOKEN"] = Config.REPLICATE_API_TOKEN
        
        # Check if dependencies are available
        if not REPLICATE_AVAILABLE:
            print("Warning: replicate module not available. Install with: pip install replicate")
        
        try:
            self.vector_store = VectorStore()
        except ImportError as e:
            print(f"Warning: {str(e)}")
            self.vector_store = None
            
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for Q&A."""
        template = """You are a helpful assistant specialized in web accessibility. Use the following context from accessibility documentation to answer the user's question. If you don't know the answer based on the provided context, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear, helpful answer based on the context above. Include specific recommendations and best practices when relevant. If citing specific guidelines or standards, mention them clearly."""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def get_answer(self, question: str) -> Dict[str, any]:
        """Get an answer to a question using the knowledge base."""
        try:
            # Check if dependencies are available
            if not REPLICATE_AVAILABLE:
                return {
                    "answer": "❌ The Replicate API is not available. Please install the required dependency: pip install replicate",
                    "sources": [],
                    "error": "replicate module not available"
                }
            
            if not self.vector_store:
                return {
                    "answer": "❌ Vector store is not available. Please install the required dependency: pip install sentence-transformers",
                    "sources": [],
                    "error": "vector store not available"
                }
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question)
            
            if not docs:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "error": None
                }
            
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Document: {doc['metadata'].get('source', 'Unknown')}\n{doc['content']}"
                for doc in docs
            ])
            
            # Generate answer using Replicate
            prompt = self.prompt_template.format(context=context, question=question)
            
            response = replicate.run(
                Config.CHAT_MODEL,
                input={
                    "prompt": prompt,
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            )
            
            # Handle response - Replicate may return a list of strings or FileOutput objects
            if isinstance(response, list):
                # Convert FileOutput objects to strings if needed
                answer = "".join([str(item) for item in response]).strip()
            else:
                answer = str(response).strip()
            
            # Prepare sources
            sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in docs]))
            
            return {
                "answer": answer,
                "sources": sources,
                "error": None
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def initialize_knowledge_base(self, force_rebuild: bool = False):
        """Initialize the knowledge base from PDFs."""
        try:
            if not self.vector_store:
                print("❌ Vector store is not available. Please install sentence-transformers")
                return False
            
            # Check if collection already has documents
            if not force_rebuild and self.vector_store.get_collection_count() > 0:
                print(f"Knowledge base already contains {self.vector_store.get_collection_count()} documents")
                return True
            
            # Import here to avoid circular imports
            from .pdf_processor import PDFProcessor
            
            # Process PDFs
            processor = PDFProcessor()
            chunks = processor.process_all_pdfs()
            
            if not chunks:
                print("No chunks created from PDFs")
                return False
            
            # Reset collection if force rebuild
            if force_rebuild:
                self.vector_store.reset_collection()
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            print(f"Knowledge base initialized with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error initializing knowledge base: {str(e)}")
            return False