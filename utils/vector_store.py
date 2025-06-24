import chromadb
from chromadb.config import Settings
from typing import List, Dict
from config import Config

# Try to import sentence_transformers, but don't fail if it's not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers module not available. Install with: pip install sentence-transformers")

class VectorStore:
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers is required. Install with: pip install sentence-transformers")
            
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(allow_reset=True)
        )
        self.collection_name = Config.COLLECTION_NAME
        self.embeddings = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, str]]):
        """Add document chunks to the vector store."""
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract texts and metadata
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [{'source': chunk['source'], 'chunk_id': chunk['chunk_id']} for chunk in chunks]
        ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embeddings.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully added {len(chunks)} chunks to vector store")
    
    def similarity_search(self, query: str, k: int = None) -> List[Dict]:
        """Search for similar documents."""
        if k is None:
            k = Config.TOP_K_DOCUMENTS
        
        # Generate query embedding
        query_embedding = self.embeddings.encode([query])[0].tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                })
        
        return documents
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def reset_collection(self):
        """Reset the collection (delete all documents)."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        print(f"Reset collection: {self.collection_name}")