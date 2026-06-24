import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearchIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load a lightweight, highly optimized Transformer embedding model
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def initialize_index(self, doc_list: list):
        """
        Extracts text from documents, generates vector embeddings, and builds the FAISS index.
        """
        self.documents = doc_list
        corpus_texts = [doc["text"] for doc in doc_list]
        
        # Convert sentences to 384-dimensional dense vectors
        embeddings = self.encoder.encode(corpus_texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        
        # Initialize an IndexFlatL2 index (Exact L2 Euclidean distance search)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

    def search(self, query: str, top_k: int = 1) -> list:
        """
        Encodes user query and returns top_k closest matching documents.
        """
        if not self.index:
            raise ValueError("Search index has not been initialized.")
            
        # Encode user query into the same vector space
        query_vector = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        
        # Query the FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx != -1: # Ensure match exists
                results.append({
                    "document": self.documents[idx],
                    "score": float(distances[0][rank])
                })
        return results