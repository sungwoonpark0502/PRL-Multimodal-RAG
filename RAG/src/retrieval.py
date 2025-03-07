# retrieval.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class TextRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def create_index(self, docs):
        """Creates a FAISS index with given documents."""
        self.documents = docs
        doc_embeddings = self.embed_model.encode(docs)
        self.index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        self.index.add(np.array(doc_embeddings))

    def retrieve(self, query, k=1):
        """Retrieves the most relevant document for a query."""
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")

        query_embedding = self.embed_model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k)
        return [self.documents[i] for i in indices[0]]

