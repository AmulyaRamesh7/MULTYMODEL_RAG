import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, k=5):
        D, I = self.index.search(query_embedding, k)
        return [self.text_chunks[i] for i in I[0]]
