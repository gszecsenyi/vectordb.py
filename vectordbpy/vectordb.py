"""
vectordb.py - In-memory vector store package

This package implements an in-memory vector database solution. 
Vectors and their associated documents are stored in memory during program execution, 
so there is no need to run or install a separate vector database (such as Pinecone, 
Weaviate, Milvus, etc.).

This approach is ideal for small projects, prototypes, or use cases 
where the amount of data does not justify using an external database.
"""
from .similarity import cosine_similarity_pure_python

class Document:
    def __init__(self, page_content : str, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __str__(self):
        return f"Document(Page Content: {self.page_content}\nMetadata: {self.metadata})"

class MemoryVectorStore:
    def __init__(self, vectors=None, documents=None, metadatas=None):
        self.vectors = vectors if vectors is not None else []
        self.documents = documents if documents is not None else []
        self.metadatas = metadatas if metadatas is not None else []

    def query_vector(self, query_vector, k=5):
        similarities = [cosine_similarity_pure_python(query_vector, v) for v in self.vectors]
        top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i],
                               reverse=True)[:k]
        results = [self.documents[i] for i in top_k_indices]
        return results
