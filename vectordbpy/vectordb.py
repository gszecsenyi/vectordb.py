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
    """
    Represents a document with content and optional metadata.

    This class encapsulates a document's textual content along with
    associated metadata for use in vector database operations.
    """

    def __init__(self, page_content : str, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __str__(self):
        return f"Document(Page Content: {self.page_content}\nMetadata: {self.metadata})"

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

class MemoryVectorStore:
    """
    An in-memory vector database for storing and querying document vectors.

    This class provides functionality to store vectors with associated documents
    and perform similarity-based queries to retrieve the most relevant documents.
    """

    def __init__(self, vectors=None, documents=None, metadatas=None):
        """
        Initialize the vector store.

        Args:
            vectors (list, optional): List of vectors. Defaults to empty list.
            documents (list, optional): List of documents. Defaults to empty list.
            metadatas (list, optional): List of metadata. Defaults to empty list.
        """
        self.vectors = vectors if vectors is not None else []
        self.documents = documents if documents is not None else []

        if metadatas is not None:
            self.metadatas = metadatas
        else:
            # Create empty metadata for each existing document
            self.metadatas = [{}] * len(self.documents)

    def add_documents(self, documents, vectors, metadatas=None):
        """
        Add documents with their corresponding vectors to the store.

        Args:
            documents (list): List of Document objects to add.
            vectors (list): List of vectors corresponding to the documents.
            metadatas (list, optional): List of metadata objects. Defaults to None.
        """
        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")

        self.documents.extend(documents)
        self.vectors.extend(vectors)

        if metadatas is None:
            metadatas = [{}] * len(documents)

        if len(metadatas) != len(documents):
            raise ValueError("Number of metadatas must match number of documents")

        self.metadatas.extend(metadatas)

    def query_vector(self, query_vector, k=5):
        """
        Query the vector store for the most similar documents.

        Args:
            query_vector (list): Vector to query for.
            k (int): Number of top similar documents to return. Defaults to 5.

        Returns:
            list: List of the k most similar documents.
        """
        if not self.vectors:
            return []

        similarities = [cosine_similarity_pure_python(query_vector, v) for v in self.vectors]
        top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i],
                               reverse=True)[:k]
        results = [self.documents[i] for i in top_k_indices]
        return results
