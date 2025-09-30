"""
vectordb.py - In-memory vector store package

This package implements an in-memory vector database solution.
Vectors and their associated documents are stored in memory during program
execution, so there is no need to run or install a separate vector database
(such as Pinecone, Weaviate, Milvus, etc.).

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

    def __init__(self, page_content: str, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __str__(self):
        return (f"Document(Page Content: {self.page_content}\n"
                f"Metadata: {self.metadata})")

    def __repr__(self):
        return (f"Document(page_content='{self.page_content}', "
                f"metadata={self.metadata})")


class MemoryVectorStore:
    """
    An in-memory vector database for storing and querying document vectors.

    This class provides functionality to store vectors with associated
    documents and perform similarity-based queries to retrieve the most
    relevant documents.
    """

    def __init__(self, vectors=None, documents=None, metadatas=None,
                 vectorizer=None):
        """
        Initialize the vector store.

        Args:
            vectors (list, optional): List of vectors. Defaults to empty list.
            documents (list, optional): List of documents. Defaults to empty.
            metadatas (list, optional): List of metadata. Defaults to empty.
            vectorizer (BaseVectorizer, optional): Vectorizer for automatic
                text vectorization.
        """
        self.vectors = vectors if vectors is not None else []
        self.documents = documents if documents is not None else []
        self.vectorizer = vectorizer

        if metadatas is not None:
            self.metadatas = metadatas
        else:
            # Create empty metadata for each existing document
            self.metadatas = [{}] * len(self.documents)

    def _extract_texts_from_documents(self, documents):
        """Extract text content from documents for vectorization."""
        return [doc.page_content if isinstance(doc, Document) else str(doc)
                for doc in documents]

    def _should_refit_vectorizer(self, all_texts):
        """Check if vectorizer should be refitted due to new vocabulary."""
        if (not hasattr(self.vectorizer, 'is_fitted') or
                not self.vectorizer.is_fitted):
            return True

        # Check if new vocabulary would be created
        old_vocab_size = len(self.vectorizer.vocabulary)

        # Create a temporary vectorizer to check new vocabulary size
        temp_vectorizer = type(self.vectorizer)()
        temp_vectorizer.fit(all_texts)
        new_vocab_size = len(temp_vectorizer.vocabulary)

        return new_vocab_size > old_vocab_size

    def _refit_vectorizer_and_update_vectors(self, all_texts):
        """Refit vectorizer and update existing vectors."""
        self.vectorizer.fit(all_texts)

        # Re-vectorize existing documents if any
        if self.documents:
            existing_texts = self._extract_texts_from_documents(
                self.documents)
            self.vectors = self.vectorizer.transform(existing_texts)

    def add_documents(self, documents, vectors=None, metadatas=None):
        """
        Add documents with their corresponding vectors to the store.

        Args:
            documents (list): List of Document objects to add.
            vectors (list, optional): List of vectors corresponding to the
                documents. If None and vectorizer is set, vectors will be
                computed.
            metadatas (list, optional): List of metadata objects. Defaults
                to None.
        """
        if vectors is None:
            if self.vectorizer is None:
                raise ValueError("Either vectors must be provided or "
                                 "vectorizer must be set")

            # Extract text content from documents for vectorization
            texts = self._extract_texts_from_documents(documents)

            # Prepare all texts (existing + new) for potential refitting
            all_texts = []
            all_texts.extend(self._extract_texts_from_documents(
                self.documents))
            all_texts.extend(texts)

            # Check if vectorizer needs fitting/refitting
            if self._should_refit_vectorizer(all_texts):
                self._refit_vectorizer_and_update_vectors(all_texts)

            # Vectorize new documents
            vectors = self.vectorizer.transform(texts)

        if len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of "
                             "vectors")

        self.documents.extend(documents)
        self.vectors.extend(vectors)

        if metadatas is None:
            metadatas = [{}] * len(documents)

        if len(metadatas) != len(documents):
            raise ValueError("Number of metadatas must match number of "
                             "documents")

        self.metadatas.extend(metadatas)

    def add_texts(self, texts, metadatas=None):
        """
        Add texts by converting them to Document objects and vectorizing them.

        Args:
            texts (list): List of text strings to add.
            metadatas (list, optional): List of metadata objects. Defaults
                to None.
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")

        # Convert texts to Document objects
        documents = [Document(text, metadata)
                     for text, metadata in zip(texts, metadatas)]

        # Add documents (this will automatically vectorize if vectorizer
        # is set)
        self.add_documents(documents, vectors=None, metadatas=None)

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

        similarities = [cosine_similarity_pure_python(query_vector, v)
                        for v in self.vectors]
        top_k_indices = sorted(range(len(similarities)),
                               key=lambda i: similarities[i],
                               reverse=True)[:k]
        results = [self.documents[i] for i in top_k_indices]
        return results

    def query_text(self, query_text, k=5):
        """
        Query the vector store for the most similar documents using text.

        Args:
            query_text (str): Text query to search for.
            k (int): Number of top similar documents to return. Defaults to 5.

        Returns:
            list: List of the k most similar documents.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be set to query with text")

        if (not hasattr(self.vectorizer, 'is_fitted') or
                not self.vectorizer.is_fitted):
            raise ValueError("Vectorizer must be fitted before querying")

        # Vectorize the query text
        query_vectors = self.vectorizer.transform([query_text])
        query_vector = query_vectors[0]

        return self.query_vector(query_vector, k)
