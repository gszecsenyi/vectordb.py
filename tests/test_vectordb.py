"""Tests for vectordb module."""

import pytest
from vectordbpy.vectordb import Document, MemoryVectorStore


class TestDocument:
    """Test cases for Document class."""

    def test_init_with_content_only(self):
        """Test Document initialization with content only."""
        content = "This is test content"
        doc = Document(content)
        assert doc.page_content == content
        assert doc.metadata == {}

    def test_init_with_content_and_metadata(self):
        """Test Document initialization with content and metadata."""
        content = "This is test content"
        metadata = {"source": "test.txt", "page": 1}
        doc = Document(content, metadata)
        assert doc.page_content == content
        assert doc.metadata == metadata

    def test_init_with_none_metadata(self):
        """Test Document initialization with explicit None metadata."""
        content = "This is test content"
        doc = Document(content, None)
        assert doc.page_content == content
        assert doc.metadata == {}

    def test_str_representation(self):
        """Test Document string representation."""
        content = "Test content"
        metadata = {"key": "value"}
        doc = Document(content, metadata)
        expected = f"Document(Page Content: {content}\nMetadata: {metadata})"
        assert str(doc) == expected

    def test_repr_representation(self):
        """Test Document repr representation."""
        content = "Test content"
        metadata = {"key": "value"}
        doc = Document(content, metadata)
        expected = f"Document(page_content='{content}', metadata={metadata})"
        assert repr(doc) == expected

    def test_empty_content(self):
        """Test Document with empty content."""
        doc = Document("")
        assert doc.page_content == ""
        assert doc.metadata == {}

    def test_complex_metadata(self):
        """Test Document with complex metadata."""
        content = "Test"
        metadata = {
            "source": "file.pdf",
            "page": 42,
            "tags": ["important", "review"],
            "nested": {"level": 1, "priority": "high"}
        }
        doc = Document(content, metadata)
        assert doc.page_content == content
        assert doc.metadata == metadata


class TestMemoryVectorStore:
    """Test cases for MemoryVectorStore class."""

    def test_init_empty(self):
        """Test MemoryVectorStore initialization with no arguments."""
        store = MemoryVectorStore()
        assert store.vectors == []
        assert store.documents == []
        assert store.metadatas == []

    def test_init_with_data(self):
        """Test MemoryVectorStore initialization with data."""
        vectors = [[1, 2], [3, 4]]
        documents = [Document("doc1"), Document("doc2")]
        metadatas = [{"id": 1}, {"id": 2}]
        
        store = MemoryVectorStore(vectors, documents, metadatas)
        assert store.vectors == vectors
        assert store.documents == documents
        assert store.metadatas == metadatas

    def test_init_with_none_values(self):
        """Test MemoryVectorStore initialization with explicit None values."""
        store = MemoryVectorStore(None, None, None)
        assert store.vectors == []
        assert store.documents == []
        assert store.metadatas == []

    def test_add_documents_valid(self):
        """Test adding documents with matching vectors."""
        store = MemoryVectorStore()
        docs = [Document("doc1"), Document("doc2")]
        vectors = [[1, 0], [0, 1]]
        metadatas = [{"id": 1}, {"id": 2}]
        
        store.add_documents(docs, vectors, metadatas)
        
        assert len(store.documents) == 2
        assert len(store.vectors) == 2
        assert len(store.metadatas) == 2
        assert store.documents == docs
        assert store.vectors == vectors
        assert store.metadatas == metadatas

    def test_add_documents_without_metadata(self):
        """Test adding documents without metadata."""
        store = MemoryVectorStore()
        docs = [Document("doc1"), Document("doc2")]
        vectors = [[1, 0], [0, 1]]
        
        store.add_documents(docs, vectors)
        
        assert len(store.documents) == 2
        assert len(store.vectors) == 2
        assert len(store.metadatas) == 2
        assert store.metadatas == [{}, {}]

    def test_add_documents_mismatched_lengths(self):
        """Test adding documents with mismatched vector lengths."""
        store = MemoryVectorStore()
        docs = [Document("doc1"), Document("doc2")]
        vectors = [[1, 0]]  # Only one vector for two documents
        
        with pytest.raises(ValueError, match="Number of documents must match number of vectors"):
            store.add_documents(docs, vectors)

    def test_add_documents_mismatched_metadata_lengths(self):
        """Test adding documents with mismatched metadata lengths."""
        store = MemoryVectorStore()
        docs = [Document("doc1"), Document("doc2")]
        vectors = [[1, 0], [0, 1]]
        metadatas = [{"id": 1}]  # Only one metadata for two documents
        
        with pytest.raises(ValueError, match="Number of metadatas must match number of documents"):
            store.add_documents(docs, vectors, metadatas)

    def test_add_documents_to_existing_store(self):
        """Test adding documents to store that already has data."""
        # Initialize with existing data
        existing_docs = [Document("existing")]
        existing_vectors = [[1, 1]]
        store = MemoryVectorStore(existing_vectors, existing_docs)
        
        # Add new documents
        new_docs = [Document("new1"), Document("new2")]
        new_vectors = [[0, 1], [1, 0]]
        store.add_documents(new_docs, new_vectors)
        
        assert len(store.documents) == 3
        assert len(store.vectors) == 3
        assert len(store.metadatas) == 3
        assert store.documents[0].page_content == "existing"
        assert store.documents[1].page_content == "new1"
        assert store.documents[2].page_content == "new2"

    def test_query_vector_empty_store(self):
        """Test querying an empty vector store."""
        store = MemoryVectorStore()
        query_vector = [1, 0]
        results = store.query_vector(query_vector)
        assert results == []

    def test_query_vector_single_document(self):
        """Test querying store with single document."""
        doc = Document("test document")
        vector = [1, 0]
        store = MemoryVectorStore([vector], [doc])
        
        query_vector = [1, 0]  # Identical vector
        results = store.query_vector(query_vector, k=1)
        
        assert len(results) == 1
        assert results[0] == doc

    def test_query_vector_multiple_documents(self):
        """Test querying store with multiple documents."""
        docs = [
            Document("doc1"),
            Document("doc2"),
            Document("doc3")
        ]
        vectors = [
            [1, 0],    # Perfect match to query
            [0, 1],    # Orthogonal to query
            [0.5, 0]   # Partial match to query
        ]
        store = MemoryVectorStore(vectors, docs)
        
        query_vector = [1, 0]
        results = store.query_vector(query_vector, k=2)
        
        assert len(results) == 2
        # Should return doc1 first (perfect match), then doc3 (partial match)
        assert results[0].page_content == "doc1"
        assert results[1].page_content == "doc3"

    def test_query_vector_k_larger_than_store(self):
        """Test querying with k larger than number of documents."""
        docs = [Document("doc1"), Document("doc2")]
        vectors = [[1, 0], [0, 1]]
        store = MemoryVectorStore(vectors, docs)
        
        query_vector = [1, 0]
        results = store.query_vector(query_vector, k=10)
        
        assert len(results) == 2  # Should return all available documents

    def test_query_vector_k_zero(self):
        """Test querying with k=0."""
        docs = [Document("doc1"), Document("doc2")]
        vectors = [[1, 0], [0, 1]]
        store = MemoryVectorStore(vectors, docs)
        
        query_vector = [1, 0]
        results = store.query_vector(query_vector, k=0)
        
        assert len(results) == 0

    def test_query_vector_default_k(self):
        """Test querying with default k value."""
        docs = [Document(f"doc{i}") for i in range(10)]
        vectors = [[i, 0] for i in range(10)]
        store = MemoryVectorStore(vectors, docs)
        
        query_vector = [5, 0]
        results = store.query_vector(query_vector)  # Default k=5
        
        assert len(results) == 5

    def test_query_vector_similarity_ordering(self):
        """Test that results are ordered by similarity (highest first)."""
        docs = [
            Document("low_similarity"),   # [1, 1] vs [1, 0] = lower similarity
            Document("high_similarity"),  # [1, 0] vs [1, 0] = perfect similarity
            Document("medium_similarity") # [1, 0.5] vs [1, 0] = medium similarity
        ]
        vectors = [
            [1, 1],
            [1, 0],
            [1, 0.5]
        ]
        store = MemoryVectorStore(vectors, docs)
        
        query_vector = [1, 0]
        results = store.query_vector(query_vector, k=3)
        
        assert len(results) == 3
        assert results[0].page_content == "high_similarity"
        assert results[1].page_content == "medium_similarity"
        assert results[2].page_content == "low_similarity"