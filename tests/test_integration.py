"""Integration tests for vectordbpy package."""

import pytest
import vectordbpy
from vectordbpy import Document, MemoryVectorStore, cosine_similarity_pure_python


class TestPackageIntegration:
    """Test the package integration and imports."""

    def test_package_imports(self):
        """Test that all main components can be imported from the package."""
        # Test direct imports work
        assert hasattr(vectordbpy, 'Document')
        assert hasattr(vectordbpy, 'MemoryVectorStore')
        assert hasattr(vectordbpy, 'cosine_similarity_pure_python')
        assert hasattr(vectordbpy, 'BaseVectorizer')
        assert hasattr(vectordbpy, 'TFIDFVectorizer')
        assert hasattr(vectordbpy, 'BagOfWordsVectorizer')
        assert hasattr(vectordbpy, 'WordCountVectorizer')
        
        # Test that they are the correct classes/functions
        assert vectordbpy.Document is Document
        assert vectordbpy.MemoryVectorStore is MemoryVectorStore
        assert vectordbpy.cosine_similarity_pure_python is cosine_similarity_pure_python

    def test_package_version(self):
        """Test that the package has a version attribute."""
        assert hasattr(vectordbpy, '__version__')
        assert isinstance(vectordbpy.__version__, str)

    def test_package_all(self):
        """Test that __all__ is defined and contains expected exports."""
        assert hasattr(vectordbpy, '__all__')
        expected_exports = [
            "Document", 
            "MemoryVectorStore", 
            "cosine_similarity_pure_python",
            "BaseVectorizer",
            "TFIDFVectorizer",
            "BagOfWordsVectorizer",
            "WordCountVectorizer"
        ]
        assert set(vectordbpy.__all__) == set(expected_exports)

    def test_end_to_end_workflow(self):
        """Test a complete end-to-end workflow using the package."""
        # Create documents
        doc1 = Document("This is about machine learning", {"topic": "ML"})
        doc2 = Document("This is about deep learning", {"topic": "DL"})
        doc3 = Document("This is about cooking recipes", {"topic": "cooking"})
        
        # Create vectors (simplified embeddings)
        vectors = [
            [1.0, 0.5, 0.0],  # ML-related
            [0.9, 0.6, 0.1],  # DL-related (similar to ML)
            [0.0, 0.1, 1.0]   # Cooking-related (different)
        ]
        
        # Create vector store and add documents
        store = MemoryVectorStore()
        store.add_documents([doc1, doc2, doc3], vectors)
        
        # Query for ML-related content
        query_vector = [1.0, 0.4, 0.0]  # Similar to ML vector
        results = store.query_vector(query_vector, k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0].page_content == "This is about machine learning"
        assert results[1].page_content == "This is about deep learning"
        
        # Test similarity function directly
        similarity = cosine_similarity_pure_python([1, 0, 0], [1, 0, 0])
        assert similarity == pytest.approx(1.0)

    def test_end_to_end_with_vectorization(self):
        """Test a complete end-to-end workflow with automatic vectorization."""
        from vectordbpy import TFIDFVectorizer, MemoryVectorStore, Document
        
        # Create a vectorizer and vector store
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        # Add documents using text input
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing deals with text analysis",
            "Computer vision processes and analyzes visual data"
        ]
        
        store.add_texts(texts)
        
        # Verify documents were added and vectorized
        assert len(store.documents) == 4
        assert len(store.vectors) == 4
        assert vectorizer.is_fitted
        
        # Test text-based querying
        results = store.query_text("artificial intelligence machine learning", k=2)
        assert len(results) == 2
        
        # First result should be the machine learning document
        assert "Machine learning" in results[0].page_content
        
        # Test adding more documents
        new_texts = ["Reinforcement learning trains agents through rewards"]
        store.add_texts(new_texts)
        
        assert len(store.documents) == 5
        assert len(store.vectors) == 5
        
        # Test querying again with updated store
        results = store.query_text("learning", k=3)
        assert len(results) == 3