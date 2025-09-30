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
        expected_exports = ["Document", "MemoryVectorStore", "cosine_similarity_pure_python"]
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