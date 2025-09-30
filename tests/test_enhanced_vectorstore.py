"""Tests for enhanced MemoryVectorStore functionality with vectorization."""

import pytest
from vectordbpy.vectordb import Document, MemoryVectorStore
from vectordbpy.vectorizers import TFIDFVectorizer, BagOfWordsVectorizer, WordCountVectorizer


class TestMemoryVectorStoreWithVectorizers:
    """Test cases for MemoryVectorStore with automatic vectorization."""
    
    def test_init_with_vectorizer(self):
        """Test initialization with a vectorizer."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        assert store.vectorizer is vectorizer
        assert len(store.documents) == 0
        assert len(store.vectors) == 0
        assert len(store.metadatas) == 0
    
    def test_add_documents_with_vectorizer(self):
        """Test adding documents with automatic vectorization."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        documents = [
            Document("hello world", {"id": 1}),
            Document("world python", {"id": 2}),
            Document("hello python", {"id": 3})
        ]
        
        # Add documents without providing vectors
        store.add_documents(documents)
        
        # Check that documents were added
        assert len(store.documents) == 3
        assert len(store.vectors) == 3
        assert len(store.metadatas) == 3
        
        # Check that vectorizer was fitted
        assert vectorizer.is_fitted
        assert len(vectorizer.vocabulary) == 3  # hello, world, python
        
        # Check that vectors were generated
        for vector in store.vectors:
            assert len(vector) == 3  # vocabulary size
            assert isinstance(vector, list)
            assert all(isinstance(val, float) for val in vector)
    
    def test_add_texts_method(self):
        """Test the add_texts convenience method."""
        vectorizer = WordCountVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = ["hello world", "world python", "hello python"]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        store.add_texts(texts, metadatas)
        
        # Check that texts were converted to documents and added
        assert len(store.documents) == 3
        assert all(isinstance(doc, Document) for doc in store.documents)
        assert store.documents[0].page_content == "hello world"
        assert store.documents[0].metadata == {"id": 1}
        
        # Check vectors were generated
        assert len(store.vectors) == 3
        assert all(len(vector) == 3 for vector in store.vectors)  # vocab size
    
    def test_add_texts_without_metadata(self):
        """Test add_texts without providing metadata."""
        vectorizer = WordCountVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = ["hello world", "world python"]
        store.add_texts(texts)
        
        # Should use empty metadata
        assert len(store.documents) == 2
        assert all(doc.metadata == {} for doc in store.documents)
    
    def test_query_text_method(self):
        """Test querying with text input."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        # Add some documents
        texts = [
            "machine learning algorithms",
            "deep learning neural networks", 
            "cooking recipes and food",
            "machine learning models"
        ]
        store.add_texts(texts)
        
        # Query with text
        results = store.query_text("machine learning", k=2)
        
        assert len(results) == 2
        # Should return documents most similar to "machine learning"
        assert "machine learning" in results[0].page_content
        assert "machine learning" in results[1].page_content
    
    def test_query_text_without_vectorizer_raises_error(self):
        """Test that query_text raises error without vectorizer."""
        store = MemoryVectorStore()
        
        with pytest.raises(ValueError, match="Vectorizer must be set to query with text"):
            store.query_text("hello world")
    
    def test_query_text_before_fitting_raises_error(self):
        """Test that query_text raises error before vectorizer is fitted."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        with pytest.raises(ValueError, match="Vectorizer must be fitted before querying"):
            store.query_text("hello world")
    
    def test_add_documents_without_vectorizer_and_vectors_raises_error(self):
        """Test that adding documents without vectorizer and vectors raises error."""
        store = MemoryVectorStore()
        documents = [Document("hello world")]
        
        with pytest.raises(ValueError, match="Either vectors must be provided or vectorizer must be set"):
            store.add_documents(documents)
    
    def test_add_documents_with_existing_documents_refits_vectorizer(self):
        """Test that adding documents to existing store refits vectorizer."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        # Add initial documents
        initial_docs = [Document("hello world"), Document("world python")]
        store.add_documents(initial_docs)
        
        initial_vocab_size = len(vectorizer.vocabulary)
        initial_vectors = [v.copy() for v in store.vectors]
        
        # Add new documents with completely new vocabulary
        new_docs = [Document("machine learning"), Document("artificial intelligence")]
        store.add_documents(new_docs)
        
        # Vocabulary should have expanded (4 new words: machine, learning, artificial, intelligence)
        assert len(vectorizer.vocabulary) > initial_vocab_size
        
        # All vectors should be recomputed and have new dimensions
        assert len(store.vectors) == 4
        for vector in store.vectors:
            # Vectors should have dimensions matching new vocabulary size
            assert len(vector) == len(vectorizer.vocabulary)
    
    def test_mixed_document_types(self):
        """Test handling of mixed document types (Document objects and strings)."""
        vectorizer = WordCountVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        # Mix of Document objects and plain strings
        documents = [
            Document("hello world", {"type": "document"}),
            "plain string text",  # This should be handled gracefully
            Document("python programming", {"type": "document"})
        ]
        
        store.add_documents(documents)
        
        assert len(store.documents) == 3
        assert len(store.vectors) == 3
        
        # Check that string was handled correctly
        assert store.documents[1] == "plain string text"
    
    def test_add_texts_mismatched_metadata_length_raises_error(self):
        """Test that mismatched texts and metadata lengths raise error."""
        vectorizer = WordCountVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = ["hello world", "python programming"]
        metadatas = [{"id": 1}]  # Only one metadata for two texts
        
        with pytest.raises(ValueError, match="Number of metadatas must match number of texts"):
            store.add_texts(texts, metadatas)
    
    def test_backward_compatibility_with_explicit_vectors(self):
        """Test that explicit vectors still work with vectorizer set."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        documents = [Document("hello world"), Document("world python")]
        vectors = [[1.0, 0.0], [0.5, 1.0]]  # Explicit vectors
        
        # Should use provided vectors instead of computing them
        store.add_documents(documents, vectors=vectors)
        
        assert store.vectors == vectors
        assert len(store.documents) == 2
        
        # Vectorizer should not be fitted since we provided explicit vectors
        assert not vectorizer.is_fitted


class TestMemoryVectorStoreVectorizerIntegration:
    """Test integration between MemoryVectorStore and different vectorizers."""
    
    def test_with_tfidf_vectorizer(self):
        """Test MemoryVectorStore with TF-IDF vectorizer."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = [
            "machine learning is powerful",
            "deep learning uses neural networks",
            "cooking recipes are delicious",
            "machine learning algorithms"
        ]
        store.add_texts(texts)
        
        # Query for machine learning related content
        results = store.query_text("machine learning", k=2)
        
        assert len(results) == 2
        # Should return the two machine learning documents
        ml_texts = [doc.page_content for doc in results]
        assert any("machine learning is powerful" in text for text in ml_texts)
        assert any("machine learning algorithms" in text for text in ml_texts)
    
    def test_with_bag_of_words_vectorizer(self):
        """Test MemoryVectorStore with Bag of Words vectorizer."""
        vectorizer = BagOfWordsVectorizer(binary=True)
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = ["hello world", "hello python", "world python"]
        store.add_texts(texts)
        
        # Query with text that shares words
        results = store.query_text("hello", k=2)
        
        assert len(results) == 2
        # Should return documents containing "hello"
        result_texts = [doc.page_content for doc in results]
        assert all("hello" in text for text in result_texts)
    
    def test_with_word_count_vectorizer(self):
        """Test MemoryVectorStore with Word Count vectorizer."""
        vectorizer = WordCountVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = [
            "python python python",  # High count of "python"
            "python programming", 
            "hello world"
        ]
        store.add_texts(texts)
        
        # Query for python
        results = store.query_text("python", k=2)
        
        assert len(results) == 2
        # Document with higher "python" count should rank higher
        assert "python python python" in results[0].page_content
    
    def test_vectorizer_consistency_across_queries(self):
        """Test that vectorizer produces consistent results across multiple queries."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = ["machine learning", "deep learning", "artificial intelligence"]
        store.add_texts(texts)
        
        # Multiple queries should be consistent
        results1 = store.query_text("learning", k=2)
        results2 = store.query_text("learning", k=2)
        
        assert len(results1) == len(results2)
        assert [doc.page_content for doc in results1] == [doc.page_content for doc in results2]
    
    def test_empty_query_handling(self):
        """Test handling of empty query text."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = ["hello world", "python programming"]
        store.add_texts(texts)
        
        # Empty query should work but may return all documents with zero similarity
        results = store.query_text("", k=2)
        assert len(results) <= 2  # Should return available documents
    
    def test_query_with_unseen_vocabulary(self):
        """Test querying with words not in the training vocabulary."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        texts = ["hello world", "python programming"]
        store.add_texts(texts)
        
        # Query with word not in vocabulary
        results = store.query_text("unseen word", k=2)
        
        # Should still return results (though with zero similarity for unseen words)
        assert len(results) <= 2


class TestVectorizerPerformance:
    """Test performance characteristics of vectorizers."""
    
    def test_large_vocabulary_handling(self):
        """Test handling of larger vocabularies."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        # Create documents with diverse vocabulary
        texts = []
        for i in range(50):
            text = f"document {i} contains unique words word{i} term{i}"
            texts.append(text)
        
        store.add_texts(texts)
        
        # Should handle large vocabulary
        assert len(vectorizer.vocabulary) > 100  # Many unique words
        assert len(store.vectors) == 50
        
        # Querying should still work
        results = store.query_text("document", k=5)
        assert len(results) == 5
    
    def test_incremental_addition(self):
        """Test incremental addition of documents."""
        vectorizer = TFIDFVectorizer()
        store = MemoryVectorStore(vectorizer=vectorizer)
        
        # Add documents incrementally
        for i in range(10):
            text = f"document {i} with content"
            store.add_texts([text])
        
        assert len(store.documents) == 10
        assert len(store.vectors) == 10
        
        # All vectors should have same dimension (vocabulary size)
        vocab_size = len(vectorizer.vocabulary)
        assert all(len(vector) == vocab_size for vector in store.vectors)