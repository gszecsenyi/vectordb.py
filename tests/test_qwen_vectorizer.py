"""Tests for QwenEmbeddingVectorizer."""

import pytest
from vectordbpy.vectorizers import QwenEmbeddingVectorizer


class TestQwenEmbeddingVectorizer:
    """Test cases for QwenEmbeddingVectorizer class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        vectorizer = QwenEmbeddingVectorizer()

        assert vectorizer.embedding_dim == 1024
        assert vectorizer.max_sequence_length == 512
        assert not vectorizer.is_fitted
        assert vectorizer.vocabulary == {}
        assert vectorizer.token_embeddings == {}
        assert vectorizer.positional_encodings == {}

    def test_init_custom_embedding_dim(self):
        """Test initialization with custom embedding dimension."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=768)

        assert vectorizer.embedding_dim == 768
        assert vectorizer.max_sequence_length == 512
        assert not vectorizer.is_fitted

    def test_fit_basic(self):
        """Test basic fit functionality."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)
        documents = ["hello world", "machine learning", "python programming"]

        vectorizer.fit(documents)

        assert vectorizer.is_fitted
        assert len(vectorizer.vocabulary) == 6  # hello, world, machine, learning, py...
        assert "hello" in vectorizer.vocabulary
        assert "world" in vectorizer.vocabulary
        assert "machine" in vectorizer.vocabulary
        assert "learning" in vectorizer.vocabulary
        assert "python" in vectorizer.vocabulary
        assert "programming" in vectorizer.vocabulary

        # Check token embeddings are generated
        for token in vectorizer.vocabulary:
            assert token in vectorizer.token_embeddings
            assert len(vectorizer.token_embeddings[token]) == 128

        # Check positional encodings are generated
        assert len(vectorizer.positional_encodings) == 512
        for pos in range(512):
            assert len(vectorizer.positional_encodings[pos]) == 128

    def test_transform_basic(self):
        """Test basic transform functionality."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=256)
        documents = ["hello world", "machine learning", "natural language processing"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        assert len(vectors) == 3
        for vector in vectors:
            assert len(vector) == 256
            assert all(isinstance(val, float) for val in vector)

    def test_fit_transform_shortcut(self):
        """Test fit_transform method."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=512)
        documents = ["hello world", "python programming"]

        vectors = vectorizer.fit_transform(documents)

        assert vectorizer.is_fitted
        assert len(vectors) == 2
        assert len(vectors[0]) == 512

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        vectorizer = QwenEmbeddingVectorizer()
        documents = ["hello world"]

        with pytest.raises(ValueError,
                           match="Vectorizer must be fitted before transform"):
            vectorizer.transform(documents)

    def test_empty_documents(self):
        """Test handling of empty documents."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)
        documents = ["hello world", "", "machine learning"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        assert len(vectors) == 3
        assert len(vectors[0]) == 128  # "hello world"
        assert len(vectors[1]) == 128  # empty document
        assert len(vectors[2]) == 128  # "machine learning"

        # Empty document should have zero vector
        assert all(val == 0.0 for val in vectors[1])

    def test_single_word_documents(self):
        """Test handling of single word documents."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=64)
        documents = ["hello", "world", "python"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        assert len(vectors) == 3
        for vector in vectors:
            assert len(vector) == 64
            # Single word documents should not be zero vectors
            assert not all(val == 0.0 for val in vector)

    def test_long_sequence_truncation(self):
        """Test that long sequences are truncated properly."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)

        # Create a very long document (more than 512 tokens)
        long_doc = " ".join([f"word{i}" for i in range(600)])
        documents = [long_doc, "short document"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        assert len(vectors) == 2
        assert len(vectors[0]) == 128
        assert len(vectors[1]) == 128

        # Should have handled the long document without error
        assert vectorizer.is_fitted

    def test_consistent_embeddings(self):
        """Test that same documents produce consistent embeddings."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=256)
        documents = ["machine learning", "artificial intelligence"]

        vectorizer.fit(documents)
        vectors1 = vectorizer.transform(documents)
        vectors2 = vectorizer.transform(documents)

        # Same documents should produce identical vectors
        for v1, v2 in zip(vectors1, vectors2):
            for val1, val2 in zip(v1, v2):
                assert abs(val1 - val2) < 1e-10

    def test_different_documents_different_vectors(self):
        """Test that different documents produce different vectors."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)
        documents = ["hello world", "goodbye moon", "machine learning"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        # Different documents should produce different vectors
        assert vectors[0] != vectors[1]
        assert vectors[1] != vectors[2]
        assert vectors[0] != vectors[2]

    def test_unknown_tokens_in_transform(self):
        """Test handling of unknown tokens during transform."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=64)

        # Fit on limited vocabulary
        train_docs = ["hello world", "python programming"]
        vectorizer.fit(train_docs)

        # Transform documents with unknown tokens
        test_docs = ["hello world", "machine learning", "unknown words everywhere"]
        vectors = vectorizer.transform(test_docs)

        assert len(vectors) == 3
        for vector in vectors:
            assert len(vector) == 64

        # Should handle unknown tokens gracefully (zero embeddings for unknown
        # tokens) But documents should still produce non-zero final vectors
        # due to positional encodings and attention mechanism

    def test_case_insensitive_tokenization(self):
        """Test that tokenization is case insensitive."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)
        documents = ["Hello World", "hello world", "HELLO WORLD"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        # All documents should produce identical vectors (case insensitive)
        for i in range(len(vectors) - 1):
            for val1, val2 in zip(vectors[i], vectors[i + 1]):
                assert abs(val1 - val2) < 1e-10

    def test_punctuation_handling(self):
        """Test handling of punctuation in documents."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=64)
        documents = ["hello, world!", "hello world", "hello... world???"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        # All documents should produce identical vectors (punctuation removed)
        for i in range(len(vectors) - 1):
            for val1, val2 in zip(vectors[i], vectors[i + 1]):
                assert abs(val1 - val2) < 1e-10


class TestQwenEmbeddingVectorizerEdgeCases:
    """Test edge cases for QwenEmbeddingVectorizer."""

    def test_empty_fit_documents(self):
        """Test fitting on empty document list."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)
        documents = []

        vectorizer.fit(documents)

        assert vectorizer.is_fitted
        assert len(vectorizer.vocabulary) == 0
        assert len(vectorizer.token_embeddings) == 0

    def test_transform_empty_document_list(self):
        """Test transforming empty document list."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)
        documents = ["hello world"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform([])

        assert vectors == []

    def test_fit_documents_with_only_empty_strings(self):
        """Test fitting on documents that are only empty strings."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=128)
        documents = ["", "", ""]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        assert vectorizer.is_fitted
        assert len(vectorizer.vocabulary) == 0
        assert len(vectors) == 3
        for vector in vectors:
            assert all(val == 0.0 for val in vector)

    def test_documents_with_only_punctuation(self):
        """Test documents containing only punctuation."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=64)
        documents = ["!@#$%", "...", "???", "hello world"]

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        assert len(vectors) == 4
        # First three should be zero vectors (only punctuation)
        for i in range(3):
            assert all(val == 0.0 for val in vectors[i])
        # Last one should be non-zero (has actual words)
        assert not all(val == 0.0 for val in vectors[3])


class TestQwenEmbeddingVectorizerIntegration:
    """Test integration with MemoryVectorStore."""

    def test_integration_with_memory_vector_store(self):
        """Test QwenEmbeddingVectorizer with MemoryVectorStore."""
        from vectordbpy import MemoryVectorStore

        vectorizer = QwenEmbeddingVectorizer(embedding_dim=256)
        store = MemoryVectorStore(vectorizer=vectorizer)

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

    def test_performance_with_large_vocabulary(self):
        """Test performance with large vocabulary."""
        vectorizer = QwenEmbeddingVectorizer(embedding_dim=512)

        # Create documents with diverse vocabulary
        documents = []
        for i in range(100):
            doc = " ".join([f"word{j}_{i}" for j in range(10)])
            documents.append(doc)

        vectorizer.fit(documents)
        vectors = vectorizer.transform(documents)

        assert len(vectors) == 100
        assert len(vectors[0]) == 512
        assert vectorizer.is_fitted
        assert len(vectorizer.vocabulary) == 1000  # 100 docs * 10 unique words each
