"""Tests for vectorizers module."""

import pytest
import math
from vectordbpy.vectorizers import (
    BaseVectorizer, 
    TFIDFVectorizer, 
    BagOfWordsVectorizer, 
    WordCountVectorizer
)


class TestBaseVectorizer:
    """Test cases for BaseVectorizer abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseVectorizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorizer()
    
    def test_tokenize_method(self):
        """Test the _tokenize method functionality."""
        # Create a concrete implementation for testing
        class ConcreteVectorizer(BaseVectorizer):
            def fit(self, documents):
                return self
            def transform(self, documents):
                return []
        
        vectorizer = ConcreteVectorizer()
        
        # Test basic tokenization
        tokens = vectorizer._tokenize("Hello world!")
        assert tokens == ["hello", "world"]
        
        # Test with punctuation and numbers
        tokens = vectorizer._tokenize("Hello, world! This is test123.")
        assert tokens == ["hello", "world", "this", "is", "test123"]
        
        # Test with empty string
        tokens = vectorizer._tokenize("")
        assert tokens == []
        
        # Test with only punctuation
        tokens = vectorizer._tokenize("!@#$%")
        assert tokens == []


class TestWordCountVectorizer:
    """Test cases for WordCountVectorizer class."""
    
    def test_fit_and_transform_basic(self):
        """Test basic fit and transform functionality."""
        vectorizer = WordCountVectorizer()
        documents = ["hello world", "world python", "hello python"]
        
        # Test fit
        vectorizer.fit(documents)
        assert vectorizer.is_fitted
        assert len(vectorizer.vocabulary) == 3
        assert "hello" in vectorizer.vocabulary
        assert "world" in vectorizer.vocabulary
        assert "python" in vectorizer.vocabulary
        
        # Test transform
        vectors = vectorizer.transform(documents)
        assert len(vectors) == 3
        assert len(vectors[0]) == 3
        
        # Check specific values
        hello_idx = vectorizer.vocabulary["hello"]
        world_idx = vectorizer.vocabulary["world"]
        python_idx = vectorizer.vocabulary["python"]
        
        # First document: "hello world"
        assert vectors[0][hello_idx] == 1.0
        assert vectors[0][world_idx] == 1.0
        assert vectors[0][python_idx] == 0.0
        
        # Second document: "world python"
        assert vectors[1][hello_idx] == 0.0
        assert vectors[1][world_idx] == 1.0
        assert vectors[1][python_idx] == 1.0
    
    def test_fit_transform_shortcut(self):
        """Test fit_transform method."""
        vectorizer = WordCountVectorizer()
        documents = ["hello world", "world python"]
        
        vectors = vectorizer.fit_transform(documents)
        assert vectorizer.is_fitted
        assert len(vectors) == 2
        assert len(vectors[0]) == 3  # hello, python, world (sorted)
    
    def test_repeated_words(self):
        """Test handling of repeated words in documents."""
        vectorizer = WordCountVectorizer()
        documents = ["hello hello world", "world world world"]
        
        vectors = vectorizer.fit_transform(documents)
        
        hello_idx = vectorizer.vocabulary["hello"]
        world_idx = vectorizer.vocabulary["world"]
        
        # First document: "hello hello world" -> hello:2, world:1
        assert vectors[0][hello_idx] == 2.0
        assert vectors[0][world_idx] == 1.0
        
        # Second document: "world world world" -> hello:0, world:3
        assert vectors[1][hello_idx] == 0.0
        assert vectors[1][world_idx] == 3.0
    
    def test_empty_documents(self):
        """Test handling of empty documents."""
        vectorizer = WordCountVectorizer()
        documents = ["hello world", "", "python"]
        
        vectors = vectorizer.fit_transform(documents)
        assert len(vectors) == 3
        
        # Empty document should have all zeros
        assert all(val == 0.0 for val in vectors[1])
    
    def test_transform_before_fit_raises_error(self):
        """Test that transform raises error before fitting."""
        vectorizer = WordCountVectorizer()
        with pytest.raises(ValueError, match="Vectorizer must be fitted before transform"):
            vectorizer.transform(["hello world"])
    
    def test_case_insensitive(self):
        """Test that vectorizer is case insensitive."""
        vectorizer = WordCountVectorizer()
        documents = ["Hello World", "hello WORLD"]
        
        vectors = vectorizer.fit_transform(documents)
        
        # Both documents should have identical vectors
        assert vectors[0] == vectors[1]


class TestBagOfWordsVectorizer:
    """Test cases for BagOfWordsVectorizer class."""
    
    def test_count_mode(self):
        """Test bag of words in count mode (default)."""
        vectorizer = BagOfWordsVectorizer(binary=False)
        documents = ["hello hello world", "world python"]
        
        vectors = vectorizer.fit_transform(documents)
        
        hello_idx = vectorizer.vocabulary["hello"]
        world_idx = vectorizer.vocabulary["world"]
        python_idx = vectorizer.vocabulary["python"]
        
        # First document: "hello hello world" -> hello:2, world:1, python:0
        assert vectors[0][hello_idx] == 2.0
        assert vectors[0][world_idx] == 1.0
        assert vectors[0][python_idx] == 0.0
    
    def test_binary_mode(self):
        """Test bag of words in binary mode."""
        vectorizer = BagOfWordsVectorizer(binary=True)
        documents = ["hello hello world", "world python"]
        
        vectors = vectorizer.fit_transform(documents)
        
        hello_idx = vectorizer.vocabulary["hello"]
        world_idx = vectorizer.vocabulary["world"]
        python_idx = vectorizer.vocabulary["python"]
        
        # First document: "hello hello world" -> hello:1, world:1, python:0 (binary)
        assert vectors[0][hello_idx] == 1.0
        assert vectors[0][world_idx] == 1.0
        assert vectors[0][python_idx] == 0.0
        
        # Second document: "world python" -> hello:0, world:1, python:1
        assert vectors[1][hello_idx] == 0.0
        assert vectors[1][world_idx] == 1.0
        assert vectors[1][python_idx] == 1.0
    
    def test_default_mode_is_count(self):
        """Test that default mode is count (not binary)."""
        vectorizer = BagOfWordsVectorizer()
        documents = ["hello hello world"]
        
        vectors = vectorizer.fit_transform(documents)
        hello_idx = vectorizer.vocabulary["hello"]
        
        # Should be 2 (count mode), not 1 (binary mode)
        assert vectors[0][hello_idx] == 2.0


class TestTFIDFVectorizer:
    """Test cases for TFIDFVectorizer class."""
    
    def test_basic_tfidf(self):
        """Test basic TF-IDF computation."""
        vectorizer = TFIDFVectorizer()
        documents = [
            "hello world",
            "hello python", 
            "world python"
        ]
        
        vectors = vectorizer.fit_transform(documents)
        assert len(vectors) == 3
        assert vectorizer.is_fitted
        
        # Check that we have IDF values
        assert len(vectorizer.idf_values) == 3
        assert "hello" in vectorizer.idf_values
        assert "world" in vectorizer.idf_values
        assert "python" in vectorizer.idf_values
    
    def test_idf_calculation(self):
        """Test IDF calculation with smooth_idf=False for exact values."""
        vectorizer = TFIDFVectorizer(smooth_idf=False)
        documents = [
            "hello world",     # hello appears in 2/3 docs
            "hello python",    # world appears in 2/3 docs  
            "world python"     # python appears in 2/3 docs
        ]
        
        vectorizer.fit(documents)
        
        # All words appear in 2 out of 3 documents
        # IDF = log(3/2) = log(1.5) ≈ 0.405
        expected_idf = math.log(3/2)
        
        assert vectorizer.idf_values["hello"] == pytest.approx(expected_idf)
        assert vectorizer.idf_values["world"] == pytest.approx(expected_idf)
        assert vectorizer.idf_values["python"] == pytest.approx(expected_idf)
    
    def test_tf_calculation(self):
        """Test term frequency calculation."""
        vectorizer = TFIDFVectorizer(use_idf=False)  # Only TF, no IDF
        documents = ["hello hello world"]  # hello appears 2/3 times, world 1/3 times
        
        vectors = vectorizer.fit_transform(documents)
        
        hello_idx = vectorizer.vocabulary["hello"]
        world_idx = vectorizer.vocabulary["world"]
        
        # TF(hello) = 2/3, TF(world) = 1/3
        assert vectors[0][hello_idx] == pytest.approx(2/3)
        assert vectors[0][world_idx] == pytest.approx(1/3)
    
    def test_smooth_idf(self):
        """Test smooth IDF calculation."""
        vectorizer = TFIDFVectorizer(smooth_idf=True)
        documents = ["hello world", "hello python", "world python"]
        
        vectorizer.fit(documents)
        
        # With smooth_idf: IDF = log((num_docs + 1) / (doc_freq + 1))
        # All words appear in 2/3 docs, so IDF = log((3+1)/(2+1)) = log(4/3)
        expected_idf = math.log(4/3)
        
        assert vectorizer.idf_values["hello"] == pytest.approx(expected_idf)
        assert vectorizer.idf_values["world"] == pytest.approx(expected_idf)
        assert vectorizer.idf_values["python"] == pytest.approx(expected_idf)
    
    def test_use_idf_false(self):
        """Test TF-IDF with use_idf=False (only term frequency)."""
        vectorizer = TFIDFVectorizer(use_idf=False)
        documents = ["hello hello world"]
        
        vectors = vectorizer.fit_transform(documents)
        
        hello_idx = vectorizer.vocabulary["hello"]
        world_idx = vectorizer.vocabulary["world"]
        
        # Should be just TF values
        assert vectors[0][hello_idx] == pytest.approx(2/3)
        assert vectors[0][world_idx] == pytest.approx(1/3)
        
        # IDF values should not be computed
        assert not vectorizer.idf_values
    
    def test_empty_document(self):
        """Test handling of empty documents."""
        vectorizer = TFIDFVectorizer()
        documents = ["hello world", "", "python"]
        
        vectors = vectorizer.fit_transform(documents)
        
        # Empty document should have all zeros
        assert all(val == 0.0 for val in vectors[1])
    
    def test_single_word_document(self):
        """Test document with single word."""
        vectorizer = TFIDFVectorizer(use_idf=False)
        documents = ["hello"]
        
        vectors = vectorizer.fit_transform(documents)
        
        hello_idx = vectorizer.vocabulary["hello"]
        
        # Single word document: TF = 1/1 = 1.0
        assert vectors[0][hello_idx] == pytest.approx(1.0)


class TestVectorizersIntegration:
    """Integration tests for all vectorizers."""
    
    def test_all_vectorizers_same_interface(self):
        """Test that all vectorizers implement the same interface."""
        documents = ["hello world", "world python", "hello python"]
        
        vectorizers = [
            WordCountVectorizer(),
            BagOfWordsVectorizer(),
            TFIDFVectorizer()
        ]
        
        for vectorizer in vectorizers:
            # Test fit method
            vectorizer.fit(documents)
            assert vectorizer.is_fitted
            
            # Test transform method
            vectors = vectorizer.transform(documents)
            assert len(vectors) == 3
            assert all(len(v) == len(vectorizer.vocabulary) for v in vectors)
            
            # Test fit_transform method
            vectors2 = type(vectorizer)().fit_transform(documents)
            assert len(vectors2) == 3
    
    def test_vocabulary_consistency(self):
        """Test that vocabulary is built consistently across vectorizers."""
        documents = ["hello world", "world python"]
        
        vectorizers = [
            WordCountVectorizer(),
            BagOfWordsVectorizer(),
            TFIDFVectorizer()
        ]
        
        vocabularies = []
        for vectorizer in vectorizers:
            vectorizer.fit(documents)
            vocabularies.append(set(vectorizer.vocabulary.keys()))
        
        # All vocabularies should be the same
        assert len(set(tuple(vocab) for vocab in vocabularies)) == 1
        
        # Should contain expected words
        expected_words = {"hello", "world", "python"}
        assert vocabularies[0] == expected_words
    
    def test_different_vector_values(self):
        """Test that different vectorizers produce different vector values."""
        documents = ["hello hello world", "world python"]
        
        word_count = WordCountVectorizer().fit_transform(documents)
        bow_binary = BagOfWordsVectorizer(binary=True).fit_transform(documents)
        tfidf = TFIDFVectorizer().fit_transform(documents)
        
        # Vectors should be different between methods
        assert word_count != bow_binary
        assert word_count != tfidf
        assert bow_binary != tfidf
        
        # But should have same dimensions
        assert len(word_count[0]) == len(bow_binary[0]) == len(tfidf[0])


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_punctuation_handling(self):
        """Test handling of punctuation and special characters."""
        vectorizer = WordCountVectorizer()
        documents = ["Hello, world!", "World & Python?", "Python-3.9"]
        
        vectors = vectorizer.fit_transform(documents)
        
        # Should normalize and handle punctuation
        expected_words = {"hello", "world", "python", "python39"}  # Python-3.9 becomes python39
        assert set(vectorizer.vocabulary.keys()) == expected_words
    
    def test_numbers_in_text(self):
        """Test handling of numbers in text."""
        vectorizer = WordCountVectorizer()
        documents = ["python 3.9", "version 123", "test 456 789"]
        
        vectorizer.fit(documents)
        
        # Numbers should be preserved as tokens (punctuation removed)
        assert "python" in vectorizer.vocabulary
        assert "39" in vectorizer.vocabulary  # 3.9 becomes 39
        assert "version" in vectorizer.vocabulary
        assert "123" in vectorizer.vocabulary
        assert "test" in vectorizer.vocabulary
        assert "456" in vectorizer.vocabulary
        assert "789" in vectorizer.vocabulary
    
    def test_whitespace_handling(self):
        """Test handling of various whitespace characters."""
        vectorizer = WordCountVectorizer()
        documents = ["hello\tworld", "python\n\ntest", "  extra  spaces  "]
        
        vectors = vectorizer.fit_transform(documents)
        
        # Should handle different whitespace properly
        expected_words = {"hello", "world", "python", "test", "extra", "spaces"}
        assert set(vectorizer.vocabulary.keys()) == expected_words
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        vectorizer = WordCountVectorizer()
        documents = ["café python", "résumé test", "naïve approach"]
        
        # Should handle unicode (though our simple regex might filter it out)
        vectors = vectorizer.fit_transform(documents)
        
        # The simple regex in _tokenize removes non-ASCII, so these should be filtered
        assert "python" in vectorizer.vocabulary
        assert "test" in vectorizer.vocabulary
        assert "approach" in vectorizer.vocabulary