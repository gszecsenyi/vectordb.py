"""
Vectorization models for converting text input to vectors.

This module provides implementations of popular vectorization models
without external dependencies, using only Python's standard library.

Classes:
    BaseVectorizer: Abstract base class for all vectorizers
    TFIDFVectorizer: Term Frequency-Inverse Document Frequency vectorizer
    BagOfWordsVectorizer: Bag of Words vectorizer
    WordCountVectorizer: Simple word count vectorizer
    QwenEmbeddingVectorizer: Qwen3-Embedding-0.6B style dense embedding vectorizer
"""

import math
import re
from abc import ABC, abstractmethod
from typing import List


class BaseVectorizer(ABC):
    """
    Abstract base class for all vectorizers.

    This class defines the interface that all vectorizers must implement.
    """

    def __init__(self):
        """Initialize the vectorizer."""
        self.vocabulary = {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, documents: List[str]) -> 'BaseVectorizer':
        """
        Fit the vectorizer to the documents.

        Args:
            documents: List of text documents to fit on

        Returns:
            Self for method chaining
        """

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform documents to vectors.

        Args:
            documents: List of text documents to transform

        Returns:
            List of vectors (each vector is a list of floats)
        """

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """
        Fit the vectorizer and transform documents in one step.

        Args:
            documents: List of text documents

        Returns:
            List of vectors (each vector is a list of floats)
        """
        return self.fit(documents).transform(documents)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text string

        Returns:
            List of tokens (words)
        """
        # Simple tokenization: lowercase, remove non-alphanumeric,
        # split on whitespace
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 0]


class WordCountVectorizer(BaseVectorizer):
    """
    Simple word count vectorizer.

    Converts text to vectors based on word frequency counts.
    Each dimension represents a word in the vocabulary.
    """

    def fit(self, documents: List[str]) -> 'WordCountVectorizer':
        """
        Fit the vectorizer by building vocabulary from documents.

        Args:
            documents: List of text documents

        Returns:
            Self for method chaining
        """
        all_words = set()

        for doc in documents:
            tokens = self._tokenize(doc)
            all_words.update(tokens)

        # Create vocabulary mapping word -> index
        self.vocabulary = {word: idx for idx,
                           word in enumerate(sorted(all_words))}
        self.is_fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform documents to word count vectors.

        Args:
            documents: List of text documents

        Returns:
            List of word count vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        vectors = []
        vocab_size = len(self.vocabulary)

        for doc in documents:
            vector = [0.0] * vocab_size
            tokens = self._tokenize(doc)

            for token in tokens:
                if token in self.vocabulary:
                    vector[self.vocabulary[token]] += 1.0

            vectors.append(vector)

        return vectors


class BagOfWordsVectorizer(BaseVectorizer):
    """
    Bag of Words vectorizer.

    Similar to WordCountVectorizer but with optional binary representation
    (1 if word present, 0 if absent).
    """

    def __init__(self, binary: bool = False):
        """
        Initialize the Bag of Words vectorizer.

        Args:
            binary: If True, use binary representation (0/1), otherwise use
                counts
        """
        super().__init__()
        self.binary = binary

    def fit(self, documents: List[str]) -> 'BagOfWordsVectorizer':
        """
        Fit the vectorizer by building vocabulary from documents.

        Args:
            documents: List of text documents

        Returns:
            Self for method chaining
        """
        all_words = set()

        for doc in documents:
            tokens = self._tokenize(doc)
            all_words.update(tokens)

        # Create vocabulary mapping word -> index
        self.vocabulary = {word: idx for idx,
                           word in enumerate(sorted(all_words))}
        self.is_fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform documents to bag of words vectors.

        Args:
            documents: List of text documents

        Returns:
            List of bag of words vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        vectors = []
        vocab_size = len(self.vocabulary)

        for doc in documents:
            vector = [0.0] * vocab_size
            tokens = self._tokenize(doc)

            if self.binary:
                # Binary representation: 1 if word present, 0 otherwise
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    if token in self.vocabulary:
                        vector[self.vocabulary[token]] = 1.0
            else:
                # Count representation
                for token in tokens:
                    if token in self.vocabulary:
                        vector[self.vocabulary[token]] += 1.0

            vectors.append(vector)

        return vectors


class TFIDFVectorizer(BaseVectorizer):
    """
    Term Frequency-Inverse Document Frequency vectorizer.

    Implements TF-IDF algorithm without external dependencies.
    TF-IDF = TF(term, doc) * IDF(term, corpus)
    where:
    - TF = (number of times term appears in doc) / (total number of terms
      in doc)
    - IDF = log(total number of docs / number of docs containing term)
    """

    def __init__(self, use_idf: bool = True, smooth_idf: bool = True):
        """
        Initialize the TF-IDF vectorizer.

        Args:
            use_idf: If True, use IDF weighting, otherwise just TF
            smooth_idf: If True, add 1 to document frequencies to avoid
                division by zero
        """
        super().__init__()
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.idf_values = {}
        self.num_docs = 0

    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        Fit the vectorizer by building vocabulary and computing IDF values.

        Args:
            documents: List of text documents

        Returns:
            Self for method chaining
        """
        all_words = set()
        doc_word_counts = []

        # Tokenize all documents and build vocabulary
        for doc in documents:
            tokens = self._tokenize(doc)
            all_words.update(tokens)
            doc_word_counts.append(set(tokens))

        # Create vocabulary mapping word -> index
        self.vocabulary = {word: idx for idx,
                           word in enumerate(sorted(all_words))}
        self.num_docs = len(documents)

        # Compute IDF values
        if self.use_idf:
            for word in self.vocabulary:
                # Count documents containing this word
                doc_freq = sum(
                    1 for doc_words in doc_word_counts if word in doc_words)

                if self.smooth_idf:
                    # Add 1 to both numerator and denominator for smoothing
                    idf = math.log((self.num_docs + 1) / (doc_freq + 1))
                else:
                    idf = math.log(self.num_docs /
                                   doc_freq) if doc_freq > 0 else 0

                self.idf_values[word] = idf

        self.is_fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform documents to TF-IDF vectors.

        Args:
            documents: List of text documents

        Returns:
            List of TF-IDF vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        vectors = []
        vocab_size = len(self.vocabulary)

        for doc in documents:
            vector = [0.0] * vocab_size
            tokens = self._tokenize(doc)
            doc_length = len(tokens)

            if doc_length == 0:
                vectors.append(vector)
                continue

            # Count term frequencies
            term_counts = {}
            for token in tokens:
                if token in self.vocabulary:
                    term_counts[token] = term_counts.get(token, 0) + 1

            # Compute TF-IDF values
            for term, count in term_counts.items():
                tf = count / doc_length  # Term frequency

                if self.use_idf and term in self.idf_values:
                    tfidf = tf * self.idf_values[term]
                else:
                    tfidf = tf

                vector[self.vocabulary[term]] = tfidf

            vectors.append(vector)

        return vectors


class QwenEmbeddingVectorizer(BaseVectorizer):
    """
    Qwen3-Embedding-0.6B style vectorizer.

    This vectorizer simulates the behavior of the Qwen/Qwen3-Embedding-0.6B
    model using only Python standard library. It creates dense embeddings that
    incorporate contextual information and positional encoding similar to
    transformer models.
    """

    def __init__(self, embedding_dim: int = 1024):
        """
        Initialize the Qwen embedding vectorizer.

        Args:
            embedding_dim: Dimension of the output embedding vectors
                (default: 1024)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_embeddings = {}
        self.positional_encodings = {}
        self.max_sequence_length = 512

    def _generate_positional_encoding(self, position: int,
                                      d_model: int) -> List[float]:
        """
        Generate positional encoding for a given position.

        Args:
            position: Position in the sequence
            d_model: Model dimension

        Returns:
            Positional encoding vector
        """
        encoding = []
        for i in range(d_model):
            if i % 2 == 0:
                encoding.append(math.sin(position / (10000 ** (i / d_model))))
            else:
                encoding.append(math.cos(position / (10000 ** ((i-1) /
                                                               d_model))))
        return encoding

    def _generate_token_embedding(self, token: str) -> List[float]:
        """
        Generate a dense embedding for a token using a hash-based approach.

        Args:
            token: Input token

        Returns:
            Dense embedding vector
        """
        # Use hash function to generate consistent embeddings for tokens
        import hashlib

        # Create multiple hash values for the token to generate embedding
        # dimensions
        embedding = []
        for i in range(self.embedding_dim):
            # Create different hash inputs by appending dimension index
            hash_input = f"{token}_{i}".encode('utf-8')
            hash_value = hashlib.md5(hash_input).hexdigest()
            # Convert hex to float in range [-1, 1]
            normalized_value = (int(hash_value[:8], 16) % 2000000 -
                                1000000) / 1000000.0
            embedding.append(normalized_value)

        return embedding

    def _apply_attention_mechanism(self, token_embeddings: List[List[float]],
                                   position_embeddings: List[List[float]]
                                   ) -> List[float]:
        """
        Apply a simplified attention mechanism to combine token and positional
        embeddings.

        Args:
            token_embeddings: List of token embedding vectors
            position_embeddings: List of positional embedding vectors

        Returns:
            Final document embedding vector
        """
        if not token_embeddings:
            return [0.0] * self.embedding_dim

        # Combine token and positional embeddings
        combined_embeddings = []
        for i, (token_emb, pos_emb) in enumerate(zip(token_embeddings,
                                                     position_embeddings)):
            combined = [t + p for t, p in zip(token_emb, pos_emb)]
            combined_embeddings.append(combined)

        # Apply simplified self-attention pooling
        # Calculate attention weights based on embedding norms
        attention_weights = []
        for emb in combined_embeddings:
            # Attention weight based on L2 norm
            norm = math.sqrt(sum(x * x for x in emb))
            attention_weights.append(norm)

        # Normalize attention weights
        total_weight = sum(attention_weights)
        if total_weight > 0:
            attention_weights = [w / total_weight for w in attention_weights]
        else:
            num_embeddings = len(combined_embeddings)
            attention_weights = [1.0 / num_embeddings] * num_embeddings

        # Weighted average of embeddings
        final_embedding = [0.0] * self.embedding_dim
        for emb, weight in zip(combined_embeddings, attention_weights):
            for i, val in enumerate(emb):
                final_embedding[i] += val * weight

        # Apply layer normalization
        mean = sum(final_embedding) / len(final_embedding)
        variance = sum((x - mean) ** 2 for x in final_embedding) / len(
            final_embedding)
        std = math.sqrt(variance + 1e-8)  # Add small epsilon for stability

        normalized_embedding = [(x - mean) / std for x in final_embedding]

        return normalized_embedding

    def fit(self, documents: List[str]) -> 'QwenEmbeddingVectorizer':
        """
        Fit the vectorizer by building vocabulary and token embeddings.

        Args:
            documents: List of text documents to fit on

        Returns:
            Self for method chaining
        """
        # Build vocabulary
        all_tokens = set()
        for doc in documents:
            tokens = self._tokenize(doc)
            all_tokens.update(tokens)

        # Create vocabulary mapping
        self.vocabulary = {token: idx for idx, token in enumerate(
            sorted(all_tokens))}

        # Generate embeddings for all tokens
        for token in self.vocabulary:
            self.token_embeddings[token] = self._generate_token_embedding(
                token)

        # Pre-compute positional encodings
        for pos in range(self.max_sequence_length):
            self.positional_encodings[pos] = (
                self._generate_positional_encoding(pos, self.embedding_dim))

        self.is_fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform documents to Qwen-style embedding vectors.

        Args:
            documents: List of text documents to transform

        Returns:
            List of dense embedding vectors
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        vectors = []

        for doc in documents:
            tokens = self._tokenize(doc)

            if not tokens:
                # Empty document gets zero vector
                vectors.append([0.0] * self.embedding_dim)
                continue

            # Limit sequence length
            tokens = tokens[:self.max_sequence_length]

            # Get token embeddings
            token_embeddings = []
            for token in tokens:
                if token in self.token_embeddings:
                    token_embeddings.append(self.token_embeddings[token])
                else:
                    # Unknown token gets zero embedding
                    token_embeddings.append([0.0] * self.embedding_dim)

            # Get positional embeddings
            position_embeddings = []
            for i in range(len(tokens)):
                position_embeddings.append(self.positional_encodings[i])

            # Apply attention mechanism to get final embedding
            final_embedding = self._apply_attention_mechanism(
                token_embeddings, position_embeddings)
            vectors.append(final_embedding)

        return vectors
