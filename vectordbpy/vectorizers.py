"""
Vectorization models for converting text input to vectors.

This module provides implementations of popular vectorization models
without external dependencies, using only Python's standard library.

Classes:
    BaseVectorizer: Abstract base class for all vectorizers
    TFIDFVectorizer: Term Frequency-Inverse Document Frequency vectorizer
    BagOfWordsVectorizer: Bag of Words vectorizer
    WordCountVectorizer: Simple word count vectorizer
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
