"""
vectordbpy - In-memory vector database package

This package provides an in-memory vector database solution for Python applications.
"""

from .vectordb import Document, MemoryVectorStore
from .similarity import cosine_similarity_pure_python
from .vectorizers import (
    BaseVectorizer,
    TFIDFVectorizer,
    BagOfWordsVectorizer,
    WordCountVectorizer
)

__version__ = "0.0.1"
__all__ = [
    "Document",
    "MemoryVectorStore",
    "cosine_similarity_pure_python",
    "BaseVectorizer",
    "TFIDFVectorizer",
    "BagOfWordsVectorizer",
    "WordCountVectorizer"
]
