#!/usr/bin/env python3
"""
Example usage of vectordb.py with automatic vectorization.

This script demonstrates how to use the four vectorization models
(TF-IDF, Bag of Words, Word Count, and Qwen Embedding) with the vectordb.py library.
"""

from vectordbpy import (
    MemoryVectorStore,
    TFIDFVectorizer,
    BagOfWordsVectorizer,
    WordCountVectorizer,
    QwenEmbeddingVectorizer,
    Document
)


def main():
    """Demonstrate vectordb.py vectorization features."""
    print("=== VectorDB.py Vectorization Example ===\n")

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with text analysis",
        "Computer vision processes and analyzes visual data",
        "Reinforcement learning trains agents through rewards",
        "Data science combines statistics and programming"
    ]

    print("Sample documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    print()

    # Demonstrate TF-IDF Vectorizer
    print("=== TF-IDF Vectorizer ===")
    tfidf_vectorizer = TFIDFVectorizer()
    tfidf_store = MemoryVectorStore(vectorizer=tfidf_vectorizer)
    tfidf_store.add_texts(documents)

    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary)}")
    print(f"Documents stored: {len(tfidf_store.documents)}")

    # Query with TF-IDF
    query = "machine learning artificial intelligence"
    results = tfidf_store.query_text(query, k=3)
    print(f"\nQuery: '{query}'")
    print("Top 3 results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    print()

    # Demonstrate Bag of Words Vectorizer
    print("=== Bag of Words Vectorizer (Binary) ===")
    bow_vectorizer = BagOfWordsVectorizer(binary=True)
    bow_store = MemoryVectorStore(vectorizer=bow_vectorizer)
    bow_store.add_texts(documents)

    query = "data processing"
    results = bow_store.query_text(query, k=2)
    print(f"Query: '{query}'")
    print("Top 2 results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    print()

    # Demonstrate Word Count Vectorizer
    print("=== Word Count Vectorizer ===")
    wc_vectorizer = WordCountVectorizer()
    wc_store = MemoryVectorStore(vectorizer=wc_vectorizer)
    wc_store.add_texts(documents)

    query = "learning"
    results = wc_store.query_text(query, k=3)
    print(f"Query: '{query}'")
    print("Top 3 results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    print()

    # Demonstrate Qwen Embedding Vectorizer
    print("=== Qwen Embedding Vectorizer ===")
    qwen_vectorizer = QwenEmbeddingVectorizer(embedding_dim=512)
    qwen_store = MemoryVectorStore(vectorizer=qwen_vectorizer)
    qwen_store.add_texts(documents)

    query = "artificial intelligence"
    results = qwen_store.query_text(query, k=3)
    print(f"Query: '{query}'")
    print("Top 3 results (dense embeddings):")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
    
    print(f"Vector dimension: {qwen_vectorizer.embedding_dim}")
    print(f"Vocabulary size: {len(qwen_vectorizer.vocabulary)}")
    print()

    # Demonstrate adding documents with metadata
    print("=== Adding Documents with Metadata ===")
    store_with_metadata = MemoryVectorStore(vectorizer=TFIDFVectorizer())

    # Add documents with metadata
    docs_with_metadata = [
        Document("Python is a programming language", {"category": "programming", "difficulty": "beginner"}),
        Document("JavaScript runs in web browsers", {"category": "programming", "difficulty": "intermediate"}),
        Document("Machine learning requires large datasets", {"category": "AI", "difficulty": "advanced"})
    ]

    store_with_metadata.add_documents(docs_with_metadata)

    query = "programming"
    results = store_with_metadata.query_text(query, k=2)
    print(f"Query: '{query}'")
    print("Results with metadata:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content}")
        print(f"     Metadata: {doc.metadata}")
    print()

    print("=== Vocabulary Comparison ===")
    print(f"TF-IDF vocabulary: {sorted(list(tfidf_vectorizer.vocabulary.keys())[:10])}...")
    print(f"Bag of Words vocabulary: {sorted(list(bow_vectorizer.vocabulary.keys())[:10])}...")
    print(f"Word Count vocabulary: {sorted(list(wc_vectorizer.vocabulary.keys())[:10])}...")
    print(f"Qwen Embedding vocabulary: {sorted(list(qwen_vectorizer.vocabulary.keys())[:10])}...")
    print("\nAll vectorizers produce the same vocabulary but different vector representations!")
    print(f"Traditional vectorizers produce sparse vectors (vocab size: {len(tfidf_vectorizer.vocabulary)})")
    print(f"Qwen Embedding produces dense vectors (fixed size: {qwen_vectorizer.embedding_dim})")


if __name__ == "__main__":
    main()