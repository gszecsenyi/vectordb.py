# vectordb.py package

`vectordb.py` is a Python library that streamlines vector database operations. It is an in-memory solution, so all vectors and documents are stored in RAM during program execution. This makes it extremely fast and removes the need to run or maintain a persistent vector store (such as Pinecone, Weaviate, Milvus, etc.).

It offers user-friendly APIs for storing, querying, and managing high-dimensional vectors, making it well-suited for machine learning, information retrieval, and AI-driven search applications.

`vectordb.py` can also be used efficiently for promptflow queries, enabling fast and flexible retrieval in LLM and prompt engineering workflows.

## Features

With `vectordb.py`, you can:
- Efficiently store and organize large collections of vector data
- Perform fast similarity searches and retrieve relevant results
- Manage and update vector datasets with ease
- Integrate vector search capabilities into your Python projects
- **Automatically vectorize text using built-in vectorization models**

## Supported Vectorization Models

`vectordb.py` includes three popular vectorization models that work without any external dependencies:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Industry-standard text vectorization that weighs terms by their importance in documents vs. the entire corpus
2. **Bag of Words**: Simple and effective word frequency-based vectorization with optional binary representation
3. **Word Count**: Basic frequency counting vectorization for straightforward text analysis

All vectorizers are implemented using only Python's standard library for maximum compatibility and minimal dependencies.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install package_name

```bash
pip install package_name
```

## Usage

### Basic Usage with Explicit Vectors

```python
from vectordbpy import Document, MemoryVectorStore

# Create documents and vectors manually
doc1 = Document("This is about machine learning", {"topic": "ML"})
doc2 = Document("This is about deep learning", {"topic": "DL"})
doc3 = Document("This is about cooking recipes", {"topic": "cooking"})

vectors = [
    [1.0, 0.5, 0.0],  # ML-related
    [0.9, 0.6, 0.1],  # DL-related (similar to ML)
    [0.0, 0.1, 1.0]   # Cooking-related (different)
]

# Create vector store and add documents
store = MemoryVectorStore()
store.add_documents([doc1, doc2, doc3], vectors)

# Query for similar content
query_vector = [1.0, 0.4, 0.0]  # Similar to ML vector
results = store.query_vector(query_vector, k=2)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### Automatic Vectorization with TF-IDF

```python
from vectordbpy import MemoryVectorStore, TFIDFVectorizer

# Create a vector store with TF-IDF vectorization
vectorizer = TFIDFVectorizer()
store = MemoryVectorStore(vectorizer=vectorizer)

# Add texts directly - they'll be automatically vectorized
texts = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing deals with text analysis",
    "Computer vision processes and analyzes visual data"
]

store.add_texts(texts)

# Query using text - it will be automatically vectorized
results = store.query_text("artificial intelligence machine learning", k=2)

for doc in results:
    print(f"Result: {doc.page_content}")
```

### Using Different Vectorizers

```python
from vectordbpy import (
    MemoryVectorStore, 
    TFIDFVectorizer, 
    BagOfWordsVectorizer, 
    WordCountVectorizer
)

# TF-IDF Vectorizer (recommended for most text applications)
tfidf_store = MemoryVectorStore(vectorizer=TFIDFVectorizer())

# Bag of Words with binary representation
bow_store = MemoryVectorStore(vectorizer=BagOfWordsVectorizer(binary=True))

# Simple word count vectorizer
wc_store = MemoryVectorStore(vectorizer=WordCountVectorizer())

# All can be used the same way
texts = ["example text", "another document"]
tfidf_store.add_texts(texts)
bow_store.add_texts(texts)
wc_store.add_texts(texts)

# Query all stores
results1 = tfidf_store.query_text("example", k=1)
results2 = bow_store.query_text("example", k=1)
results3 = wc_store.query_text("example", k=1)
```

### Adding Documents with Metadata

```python
from vectordbpy import Document, MemoryVectorStore, TFIDFVectorizer

store = MemoryVectorStore(vectorizer=TFIDFVectorizer())

# Add documents with rich metadata
documents = [
    Document("Python programming tutorial", {
        "category": "programming", 
        "difficulty": "beginner",
        "tags": ["python", "tutorial"]
    }),
    Document("Advanced machine learning concepts", {
        "category": "AI", 
        "difficulty": "advanced",
        "tags": ["ML", "AI", "algorithms"]
    })
]

store.add_documents(documents)

# Query and access metadata
results = store.query_text("programming", k=1)
doc = results[0]
print(f"Content: {doc.page_content}")
print(f"Category: {doc.metadata['category']}")
print(f"Difficulty: {doc.metadata['difficulty']}")
```

## Examples

See the `examples/` directory for complete working examples:

- `vectorization_demo.py`: Comprehensive demonstration of all vectorization models

## API Reference

### Classes

- **`Document`**: Represents a document with content and metadata
- **`MemoryVectorStore`**: In-memory vector database with optional automatic vectorization
- **`TFIDFVectorizer`**: TF-IDF text vectorization
- **`BagOfWordsVectorizer`**: Bag of words text vectorization
- **`WordCountVectorizer`**: Word frequency vectorization
- **`BaseVectorizer`**: Abstract base class for custom vectorizers

### Functions

- **`cosine_similarity_pure_python()`**: Compute cosine similarity between vectors

## Architecture

The library is designed with modularity in mind:

- **Vectorizers** implement the `BaseVectorizer` interface, making it easy to add new models
- **MemoryVectorStore** can work with any vectorizer or with explicit vectors
- **Document** class provides a consistent interface for text and metadata
- All components use only Python standard library (no external dependencies)

## Performance

- **In-memory storage**: Extremely fast queries with no database overhead
- **Pure Python**: No compiled dependencies, works anywhere Python runs
- **Efficient algorithms**: Optimized similarity calculations and vector operations
- **Scalable**: Suitable for thousands to hundreds of thousands of documents

## Author

My_name

## License

[MIT](https://choosealicense.com/licenses/mit/)