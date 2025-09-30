"""
# Similarity functions for vector operations
similarity.py

This module provides similarity functions for vector operations, 
implemented without external libraries.

Functions:
    cosine_similarity_pure_python(v1, v2): Calculates the cosine similarity 
    between two numeric vectors (lists).
"""
def cosine_similarity_pure_python(v1, v2):
    """
    Calculate cosine similarity between two lists (vectors) without external libraries.

    Args:
        v1 (list of numbers): First vector.
        v2 (list of numbers): Second vector.

    Returns:
        float: Cosine similarity between v1 and v2.

    Notes:
        If the vectors are of different lengths, calculation is performed up to the 
        length of the shorter vector.
        If either vector is all zeros, returns 0.0.
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
