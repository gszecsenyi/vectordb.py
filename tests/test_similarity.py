"""Tests for similarity module."""

import pytest
from vectordbpy.similarity import cosine_similarity_pure_python


class TestCosineSimilarityPurePython:
    """Test cases for cosine_similarity_pure_python function."""

    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1.0."""
        v1 = [1, 2, 3]
        v2 = [1, 2, 3]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0.0."""
        v1 = [1, 0]
        v2 = [0, 1]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity of -1.0."""
        v1 = [1, 0]
        v2 = [-1, 0]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(-1.0)

    def test_zero_vector_first(self):
        """Test that zero vector as first argument returns 0.0."""
        v1 = [0, 0, 0]
        v2 = [1, 2, 3]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == 0.0

    def test_zero_vector_second(self):
        """Test that zero vector as second argument returns 0.0."""
        v1 = [1, 2, 3]
        v2 = [0, 0, 0]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == 0.0

    def test_both_zero_vectors(self):
        """Test that both zero vectors return 0.0."""
        v1 = [0, 0]
        v2 = [0, 0]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == 0.0

    def test_different_length_vectors(self):
        """Test similarity calculation with different length vectors."""
        v1 = [1, 2, 3, 4]
        v2 = [1, 2]  # Only first two elements will be used
        # When v1=[1,2,3,4] and v2=[1,2], zip gives [(1,1), (2,2)]
        # dot = 1*1 + 2*2 = 5
        # norm1 = sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
        # norm2 = sqrt(1^2 + 2^2) = sqrt(5)
        # similarity = 5 / (sqrt(30) * sqrt(5)) = 5 / sqrt(150)
        import math
        expected = 5 / math.sqrt(150)
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(expected)

    def test_single_element_vectors(self):
        """Test similarity with single element vectors."""
        v1 = [5]
        v2 = [3]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(1.0)  # Same direction

    def test_negative_values(self):
        """Test similarity with negative values."""
        v1 = [-1, -2, -3]
        v2 = [1, 2, 3]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(-1.0)  # Opposite direction

    def test_mixed_positive_negative(self):
        """Test similarity with mixed positive and negative values."""
        v1 = [1, -2, 3]
        v2 = [2, -4, 6]
        # These vectors are in the same direction (v2 = 2 * v1)
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(1.0)

    def test_floating_point_vectors(self):
        """Test similarity with floating point vectors."""
        v1 = [1.5, 2.7, 3.1]
        v2 = [1.5, 2.7, 3.1]
        result = cosine_similarity_pure_python(v1, v2)
        assert result == pytest.approx(1.0)

    def test_empty_vectors(self):
        """Test similarity with empty vectors."""
        v1 = []
        v2 = []
        result = cosine_similarity_pure_python(v1, v2)
        assert result == 0.0