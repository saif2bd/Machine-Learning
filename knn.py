"""Simple K-Nearest Neighbors (KNN) implementation."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class KNNClassifier:
    """A basic K-Nearest Neighbors classifier."""

    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self._features: List[Sequence[float]] = []
        self._labels: List[T] = []

    def fit(self, features: Iterable[Sequence[float]], labels: Iterable[T]) -> None:
        """Store the training data for later prediction."""
        self._features = [tuple(x) for x in features]
        self._labels = list(labels)
        if len(self._features) != len(self._labels):
            raise ValueError("Number of feature vectors and labels must match")

    def _find_neighbors(self, point: Sequence[float]) -> List[int]:
        """Find the indices of the k nearest neighbors to the given point."""
        if not self._features:
            raise ValueError("Classifier has not been fitted yet")

        distances = [
            (index, euclidean_distance(point, sample))
            for index, sample in enumerate(self._features)
        ]
        distances.sort(key=lambda item: item[1])
        return [index for index, _ in distances[: self.k]]

    def predict(self, points: Iterable[Sequence[float]]) -> List[T]:
        """Predict labels for one or more points."""
        return [self._predict_one(point) for point in points]

    def _predict_one(self, point: Sequence[float]) -> T:
        neighbors = self._find_neighbors(point)
        neighbor_labels = [self._labels[i] for i in neighbors]
        most_common = Counter(neighbor_labels).most_common(1)
        if not most_common:
            raise ValueError("No neighbors found for prediction")
        return most_common[0][0]

    def score(self, features: Iterable[Sequence[float]], labels: Iterable[T]) -> float:
        """Return the accuracy of the classifier on the given dataset."""
        predicted = self.predict(features)
        actual = list(labels)
        if len(predicted) != len(actual):
            raise ValueError("Number of predictions and labels must match")
        correct = sum(1 for p, a in zip(predicted, actual) if p == a)
        return correct / len(actual)


if __name__ == "__main__":
    # Example usage with a tiny dataset.
    dataset = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.0],
        [6.0, 5.0],
        [7.0, 8.0],
    ]
    labels = ["red", "red", "red", "blue", "blue"]
    model = KNNClassifier(k=3)
    model.fit(dataset, labels)

    test_points = [[2.5, 2.5], [6.5, 6.0]]
    predictions = model.predict(test_points)
    print("Test points:", test_points)
    print("Predictions:", predictions)
    print("Accuracy on training data:", model.score(dataset, labels))
