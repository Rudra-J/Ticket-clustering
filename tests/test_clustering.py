import numpy as np
import pytest
from pipeline.clustering import cluster

MOCK_CONFIG = {
    'clustering': {
        'kmeans': {
            'k': 3,
            'init': 'k-means++',
            'n_init': 3,
            'random_state': 42,
        },
        'hdbscan': {
            'min_cluster_size': 3,
            'min_samples': 2,
            'metric': 'euclidean',
        }
    }
}

# 30 synthetic vectors with 3 clear clusters
np.random.seed(42)
VECTORS = np.vstack([
    np.random.randn(10, 8) + np.array([5, 0, 0, 0, 0, 0, 0, 0]),
    np.random.randn(10, 8) + np.array([0, 5, 0, 0, 0, 0, 0, 0]),
    np.random.randn(10, 8) + np.array([0, 0, 5, 0, 0, 0, 0, 0]),
]).astype(np.float32)


def test_cluster_kmeans_returns_array_same_length():
    labels = cluster(VECTORS, 'kmeans', MOCK_CONFIG)
    assert len(labels) == len(VECTORS)


def test_cluster_kmeans_returns_k_unique_labels():
    labels = cluster(VECTORS, 'kmeans', MOCK_CONFIG)
    assert len(set(labels)) == MOCK_CONFIG['clustering']['kmeans']['k']


def test_cluster_hdbscan_returns_array_same_length():
    labels = cluster(VECTORS, 'hdbscan', MOCK_CONFIG)
    assert len(labels) == len(VECTORS)


def test_cluster_hdbscan_finds_clusters():
    labels = cluster(VECTORS, 'hdbscan', MOCK_CONFIG)
    unique = set(labels) - {-1}
    assert len(unique) >= 1


def test_cluster_raises_on_unknown_method():
    with pytest.raises(ValueError, match='Unknown method'):
        cluster(VECTORS, 'invalid_method', MOCK_CONFIG)
