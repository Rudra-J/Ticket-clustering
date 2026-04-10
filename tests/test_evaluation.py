import numpy as np
import pytest
from pipeline.evaluation import evaluate, business_interpret

MOCK_CONFIG = {
    'evaluation': {
        'top_n_keywords': 5,
        'n_sample_tickets': 2,
    }
}

N = 30
D = 8
np.random.seed(0)
VECTORS = np.random.rand(N, D).astype(np.float32)
LABELS = np.array([i % 3 for i in range(N)])
TEXTS = [
    'password reset account locked user',
    'vpn connection failed network error',
    'software install update error crash',
] * 10


def test_evaluate_returns_dict():
    result = evaluate(VECTORS, LABELS, TEXTS, MOCK_CONFIG)
    assert isinstance(result, dict)


def test_evaluate_has_required_keys():
    result = evaluate(VECTORS, LABELS, TEXTS, MOCK_CONFIG)
    assert 'silhouette' in result
    assert 'cluster_sizes' in result
    assert 'cluster_keywords' in result


def test_evaluate_cluster_sizes_sum_to_n():
    result = evaluate(VECTORS, LABELS, TEXTS, MOCK_CONFIG)
    total = sum(v for k, v in result['cluster_sizes'].items() if k >= 0)
    assert total == N


def test_evaluate_cluster_keywords_per_cluster():
    result = evaluate(VECTORS, LABELS, TEXTS, MOCK_CONFIG)
    for cid in [0, 1, 2]:
        assert cid in result['cluster_keywords']
        assert len(result['cluster_keywords'][cid]) <= MOCK_CONFIG['evaluation']['top_n_keywords']


def test_business_interpret_returns_dict():
    keywords = {0: ['password', 'reset', 'login'], 1: ['vpn', 'network'], 2: ['install']}
    result = business_interpret(keywords, MOCK_CONFIG)
    assert isinstance(result, dict)
    assert 0 in result and 1 in result and 2 in result


def test_business_interpret_has_required_fields():
    keywords = {0: ['password', 'reset', 'login', 'account', 'locked']}
    result = business_interpret(keywords, MOCK_CONFIG)
    for field in ['category', 'team', 'automation', 'confidence', 'top_keywords']:
        assert field in result[0]
