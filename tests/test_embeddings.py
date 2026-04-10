import numpy as np
import os
import pytest
from unittest.mock import patch, MagicMock
from pipeline.embeddings import generate_embeddings

MOCK_CONFIG = {
    'embeddings': {
        'model': 'all-MiniLM-L6-v2',
        'cache_path': '/tmp/test_embeddings_cache.npy',
        'batch_size': 8,
    }
}

SAMPLE_TEXTS = [
    'vpn connection failed cannot connect',
    'password reset account locked',
    'software installation error occurred',
]


def test_generate_embeddings_returns_numpy_array():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
    with patch('pipeline.embeddings.SentenceTransformer', return_value=mock_model):
        result = generate_embeddings(SAMPLE_TEXTS, MOCK_CONFIG)
    assert isinstance(result, np.ndarray)


def test_generate_embeddings_correct_shape():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
    with patch('pipeline.embeddings.SentenceTransformer', return_value=mock_model):
        result = generate_embeddings(SAMPLE_TEXTS, MOCK_CONFIG)
    assert result.shape[0] == len(SAMPLE_TEXTS)
    assert result.shape[1] > 0


def test_generate_embeddings_uses_cache(tmp_path):
    cache_path = str(tmp_path / 'cache.npy')
    config = {**MOCK_CONFIG, 'embeddings': {**MOCK_CONFIG['embeddings'], 'cache_path': cache_path}}
    cached = np.random.rand(3, 384).astype(np.float32)
    np.save(cache_path, cached)

    with patch('pipeline.embeddings.SentenceTransformer') as mock_st:
        result = generate_embeddings(SAMPLE_TEXTS, config)
        mock_st.assert_not_called()

    np.testing.assert_array_equal(result, cached)


def test_generate_embeddings_saves_cache(tmp_path):
    cache_path = str(tmp_path / 'new_cache.npy')
    config = {**MOCK_CONFIG, 'embeddings': {**MOCK_CONFIG['embeddings'], 'cache_path': cache_path}}

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
    with patch('pipeline.embeddings.SentenceTransformer', return_value=mock_model):
        generate_embeddings(SAMPLE_TEXTS, config)

    assert os.path.exists(cache_path)
