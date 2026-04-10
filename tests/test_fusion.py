import numpy as np
import pytest
from pipeline.fusion import concatenate_fusion, weighted_fusion

N = 20
D_SEM = 10
D_OP = 4
np.random.seed(42)
SEM = np.random.rand(N, D_SEM).astype(np.float32)
OP = np.random.rand(N, D_OP).astype(np.float32)
SEM_LABELS = np.array([i % 3 for i in range(N)])
OP_LABELS = np.array([i % 3 for i in range(N)])


def test_concatenate_fusion_shape():
    result = concatenate_fusion(SEM, OP)
    assert result.shape == (N, D_SEM + D_OP)


def test_concatenate_fusion_l2_normalized():
    result = concatenate_fusion(SEM, OP)
    sem_half = result[:, :D_SEM]
    norms = np.linalg.norm(sem_half, axis=1)
    np.testing.assert_allclose(norms, np.ones(N), atol=1e-5)


def test_weighted_fusion_shape():
    result = weighted_fusion(SEM, OP, alpha=0.7)
    assert result.shape == (N, D_SEM + D_OP)


def test_weighted_fusion_alpha_1_is_semantic_only():
    result = weighted_fusion(SEM, OP, alpha=1.0)
    op_half = result[:, D_SEM:]
    np.testing.assert_allclose(op_half, np.zeros((N, D_OP)), atol=1e-5)


def test_weighted_fusion_alpha_0_is_operational_only():
    result = weighted_fusion(SEM, OP, alpha=0.0)
    sem_half = result[:, :D_SEM]
    np.testing.assert_allclose(sem_half, np.zeros((N, D_SEM)), atol=1e-5)


