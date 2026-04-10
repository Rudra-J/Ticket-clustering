import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment


def concatenate_fusion(semantic: np.ndarray, operational: np.ndarray) -> np.ndarray:
    """
    Fuse semantic and operational vectors by L2-normalizing then stacking.

    Both vectors are L2-normalized before concatenation so neither dominates
    by magnitude. The result has shape (n, semantic_dim + operational_dim).

    Signal balance: equal -- both halves contribute unit-normalized vectors.
    Appropriate when: both semantic meaning and operational context matter equally,
    or when you don't have a prior on which matters more.
    Failure mode: if operational features have many one-hot columns (high cardinality
    product encoding), the operational half is higher-dimensional and can
    geometrically crowd out the semantic signal even after normalization.

    Args:
        semantic: (n, d_s) sentence embedding array.
        operational: (n, d_o) encoded metadata array.

    Returns:
        (n, d_s + d_o) fused array with each half L2-normalized.
    """
    sem_norm = normalize(semantic, norm='l2')
    op_norm = normalize(operational, norm='l2')
    return np.hstack([sem_norm, op_norm])


def weighted_fusion(
    semantic: np.ndarray,
    operational: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Fuse by applying scalar weights to each normalized vector half.

    final_vector = [alpha * normalized_semantic | (1-alpha) * normalized_operational]

    At alpha=1.0, the operational half is zeroed -- pure semantic clustering.
    At alpha=0.0, the semantic half is zeroed -- pure operational clustering.
    At alpha=0.7 (default), semantic signal dominates while operational adds context.

    Signal balance: semantic-dominant when alpha > 0.5.
    Appropriate when: you have domain knowledge that issue type matters more (or less)
    than operational urgency for your use case.
    Failure mode: a wrong alpha completely suppresses one signal. Always compare with
    concatenation fusion to validate the chosen alpha.

    Args:
        semantic: (n, d_s) sentence embedding array.
        operational: (n, d_o) encoded metadata array.
        alpha: scalar in [0, 1] -- weight for the semantic signal.

    Returns:
        (n, d_s + d_o) fused array.
    """
    sem_norm = normalize(semantic, norm='l2')
    op_norm = normalize(operational, norm='l2')
    return np.hstack([alpha * sem_norm, (1.0 - alpha) * op_norm])

