import numpy as np
from sklearn.cluster import KMeans
import hdbscan


def cluster(vectors: np.ndarray, method: str, config: dict) -> np.ndarray:
    """
    Cluster a feature matrix using the specified algorithm.

    Args:
        vectors: (n, d) array of feature vectors (fused or single-source).
        method: 'kmeans' or 'hdbscan'.
        config: Full config dict.

    Returns:
        (n,) integer array of cluster labels.
        Note: HDBSCAN may assign -1 to noise points.

    Raises:
        ValueError: If method is not 'kmeans' or 'hdbscan'.
    """
    if method == 'kmeans':
        return _kmeans(vectors, config['clustering']['kmeans'])
    elif method == 'hdbscan':
        return _hdbscan(vectors, config['clustering']['hdbscan'])
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'kmeans' or 'hdbscan'.")


def _kmeans(vectors: np.ndarray, cfg: dict) -> np.ndarray:
    """
    KMeans baseline clustering.

    Initialization -- k-means++: seeds centroids to be spread across the space,
    avoiding degenerate initializations where multiple centroids start near the
    same dense region. Significantly reduces variance across runs compared to
    random initialization.

    Hyperparameter notes:
    - k: number of clusters. Too low -> heterogeneous clusters (e.g., password
      reset and VPN issues mixed together). Too high -> over-fragmented, small
      clusters that are hard to interpret. Tune with elbow + silhouette in
      hyperparameter_analysis.ipynb.
    - n_init: run the full KMeans algorithm N times and keep the best result
      by inertia. Reduces sensitivity to initialization randomness.
    - Weakness: KMeans assumes spherical, equally-sized clusters. Real-world
      ticket distributions often have large dominant clusters (common issues)
      and small rare ones -- compare with HDBSCAN results.
    """
    km = KMeans(
        n_clusters=cfg['k'],
        init=cfg['init'],
        n_init=cfg['n_init'],
        random_state=cfg['random_state'],
    )
    return km.fit_predict(vectors)


def _hdbscan(vectors: np.ndarray, cfg: dict) -> np.ndarray:
    """
    HDBSCAN density-based clustering.

    Discovers clusters of arbitrary shape and size without specifying k upfront.
    Tickets that don't fit any cluster are labeled -1 (noise) -- this is
    realistic for ITSM data where some tickets are genuinely one-off.

    Hyperparameter notes:
    - min_cluster_size: minimum number of tickets to form a cluster. Too small
      -> fragmented, many tiny clusters. Too large -> valid small clusters absorbed
      into noise. Start at ~1% of dataset size, tune visually with UMAP.
    - min_samples: controls how conservative noise labeling is. Higher values
      -> more noise points, smaller but more confident clusters. Lower -> more
      aggressive clustering, fewer noise points.
    - Failure modes: 'noise explosion' (too many -1 labels) means min_cluster_size
      is too large; 'cluster collapse' (everything in one cluster) means too small.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg['min_cluster_size'],
        min_samples=cfg['min_samples'],
        metric=cfg['metric'],
    )
    return clusterer.fit_predict(vectors)
