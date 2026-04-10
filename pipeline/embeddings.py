import os
import numpy as np


# ---------------------------------------------------------------------------
# Provider: Voyage AI
# ---------------------------------------------------------------------------

def _voyage_embed(texts: list, config: dict) -> np.ndarray:
    """
    Embed texts via Voyage AI API in batches.

    Voyage AI limits each request to 128 inputs (voyage-3). We batch
    automatically so callers don't need to think about limits.

    Requires VOYAGE_API_KEY environment variable.
    """
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY environment variable not set. "
            "Get your key from https://dash.voyageai.com and run: "
            "set VOYAGE_API_KEY=<your_key>"
        )

    cfg = config['embeddings']
    model = cfg.get('voyage_model', 'voyage-3')
    batch_size = cfg.get('batch_size', 128)

    client = voyageai.Client(api_key=api_key)
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        result = client.embed(batch, model=model, input_type="document")
        all_vecs.append(np.array(result.embeddings, dtype=np.float32))
        if (i // batch_size) % 10 == 0:
            print(f"    Voyage: {min(i + batch_size, len(texts)):,}/{len(texts):,}", flush=True)

    return np.vstack(all_vecs)


# ---------------------------------------------------------------------------
# Provider: sentence-transformers (local)
# ---------------------------------------------------------------------------

def _resolve_device(cfg: dict) -> str:
    """CUDA > MPS > CPU, overridable via config."""
    if 'device' in cfg:
        return cfg['device']
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def load_model(config: dict):
    """
    Load the local sentence-transformer model onto the resolved device.
    Only used when provider is 'sentence-transformers'.
    Returns None for other providers.
    """
    if config['embeddings'].get('provider', 'sentence-transformers') != 'sentence-transformers':
        return None

    from sentence_transformers import SentenceTransformer
    cfg = config['embeddings']
    device = _resolve_device(cfg)
    print(f"  Embedding device: {device}")
    return SentenceTransformer(cfg['model'], device=device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: list,
    config: dict,
    model=None,
) -> np.ndarray:
    """
    Generate sentence-level embeddings for a list of ticket texts.

    Supports two providers, selected via config['embeddings']['provider']:
    - 'voyage'               : Voyage AI API (voyage-3). Fast, no GPU needed,
                               32K context. Requires VOYAGE_API_KEY env var.
    - 'sentence-transformers': Local model (all-mpnet-base-v2). No API key,
                               but slow on CPU (~4h for 61k tickets).

    Embeddings are cached to disk after first generation.

    Args:
        texts: List of preprocessed ticket text strings.
        config: Config dict with 'embeddings' section.
        model: Pre-loaded SentenceTransformer (ignored for Voyage provider).

    Returns:
        numpy array of shape (n_tickets, embedding_dim).
    """
    cfg = config['embeddings']
    cache_path = cfg['cache_path']
    provider = cfg.get('provider', 'sentence-transformers')

    if os.path.exists(cache_path):
        return np.load(cache_path)

    if provider == 'voyage':
        embeddings = _voyage_embed(texts, config)
    else:
        if model is None:
            model = load_model(config)
        embeddings = model.encode(
            texts,
            batch_size=cfg.get('batch_size', 32),
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def enrich_with_tag_embeddings(
    semantic_vecs: np.ndarray,
    df,
    config: dict,
    model=None,
) -> np.ndarray:
    """
    Enrich semantic vectors by concatenating tag_1 embeddings where present.

    Embeds the ~211 unique tag values once (cached), looks up per row.
    Unknown rows receive a zero vector — no tag signal contributed.
    Uses the same provider as generate_embeddings.

    Args:
        semantic_vecs: (n, embedding_dim) from generate_embeddings().
        df: DataFrame with the tag column (post-imputation).
        config: Config dict with 'embeddings.tag_enrichment' section.
        model: Pre-loaded SentenceTransformer (ignored for Voyage provider).

    Returns:
        (n, embedding_dim + tag_dim) array. Untagged rows have zeros in tag dims.
    """
    cfg = config['embeddings']['tag_enrichment']
    tag_col = cfg['column']
    weight = cfg.get('weight', 1.0)
    cache_path = cfg.get('cache_path', 'outputs/tag_embeddings_cache.npy')
    provider = config['embeddings'].get('provider', 'sentence-transformers')
    sentinel = 'Unknown'

    tag_values = df[tag_col].fillna(sentinel).tolist()
    unique_tags = [t for t in set(tag_values) if t != sentinel]

    if os.path.exists(cache_path):
        tag_lookup_matrix = np.load(cache_path, allow_pickle=True).item()
    else:
        if provider == 'voyage':
            unique_vecs = _voyage_embed(unique_tags, config)
        else:
            if model is None:
                model = load_model(config)
            unique_vecs = model.encode(
                unique_tags,
                batch_size=config['embeddings'].get('batch_size', 32),
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        tag_lookup_matrix = dict(zip(unique_tags, unique_vecs))
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, tag_lookup_matrix)

    embedding_dim = next(iter(tag_lookup_matrix.values())).shape[0]
    zero_vec = np.zeros(embedding_dim, dtype=np.float32)

    tag_matrix = np.vstack([
        tag_lookup_matrix.get(t, zero_vec) for t in tag_values
    ])

    return np.hstack([semantic_vecs, weight * tag_matrix])
