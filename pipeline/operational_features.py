import numpy as np
import pandas as pd


def encode_metadata(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Encode structured ticket metadata into a normalized numeric feature matrix.

    Each field captures a different operational dimension:
    - Priority: urgency level -- ordinal (Low < Medium < High < Critical).
      Ordinal because the ordering is meaningful; one-hot would lose this.
    - Ticket Type: issue category -- one-hot. No natural ordering.
    - Product: which system/product -- frequency encoding by default.
      High cardinality (many products) makes one-hot impractical.
      Frequency encoding assigns each product its relative frequency in the
      dataset, capturing "how common is this product's tickets" as a signal.
    - Channel: submission channel -- one-hot. No natural ordering.

    All features are min-max normalized to [0,1] after encoding so that the
    high-magnitude semantic embedding dimensions don't overwhelm low-magnitude
    metadata in the fusion layer.

    Args:
        df: DataFrame with ticket metadata columns.
        config: Config dict with 'operational_features' and 'data.metadata_columns'.

    Returns:
        numpy array of shape (n_tickets, n_features), all values in [0, 1].
    """
    cfg = config['operational_features']
    meta = config['data']['metadata_columns']
    parts = []

    # Priority: ordinal encoding (preserves ranking signal)
    priority_map = cfg['priority_map']
    priority_encoded = (
        df[meta['priority']]
        .map(priority_map)
        .fillna(0)          # unknown priorities -> 0 (treat as missing, not Low)
        .values
        .reshape(-1, 1)
        .astype(float)
    )
    parts.append(priority_encoded)

    # Ticket Type: one-hot
    type_dummies = pd.get_dummies(df[meta['ticket_type']], prefix='type')
    parts.append(type_dummies.values.astype(float))

    # Product: frequency or one-hot
    if cfg['product_encoding'] == 'frequency':
        # Replace each product with its relative frequency in the dataset.
        # High-frequency products get higher values -- useful signal for routing.
        freq_map = df[meta['product']].value_counts(normalize=True)
        product_encoded = (
            df[meta['product']]
            .map(freq_map)
            .fillna(0)
            .values
            .reshape(-1, 1)
            .astype(float)
        )
        parts.append(product_encoded)
    else:
        product_dummies = pd.get_dummies(df[meta['product']], prefix='product')
        parts.append(product_dummies.values.astype(float))

    # Channel: one-hot
    channel_dummies = pd.get_dummies(df[meta['channel']], prefix='channel')
    parts.append(channel_dummies.values.astype(float))

    features = np.hstack(parts)

    # Min-max normalize per column so no feature dominates by scale
    if cfg.get('normalize', True):
        col_min = features.min(axis=0)
        col_max = features.max(axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1   # constant columns stay 0 after normalization
        features = (features - col_min) / denom

    return features
