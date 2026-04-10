import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def impute(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Impute missing values across configured columns using confidence-thresholded
    TF-IDF + Logistic Regression classifiers.

    One vectorizer is fit on labeled-row text and reused for every column,
    so the cost is one TF-IDF fit + one LR fit per column rather than one
    full fit-transform cycle per column.

    Rows where max class probability < threshold receive a configurable sentinel
    ('Unknown' by default) instead of a predicted value. This keeps every
    downstream one-hot encoding complete -- pd.get_dummies silently drops NaN
    rows, which would silently shrink the dataset.

    Args:
        df: Raw DataFrame from load_data().
        config: Config dict with 'imputation', 'data.text_columns', and
                'data.metadata_columns'.

    Returns:
        Copy of df with imputed columns partially filled and boolean
        '<col>_imputed' columns marking machine-filled rows.
    """
    cfg = config['imputation']
    subj_col, body_col = config['data']['text_columns']
    col_configs = cfg.get('columns', {})

    df = df.copy()

    # Combined text used as features for every classifier.
    # Nulls become empty strings so the vectorizer sees a consistent corpus.
    text = (df[subj_col].fillna('') + ' ' + df[body_col].fillna('')).tolist()

    # Identify the labeled anchor: rows where the *first* configured column
    # is non-null. For this dataset that is 'type' -- the segment boundary.
    # All tag nulls in the labeled segment are a small subset handled column-by-column.
    anchor_col = next(iter(col_configs))  # first column in config order
    labeled_mask = df[anchor_col].notna()

    if labeled_mask.sum() == 0:
        print("  Imputation: no labeled rows found, skipping.")
        return df

    # --- Fit vectorizer once on labeled text ---
    vect = TfidfVectorizer(
        max_features=cfg.get('max_features', 15000),
        ngram_range=tuple(cfg.get('ngram_range', [1, 2])),
        sublinear_tf=True,
    )
    labeled_idx = np.where(labeled_mask.values)[0]
    X_labeled_all = vect.fit_transform([text[i] for i in labeled_idx])
    X_all = vect.transform(text)  # full matrix reused per column

    default_threshold = cfg.get('confidence_threshold', 0.7)
    default_sentinel = cfg.get('low_confidence_fill', 'Unknown')
    min_class_size = cfg.get('min_class_size', 5)

    for col, col_cfg in col_configs.items():
        if not col_cfg.get('enabled', True):
            continue

        threshold = col_cfg.get('threshold', default_threshold)
        sentinel = col_cfg.get('low_confidence_fill', default_sentinel)

        null_mask = df[col].isnull()
        n_null = null_mask.sum()
        if n_null == 0:
            print(f"  {col}: no nulls, skipping.")
            continue

        df[f'{col}_imputed'] = False

        # Build training set: labeled rows that also have this column filled.
        # For tag_2/tag_3 a small number of labeled rows may also be missing.
        train_mask = labeled_mask & df[col].notna()
        train_idx = np.where(train_mask.values)[0]

        # Drop rare classes -- singletons cause stratification failures and
        # produce unreliable probability estimates.
        y_train_raw = df.loc[train_mask, col]
        vc = y_train_raw.value_counts()
        valid_classes = vc[vc >= min_class_size].index
        class_mask = y_train_raw.isin(valid_classes)
        final_train_idx = train_idx[class_mask.values]
        y_train = y_train_raw[class_mask].values

        if len(np.unique(y_train)) < 2:
            print(f"  {col}: too few valid classes after min_class_size={min_class_size} filter, skipping.")
            continue

        X_train = X_all[final_train_idx]

        # --- Train classifier ---
        clf = LogisticRegression(
            max_iter=cfg.get('max_iter', 1000),
            C=cfg.get('C', 1.0),
            class_weight='balanced',
            solver='lbfgs',     # faster than saga on this scale; supports multi-class natively
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # --- Predict for all null rows ---
        null_idx = np.where(null_mask.values)[0]
        X_null = X_all[null_idx]
        proba = clf.predict_proba(X_null)
        max_conf = proba.max(axis=1)
        predictions = clf.classes_[proba.argmax(axis=1)]

        high_conf = max_conf >= threshold

        # Write high-confidence predictions
        df.iloc[null_idx[high_conf], df.columns.get_loc(col)] = predictions[high_conf]
        df.iloc[null_idx[high_conf], df.columns.get_loc(f'{col}_imputed')] = True

        # Write sentinel for low-confidence rows
        if sentinel:
            df.iloc[null_idx[~high_conf], df.columns.get_loc(col)] = sentinel

        n_imputed = int(high_conf.sum())
        n_sentinel = int((~high_conf).sum())
        dropped_classes = len(vc) - len(valid_classes)
        print(
            f"  {col} (threshold={threshold}): "
            f"{n_imputed:,} imputed, "
            f"{n_sentinel:,} -> '{sentinel}'"
            + (f", {dropped_classes} rare classes dropped from training" if dropped_classes else "")
        )

    return df
