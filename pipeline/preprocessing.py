import re
import pandas as pd


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Normalize ticket text while preserving domain-specific keywords.

    Design decisions:
    - We combine Subject + Description because subject provides
      compressed context and description has the detail. Together
      they give the embedding model more signal than either alone.
    - We do minimal cleaning (lowercase + whitespace normalization +
      light punctuation removal) rather than aggressive stemming or
      stopword removal. Sentence transformers like mpnet are trained
      on natural language and perform best on text that resembles
      their training distribution. Over-cleaning (e.g., removing 'not',
      'cannot') destroys semantic content.
    - Domain keywords (VPN, MFA, DNS, etc.) are lowercase after
      processing but remain present -- we do not remove short tokens
      or apply domain-blind stopword lists.

    Args:
        df: Raw DataFrame from load_data().
        config: Config dict with 'data.text_columns' and 'preprocessing'.

    Returns:
        Copy of df with a new 'text' column containing the cleaned,
        combined ticket text.
    """
    cfg = config['preprocessing']
    subj_col, desc_col = config['data']['text_columns']
    sep = cfg['combine_separator']

    df = df.copy()

    # Combine subject and description, treating nulls as empty strings
    df['text'] = (
        df[subj_col].fillna('').astype(str)
        + sep
        + df[desc_col].fillna('').astype(str)
    )

    # Lowercase -- normalizes surface forms without losing meaning
    df['text'] = df['text'].str.lower()

    # Remove characters that are not letters, digits, or spaces.
    # We keep digits because ticket IDs, error codes, and version numbers
    # are semantically meaningful in ITSM context.
    df['text'] = df['text'].str.replace(r'[^a-z0-9\s]', ' ', regex=True)

    # Collapse multiple whitespace into single space
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df
