import os
import pandas as pd


def load_data(path: str, config: dict) -> pd.DataFrame:
    """
    Load the ticket dataset from a CSV file or HuggingFace Hub and validate required columns.

    Ingestion only -- no transformation. Downstream modules handle cleaning.
    Keeping ingestion separate means the data source can change without touching
    preprocessing logic.

    Args:
        path: Path to the CSV file (used when config['data']['source'] == 'csv').
        config: Config dict containing 'data.text_columns', 'data.metadata_columns',
                and optionally 'data.source' and 'data.hf_dataset'.

    Returns:
        Raw DataFrame with all columns from the source.

    Raises:
        FileNotFoundError: If source is 'csv' and the file does not exist at path.
        ValueError: If any required columns are absent from the loaded data.
    """
    source = config['data'].get('source', 'csv')

    if source == 'huggingface':
        # HF datasets are streamed/downloaded on first call; subsequent runs use cache.
        # Importing here so the CSV-only path has no hard dependency on `datasets`.
        from datasets import load_dataset
        hf_dataset = config['data']['hf_dataset']
        ds = load_dataset(hf_dataset)
        # load_dataset returns a DatasetDict; use the first split (usually 'train').
        split = list(ds.keys())[0]
        df = ds[split].to_pandas()
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pd.read_csv(path)

    # Validate all columns the pipeline will need are present.
    # Fail early so downstream errors are not confusingly attributed to missing data.
    text_cols = config['data']['text_columns']
    meta_cols = list(config['data']['metadata_columns'].values())
    required = text_cols + meta_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df
