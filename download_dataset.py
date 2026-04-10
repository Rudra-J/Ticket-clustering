"""
download_dataset.py — Fetch the HuggingFace dataset and save it as a CSV.

Usage:
    python download_dataset.py

The output CSV is written to data_source/customer_support_tickets.csv so the
pipeline can be switched back to source: csv in config.yaml without any other
changes. Run this once (or whenever the upstream dataset updates).
"""

import os
import yaml
from datasets import load_dataset


def download(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hf_dataset = config["data"]["hf_dataset"]
    out_path = config["data"]["path"]

    print(f"Loading '{hf_dataset}' from HuggingFace Hub...")
    ds = load_dataset(hf_dataset)

    # Use the first available split (typically 'train')
    split = list(ds.keys())[0]
    df = ds[split].to_pandas()
    print(f"  {len(df):,} rows loaded from split '{split}'")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    download()
