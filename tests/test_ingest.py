import pytest
import pandas as pd
from pathlib import Path
from pipeline.ingest import load_data

MOCK_CONFIG = {
    'data': {
        'text_columns': ['Ticket Subject', 'Ticket Description'],
        'metadata_columns': {
            'priority': 'Ticket Priority',
            'ticket_type': 'Ticket Type',
            'product': 'Product Purchased',
            'channel': 'Ticket Channel',
        }
    }
}


def make_csv(tmp_path: Path) -> str:
    df = pd.DataFrame({
        'Ticket Subject': ['VPN issue', 'Password reset'],
        'Ticket Description': ['Cannot connect to VPN', 'Forgot my password'],
        'Ticket Priority': ['High', 'Medium'],
        'Ticket Type': ['Technical issue', 'Account access'],
        'Product Purchased': ['ProductA', 'ProductB'],
        'Ticket Channel': ['Email', 'Phone'],
    })
    path = str(tmp_path / 'tickets.csv')
    df.to_csv(path, index=False)
    return path


def test_load_data_returns_dataframe(tmp_path):
    path = make_csv(tmp_path)
    df = load_data(path, MOCK_CONFIG)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_data_preserves_all_columns(tmp_path):
    path = make_csv(tmp_path)
    df = load_data(path, MOCK_CONFIG)
    for col in ['Ticket Subject', 'Ticket Description', 'Ticket Priority']:
        assert col in df.columns


def test_load_data_raises_on_missing_file():
    with pytest.raises(FileNotFoundError, match='not found'):
        load_data('nonexistent_file.csv', MOCK_CONFIG)


def test_load_data_raises_on_missing_columns(tmp_path):
    # CSV missing required columns
    bad_df = pd.DataFrame({'Ticket Subject': ['x'], 'Ticket Description': ['y']})
    path = str(tmp_path / 'bad.csv')
    bad_df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Missing expected columns'):
        load_data(path, MOCK_CONFIG)
