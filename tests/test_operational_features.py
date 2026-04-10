import numpy as np
import pandas as pd
import pytest
from pipeline.operational_features import encode_metadata

MOCK_CONFIG = {
    'data': {
        'metadata_columns': {
            'priority': 'Ticket Priority',
            'ticket_type': 'Ticket Type',
            'product': 'Product Purchased',
            'channel': 'Ticket Channel',
        }
    },
    'operational_features': {
        'priority_map': {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4},
        'product_encoding': 'frequency',
        'normalize': True,
    }
}


def make_df():
    return pd.DataFrame({
        'Ticket Priority': ['High', 'Medium', 'Low', 'Critical'],
        'Ticket Type': ['Technical issue', 'Account access', 'Technical issue', 'Billing'],
        'Product Purchased': ['ProductA', 'ProductB', 'ProductA', 'ProductC'],
        'Ticket Channel': ['Email', 'Phone', 'Email', 'Chat'],
    })


def test_encode_metadata_returns_numpy_array():
    result = encode_metadata(make_df(), MOCK_CONFIG)
    assert isinstance(result, np.ndarray)


def test_encode_metadata_correct_row_count():
    df = make_df()
    result = encode_metadata(df, MOCK_CONFIG)
    assert result.shape[0] == len(df)


def test_encode_metadata_normalized_to_unit_range():
    result = encode_metadata(make_df(), MOCK_CONFIG)
    assert result.min() >= 0.0 - 1e-9
    assert result.max() <= 1.0 + 1e-9


def test_encode_metadata_handles_unknown_priority():
    df = make_df()
    df.loc[0, 'Ticket Priority'] = 'Unknown'
    # Should not raise -- unknown priority maps to 0
    result = encode_metadata(df, MOCK_CONFIG)
    assert result.shape[0] == len(df)
