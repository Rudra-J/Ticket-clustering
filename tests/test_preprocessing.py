import pandas as pd
from pipeline.preprocessing import preprocess

MOCK_CONFIG = {
    'data': {
        'text_columns': ['Ticket Subject', 'Ticket Description'],
    },
    'preprocessing': {
        'combine_separator': ' ',
        'domain_keywords': ['VPN', 'DNS', 'SSO', 'MFA'],
    }
}


def make_df():
    return pd.DataFrame({
        'Ticket Subject': ['VPN Connection Issue', 'Password Reset Request'],
        'Ticket Description': ['Cannot connect to VPN server', 'User forgot MFA code'],
        'Ticket Priority': ['High', 'Medium'],
    })


def test_preprocess_adds_text_column():
    df = preprocess(make_df(), MOCK_CONFIG)
    assert 'text' in df.columns


def test_preprocess_combines_subject_and_description():
    df = preprocess(make_df(), MOCK_CONFIG)
    # Both subject and description content should appear in combined text
    assert 'vpn' in df['text'].iloc[0]
    assert 'cannot' in df['text'].iloc[0]


def test_preprocess_lowercases_text():
    df = preprocess(make_df(), MOCK_CONFIG)
    assert df['text'].iloc[0] == df['text'].iloc[0].lower()


def test_preprocess_does_not_drop_original_columns():
    df = preprocess(make_df(), MOCK_CONFIG)
    assert 'Ticket Subject' in df.columns
    assert 'Ticket Priority' in df.columns


def test_preprocess_handles_missing_text_gracefully():
    df = pd.DataFrame({
        'Ticket Subject': [None, 'Normal subject'],
        'Ticket Description': ['Some description', None],
        'Ticket Priority': ['High', 'Low'],
    })
    result = preprocess(df, MOCK_CONFIG)
    # Should not raise, should produce non-null text
    assert result['text'].notna().all()
