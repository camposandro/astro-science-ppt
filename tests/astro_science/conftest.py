import os

import pandas as pd
import pytest


@pytest.fixture
def test_df():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root_dir, "data", "gd1_data.csv")
    return pd.read_csv(path)
