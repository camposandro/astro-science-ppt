"""Tests the example module"""
import os

import pandas as pd
import pytest

from example import Query, apply_transformation, query_by_column


@pytest.fixture
def test_df():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root_dir, "data", "gd1_data.csv")
    return pd.read_csv(path)


def test_create_query():
    """Tests creation of query objects."""
    with pytest.raises(ValueError):
        Query(min=150, max=160)
    with pytest.raises(ValueError):
        Query(column_name="ra", min=160, max=150)
    Query(column_name="ra", min=150, max=160)


def test_query(test_df):
    """Tests running a query on the dataframe."""
    query = Query(column_name="ra", min=130, max=140)
    results = query_by_column(test_df, query)
    assert isinstance(results, pd.DataFrame)
    assert all(results["ra"].between(130, 140))


def test_apply_transformation(test_df):
    """Tests the creation of a new column via transformation"""
    result_df = apply_transformation(
        test_df,
        col_1="phi1",
        col_2="phi2",
        result_column="result",
        operation=sum,
    )
    # New column exists
    assert "result" in result_df.columns
    # The computed result is the one expected
    pd.testing.assert_series_equal(
        sum(test_df["phi1"], test_df["phi2"]), result_df["result"], check_names=False
    )
