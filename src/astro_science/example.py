"""Example module that performs operations on a Dataframe"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from pandas import Series


@dataclass
class Query:
    """Parameters to query float columns of the Dataframe."""

    column_name: str = ""
    min: float = 0
    max: float = 0

    def __post_init__(self):
        if self.column_name == "":
            raise ValueError("Column name is invalid")
        if self.min > self.max:
            raise ValueError("Range is invalid")


def query_by_column(df: pd.DataFrame, query: Query):
    """Queries float columns of a Dataframe."""
    return df.loc[
        (df[query.column_name] >= query.min) & (df[query.column_name] <= query.max)
        ]


def apply_transformation(
        df: pd.DataFrame,
        col_1: str,
        col_2: str,
        result_column: str,
        operation: Callable[[Series, Series], Series],
):
    """Applies transformation function involving two Dataframe columns."""
    kwargs = {result_column: operation(df[col_1], df[col_2])}
    return df.assign(**kwargs)
