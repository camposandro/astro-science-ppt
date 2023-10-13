"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""
import os
import random

import pandas as pd

from astro_science.example import apply_transformation


def load_test_df():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root_dir, "data", "gd1_data.csv")
    test_df = pd.read_csv(path)
    random_size = random.randint(0, len(test_df.index))
    test_df = test_df.iloc[:random_size]
    return test_df


def time_computation():
    """Time computations are prefixed with 'time'."""
    test_df = load_test_df()
    apply_transformation(
        test_df,
        col_1="phi1",
        col_2="phi2",
        result_column="result",
        operation=sum,
    )


def peakmem_computation():
    """Memory computations are prefixed with 'mem' or 'peakmem'."""
    test_df = load_test_df()
    return apply_transformation(
        test_df,
        col_1="phi1",
        col_2="phi2",
        result_column="result",
        operation=sum,
    )
