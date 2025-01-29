import numpy as np
import pandas as pd

from generate_data import (
    circular_distribution,
    generate_data,
    normal_distribution,
)


def test_normal_distribution():
    rng = np.random.default_rng(seed=42)
    data = normal_distribution(rng, 0.0, 1.0, 100, 0)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert data["label"].nunique() == 1
    assert data["label"].unique()[0] == 0


def test_circular_distribution():
    rng = np.random.default_rng(seed=42)
    data = circular_distribution(rng, 9.0, 12.0, 100, 0)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert data["label"].nunique() == 1
    assert data["label"].unique()[0] == 0


def test_generate_data():
    rng = np.random.default_rng(seed=42)
    data = generate_data(rng, 100)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100 * 2
    assert data["label"].nunique() == 2
    assert set(data["label"].unique()) == {0, 1}
