import numpy as np
import pandas as pd

from generate_predictions import (
    get_majority_label,
    get_nearest_labels,
    nearest_neighbor_predictor,
)


def test_get_nearest_labels():
    row = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 2, 3])
    num_neighbors = 2

    nearest_labels = get_nearest_labels(row, labels, num_neighbors)

    assert nearest_labels == [0, 1]


def test_get_majority_label():
    nearest_labels = [0, 1, 1, 1, 0, 0, 1, 1, 1]

    majority_label = get_majority_label(nearest_labels)

    assert majority_label == 1


def test_nearest_neighbor_predictor():
    df_train = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "label": [0, 1, 0, 1],
        }
    )
    df_test = pd.DataFrame(
        {
            "x": [0.1, 1.1, 2.1, 3.1],
            "y": [0.1, 1.1, 2.1, 3.1],
        }
    )
    num_neighbors = 1

    predictions = nearest_neighbor_predictor(df_train, df_test, num_neighbors)

    assert predictions == [0, 1, 0, 1]
