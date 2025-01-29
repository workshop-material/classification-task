import click
import numpy as np
import pandas as pd
import sys

from scipy.spatial.distance import cdist


def get_nearest_labels(row, labels, num_neighbors):
    nearest_labels = []
    for _distance, label in sorted(zip(row, labels)):
        nearest_labels.append(label)
        if len(nearest_labels) == num_neighbors:
            return nearest_labels


def get_maj_ind(index_list) -> int:
    return np.bincount(index_list).argmax()

def test_get_maj_ind():
    assert get_maj_ind([0,1,0,1,1,1,0,1]) == 1

def nearest_neighbor_predictor(df_train, df_test, num_neighbors):
    test_positions = df_test[["x", "y"]].values
    train_positions = df_train[["x", "y"]].values
    train_labels = df_train["label"].values

    distance_matrix = cdist(test_positions, train_positions)

    predictions = []
    for row in distance_matrix:
        nearest_labels = get_nearest_labels(row, train_labels, num_neighbors)

        # this finds the most common nearest label by majority vote
        majority_index = get_maj_ind(nearest_labels)
        predictions.append(majority_index)

    return predictions


@click.command()
@click.option(
    "--num-neighbors",
    type=int,
    required=True,
    help="Number of nearest neighbors in classifier. Should be an odd number to avoid ties.",
)
@click.option(
    "--training-data",
    type=str,
    required=True,
    help="We read the training data from this file.",
)
@click.option(
    "--test-data", type=str, required=True, help="We read the test data from this file."
)
@click.option(
    "--predictions",
    type=str,
    required=True,
    help="Predictions on test data are stored in this file.",
)
def main(num_neighbors, training_data, test_data, predictions):
    """
    Creates predictions on test data with a nearest neighbor classifier.
    """
    if num_neighbors % 2 == 0:
        sys.exit("number of errors has to be uneven!")


    df_train = pd.read_csv(training_data)
    df_test = pd.read_csv(test_data)

    df_test["prediction"] = nearest_neighbor_predictor(df_train, df_test, num_neighbors)

    df_test.to_csv(predictions, index=False)
    print(f"Predictions saved to {predictions}")


if __name__ == "__main__":
    main()
