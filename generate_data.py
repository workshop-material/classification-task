import click
import numpy as np
import pandas as pd


def normal_distribution(rng, loc, scale, num_samples, label):
    xs = rng.normal(loc, scale, num_samples)
    ys = rng.normal(loc, scale, num_samples)

    data = pd.DataFrame({"x": xs, "y": ys, "label": [label] * len(xs)})

    return data


def test_normal_distribution():
    rng = np.random.default_rng(seed=42)
    data = normal_distribution(rng, 0.0, 1.0, 100, 0)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert data["label"].nunique() == 1
    assert data["label"].unique()[0] == 0


def circular_distribution(rng, r_min, r_max, num_samples, label):
    angles = np.linspace(0.0, 2.0 * np.pi, num_samples)
    radii = rng.uniform(r_min, r_max, num_samples)

    xs = []
    ys = []
    for r, angle in zip(radii, angles):
        xs.append(r * np.cos(angle))
        ys.append(r * np.sin(angle))

    data = pd.DataFrame({"x": xs, "y": ys, "label": [label] * len(xs)})

    return data


def test_circular_distribution():
    rng = np.random.default_rng(seed=42)
    data = circular_distribution(rng, 9.0, 12.0, 100, 0)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert data["label"].nunique() == 1
    assert data["label"].unique()[0] == 0


# generates a fixed number of 2-dimensional data points with a corresponding
# binary class label (0 or 1)
def generate_data(rng, num_samples):
    data1 = normal_distribution(rng, 0.0, 1.0, num_samples // 2, 0)
    data2 = circular_distribution(rng, 9.0, 12.0, num_samples // 2, 0)
    data3 = circular_distribution(rng, 4.0, 6.0, num_samples, 1)

    return pd.concat([data1, data2, data3])


def test_generate_data():
    rng = np.random.default_rng(seed=42)
    data = generate_data(rng, 100)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100 * 2
    assert data["label"].nunique() == 2
    assert set(data["label"].unique()) == {0, 1}


@click.command()
@click.option(
    "--num-samples", type=int, required=True, help="Number of samples for each class."
)
@click.option(
    "--training-data",
    type=str,
    required=True,
    help="Training data is written to this file.",
)
@click.option(
    "--test-data",
    type=str,
    required=True,
    help="Test data is written to this file.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for the random number generator.",
)
def main(num_samples, training_data, test_data, seed):
    """
    Program that generates a set of training and test samples for a non-linear classification task.
    """

    rng = np.random.default_rng(seed=seed)

    for output_file in [training_data, test_data]:
        data = generate_data(rng, num_samples)
        data.to_csv(output_file, index=False)

    print(
        f"Generated {num_samples} training samples ({training_data}) and test samples ({test_data})."
    )


if __name__ == "__main__":
    main()
