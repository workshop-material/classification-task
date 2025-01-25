import click
import pandas as pd
import altair as alt


def scatter_plot(data, title, color_column, color_scheme):
    chart = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            x="x",
            y="y",
            color=alt.Color(f"{color_column}:N", scale=alt.Scale(scheme=color_scheme)),
            column="label",
        )
        .interactive()
        .properties(title=title)
    )

    return chart


@click.command()
@click.option(
    "--training-data",
    type=str,
    required=True,
    help="We read the training data from this file.",
)
@click.option(
    "--predictions",
    type=str,
    required=True,
    help="We read the test data predictions from this file.",
)
@click.option(
    "--output-chart",
    type=str,
    required=True,
    help="We store the visualized results.",
)
def main(
    training_data,
    predictions,
    output_chart,
):
    """
    Reads training data and predictions and creates a scatter plot to visualize the results.
    """

    data = pd.read_csv(predictions)

    # create a new column "correct" where column label matches column prediction
    data["correct"] = data["label"] == data["prediction"]

    chart1 = scatter_plot(data, "Are predictions correct?", "correct", "set1")

    data = pd.read_csv(training_data)

    chart2 = scatter_plot(data, "Training data", "label", "set2")

    # the resolve part makes sure that the legends don't get merged and that
    # the color scales remain independent
    combined_chart = alt.vconcat(chart1, chart2).properties(
        resolve=alt.Resolve(
            scale=alt.LegendResolveMap(color=alt.ResolveMode("independent"))
        ),
    )

    combined_chart.save(output_chart)
    print(f"Saved chart to {output_chart}")


if __name__ == "__main__":
    main()
