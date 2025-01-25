import click
import pandas as pd
import numpy as np
import sys


@click.command()
@click.argument("file1")
@click.argument("file2")
def main(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    tolerance = 1.0e-8
    comparison = np.isclose(df1.values, df2.values, rtol=tolerance)

    mismatched = ~comparison.all(axis=1)
    mismatched_rows = df1[mismatched]

    if len(mismatched_rows) > 0:
        print("Mismatched rows:", mismatched_rows)
        sys.exit(1)
    else:
        print(f"Files {file1} and {file2} are identical (within numerical tolerance)")


if __name__ == "__main__":
    main()
