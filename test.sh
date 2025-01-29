#!/usr/bin/env bash

# create the data directory if it does not exist
echo "Creating data directory"
pwd 
mkdir -p data
echo "Created data directory"

python generate_data.py \
        --num-samples 50 \
        --training-data data/train.csv \
        --test-data data/test.csv

python reference/compare-csv-files.py data/train.csv reference/train.csv
python reference/compare-csv-files.py data/test.csv  reference/test.csv

python generate_predictions.py \
        --num-neighbors 7 \
        --training-data data/train.csv \
        --test-data data/test.csv \
        --predictions data/predictions.csv

python reference/compare-csv-files.py data/predictions.csv reference/predictions.csv
