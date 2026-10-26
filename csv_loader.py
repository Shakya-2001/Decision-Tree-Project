import csv
import numpy as np


def load_csv(file_path, target_column):

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV must contain column headers.")

        columns = reader.fieldnames

        if len(columns) != len(set(columns)):
            raise ValueError("CSV contains duplicate column names.")

        if target_column not in columns:
            raise ValueError(
                f"Target '{target_column}' not found. "
                f"Available columns: {columns}"
            )

        feature_names = [
            column for column in columns
            if column != target_column
        ]

        if not feature_names:
            raise ValueError("At least one feature column is required.")

        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    X = {}

    for feature in feature_names:

        values = [row[feature].strip() for row in rows]

        if any(value == "" for value in values):
            raise ValueError(
                f"Missing values found in '{feature}'."
            )

        try:
            X[feature] = np.array(
                [float(value) for value in values]
            )
        except ValueError:
            X[feature] = np.array(values)

    labels = [row[target_column].strip() for row in rows]

    if any(label == "" for label in labels):
        raise ValueError("Target column contains missing values.")

    y = np.array(labels)

    return X, y