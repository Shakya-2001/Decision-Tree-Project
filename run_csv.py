from pathlib import Path
import numpy as np

from csv_loader import load_csv
from decision_tree import DecisionTreeClassifier


def main():
    print("\n===== Decision Tree Classifier =====")

    file_path = input("Enter CSV file path: ").strip().strip('"')
    target_column = input("Enter target column name: ").strip()

    max_depth = int(input("Enter max_depth (e.g. 5): "))
    min_samples_split = int(input("Enter min_samples_split (e.g. 2): "))
    min_samples_leaf = int(input("Enter min_samples_leaf (e.g. 1): "))

    if not Path(file_path).is_file():
        print("Error: CSV file not found.")
        return

    try:
        X, y = load_csv(file_path, target_column)

        print("\nDataset Information")
        print("Total samples:", len(y))
        print("Features:", list(X.keys()))
        print("Classes:", np.unique(y))

        if len(y) < 2:
            raise ValueError("At least two samples are required.")

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(y))

        split_index = max(1, min(len(y) - 1, int(0.8 * len(y))))

        train_idx = indices[:split_index]
        test_idx = indices[split_index:]

        X_train = {
            feature: values[train_idx]
            for feature, values in X.items()
        }

        X_test = {
            feature: values[test_idx]
            for feature, values in X.items()
        }

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        model.fit(X_train, y_train)

        print("\n===== Evaluation =====")
        print("Training samples:", len(y_train))
        print("Testing samples:", len(y_test))
        print("Training accuracy:", round(model.score(X_train, y_train), 4))
        print("Testing accuracy:", round(model.score(X_test, y_test), 4))

        print("\n===== Tree Statistics =====")
        print(model.get_tree_stats())

        print("\n===== Tree Structure =====")
        model.print_tree()

        visualize = input("\nVisualize tree? (yes/no): ").strip().lower()

        if visualize in ("yes", "y"):
            model.plot_tree(filename="custom_decision_tree.png")

    except (ValueError, OSError, TypeError) as error:
        print("Error:", error)


if __name__ == "__main__":
    main()