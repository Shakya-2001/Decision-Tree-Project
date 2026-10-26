import numpy as np
import matplotlib.pyplot as plt

def entropy(y):
    y = np.asarray(y)

    if len(y) == 0:
        return 0.0

    # Count samples belonging to each class
    classes, counts = np.unique(y, return_counts=True)

    # Calculate probability of each class
    probabilities = counts / len(y)

    # Calculate entropy
    entropy_value = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy_value)


def information_gain(parent, left_child, right_child):
    # Calculate information gain for a binary split
    parent = np.asarray(parent)
    left_child = np.asarray(left_child)
    right_child = np.asarray(right_child)

    n = len(parent)
    n_left = len(left_child)
    n_right = len(right_child)

    # Reject invalid splits
    if n == 0 or n_left == 0 or n_right == 0:
        return 0.0

    if n_left + n_right != n:
        raise ValueError(
            "Child sample counts must equal parent sample count."
        )

    # Parent entropy
    parent_entropy = entropy(parent)

    # Weighted entropy of children
    weighted_entropy = (
        (n_left / n) * entropy(left_child)
        + (n_right / n) * entropy(right_child)
    )

    # Information gain
    gain = parent_entropy - weighted_entropy

    return float(gain)


def best_split(X, y, min_samples_leaf=1):
    """
    Find the best binary split using Information Gain.

    X: Dictionary containing feature names and values
    y: Target labels
    min_samples_leaf: Minimum samples required in each child
    """

    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be >= 1")

    y = np.asarray(y)

    best_gain = 0.0
    best_feature = None
    best_value = None
    best_mask = None
    best_split_type = None

    for feature, values in X.items():

        values = np.asarray(values)

        if len(values) != len(y):
            raise ValueError("Feature and target lengths must match.")

        # Check whether feature is numerical
        is_numeric = all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_))
            for v in values
        )

        if is_numeric:

            unique_values = np.sort(np.unique(values))

            # Midpoints between consecutive values
            thresholds = (
                unique_values[:-1] + unique_values[1:]
            ) / 2

            for threshold in thresholds:

                mask = values <= threshold

                # Ensure both children have enough samples
                if (
                    np.sum(mask) < min_samples_leaf
                    or np.sum(~mask) < min_samples_leaf
                ):
                    continue

                gain = information_gain(
                    y, y[mask], y[~mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = threshold
                    best_mask = mask
                    best_split_type = "numerical"

        else:

            for category in np.unique(values):

                mask = values == category

                # Ensure both children have enough samples
                if (
                    np.sum(mask) < min_samples_leaf
                    or np.sum(~mask) < min_samples_leaf
                ):
                    continue

                gain = information_gain(
                    y, y[mask], y[~mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = category
                    best_mask = mask
                    best_split_type = "categorical"

    return {
        "feature": best_feature,
        "value": best_value,
        "gain": best_gain,
        "mask": best_mask,
        "split_type": best_split_type
    }


class DecisionTreeClassifier:

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1
    ):

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):

        y = np.asarray(y)

        if not isinstance(X, dict):
            raise TypeError("X must be a dictionary")

        if len(y) == 0:
            raise ValueError("Dataset cannot be empty")

        if not X:
            raise ValueError("X must contain at least one feature")

        for feature, values in X.items():
            if len(values) != len(y):
                raise ValueError(
                    f"Length mismatch in feature: {feature}"
                )

        self.tree = self._build_tree(X, y, depth=0)

        return self

    def _build_tree(self, X, y, depth):

        classes, counts = np.unique(
            y, return_counts=True
        )

        majority_class = classes[np.argmax(counts)]

        # Stopping conditions
        if (
            len(classes) == 1
            or len(y) < self.min_samples_split
            or (
                self.max_depth is not None
                and depth >= self.max_depth
            )
        ):
            return {
                "leaf": True,
                "prediction": majority_class
            }

        # Find best split
        split = best_split(
            X,
            y,
            min_samples_leaf=self.min_samples_leaf
        )

        if split["feature"] is None or split["gain"] <= 0:
            return {
                "leaf": True,
                "prediction": majority_class
            }

        mask = split["mask"]

        # Divide dataset into left and right groups
        left_X = {
            feature: np.asarray(values)[mask]
            for feature, values in X.items()
        }

        right_X = {
            feature: np.asarray(values)[~mask]
            for feature, values in X.items()
        }

        # Recursively build child nodes
        left_tree = self._build_tree(
            left_X, y[mask], depth + 1
        )

        right_tree = self._build_tree(
            right_X, y[~mask], depth + 1
        )

        return {
            "leaf": False,
            "feature": split["feature"],
            "value": split["value"],
            "split_type": split["split_type"],
            "gain": split["gain"],
            "prediction": majority_class,
            "left": left_tree,
            "right": right_tree
        }

    def _predict_one(self, sample, node):

        if node["leaf"]:
            return node["prediction"]

        feature = node["feature"]
        value = node["value"]

        if node["split_type"] == "numerical":
            go_left = sample[feature] <= value
        else:
            go_left = sample[feature] == value

        if go_left:
            return self._predict_one(sample, node["left"])
        else:
            return self._predict_one(sample, node["right"])

    def predict(self, X):

        if self.tree is None:
            raise ValueError("Train the model before predicting.")

        if not isinstance(X, dict):
            raise TypeError("X must be a dictionary")

        if not X:
            raise ValueError("X cannot be empty")

        lengths = [len(values) for values in X.values()]

        if len(set(lengths)) != 1:
            raise ValueError("All features must have equal lengths")

        predictions = []

        for i in range(lengths[0]):

            sample = {
                feature: values[i]
                for feature, values in X.items()
            }

            prediction = self._predict_one(
                sample, self.tree
            )

            predictions.append(prediction)

        return np.asarray(predictions)

    def score(self, X, y):

        y = np.asarray(y)

        predictions = self.predict(X)

        if len(y) == 0 or len(y) != len(predictions):
            raise ValueError("Invalid target length")

        accuracy = np.mean(predictions == y)

        return float(accuracy)

    def print_tree(self):
        if self.tree is None:
            raise ValueError("Train the model before displaying the tree.")
        self._print_node(self.tree, depth=0)

    def _print_node(self, node, depth):

        indent = "    " * depth

        if node["leaf"]:
            print(f"{indent}Predict: {node['prediction']}")
            return

        feature = node["feature"]
        value = node["value"]
        gain = node["gain"]

        if node["split_type"] == "numerical":
            condition = f"{feature} <= {value}"
        else:
            condition = f"{feature} == {value}"

        print(f"{indent}IF {condition} (Gain: {gain:.4f})")

        print(f"{indent}  TRUE:")
        self._print_node(node["left"], depth + 1)

        print(f"{indent}  FALSE:")
        self._print_node(node["right"], depth + 1)

    def get_tree_stats(self):

        if self.tree is None:
            raise ValueError("Train the model first.")

        return self._calculate_stats(self.tree)

    def _calculate_stats(self, node):

        if node["leaf"]:
            return {
                "depth": 0,
                "nodes": 1,
                "leaves": 1
            }

        left = self._calculate_stats(node["left"])
        right = self._calculate_stats(node["right"])

        return {
            "depth": 1 + max(left["depth"], right["depth"]),
            "nodes": 1 + left["nodes"] + right["nodes"],
            "leaves": left["leaves"] + right["leaves"]
        }

    def plot_tree(self, filename=None):

        if self.tree is None:
            raise ValueError("Train the model first.")

        stats = self.get_tree_stats()

        fig_width = max(10, stats["leaves"] * 2.5)
        fig_height = max(5, (stats["depth"] + 1) * 2)

        fig, ax = plt.subplots(
            figsize=(fig_width, fig_height)
        )

        ax.axis("off")

        def count_leaves(node):
            if node["leaf"]:
                return 1

            return (
                count_leaves(node["left"])
                + count_leaves(node["right"])
            )

        def draw_node(node, x, y, width):

            if node["leaf"]:
                label = f"Predict: {node['prediction']}"
                color = "#C8E6C9"

            else:
                operator = (
                    "<=" if node["split_type"] == "numerical"
                    else "=="
                )

                label = (
                    f"{node['feature']} {operator} {node['value']}\n"
                    f"Gain: {node['gain']:.3f}"
                )

                color = "#BBDEFB"

            ax.text(
                x, y, label,
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=color,
                    edgecolor="black"
                ),
                transform=ax.transAxes
            )

            if node["leaf"]:
                return

            left_count = count_leaves(node["left"])
            right_count = count_leaves(node["right"])
            total = left_count + right_count

            left_width = width * left_count / total
            right_width = width * right_count / total

            left_x = x - width / 2 + left_width / 2
            right_x = x + width / 2 - right_width / 2

            next_y = y - 0.8 / (stats["depth"] + 1)

            for child_x, branch in [
                (left_x, "True"),
                (right_x, "False")
            ]:
                ax.annotate(
                    "",
                    xy=(child_x, next_y + 0.025),
                    xytext=(x, y - 0.025),
                    xycoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray"
                    )
                )

                ax.text(
                    (x + child_x) / 2,
                    (y + next_y) / 2,
                    branch,
                    fontsize=8,
                    ha="center",
                    transform=ax.transAxes
                )

            draw_node(
                node["left"],
                left_x,
                next_y,
                left_width
            )

            draw_node(
                node["right"],
                right_x,
                next_y,
                right_width
            )

        draw_node(self.tree, 0.5, 0.92, 0.9)

        if filename:
            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight"
            )
            print(f"Tree saved to {filename}")

        plt.show()