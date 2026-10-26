import unittest
import numpy as np

from decision_tree import (
    entropy,
    information_gain,
    best_split,
    DecisionTreeClassifier
)


class TestDecisionTree(unittest.TestCase):

    def test_entropy_pure(self):
        self.assertAlmostEqual(
            entropy(["Yes", "Yes", "Yes"]),
            0.0
        )

    def test_entropy_balanced(self):
        self.assertAlmostEqual(
            entropy(["Yes", "No"]),
            1.0
        )

    def test_information_gain(self):
        parent = ["Yes", "Yes", "No", "No"]
        left = ["Yes", "Yes"]
        right = ["No", "No"]

        self.assertAlmostEqual(
            information_gain(parent, left, right),
            1.0
        )

    def test_best_split(self):
        X = {
            "Age": [22, 25, 47, 52, 46, 56]
        }

        y = ["No", "No", "Yes", "Yes", "Yes", "Yes"]

        result = best_split(X, y)

        self.assertEqual(result["feature"], "Age")
        self.assertAlmostEqual(result["value"], 35.5)

    def test_predictions(self):
        X = {
            "Age": [22, 25, 47, 52, 46, 56]
        }

        y = ["No", "No", "Yes", "Yes", "Yes", "Yes"]

        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)

        predictions = model.predict(X)

        np.testing.assert_array_equal(
            predictions,
            np.asarray(y)
        )

    def test_max_depth(self):
        X = {
            "Age": [22, 25, 47, 52, 46, 56]
        }

        y = ["No", "No", "Yes", "Yes", "Yes", "Yes"]

        model = DecisionTreeClassifier(max_depth=0)
        model.fit(X, y)

        stats = model.get_tree_stats()

        self.assertEqual(stats["depth"], 0)
        self.assertEqual(stats["nodes"], 1)

    def test_min_samples_leaf(self):
        X = {
            "Age": [10, 20, 30, 40]
        }

        y = ["No", "No", "No", "Yes"]

        model = DecisionTreeClassifier(
            min_samples_leaf=2
        )

        model.fit(X, y)

        stats = model.get_tree_stats()

        # A valid split creates two leaves of two samples each
        self.assertEqual(stats["leaves"], 2)

        # With min_samples_leaf=3, no valid split is possible
        restricted_model = DecisionTreeClassifier(
            min_samples_leaf=3
        )

        restricted_model.fit(X, y)

        self.assertEqual(
            restricted_model.get_tree_stats()["leaves"],
            1
        )


if __name__ == "__main__":
    unittest.main()