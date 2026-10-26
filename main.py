from decision_tree import entropy, information_gain, best_split, DecisionTreeClassifier

# Test 1: Play Tennis
y1 = ["Yes"] * 9 + ["No"] * 5

print("Play Tennis Entropy:", round(entropy(y1), 4))

# Test 2: Pure dataset
y2 = ["Yes", "Yes", "Yes", "Yes"]

print("Pure Dataset Entropy:", round(entropy(y2), 4))

# Test 3: Balanced dataset
y3 = ["Yes", "Yes", "No", "No"]

print("Balanced Dataset Entropy:", round(entropy(y3), 4))

# Test 4: Multiclass dataset
y4 = ["Cat", "Dog", "Bird", "Cat", "Dog", "Bird"]

print("Multiclass Entropy:", round(entropy(y4), 4))

# Parent: 9 Yes, 5 No
parent = ["Yes"] * 9 + ["No"] * 5

# Example split: Humidity = High vs Normal
high = ["Yes"] * 3 + ["No"] * 4
normal = ["Yes"] * 6 + ["No"] * 1

gain = information_gain(parent, high, normal)

print("Humidity Information Gain:", round(gain, 4))

X = {
    "Age": [22, 25, 47, 52, 46, 56],
    "City": [
        "Paris", "London", "Paris",
        "Delhi", "London", "Delhi"
    ]
}

y = ["No", "No", "Yes", "Yes", "Yes", "Yes"]

result = best_split(X, y)

print("\nBest Split Results")
print("Feature:", result["feature"])
print("Value:", result["value"])
print("Information Gain:", round(result["gain"], 4))
print("Split Type:", result["split_type"])

model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2
)

model.fit(X, y)

print("\nDecision Tree Structure:")
print(model.tree)

# Predict on training data
train_predictions = model.predict(X)

print("\nTraining Predictions:", train_predictions)
print("Actual Labels:", y)
print("Training Accuracy:", model.score(X, y))

# Predict on unseen data
new_data = {
    "Age": [20, 30, 40, 60],
    "City": ["Paris", "Delhi", "London", "Paris"]
}

new_predictions = model.predict(new_data)

print("\nNew Data Predictions:", new_predictions)

# Step 7: Play Tennis Dataset

X_tennis = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain", "Rain",
        "Rain", "Overcast", "Sunny", "Sunny", "Rain",
        "Sunny", "Overcast", "Overcast", "Rain"
    ],

    "Temperature": [
        "Hot", "Hot", "Hot", "Mild", "Cool",
        "Cool", "Cool", "Mild", "Cool", "Mild",
        "Mild", "Mild", "Hot", "Mild"
    ],

    "Humidity": [
        "High", "High", "High", "High", "Normal",
        "Normal", "Normal", "High", "Normal", "Normal",
        "Normal", "High", "Normal", "High"
    ],

    "Wind": [
        "Weak", "Strong", "Weak", "Weak", "Weak",
        "Strong", "Strong", "Weak", "Weak", "Weak",
        "Strong", "Strong", "Weak", "Strong"
    ]
}

y_tennis = [
    "No", "No", "Yes", "Yes", "Yes",
    "No", "Yes", "No", "Yes", "Yes",
    "Yes", "Yes", "Yes", "No"
]

tennis_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=2
)

tennis_model.fit(X_tennis, y_tennis)

print("\nPlay Tennis Decision Tree")
print(tennis_model.tree)

print(
    "\nPlay Tennis Training Accuracy:",
    tennis_model.score(X_tennis, y_tennis)
)

# Predict new weather conditions
new_weather = {
    "Outlook": ["Sunny", "Rain", "Overcast"],
    "Temperature": ["Cool", "Mild", "Hot"],
    "Humidity": ["High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak"]
}

print(
    "New Weather Predictions:",
    tennis_model.predict(new_weather)
)

print("\nTesting min_samples_leaf")

for leaf_size in [1, 2, 3, 5]:

    test_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=leaf_size
    )

    test_model.fit(X_tennis, y_tennis)

    print(
        f"min_samples_leaf={leaf_size}, "
        f"Accuracy={test_model.score(X_tennis, y_tennis):.4f}"
    )

print("\nNumerical Decision Tree:")
model.print_tree()

print("\nPlay Tennis Decision Tree:")
tennis_model.print_tree()

print("\nNumerical Tree Statistics:")
print(model.get_tree_stats())

print("\nPlay Tennis Tree Statistics:")
print(tennis_model.get_tree_stats())

print("\nTree Size Comparison:")

for depth in [1, 2, 3, 5]:

    test_model = DecisionTreeClassifier(max_depth=depth)
    test_model.fit(X_tennis, y_tennis)

    stats = test_model.get_tree_stats()

    print(
        f"max_depth={depth} | "
        f"Actual Depth={stats['depth']} | "
        f"Nodes={stats['nodes']} | "
        f"Leaves={stats['leaves']} | "
        f"Accuracy={test_model.score(X_tennis, y_tennis):.4f}"
    )

print("\nGenerating Decision Tree Visualizations...")

model.plot_tree("numerical_tree.png")

tennis_model.plot_tree("play_tennis_tree.png")

#Train/Test Evaluation

import numpy as np

rng = np.random.default_rng(42)

n_samples = len(y_tennis)
indices = rng.permutation(n_samples)

split_index = int(0.7 * n_samples)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = {
    feature: np.asarray(values)[train_indices]
    for feature, values in X_tennis.items()
}

X_test = {
    feature: np.asarray(values)[test_indices]
    for feature, values in X_tennis.items()
}

y_array = np.asarray(y_tennis)

y_train = y_array[train_indices]
y_test = y_array[test_indices]

evaluation_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1
)

evaluation_model.fit(X_train, y_train)

print("\nStep 12: Train/Test Evaluation")
print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))
print("Training accuracy:", round(evaluation_model.score(X_train, y_train), 4))
print("Testing accuracy:", round(evaluation_model.score(X_test, y_test), 4))
print("Actual test labels:", y_test)
print("Predicted test labels:", evaluation_model.predict(X_test))

# Step 13: Larger Dataset Evaluation

rng = np.random.default_rng(42)

n_samples = 500

age = rng.integers(18, 70, n_samples)
income = rng.integers(20000, 100000, n_samples)

# Synthetic classification rule
# Label is Yes when age > 30 and income > 50000
labels = np.where(
    (age > 30) & (income > 50000),
    "Yes",
    "No"
)

X_large = {
    "Age": age,
    "Income": income
}

indices = rng.permutation(n_samples)

train_idx = indices[:400]
test_idx = indices[400:]

X_large_train = {
    feature: values[train_idx]
    for feature, values in X_large.items()
}

X_large_test = {
    feature: values[test_idx]
    for feature, values in X_large.items()
}

y_large_train = labels[train_idx]
y_large_test = labels[test_idx]

large_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1
)

large_model.fit(X_large_train, y_large_train)

print("\nStep 13: Larger Dataset Evaluation")

print(
    "Training Accuracy:",
    round(large_model.score(X_large_train, y_large_train), 4)
)

print(
    "Testing Accuracy:",
    round(large_model.score(X_large_test, y_large_test), 4)
)

print("Tree Statistics:", large_model.get_tree_stats())

from csv_loader import load_csv

X_csv, y_csv = load_csv(
    "sample.csv",
    target_column="Purchased"
)

csv_model = DecisionTreeClassifier(max_depth=3)
csv_model.fit(X_csv, y_csv)

print("\nCSV Dataset Test")
print("Features:", list(X_csv.keys()))
print("Samples:", len(y_csv))
print("Training Accuracy:", csv_model.score(X_csv, y_csv))

csv_model.print_tree()