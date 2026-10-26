# Decision Tree Classifier from Scratch — Project Report

## 1. GitHub Repository

**Repository Link:** https://github.com/Shakya-2001/Decision-Tree-Project

## 2. Project Description

The objective of this project was to implement a Decision Tree Classifier from scratch in Python without using prebuilt machine learning classification algorithms.

The classifier uses entropy and information gain to identify the best feature splits and recursively construct a binary decision tree.

I also developed an interactive web application using Streamlit, allowing users to upload cleaned CSV datasets, configure model parameters, train the classifier, evaluate its performance, and visualize the resulting decision tree.

## 3. Implementation

The following components were implemented:

**Entropy and Information Gain**

Implemented entropy to measure uncertainty in class labels and information gain to evaluate possible splits.

**Decision Tree Construction**

Implemented recursive binary splitting supporting both numerical and categorical features. Numerical features use threshold-based comparisons, while categorical features use equality comparisons.

**Hyperparameter Support**

Implemented three parameters to control tree complexity:

- `max_depth` — Maximum allowed tree depth
- `min_samples_split` — Minimum samples required to split a node
- `min_samples_leaf` — Minimum samples allowed in each leaf

**Prediction and Evaluation**

Implemented prediction traversal, classification accuracy, and training/testing evaluation.

**CSV Dataset Support**

Developed a CSV loader that detects numerical and categorical feature columns, allowing users to train the classifier on different cleaned classification datasets.

**Visualization**

Implemented tree visualization using Matplotlib, displaying decision nodes, splitting conditions, and predicted classes.

**Streamlit User Interface**

Developed a web interface with CSV upload, target-column selection, hyperparameter configuration, training, accuracy metrics, tree statistics, visualization, and PNG download.

## 4. Technologies Used

- Python
- NumPy
- Matplotlib
- Streamlit
- Python `unittest`
- Python `csv`

The Decision Tree learning algorithm was implemented manually without using Scikit-learn's DecisionTreeClassifier.

## 5. Testing and Results

### Play Tennis Dataset

| Metric | Result |
|---|---:|
| Total samples | 14 |
| Full-dataset training accuracy | 100% |
| Training accuracy (70/30 split) | 88.89% |
| Testing accuracy (70/30 split) | 60.00% |
| Full tree depth | 4 |
| Full tree nodes | 13 |
| Full tree leaves | 7 |

The relatively low testing accuracy reflects the very small test set of five samples and should not be interpreted as a reliable estimate of generalization performance.

### Synthetic Classification Dataset

| Metric | Result |
|---|---:|
| Total samples | 500 |
| Training samples | 400 |
| Testing samples | 100 |
| Training accuracy | 100% |
| Testing accuracy | 100% |
| Tree depth | 2 |
| Total nodes | 5 |
| Leaf nodes | 3 |

The synthetic dataset followed a deterministic rule involving age and income, which the classifier successfully learned.

### Automated Unit Tests

Seven automated tests were executed successfully, covering entropy, information gain, best split selection, predictions, maximum depth, and minimum leaf size.

**Result: 7/7 tests passed.**

## 6. Challenges and Solutions

One challenge was supporting both numerical and categorical features without relying on a prebuilt machine learning library. This was addressed by implementing separate splitting strategies for numerical thresholds and categorical equality comparisons.

Another challenge was making the implementation reusable across datasets. I addressed this by creating a generic CSV loader and a Streamlit interface.

During development, Windows Application Control blocked a Pandas dependency. To avoid relying on Pandas for dataset previews, I used Python's built-in CSV library and HTML table rendering.

## 7. Limitations

The current implementation requires cleaned classification datasets without missing values. It does not automatically preprocess dates or free-text features.

The classifier uses information gain for splitting, and large trees may produce difficult-to-read visualizations. Performance evaluation currently uses a single random train/test split.

## 8. Conclusion

This project provided practical experience implementing a Decision Tree Classifier from first principles, including entropy, information gain, recursive splitting, prediction, model evaluation, and tree complexity control.

The completed implementation also includes a functional Streamlit interface, allowing users to train and visualize custom Decision Trees using their own cleaned CSV datasets.