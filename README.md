# Decision Tree Classifier from Scratch

A Python implementation of a Decision Tree Classifier built from scratch using NumPy, with an interactive Streamlit web interface.

The project supports numerical and categorical features, configurable tree-size hyperparameters, CSV dataset loading, model evaluation, and decision tree visualization.

## Features

- Decision Tree Classifier implemented without Scikit-learn
- Entropy and information gain calculations
- Automatic splitting for numerical and categorical features
- Recursive binary tree construction
- Configurable `max_depth`, `min_samples_split`, and `min_samples_leaf`
- Upload cleaned CSV datasets
- Select the target column
- Training and testing accuracy evaluation
- Tree statistics and visualization
- Download decision tree diagrams as PNG
- Automated unit tests using Python's `unittest`

## Technologies Used

- Python
- NumPy
- Matplotlib
- Streamlit

## Project Structure

```text
Decision-Tree-Project/
├── app.py
├── decision_tree.py
├── csv_loader.py
├── main.py
├── run_csv.py
├── test_decision_tree.py
├── sample.csv
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Decision-Tree-Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## How to Use

1. Upload a cleaned CSV dataset.
2. Select the target column.
3. Configure the tree hyperparameters.
4. Click **Train Decision Tree**.
5. Review training and testing accuracy.
6. Explore the generated decision tree visualization.
7. Download the tree diagram as a PNG image.

## Run Using the Terminal

```bash
python run_csv.py
```

Follow the prompts to provide the CSV file path, target column, and hyperparameters.

## Algorithm

The classifier uses entropy to measure class uncertainty:

\[
H(S)=-\sum_{i=1}^{k}p_i\log_2(p_i)
\]

Information gain measures the reduction in entropy after splitting:

\[
IG(S)=H(S)-\frac{|S_L|}{|S|}H(S_L)-\frac{|S_R|}{|S|}H(S_R)
\]

At each node, the algorithm selects the valid split with the highest information gain and recursively builds the left and right subtrees.

Tree growth is controlled by maximum depth, minimum samples required to split, minimum samples per leaf, and class purity.

## Testing

Run the automated tests:

```bash
python -m unittest test_decision_tree.py -v
```

The project includes tests for entropy, information gain, splitting, predictions, and tree-size constraints.

## Dataset Requirements

The application accepts cleaned CSV datasets with:

- Column headers
- Numerical or categorical features
- A classification target column
- No missing values

Dates, free-text fields, and complex data types may require preprocessing.

## Limitations

- The classifier currently uses information gain as its splitting criterion.
- Categorical features are split using binary equality comparisons.
- Missing values are not automatically imputed.
- Very large trees may produce complex diagrams.
- A single random train/test split may not reliably estimate performance on small datasets.

## Project Objective

To understand and implement the internal workings of a Decision Tree Classifier without relying on a prebuilt machine learning classifier, while providing an accessible interface for training and visualization.