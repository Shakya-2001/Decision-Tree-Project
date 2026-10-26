
import streamlit as st
import csv
import io
import html
import os
import tempfile
import numpy as np

from csv_loader import load_csv
from decision_tree import DecisionTreeClassifier


# Configure Streamlit page
st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌳",
    layout="wide"
)

st.title("🌳 Decision Tree Classifier")
st.write(
    "Train and visualize a Decision Tree Classifier "
    "built from scratch using your own cleaned CSV dataset."
)


# Upload CSV dataset
uploaded_file = st.file_uploader(
    "Upload your cleaned CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))

        if not reader.fieldnames:
            st.error("CSV file must contain column headers.")
            st.stop()

        columns = reader.fieldnames
        rows = list(reader)

        if not rows:
            st.error("The uploaded CSV contains no data.")
            st.stop()

        st.success(
            f"Dataset uploaded successfully! {len(rows)} rows loaded."
        )

        # Display dataset preview
        st.subheader("📋 Dataset Preview")

        preview_rows = rows[:10]

        table_html = """
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;">
        <thead><tr>
        """

        for column in columns:
            table_html += (
                "<th style='padding:10px;border:1px solid gray;'>"
                f"{html.escape(column)}</th>"
            )

        table_html += "</tr></thead><tbody>"

        for row in preview_rows:
            table_html += "<tr>"

            for column in columns:
                value = row[column]

                table_html += (
                    "<td style='padding:10px;border:1px solid gray;'>"
                    f"{html.escape(str(value))}</td>"
                )

            table_html += "</tr>"

        table_html += "</tbody></table></div>"

        st.html(table_html)
        st.caption("Showing the first 10 rows.")

        # Display dataset information
        st.subheader("📊 Dataset Information")

        info1, info2 = st.columns(2)

        info1.metric("Total Samples", len(rows))
        info2.metric("Total Columns", len(columns))

        # Configure model parameters
        st.subheader("⚙️ Model Configuration")

        target_column = st.selectbox(
            "Select Target Column",
            columns
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            max_depth = st.number_input(
                "Maximum Depth",
                min_value=0,
                max_value=30,
                value=5,
                step=1
            )

        with col2:
            min_samples_split = st.number_input(
                "Minimum Samples Split",
                min_value=2,
                value=2,
                step=1
            )

        with col3:
            min_samples_leaf = st.number_input(
                "Minimum Samples Leaf",
                min_value=1,
                value=1,
                step=1
            )

        st.info(
            "Maximum Depth controls how deep the tree grows. "
            "Minimum Samples Split controls when a node can split. "
            "Minimum Samples Leaf controls the minimum records in a leaf."
        )

        # Train decision tree
        if st.button(
            "🌳 Train Decision Tree",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner("Training Decision Tree..."):

                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        suffix=".csv",
                        delete=False
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    try:
                        X, y = load_csv(
                            temp_path,
                            target_column
                        )
                    finally:
                        os.remove(temp_path)

                    if len(y) < 2:
                        raise ValueError(
                            "Dataset must contain at least 2 samples."
                        )

                    # Split data into training and testing sets
                    rng = np.random.default_rng(42)
                    indices = rng.permutation(len(y))

                    split_index = max(
                        1,
                        min(len(y) - 1, int(0.8 * len(y)))
                    )

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

                    # Initialize and train classifier
                    model = DecisionTreeClassifier(
                        max_depth=int(max_depth),
                        min_samples_split=int(min_samples_split),
                        min_samples_leaf=int(min_samples_leaf)
                    )

                    model.fit(X_train, y_train)

                    # Calculate model accuracy
                    train_accuracy = model.score(X_train, y_train)
                    test_accuracy = model.score(X_test, y_test)

                    # Get tree statistics
                    stats = model.get_tree_stats()

                st.success("Decision Tree trained successfully!")

                # Display model performance
                st.subheader("🎯 Model Performance")

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Training Accuracy",
                    f"{train_accuracy * 100:.2f}%"
                )

                metric2.metric(
                    "Testing Accuracy",
                    f"{test_accuracy * 100:.2f}%"
                )

                st.write(f"Training Samples: {len(y_train)}")
                st.write(f"Testing Samples: {len(y_test)}")

                # Display tree statistics
                st.subheader("🌲 Tree Statistics")

                stat1, stat2, stat3 = st.columns(3)

                stat1.metric("Tree Depth", stats["depth"])
                stat2.metric("Total Nodes", stats["nodes"])
                stat3.metric("Leaf Nodes", stats["leaves"])

                # Generate decision tree visualization
                st.subheader("🌳 Decision Tree Visualization")

                if stats["nodes"] > 100:
                    st.warning(
                        "This tree contains more than 100 nodes. "
                        "The diagram may be difficult to read."
                    )

                with st.spinner("Generating tree diagram..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        tree_path = os.path.join(
                            temp_dir,
                            "decision_tree.png"
                        )

                        model.plot_tree(filename=tree_path)

                        with open(tree_path, "rb") as file:
                            tree_image = file.read()

                # Display tree image
                st.image(
                    tree_image,
                    caption="Trained Decision Tree",
                    use_container_width=True
                )

                # Download tree visualization
                st.download_button(
                    "⬇️ Download Decision Tree PNG",
                    data=tree_image,
                    file_name="decision_tree.png",
                    mime="image/png"
                )

                # Generate predictions on test data
                st.subheader("🔍 Test Predictions")

                predictions = model.predict(X_test)

                correct = int(np.sum(predictions == y_test))

                st.write(
                    f"Correct Predictions: {correct} / {len(y_test)}"
                )

                # Create predictions table
                prediction_html = """
                <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;">
                <tr>
                <th style="padding:8px;border:1px solid gray;">Actual</th>
                <th style="padding:8px;border:1px solid gray;">Predicted</th>
                <th style="padding:8px;border:1px solid gray;">Result</th>
                </tr>
                """

                for actual, predicted in zip(
                    y_test[:20],
                    predictions[:20]
                ):
                    result = (
                        "Correct"
                        if actual == predicted
                        else "Incorrect"
                    )

                    prediction_html += (
                        "<tr>"
                        f"<td style='padding:8px;border:1px solid gray;'>{html.escape(str(actual))}</td>"
                        f"<td style='padding:8px;border:1px solid gray;'>{html.escape(str(predicted))}</td>"
                        f"<td style='padding:8px;border:1px solid gray;'>{result}</td>"
                        "</tr>"
                    )

                prediction_html += "</table></div>"

                st.html(prediction_html)
                st.caption("Showing up to 20 test predictions.")

                if len(y_test) < 20:
                    st.warning(
                        "The test set is small, so accuracy may not "
                        "reliably estimate real-world performance."
                    )

            except (
                ValueError,
                OSError,
                TypeError,
                RecursionError
            ) as error:
                st.error(f"Training failed: {error}")

    except (
        UnicodeError,
        csv.Error,
        ValueError
    ) as error:
        st.error(f"Unable to read CSV: {error}")

else:
    st.info("Upload a cleaned CSV file to begin training.")
