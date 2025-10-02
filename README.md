# Explainable by Design: Random Forest Classifier

This repository contains the implementation of an **Explainable Random Forest (RF) Classifier** that integrates explainability directly into the model design. Unlike post-hoc methods such as SHAP or LIME, this classifier provides **decision paths** and **feature contribution scores** as part of the prediction process.

---

## Features
- **Custom Random Forest implementation** built from scratch.
- **Explainability by design**:
  - Trace decision paths for each sample across all trees.
  - Compute feature contribution scores that quantify how much each feature influenced the prediction.
  - Export results in a human-readable and machine-friendly **JSON format**.
- **Evaluation tools** comparing performance and explanations against `scikit-learn`’s RandomForestClassifier.

---

## Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/if19b020/explainable_rf_thesis
pip install -r requirements.txt
```

## Example Usage
The following example shows how to train the Explainable Random Forest on the Wine dataset, make predictions, and export per-sample explanations to a JSON file:

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from random_forest import RandomForest

# Load dataset
wine_df = load_wine()
X, y = wine_df.data, wine_df.target
feature_names = wine_df.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Explainable Random Forest
forest = RandomForest(n_trees=5, max_depth=3, random_state=2)
forest.fit(X_train, y_train)

# Predict and export results with explanations
y_pred = forest.predict_and_export(
    X_test, 
    y_test, 
    feature_names=feature_names, 
    filename="../reports/wine_paths_contributions.json"
)
```