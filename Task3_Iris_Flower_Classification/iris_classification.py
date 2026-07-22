"""
Iris Flower Classification
Classifies iris flowers into setosa / versicolor / virginica based on
sepal and petal measurements.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("IRIS.csv")
print(f"Shape: {df.shape}")
print(df["species"].value_counts())

X = df.drop(columns=["species"])
y = df["species"]

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified to keep class balance in both sets)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale features - helps distance/margin-based models (KNN, SVM, Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Train and compare several classifiers
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
    results.append({
        "Model": name,
        "Test Accuracy": round(acc, 4),
        "CV Accuracy (mean)": round(cv_scores.mean(), 4),
        "CV Accuracy (std)": round(cv_scores.std(), 4),
    })
    fitted[name] = model

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Closer look at the best model
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
best_preds = best_model.predict(X_test_scaled)
print(f"\nBest model: {best_name}")
print("\nClassification report:")
print(classification_report(y_test, best_preds))
print("Confusion matrix:")
print(confusion_matrix(y_test, best_preds))

# Feature importance (if available)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).to_string())

results_df.to_csv("iris_model_comparison.csv", index=False)
