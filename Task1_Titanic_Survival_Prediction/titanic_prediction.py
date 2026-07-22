"""
Titanic Survival Prediction
Data Science Internship - Task 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

sns.set_style("whitegrid")

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Titanic-Dataset-selected-columns.csv")
print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# 2. EXPLORATORY DATA ANALYSIS
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Overall survival counts
sns.countplot(x="Survived", data=df, ax=axes[0, 0], palette=["#c0392b", "#27ae60"])
axes[0, 0].set_title("Overall Survival Counts")
axes[0, 0].set_xticklabels(["Died", "Survived"])

# Survival by sex
sns.countplot(x="Sex", hue="Survived", data=df, ax=axes[0, 1], palette=["#c0392b", "#27ae60"])
axes[0, 1].set_title("Survival by Sex")
axes[0, 1].legend(title="Survived", labels=["Died", "Survived"])

# Survival by passenger class
sns.countplot(x="Pclass", hue="Survived", data=df, ax=axes[0, 2], palette=["#c0392b", "#27ae60"])
axes[0, 2].set_title("Survival by Passenger Class")
axes[0, 2].legend(title="Survived", labels=["Died", "Survived"])

# Age distribution by survival
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=30, ax=axes[1, 0], palette=["#c0392b", "#27ae60"])
axes[1, 0].set_title("Age Distribution by Survival")

# Fare distribution by survival
sns.boxplot(x="Survived", y="Fare", data=df, ax=axes[1, 1], palette=["#c0392b", "#27ae60"])
axes[1, 1].set_title("Fare by Survival")
axes[1, 1].set_xticklabels(["Died", "Survived"])

# Family size vs survival
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
sns.countplot(x="FamilySize", hue="Survived", data=df, ax=axes[1, 2], palette=["#c0392b", "#27ae60"])
axes[1, 2].set_title("Survival by Family Size")
axes[1, 2].legend(title="Survived", labels=["Died", "Survived"])

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=120)
plt.close()
print("\nSaved EDA visualization -> eda_overview.png")

# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------

# Extract title from Name (Mr, Mrs, Miss, Master, Rare, etc.)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"
}
df["Title"] = df["Title"].replace(title_map)
df["Title"] = df["Title"].apply(lambda t: t if t in ["Mr", "Mrs", "Miss", "Master", "Rare"] else "Rare")

# Impute Age using median age per Title (more accurate than a single global median)
df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

# Family features
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Fare per person (accounts for shared tickets)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# Encode categorical variables
le_sex = LabelEncoder()
df["Sex_enc"] = le_sex.fit_transform(df["Sex"])  # female=0, male=1

le_title = LabelEncoder()
df["Title_enc"] = le_title.fit_transform(df["Title"])

# Age bins (children/teens often had different survival odds)
df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)

# -----------------------------
# 4. SELECT FEATURES
# -----------------------------
features = ["Pclass", "Sex_enc", "Age", "SibSp", "Parch", "Fare",
            "FamilySize", "IsAlone", "FarePerPerson", "Title_enc", "AgeBin"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. TRAIN MODELS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    results[name] = {
        "model": model, "preds": preds, "probs": probs,
        "accuracy": acc, "auc": auc, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    }

    print(f"\n=== {name} ===")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"ROC AUC       : {auc:.4f}")
    print(f"5-fold CV Acc : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, preds, target_names=["Died", "Survived"]))

# Pick the best model by test accuracy
best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]
print(f"\nBest model: {best_name} (Accuracy: {best['accuracy']:.4f})")

# -----------------------------
# 6. VISUALIZE RESULTS
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Confusion matrix for best model
cm = confusion_matrix(y_test, best["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
axes[0].set_title(f"Confusion Matrix - {best_name}")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# ROC curves for both models
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC Curves")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances.values, y=importances.index, ax=axes[2], palette="viridis")
axes[2].set_title("Feature Importance (Random Forest)")

plt.tight_layout()
plt.savefig("model_results.png", dpi=120)
plt.close()
print("\nSaved model evaluation visualization -> model_results.png")

# -----------------------------
# 7. SAVE SUMMARY REPORT
# -----------------------------
with open("model_summary.txt", "w") as f:
    f.write("TITANIC SURVIVAL PREDICTION - MODEL SUMMARY\n")
    f.write("=" * 45 + "\n\n")
    for name, res in results.items():
        f.write(f"{name}\n")
        f.write(f"  Test Accuracy : {res['accuracy']:.4f}\n")
        f.write(f"  ROC AUC       : {res['auc']:.4f}\n")
        f.write(f"  5-fold CV Acc : {res['cv_mean']:.4f} (+/- {res['cv_std']:.4f})\n\n")
    f.write(f"Best Model: {best_name}\n\n")
    f.write("Top Features (Random Forest importance):\n")
    for feat, imp in importances.items():
        f.write(f"  {feat}: {imp:.4f}\n")

print("\nSaved summary -> model_summary.txt")
print("\nDone!")
