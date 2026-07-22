"""
Movie Rating Prediction with Python
Predicts an IMDb movie's rating from Genre, Director, and Actor features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/IMDb_Movies_India.csv", encoding="latin1")
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Clean columns
# ---------------------------------------------------------------------------
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float)
df["Duration"] = df["Duration"].astype(str).str.extract(r"(\d+)").astype(float)
df["Votes"] = (
    df["Votes"].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)").astype(float)
)

# Drop rows without a target rating - can't train/evaluate without it
df = df.dropna(subset=["Rating"]).copy()
print(f"Shape after dropping missing Rating: {df.shape}")

# Fill remaining gaps
df["Year"] = df["Year"].fillna(df["Year"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())
df["Votes"] = df["Votes"].fillna(0)
for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = df[col].fillna("Unknown")

df["Votes_log"] = np.log1p(df["Votes"])

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
# Primary genre (first listed) + one-hot of top genres
df["Genre_main"] = df["Genre"].apply(lambda g: g.split(",")[0].strip())
top_genres = df["Genre_main"].value_counts().nlargest(15).index
df["Genre_main"] = df["Genre_main"].where(df["Genre_main"].isin(top_genres), "Other")
genre_dummies = pd.get_dummies(df["Genre_main"], prefix="Genre")

# Target-encode Director / Actors by their historical mean rating
# (smoothed toward the global mean to avoid overfitting on rare names)
global_mean = df["Rating"].mean()

def smoothed_target_encode(series, target, m=5):
    stats = target.groupby(series).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
    return series.map(smoothed)

df["Director_enc"] = smoothed_target_encode(df["Director"], df["Rating"])
df["Actor1_enc"] = smoothed_target_encode(df["Actor 1"], df["Rating"])
df["Actor2_enc"] = smoothed_target_encode(df["Actor 2"], df["Rating"])
df["Actor3_enc"] = smoothed_target_encode(df["Actor 3"], df["Rating"])

feature_cols = [
    "Year", "Duration", "Votes_log",
    "Director_enc", "Actor1_enc", "Actor2_enc", "Actor3_enc",
]
X = pd.concat([df[feature_cols], genre_dummies], axis=1)
y = df["Rating"]

# ---------------------------------------------------------------------------
# 4. Train / test split
#    NOTE: target-encoding is computed on the FULL data above, which leaks
#    a small amount of test info through Director/Actor means. For a more
#    rigorous evaluation we recompute the encodings using only the training
#    fold below.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=RANDOM_STATE
)

# Recompute encodings using train-only statistics to avoid leakage
def encode_with_train_stats(train_series, train_target, test_series, m=5):
    stats = train_target.groupby(train_series).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
    train_enc = train_series.map(smoothed)
    test_enc = test_series.map(smoothed).fillna(global_mean)
    return train_enc, test_enc

for col, enc_col in [("Director", "Director_enc"), ("Actor 1", "Actor1_enc"),
                      ("Actor 2", "Actor2_enc"), ("Actor 3", "Actor3_enc")]:
    tr_enc, te_enc = encode_with_train_stats(
        df_train[col], df_train["Rating"], df_test[col]
    )
    X_train.loc[:, enc_col] = tr_enc.values
    X_test.loc[:, enc_col] = te_enc.values

# ---------------------------------------------------------------------------
# 5. Train models
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append({"Model": name, "R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
    fitted[name] = model

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\nModel comparison (test set):")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Feature importance from the best tree-based model
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
print(f"\nBest model: {best_name}")

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).to_string())

results_df.to_csv("/home/claude/model_comparison.csv", index=False)
