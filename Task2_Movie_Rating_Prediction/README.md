# Movie Rating Prediction with Python

Predicts an IMDb movie's rating from its **genre**, **director**, and **cast**, using historical data on Indian movies. Built with pandas and scikit-learn.

## Files

| File | Description |
|---|---|
| `Movie_Rating_Prediction.ipynb` | Jupyter notebook — full walkthrough with explanations and pre-run outputs |
| `movie_rating_prediction.py` | Same pipeline as a standalone script |
| `model_comparison.csv` | Final metrics for each model tried |
| `IMDb_Movies_India.csv` | Dataset (place in the same folder as the notebook/script) |

## How to run

```bash
pip install pandas numpy scikit-learn
python movie_rating_prediction.py
```

or open `Movie_Rating_Prediction.ipynb` in Jupyter and run all cells. Make sure `IMDb_Movies_India.csv` is in the same directory (the notebook/script expects a relative path).

## Dataset

15,509 movies with: `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`, `Director`, `Actor 1-3`. Rows with a missing `Rating` (the prediction target) are dropped, leaving 7,919 usable rows.

## Approach

1. **Cleaning** — strip non-numeric characters from `Year` (`(2019)` → `2019`), `Duration` (`109 min` → `109`), and `Votes` (`1,086` → `1086`); fill remaining gaps with medians/"Unknown".
2. **Feature engineering**
   - **Genre**: first-listed genre per movie, one-hot encoded (top 15 genres, rest grouped as "Other")
   - **Director / Actor 1-3**: too many unique names to one-hot encode, so each is **target-encoded** — replaced with that person's historical average rating, smoothed toward the overall average so names with few movies don't get an unreliable estimate
   - **Votes**: log-transformed to reduce skew
3. **Train/test split** (80/20). Director and actor encodings are recomputed using **only the training fold's** ratings, so no test-set information leaks into the features.
4. **Models compared**: Linear Regression, Random Forest, Gradient Boosting.

## Results (test set)

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.266 | 0.91 | 1.17 |
| Gradient Boosting | 0.264 | 0.90 | 1.17 |
| Random Forest | 0.235 | 0.91 | 1.19 |

Linear Regression edges out the tree-based models here, suggesting the relationship between these features and rating is close to additive once genre/director/actor are properly encoded.

## Limitations & notes

- R² of ~0.25-0.27 is expected for this feature set — genre/director/cast explain only part of a rating; things like script quality, marketing, and release timing aren't in the data.
- Target encoding is evaluated leakage-free (train-only statistics), which is stricter than encoding on the full dataset before splitting. If you see much higher R² elsewhere on this dataset, that's likely why.
- Directors/actors not seen during training fall back to the global average rating.
