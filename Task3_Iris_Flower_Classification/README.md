# Iris Flower Classification

Classifies iris flowers into **setosa**, **versicolor**, or **virginica** based on sepal and petal measurements. Built with pandas and scikit-learn.

## Files

| File | Description |
|---|---|
| `Iris_Flower_Classification.ipynb` | Jupyter notebook — full walkthrough with explanations, plots, and pre-run outputs |
| `iris_classification.py` | Same pipeline as a standalone script |
| `iris_model_comparison.csv` | Final metrics for each model tried |
| `iris_eda.png` | Scatter plots of petal/sepal measurements by species |
| `IRIS.csv` | Dataset (place in the same folder as the notebook/script) |

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python iris_classification.py
```

or open `Iris_Flower_Classification.ipynb` in Jupyter and run all cells. Make sure `IRIS.csv` is in the same directory.

## Dataset

150 samples, 50 of each species, with four measurements: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (all in cm). No missing values.

## Approach

1. **EDA** — scatter plots show petal measurements separate the three species far more cleanly than sepal measurements.
2. **Preprocessing** — stratified 80/20 train/test split (preserves class balance in both sets); features standardized (zero mean, unit variance) since KNN, SVM, and Logistic Regression are scale-sensitive.
3. **Models compared**: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, SVM — evaluated on both a held-out test set and 5-fold cross-validation (more stable on a dataset this small).

## Results

| Model | Test Accuracy | CV Accuracy (mean ± std) |
|---|---|---|
| SVM | 0.967 | 0.967 ± 0.021 |
| Logistic Regression | 0.933 | 0.960 ± 0.039 |
| K-Nearest Neighbors | 0.933 | 0.960 ± 0.025 |
| Random Forest | 0.900 | 0.960 ± 0.025 |
| Decision Tree | 0.900 | 0.953 ± 0.034 |

**Best model: SVM (RBF kernel)** — 96.7% test accuracy. The only confusion is between versicolor and virginica (one misclassification each way); setosa is separated perfectly by every model, consistent with the EDA plots.

## Notes

- Iris is a small, clean, well-separated dataset, so most standard classifiers reach 90%+ accuracy — the main differentiator here is consistency (low CV std) rather than raw accuracy.
- Petal length/width alone would likely classify setosa vs. the other two species almost perfectly; sepal measurements add most of the value for separating versicolor from virginica.
