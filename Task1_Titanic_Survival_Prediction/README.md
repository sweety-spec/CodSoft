# Task 1 - Titanic Survival Prediction

## CodSoft Data Science Internship

### Overview
This project builds a machine learning model to predict whether a passenger survived the Titanic disaster, using features such as age, sex, passenger class, fare, and family size.

### Dataset
The dataset contains 891 passenger records with the following original columns:
`PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare`

### Approach
1. **Exploratory Data Analysis (EDA)** — examined survival rates by sex, passenger class, age, fare, and family size.
2. **Feature Engineering**
   - Extracted passenger **Title** (Mr, Mrs, Miss, Master, Rare) from the Name field
   - Imputed missing Age values using the median age within each Title group
   - Created `FamilySize`, `IsAlone`, and `FarePerPerson` features
   - Binned Age into groups (child, teen, adult, middle-aged, senior)
   - Label-encoded categorical variables (Sex, Title)
3. **Modeling** — trained and compared two classifiers:
   - Logistic Regression
   - Random Forest
4. **Evaluation** — accuracy, ROC-AUC, 5-fold cross-validation, confusion matrix, and feature importance

### Results
| Model | Test Accuracy | ROC AUC | 5-Fold CV Accuracy |
|---|---|---|---|
| Logistic Regression | 81.6% | 0.871 | 79.5% (±1.8%) |
| Random Forest | **82.1%** | 0.855 | **83.0%** (±1.5%) |

**Best model:** Random Forest

**Top predictors:** Sex, Title, Fare, and Passenger Class — consistent with the historical "women and children first" evacuation pattern.

### Files
- `titanic_prediction.py` — full pipeline script
- `Titanic_Survival_Prediction.ipynb` — Jupyter notebook version
- `eda_overview.png` — exploratory data analysis charts
- `model_results.png` — confusion matrix, ROC curves, feature importance
- `model_summary.txt` — text summary of model scores

### Tech Stack
Python, pandas, NumPy, scikit-learn, seaborn, matplotlib

### How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
python titanic_prediction.py
```

---
#codsoft #internship #datascience
