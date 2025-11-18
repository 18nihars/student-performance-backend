import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocess import build_preprocessor


# =======================
# Load Dataset
# =======================

df = pd.read_csv("../dataset/sample_students.csv")
X = df.drop("target", axis=1)
y = df["target"]

preprocessor = build_preprocessor()


# =======================
# 1. RandomForest Tuning
# =======================

rf_params = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 8, None],
    "model__min_samples_split": [2, 5, 10]
}

rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier())
])

rf_grid = GridSearchCV(rf_pipe, rf_params, cv=5, scoring="accuracy", n_jobs=-1)
rf_grid.fit(X, y)

print("\nBest RandomForest Accuracy:", rf_grid.best_score_)
print("Best Parameters:", rf_grid.best_params_)


# =======================
# 2. XGBoost Model
# =======================

xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
    ))
])

xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring="accuracy")
print("\nXGBoost Mean Accuracy:", xgb_scores.mean())


# =======================
# 3. Compare Models
# =======================

results = {
    "RandomForest": rf_grid.best_score_,
    "XGBoost": xgb_scores.mean()
}

plt.bar(results.keys(), results.values(), color=["skyblue", "orange"])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.show()


# =======================
# 4. Pick Best Model
# =======================

best_model = (
    rf_grid.best_estimator_
    if results["RandomForest"] >= results["XGBoost"]
    else xgb_model.fit(X, y)
)

pickle.dump(best_model, open("best_advanced_model.pkl", "wb"))
print("\nSaved final model as best_advanced_model.pkl")
