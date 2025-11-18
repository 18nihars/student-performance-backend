import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor

df = pd.read_csv("../dataset/sample_students.csv")

X = df.drop("target", axis=1)
y = df["target"]

preprocessor = build_preprocessor()

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=200)
}

best_score = -1
best_model = None
best_name = ""

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    avg = scores.mean()

    print(f"{name}: {avg:.4f}")

    if avg > best_score:
        best_score = avg
        best_model = pipe
        best_name = name

print(f"\nBest model: {best_name} ({best_score:.4f})")

best_model.fit(X, y)
pickle.dump(best_model, open("best_pipeline.pkl", "wb"))

print("Saved best model to best_pipeline.pkl")
