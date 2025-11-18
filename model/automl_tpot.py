from tpot import TPOTClassifier
import pandas as pd
import pickle

df = pd.read_csv("../dataset/sample_students.csv")
X = df.drop("target", axis=1)
y = df["target"]

tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    cv=5,
    verbosity=2,
    scoring="accuracy"
)

tpot.fit(X, y)
tpot.export("tpot_best_model.py")

print("AutoML complete. Model saved as tpot_best_model.py")
