import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../dataset/sample_students.csv")

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

pickle.dump(model, open("best_pipeline.pkl", "wb"))

print("Model trained & saved!")

