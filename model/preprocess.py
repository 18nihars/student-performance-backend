import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pickle

def build_preprocessor():

    numeric_features = [
        "attendance_percent",
        "study_hours",
        "internal_marks",
        "assignments_submitted"
    ]

    categorical_features = [
        "activities_participation"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
            # no encoding needed for 0/1
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def fit_save_preprocessor(csv_path="../dataset/sample_students.csv",
                          out_path="preprocessor.pkl"):

    df = pd.read_csv(csv_path)
    X = df.drop("target", axis=1)

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    pickle.dump(preprocessor, open(out_path, "wb"))
    print(f"Preprocessor saved to {out_path}")


if __name__ == "__main__":
    fit_save_preprocessor()
