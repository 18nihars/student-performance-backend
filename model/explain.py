import pickle
import numpy as np
import pandas as pd
import os

_model_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_model_dir, "best_advanced_model.pkl")
pipeline = pickle.load(open(_model_path, "rb"))
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

FEATURES = [
    "attendance_percent",
    "study_hours",
    "internal_marks",
    "assignments_submitted",
    "activities_participation"
]


def local_explain(df_raw):
    """
    LIME-style local explanation:
    - Create small perturbations around the input
    - Observe how prediction changes
    - Estimate feature contribution
    """

    X0 = df_raw.copy()
    prob0 = pipeline.predict_proba(X0)[0][1]

    contributions = []

    for feature in FEATURES:
        X_perturb = X0.copy()
        # Add small noise to simulate local neighborhood
        change = 1 if feature != "activities_participation" else 0.2
        X_perturb[feature] = X0[feature] + change

        prob_new = pipeline.predict_proba(X_perturb)[0][1]

        # Contribution = how much probability changes when feature changes
        contributions.append({
            "feature": feature,
            "contribution": float(prob_new - prob0)
        })

    # Sort by absolute impact
    contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    
    return contributions


def explain_single(data):
    df_raw = pd.DataFrame([data])

    pred = int(pipeline.predict(df_raw)[0])
    prob = float(pipeline.predict_proba(df_raw)[0][1])

    # LOCAL explainability (always different for different inputs)
    top_feats = local_explain(df_raw)[:5]

    return {
        "prediction": "Pass" if pred == 1 else "Fail",
        "confidence": prob,
        "top_features": top_feats,
        "explain_method": "local_perturbation"
    }


if __name__ == "__main__":
    sample = {
        "attendance_percent": 80,
        "study_hours": 2,
        "internal_marks": 30,
        "assignments_submitted": 5,
        "activities_participation": 1
    }
    print(explain_single(sample))
