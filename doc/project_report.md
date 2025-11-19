A complete Machine Learning + Full-Stack Web Application that predicts whether a student will Pass or Fail, based on:

Attendance %

Study Hours

Internal Marks

Assignments Submitted

Activities Participation

The system includes:

🔥 Advanced ML Pipeline (RandomForest + XGBoost)

🧠 Explainability with Local Influence (LIME-style)

🗄 SQLite prediction history

🌐 Flask backend

🎨 React frontend

📊 Prediction history table + CSV export

🧩 Modular architecture

☁ Deployment ready





🛠 Tech Stack
Frontend

React.js

Fetch API

Custom components

CSV export button

Modern responsive UI

Backend

Flask

SQLite

Python ML pipeline

Explainability module

Machine Learning

Pandas / NumPy

Scikit-Learn Pipelines

RandomForestClassifier

XGBoostClassifier

Local Perturbation Explainability (LIME-style)



🤖 Machine Learning Pipeline
1. Preprocessing

StandardScaler (numeric features)

Missing value handling

Single ColumnTransformer pipeline

2. Models trained

Logistic Regression

Decision Tree

Random Forest

XGBoost

AutoML (TPOT optional)

3. Model Selection

5-fold cross validation chooses the best model → saved as:

model/best_advanced_model.pkl



🧠 Explainability (Local Influence)

The system explains each prediction with LIME-style local perturbation:

For each feature:

Slightly modify input

Measure change in model probability

Contribution = probability change

Example:

internal_marks: +0.182
attendance_percent: +0.101
study_hours: +0.042



🗄 Database Logging (SQLite)

Every prediction is stored automatically:

id | timestamp | attendance_percent | study_hours |
internal_marks | assignments_submitted | activities |
prediction | confidence


History endpoint:

GET /predictions


Returns last 20 predictions.



API Documentation
✅ POST /predict

Input JSON:

{
  "attendance_percent": 75,
  "study_hours": 3,
  "internal_marks": 42,
  "assignments_submitted": 6,
  "activities_participation": 1
}


Response:

{
  "prediction": "Pass",
  "confidence": 0.87,
  "top_features": [
    {"feature": "internal_marks", "contribution": 0.12},
    {"feature": "attendance_percent", "contribution": 0.08}
  ]
}

✅ GET /predictions

Returns last 20 entries.




💻 Frontend Features
1. Prediction UI

Form inputs

Validation

Predict button

Clear error handling

2. Confidence Gauge

Circular animated confidence indicator.

3. Explanation Section

Shows top contributing features.

4. History Table

Displays recent predictions.

5. CSV Export

One-click export of all history:

prediction_history.csv




System Architecture
             ┌────────────────────────┐
             │        FRONTEND         │
             │        (React)          │
             │ Form Input              │
             │ Prediction Result       │
             │ Confidence Gauge        │
             │ History Table + CSV     │
             └───────────┬────────────┘
                         │ (JSON API)
                         ▼
             ┌────────────────────────┐
             │         BACKEND         │
             │        (Flask)          │
             │ /predict → Model + Explain
             │ /predictions → History
             │ Saves to SQLite DB      │
             └───────────┬────────────┘
                         │
                         ▼
             ┌────────────────────────┐
             │       ML PIPELINE       │
             │ Preprocessor (Scaling) │
             │ Model (RF/XGB)         │
             │ Explainability Module  │
             └───────────┬────────────┘
                         │
                         ▼
             ┌────────────────────────┐
             │         DATABASE        │
             │     SQLite (local)      │
             │   predictions.db        │
             └────────────────────────┘