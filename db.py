import sqlite3
import os
from datetime import datetime

DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            attendance_percent REAL,
            study_hours REAL,
            internal_marks REAL,
            assignments_submitted REAL,
            activities_participation INTEGER,
            prediction TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(data, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT INTO predictions (
            timestamp, attendance_percent, study_hours, internal_marks,
            assignments_submitted, activities_participation, prediction, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        data["attendance_percent"],
        data["study_hours"],
        data["internal_marks"],
        data["assignments_submitted"],
        data["activities_participation"],
        prediction,
        confidence
    ))

    conn.commit()
    conn.close()


def get_latest_predictions(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = c.fetchall()
    conn.close()
    return rows
