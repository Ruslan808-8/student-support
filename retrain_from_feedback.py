# -*- coding: utf-8 -*-
"""Train a candidate model from opt-in, user-labeled SQLite examples.

The app calls ``retrain()`` after an opted-in ticket receives a verified label.
The production model is replaced only when the candidate does not regress on
the original held-out set. The script can also be run manually.
"""

import json
import os
import re
from collections import Counter

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import database
from train_model import CATEGORIES, build_vectorizer, clean_text


ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT, "data", "dataset.csv")
MODEL_PATH = os.path.join(ROOT, "models", "best_model_pipeline.joblib")
REPORT_PATH = os.path.join(ROOT, "models", "retraining_report.json")
MIN_EXAMPLES_PER_CATEGORY = 2
MAX_EXAMPLES_PER_CATEGORY = 100

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s().-]*\d(?:[\s().-]*\d){9}(?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")
CREDENTIAL_RE = re.compile(
    r"(?i)\b(password|пароль|құпия\s*сөз)\s*[:=]?\s*\S+"
)


def redact_for_training(text):
    """Remove common contact, national-ID, and password patterns before training."""
    text = EMAIL_RE.sub(" [EMAIL] ", text)
    text = PHONE_RE.sub(" [PHONE] ", text)
    text = LONG_NUMBER_RE.sub(" [ID] ", text)
    return CREDENTIAL_RE.sub(r"\1 [REDACTED]", text)


def make_pipeline():
    return Pipeline([
        ("vec", build_vectorizer()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])


def collect_examples(holdout_texts, source_train_texts):
    raw_rows = database.get_training_examples()
    counts = Counter()
    accepted = []
    seen_feedback = {}
    holdout = {clean_text(text) for text in holdout_texts}
    source_train = {clean_text(text) for text in source_train_texts}

    for row in raw_rows:
        label = row["corrected_category"]
        text = clean_text(redact_for_training(row["text"] or ""))
        if label not in CATEGORIES or len(text) < 3 or text in holdout:
            continue
        if text in source_train:
            continue
        if text in seen_feedback:
            prior_label = seen_feedback[text]
            if prior_label is not None and prior_label != label:
                # Conflicting labels for the same text are not safe training data.
                seen_feedback[text] = None
            continue
        seen_feedback[text] = label
        if counts[label] >= MAX_EXAMPLES_PER_CATEGORY:
            continue
        counts[label] += 1
        accepted.append((text, label))

    conflicted = {text for text, label in seen_feedback.items() if label is None}
    if conflicted:
        accepted = [(text, label) for text, label in accepted if text not in conflicted]
        counts = Counter(label for _, label in accepted)
    return accepted, counts, len(raw_rows)


def evaluate(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(
            y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0
        )), 4),
    }


def retrain():
    database.init_db()
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing base dataset: {DATASET_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing production model: {MODEL_PATH}. Run train_model.py first.")

    source = pd.read_csv(DATASET_PATH)
    if not {"text", "category"}.issubset(source.columns):
        raise ValueError("data/dataset.csv must contain text and category columns.")
    source = source.dropna(subset=["text", "category"])
    source = source[source["category"].isin(CATEGORIES)].copy()
    x = source["text"].astype(str).apply(clean_text)
    y = source["category"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    examples, counts, total_feedback = collect_examples(x_test, x_train)
    missing = [cat for cat in CATEGORIES if counts[cat] < MIN_EXAMPLES_PER_CATEGORY]
    if missing:
        return {
            "status": "collecting_examples",
            "candidate_promoted": False,
            "training_examples_used": len(examples),
            "examples_by_category": dict(counts),
            "minimum_per_category": MIN_EXAMPLES_PER_CATEGORY,
        }

    baseline = make_pipeline()
    baseline.fit(x_train, y_train)
    baseline_metrics = evaluate(y_test, baseline.predict(x_test))

    feedback_x = [text for text, _ in examples]
    feedback_y = [label for _, label in examples]
    candidate_x = pd.concat([x_train.reset_index(drop=True), pd.Series(feedback_x)])
    candidate_y = pd.concat([y_train.reset_index(drop=True), pd.Series(feedback_y)])
    candidate = make_pipeline()
    candidate.fit(candidate_x, candidate_y)
    candidate_metrics = evaluate(y_test, candidate.predict(x_test))

    # Reject a candidate that materially reduces macro F1 or accuracy.
    accepted = (
        candidate_metrics["macro_f1"] >= baseline_metrics["macro_f1"]
        and candidate_metrics["accuracy"] >= baseline_metrics["accuracy"] - 0.01
    )
    report = {
        "feedback_rows_in_database": total_feedback,
        "training_examples_used": len(examples),
        "examples_by_category": dict(counts),
        "evaluation": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "candidate_promoted": accepted,
            "holdout": "fixed stratified 20% split of data/dataset.csv; feedback excluded",
        },
    }

    if accepted:
        # Refit on the full base dataset plus consented feedback after the holdout review.
        full_x = pd.concat([x.reset_index(drop=True), pd.Series(feedback_x)])
        full_y = pd.concat([y.reset_index(drop=True), pd.Series(feedback_y)])
        final_model = make_pipeline()
        final_model.fit(full_x, full_y)
        temp_model_path = MODEL_PATH + ".candidate"
        joblib.dump(final_model, temp_model_path)
        os.replace(temp_model_path, MODEL_PATH)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    return {
        "status": "model_updated" if accepted else "candidate_rejected",
        "candidate_promoted": accepted,
        "training_examples_used": len(examples),
        "examples_by_category": dict(counts),
        "evaluation": report["evaluation"],
    }


def main():
    result = retrain()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"model_updated", "candidate_rejected"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
