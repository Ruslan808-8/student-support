# -*- coding: utf-8 -*-
"""
Model training and evaluation pipeline for Smart Student Desk (ICT1103 QAIRU 2026-2027).
Implements TF-IDF vectorization with Word + Char_wb Feature Union.
Compares Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
Generates metrics, classification reports, 4x4 confusion matrix, and serializes the best model.
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Fix Windows console encoding if needed
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CATEGORIES = ["academic", "docs", "finance", "tech_support"]

CATEGORY_INFO = {
    "academic": {
        "ru": "Академический вопрос",
        "kk": "Академиялық сұрақ",
        "en": "Academic Inquiry",
        "dept_ru": "Деканат / Эдвайзер",
        "dept_kk": "Деканат / Эдвайзер",
        "dept_en": "Dean's Office / Academic Advisor",
        "sla": "12-24 часа",
        "badge": "primary",
        "color": "#0d6efd",
        "icon": "bi-mortarboard-fill"
    },
    "docs": {
        "ru": "Документы и справки",
        "kk": "Құжаттар мен анықтамалар",
        "en": "Documents & Certificates",
        "dept_ru": "Центр обслуживания студентов (ЦОС)",
        "dept_kk": "Студенттерге қызмет көрсету орталығы (СҚКО)",
        "dept_en": "Student Service Center (SSC)",
        "sla": "24-48 сағат",
        "badge": "success",
        "color": "#198754",
        "icon": "bi-file-earmark-text-fill"
    },
    "finance": {
        "ru": "Финансовый отдел",
        "kk": "Қаржы және төлем бөлімі",
        "en": "Finance & Accounting",
        "dept_ru": "Бухгалтерия и финотдел",
        "dept_kk": "Бухгалтерия және қаржы бөлімі",
        "dept_en": "Accounting & Finance Office",
        "sla": "24 сағат",
        "badge": "warning",
        "color": "#ffc107",
        "icon": "bi-credit-card-fill"
    },
    "tech_support": {
        "ru": "Техподдержка",
        "kk": "Техникалық қолдау",
        "en": "IT Helpdesk",
        "dept_ru": "IT Департамент (Helpdesk L1)",
        "dept_kk": "IT Департаменті (Helpdesk L1)",
        "dept_en": "IT Helpdesk & Systems",
        "sla": "2-4 часа",
        "badge": "danger",
        "color": "#dc3545",
        "icon": "bi-hdd-network-fill"
    }
}

CUSTOM_STOPWORDS = [
    "здравствуйте", "подскажите", "пожалуйста", "скажите", "как", "где", "куда", "кому", "что", "для",
    "сәлеметсіз", "сәлеметсіздер", "айтыңызшы", "қалай", "қайда", "кімге", "не", "үшін",
    "болады", "бола", "ма", "ба", "ме", "бе", "керек", "қажет", "help", "hello", "hi", "how"
]

def clean_text(text: str) -> str:
    """Preprocess text: lowercase, normalize spaces."""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()

def build_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            stop_words=CUSTOM_STOPWORDS,
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char_wb",
            sublinear_tf=True
        ))
    ])

def determine_priority(text: str):
    """
    Determines ticket priority based on critical keywords.
    High: session/exam issues, login blocks, 500 error, pending debt blocks, deadlines.
    Low: general certificates, working hours, duplicate badge inquiries.
    Medium: normal course / payment / scholarship queries.
    """
    t = text.lower()
    urgent_keywords = [
        "завис", "заблокирован", "қатып", "бұғатталды", "дедлайн", "қате 500", "error 500",
        "экзамен", "емтихан", "бүгін", "сегодня", "срочно", "urgent", "шығып кетті",
        "апелляция", "бос грант", "өсімпұл", "пеня", "штраф", "отчислени", "шығарылу"
    ]
    routine_keywords = [
        "анықтама", "справка", "жұмыс уақыты", "график работы", "режим работы", "где забрать",
        "қайдан аламын", "характеристика", "көшірме", "дубликат", "verification letter",
        "время работы", "қай бөлме", "какой кабинет"
    ]
    
    for kw in urgent_keywords:
        if kw in t:
            return "high", "Жоғары (High) - Сессия/жүйелік қате немесе дедлайнға байланысты жедел қарау қажет"
            
    for kw in routine_keywords:
        if kw in t:
            return "low", "Төмен (Low) - Стандартты регламентпен өңделетін анықтама/ақпараттық өтініш"
            
    return "medium", "Орташа (Medium) - Күнделікті оқу немесе төлем процесі бойынша өтініш"

def detect_language(text: str) -> str:
    """Simple heuristic language detector (KZ / RU / EN)."""
    t = text.lower()
    kz_specific = ["ә", "і", "ң", "ғ", "ү", "ұ", "қ", "ө", "һ"]
    if any(c in t for c in kz_specific):
        return "kk"
    # Check Cyrillic vs Latin
    has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in t)
    has_latin = any('a' <= c <= 'z' for c in t)
    if has_cyrillic:
        # Check common Kazakh words without specific letters
        if any(w in t.split() for w in ["бола", "ма", "ба", "жоқ", "бар", "оқу", "балл", "қашан"]):
            return "kk"
        return "ru"
    if has_latin:
        return "en"
    return "ru"

def main():
    print("=" * 65)
    print("  Smart Student Desk: ML Pipeline Training (TF-IDF + Models)")
    print("=" * 65)

    dataset_path = "data/dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {len(df)} records.")
    print("Class distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"  - {cat:14}: {count}")

    X = df["text"].apply(clean_text)
    y = df["category"]

    # Stratified Train/Test Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Benchmark candidates
    candidates = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Linear SVM (LinearSVC)": LinearSVC(C=1.0, random_state=42)
    }

    comparison = {}
    fitted_models = {}

    for name, clf in candidates.items():
        pipe = Pipeline([
            ("vec", build_vectorizer()),
            ("clf", clf)
        ])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_duration = time.time() - t0

        t1 = time.time()
        y_pred = pipe.predict(X_test)
        latency_ms = (time.time() - t1) / len(X_test) * 1000

        acc = accuracy_score(y_test, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES).tolist()

        comparison[name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_macro), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "train_time_sec": round(float(fit_duration), 4),
            "latency_ms_per_sample": round(float(latency_ms), 3),
            "confusion_matrix": cm
        }
        fitted_models[name] = pipe

        print(f"\nModel: {name}")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Macro F1:  {f1_macro:.4f}")
        print(f"  Weighted F1: {f1_weighted:.4f}")
        print(f"  Latency:   {latency_ms:.2f} ms/query")

    # Primary chosen model for deployment: Logistic Regression (provides calibrated class probabilities)
    chosen_name = "Logistic Regression"
    print(f"\n--> Selected Production Architecture: {chosen_name}")
    print("    (Provides well-calibrated class probability distribution & high F1 score)")

    # Retrain chosen model on full dataset for maximum real-world generalization
    final_pipeline = Pipeline([
        ("vec", build_vectorizer()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    final_pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_pipeline, "models/best_model_pipeline.joblib")
    print("Saved production model to: models/best_model_pipeline.joblib")

    # Test best pipeline predictions for detailed classification report
    test_eval_pipeline = fitted_models[chosen_name]
    y_test_pred = test_eval_pipeline.predict(X_test)
    class_report = classification_report(
        y_test, y_test_pred, labels=CATEGORIES, target_names=CATEGORIES, output_dict=True
    )
    final_cm = confusion_matrix(y_test, y_test_pred, labels=CATEGORIES).tolist()

    # Required 3 Test Cases
    test_cases_defs = [
        {
            "id": 1,
            "title": "Обычный запрос по учебе",
            "text": "Не отображается оценка за лабораторную работу в LMS",
            "expected": "academic",
            "description": "Студент интересуется выставлением оценки в электронной системе."
        },
        {
            "id": 2,
            "title": "Техническая ошибка",
            "text": "Система тестирования зависла во время прохождения экзамена",
            "expected": "tech_support",
            "description": "Сбой в ПО во время прохождения контрольного тестирования."
        },
        {
            "id": 3,
            "title": "Запрос справки",
            "text": "Где я могу получить справку с места учебы для Egov?",
            "expected": "docs",
            "description": "Запрос официального документа об обучении для сторонней организации."
        }
    ]

    test_case_results = []
    print("\n--- Verifying 3 Mandatory Test Cases ---")
    for tc in test_cases_defs:
        clean_q = clean_text(tc["text"])
        pred = final_pipeline.predict([clean_q])[0]
        probs = final_pipeline.predict_proba([clean_q])[0]
        prob_dict = {cat: round(float(prob), 4) for cat, prob in zip(final_pipeline.classes_, probs)}
        max_prob = max(probs)
        priority_code, priority_desc = determine_priority(tc["text"])
        
        passed = (pred == tc["expected"])
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] Case {tc['id']}: '{tc['text']}'")
        print(f"       Expected: {tc['expected']} | Predicted: {pred} ({max_prob*100:.1f}%) | Priority: {priority_code}")

        test_case_results.append({
            "id": tc["id"],
            "title": tc["title"],
            "text": tc["text"],
            "expected": tc["expected"],
            "predicted": pred,
            "confidence": round(float(max_prob), 4),
            "status": status,
            "probabilities": prob_dict,
            "priority": priority_code,
            "priority_description": priority_desc
        })

    # Error Case Deep Dive (Required by Rubric)
    error_text = "Не могу оплатить повторный курс из-за ошибки на сайте"
    clean_err = clean_text(error_text)
    err_pred = final_pipeline.predict([clean_err])[0]
    err_probs = final_pipeline.predict_proba([clean_err])[0]
    err_prob_dict = {cat: round(float(p), 4) for cat, p in zip(final_pipeline.classes_, err_probs)}

    error_analysis = {
        "text": error_text,
        "predicted": err_pred,
        "probabilities": err_prob_dict,
        "intended_category": "tech_support",
        "conflict_reason": (
            "Запрос содержит пересекающиеся семантические маркеры двух разных подразделений: "
            "'оплатить', 'повторный курс' (Финансы / Оплата) против 'ошибки на сайте', 'не могу' (Техподдержка). "
            f"В связи с сильным весом n-грамм оплаты модель классифицирует обращение как '{err_pred}' "
            f"с уверенностью {err_prob_dict.get(err_pred, 0)*100:.1f}%, хотя первопричиной является веб-ошибка в UI."
        ),
        "mitigation_solution": (
            "1) Использование многозадачной классификации (Multi-label classification), "
            "позволяющей направлять копию тикета и в IT Helpdesk, и в Бухгалтерию.\n"
            "2) Внедрение пред-фильтрации ключевых слов критических сбоев ('ошибка на сайте', '500 error')."
        )
    }
    print(f"\n--- Error Case Analysis ---\nText: '{error_text}'\nPredicted: {err_pred}\nProbabilities: {err_prob_dict}")

    # Build metrics payload
    metrics_data = {
        "course": "ICT1103 · QAIRU · 2026–2027",
        "project_name": "Smart Student Desk: Сортировщик обращений студентов",
        "week": "7-я неделя (NLP, өңдеу және мәтінді жіктеу)",
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split_ratio": "80/20 Stratified",
            "languages": {
                "ru": int((df["language"] == "ru").sum()),
                "kk": int((df["language"] == "kk").sum()),
                "en": int((df["language"] == "en").sum())
            },
            "class_distribution": {k: int(v) for k, v in df["category"].value_counts().items()}
        },
        "chosen_model": chosen_name,
        "model_comparison": comparison,
        "test_metrics": {
            "accuracy": comparison[chosen_name]["accuracy"],
            "macro_f1": comparison[chosen_name]["macro_f1"],
            "macro_precision": comparison[chosen_name]["macro_precision"],
            "macro_recall": comparison[chosen_name]["macro_recall"],
            "weighted_f1": comparison[chosen_name]["weighted_f1"],
            "confusion_matrix": final_cm
        },
        "categories": CATEGORIES,
        "category_details": CATEGORY_INFO,
        "classification_report": class_report,
        "test_cases": test_case_results,
        "error_case": error_analysis,
        "limitations": [
            {
                "title": "Сленг және опечаткалар (Slang & Heavy Typos)",
                "description": "TF-IDF векторлау әдісі сөздікте жоқ (OOV) сленгтік сөздер мен ауыр қателерді өткізіп алады."
            },
            {
                "title": "Күрделі қос мақсатты өтініштер (Multi-Intent Queries)",
                "description": "Бір сөйлемде әрі ақы төлеу, әрі веб-сайт қатесі жазылса, біржақты жіктеуде шекаралық ықтималдық туады."
            },
            {
                "title": "Тым қысқа немесе контекстсіз сөз тіркестері",
                "description": "'Бағам жоқ' немесе 'қашан болады?' секілді 1-2 сөзден тұратын өтініштерде ақпарат тым аз."
            }
        ]
    }

    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)

    print("\n[OK] Saved all pipeline artifacts and metrics to models/metrics.json")

if __name__ == "__main__":
    main()
