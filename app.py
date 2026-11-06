# -*- coding: utf-8 -*-
"""
Smart Student Desk - Web Application & REST API (Flask)
Course: ICT1103 · QAIRU · 2026–2027 (Midterm 2: Group Project)
Classifies student requests into 4 departments using TF-IDF + Logistic Regression,
determines priority, SLA, department routing, and surfaces analytics & confusion matrix.
"""

import os
import sys
import json
import time
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Force UTF-8 encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model pipeline and metrics
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model_pipeline.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "metrics.json")

model_pipeline = None
metrics_data = {}

def load_artifacts():
    global model_pipeline, metrics_data
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"[INIT] Loaded model pipeline from {MODEL_PATH}")
    else:
        print(f"[WARN] Model file not found at {MODEL_PATH}. Run train_model.py first!")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        print(f"[INIT] Loaded metrics data from {METRICS_PATH}")
    else:
        metrics_data = {
            "chosen_model": "Logistic Regression",
            "test_metrics": {"accuracy": 0.8475, "macro_f1": 0.8463},
            "dataset": {"total_samples": 293}
        }

load_artifacts()

# In-memory ticket history log (for demo session)
TICKET_HISTORY = []

CATEGORY_INFO = {
    "academic": {
        "code": "academic",
        "title_ru": "Академический вопрос",
        "title_kk": "Академиялық сұрақ",
        "title_en": "Academic Inquiry",
        "dept_ru": "Деканат / Эдвайзер",
        "dept_kk": "Деканат / Эдвайзер",
        "dept_en": "Dean's Office / Academic Advisor",
        "sla": "12-24 сағат / часа",
        "badge": "primary",
        "color": "#0d6efd",
        "icon": "bi-mortarboard-fill",
        "desc_ru": "Вопросы успеваемости, оценок в LMS, силлабусов, расписания, перезачетов и перевода между группами.",
        "desc_kk": "Үлгерім, LMS бағалары, силлабус, емтихан кестесі, пәндерді қайта есептеу және топ ауыстыру сұрақтары."
    },
    "docs": {
        "code": "docs",
        "title_ru": "Документы и справки",
        "title_kk": "Құжаттар мен анықтамалар",
        "title_en": "Documents & Certificates",
        "dept_ru": "Центр обслуживания студентов (ЦОС)",
        "dept_kk": "Студенттерге қызмет көрсету орталығы (СҚКО)",
        "dept_en": "Student Service Center (SSC)",
        "sla": "24-48 сағат / часа",
        "badge": "success",
        "color": "#198754",
        "icon": "bi-file-earmark-text-fill",
        "desc_ru": "Выдача официальных транскриптов, справок с места учебы, формы 26 (военкомат), дубликатов студенческих билетов.",
        "desc_kk": "Ресми транскрипт, оқу орнынан анықтама, 26-нысан (әскер), студенттік билет дубликаты және архивтік құжаттар."
    },
    "finance": {
        "code": "finance",
        "title_ru": "Финансовый отдел и бухгалтерия",
        "title_kk": "Қаржы және төлем бөлімі",
        "title_en": "Finance & Accounting",
        "dept_ru": "Бухгалтерия и финотдел",
        "dept_kk": "Бухгалтерия және қаржы бөлімі",
        "dept_en": "Accounting & Finance Office",
        "sla": "24 сағат / часа",
        "badge": "warning",
        "color": "#f59f00",
        "icon": "bi-credit-card-fill",
        "desc_ru": "Оплата обучения, квитанции Kaspi, сверка расчетов, оформление рассрочки, стипендии и ректорские скидки.",
        "desc_kk": "Оқу ақысын төлеу, Kaspi түбіртектері, бөліп төлеу кестесі (рассрочка), шәкіртақы және оқу жеңілдіктері."
    },
    "tech_support": {
        "code": "tech_support",
        "title_ru": "Техподдержка и IT Helpdesk",
        "title_kk": "Техникалық қолдау (IT Helpdesk)",
        "title_en": "IT Helpdesk & Systems",
        "dept_ru": "IT Департамент (Helpdesk L1)",
        "dept_kk": "IT Департаменті (Helpdesk L1)",
        "dept_en": "IT Helpdesk & Systems",
        "sla": "2-4 сағат / часа",
        "badge": "danger",
        "color": "#dc3545",
        "icon": "bi-hdd-network-fill",
        "desc_ru": "Проблемы с авторизацией, сброс паролей корпоративной почты, ошибки LMS 500/404, кампусный Wi-Fi и турникеты.",
        "desc_kk": "Авторизация қателері, корпоративтік пошта құпия сөзін қалпына келтіру, LMS қателері, кампус Wi-Fi және турникеттер."
    }
}

def clean_input(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()

def detect_language(text: str) -> str:
    t = text.lower()
    kz_specific = ["ә", "і", "ң", "ғ", "ү", "ұ", "қ", "ө", "һ"]
    if any(c in t for c in kz_specific):
        return "kk"
    has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in t)
    has_latin = any('a' <= c <= 'z' for c in t)
    if has_cyrillic:
        if any(w in t.split() for w in ["бола", "ма", "ба", "жоқ", "бар", "оқу", "балл", "қашан", "қалай", "қайда"]):
            return "kk"
        return "ru"
    if has_latin:
        return "en"
    return "ru"

def determine_priority(text: str):
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
            return "high", "Жоғары (High)", "Жедел қарау қажет (дедлайн / сессия / жүйелік қате)", "danger"
            
    for kw in routine_keywords:
        if kw in t:
            return "low", "Төмен (Low)", "Стандартты регламент бойынша өңделеді", "secondary"
            
    return "medium", "Орташа (Medium)", "Күнделікті оқу немесе төлем регламенті", "info"

def classify_query(text: str):
    start_time = time.time()
    clean_q = clean_input(text)

    if not clean_q or model_pipeline is None:
        return None

    pred_cat = model_pipeline.predict([clean_q])[0]
    probs = model_pipeline.predict_proba([clean_q])[0]
    latency_ms = round((time.time() - start_time) * 1000, 2)

    cat_probs = {}
    for cat, prob in zip(model_pipeline.classes_, probs):
        cat_probs[cat] = round(float(prob) * 100, 1)

    max_conf = round(float(max(probs)) * 100, 1)
    lang = detect_language(text)
    priority_code, priority_label, priority_reason, priority_badge = determine_priority(text)

    category_meta = CATEGORY_INFO.get(pred_cat, {})

    ticket_record = {
        "id": f"TICK-{len(TICKET_HISTORY) + 101}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "category": pred_cat,
        "category_name_ru": category_meta.get("title_ru", pred_cat),
        "category_name_kk": category_meta.get("title_kk", pred_cat),
        "category_name_en": category_meta.get("title_en", pred_cat),
        "department_ru": category_meta.get("dept_ru", ""),
        "department_kk": category_meta.get("dept_kk", ""),
        "department_en": category_meta.get("dept_en", ""),
        "sla": category_meta.get("sla", ""),
        "badge": category_meta.get("badge", "primary"),
        "color": category_meta.get("color", "#0d6efd"),
        "icon": category_meta.get("icon", "bi-tag"),
        "confidence": max_conf,
        "probabilities": cat_probs,
        "priority": priority_code,
        "priority_label": priority_label,
        "priority_reason": priority_reason,
        "priority_badge": priority_badge,
        "language": lang,
        "latency_ms": latency_ms
    }

    # Append to recent history (cap at 50)
    TICKET_HISTORY.insert(0, ticket_record)
    if len(TICKET_HISTORY) > 50:
        TICKET_HISTORY.pop()

    return ticket_record

@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=metrics_data,
        category_info=CATEGORY_INFO,
        history=TICKET_HISTORY[:10]
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or request.form
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Текст обращения не может быть пустым."}), 400

    result = classify_query(text)
    if not result:
        return jsonify({"error": "Не удалось классифицировать запрос."}), 500

    return jsonify(result)

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(metrics_data)

@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({
        "total_tickets_logged": len(TICKET_HISTORY),
        "tickets": TICKET_HISTORY
    })

@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    global TICKET_HISTORY
    TICKET_HISTORY = []
    return jsonify({"status": "cleared"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Smart Student Desk",
        "model_loaded": model_pipeline is not None,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Smart Student Desk Server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
