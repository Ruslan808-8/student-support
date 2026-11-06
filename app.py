# -*- coding: utf-8 -*-
"""
student-support - Web Application & REST API (Flask)
Course: ICT1103 · QAIRU · 2026–2027 (Midterm 2: Group Project)
Classifies student requests into 4 departments using TF-IDF + Logistic Regression,
determines priority, SLA, department routing, and surfaces analytics & confusion matrix.
Persists tickets to SQLite and uses cookies for per-student history.
"""

import os
import sys
import json
import secrets
import time
import uuid
import joblib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, make_response, session

import database

# Force UTF-8 encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = Flask(__name__, static_folder="static", template_folder="templates")
SECRET_PATH = os.path.join(os.path.dirname(__file__), "data", ".app_secret_key")


def load_or_create_secret_key():
    """Use a deployment secret or keep a random local key outside source control."""
    configured_key = os.environ.get("SECRET_KEY")
    if configured_key:
        return configured_key

    os.makedirs(os.path.dirname(SECRET_PATH), exist_ok=True)
    try:
        with open(SECRET_PATH, "x", encoding="ascii") as secret_file:
            secret_file.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    with open(SECRET_PATH, "r", encoding="ascii") as secret_file:
        return secret_file.read().strip()


app.secret_key = load_or_create_secret_key()

# Cookie settings
COOKIE_NAME = "student_id"
COOKIE_MAX_AGE = 365 * 24 * 3600  # 1 year
cookie_secure_default = "true" if os.environ.get("RENDER") else "false"
app.config.update(
    SESSION_COOKIE_NAME=COOKIE_NAME,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", cookie_secure_default).lower() in {"1", "true", "yes"},
    PERMANENT_SESSION_LIFETIME=timedelta(seconds=COOKIE_MAX_AGE),
)

# Load model pipeline and metrics
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model_pipeline.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "metrics.json")

model_pipeline = None
metrics_data = {}

# Global ticket counter (persisted from DB)
ticket_counter = 100


def load_artifacts():
    global model_pipeline, metrics_data, ticket_counter
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

    # Initialize database
    database.init_db()

    # Set ticket counter from DB
    stats = database.get_stats()
    ticket_counter = 100 + stats["total_tickets"]
    print(f"[INIT] Ticket counter starts at TICK-{ticket_counter + 1}")


load_artifacts()


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


def get_student_id():
    """Get a server-signed per-browser id, or create one."""
    sid = session.get("student_id")
    try:
        uuid.UUID(str(sid))
    except (TypeError, ValueError, AttributeError):
        legacy_sid = request.cookies.get(COOKIE_NAME, "")
        try:
            legacy_sid = str(uuid.UUID(legacy_sid))
        except (TypeError, ValueError, AttributeError):
            legacy_sid = ""
        if legacy_sid and database.student_has_tickets(legacy_sid):
            # One-time migration of this browser's existing unsigned cookie.
            sid = legacy_sid
        else:
            sid = str(uuid.uuid4())
        session["student_id"] = sid
    session.permanent = True
    return sid


def set_student_cookie(response, student_id):
    """Persist the signed session cookie for this browser."""
    session["student_id"] = student_id
    session.permanent = True
    return response


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
    global ticket_counter
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

    ticket_counter += 1
    ticket_record = {
        "id": f"TICK-{ticket_counter}-{uuid.uuid4().hex[:10].upper()}",
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

    return ticket_record


@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=metrics_data,
        category_info=CATEGORY_INFO,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if data is None:
        data = request.form.to_dict()
    if not isinstance(data, dict):
        return jsonify({"error": "Некорректный формат запроса."}), 400
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Текст обращения не может быть пустым."}), 400
    if len(text) > 2000:
        return jsonify({"error": "Текст обращения не должен превышать 2000 символов."}), 400

    training_opt_in = data.get("training_opt_in", False)
    if not isinstance(training_opt_in, bool):
        return jsonify({"error": "training_opt_in must be a boolean."}), 400

    result = classify_query(text)
    if not result:
        return jsonify({"error": "Не удалось классифицировать запрос."}), 500

    # Get or create student_id from cookie
    student_id = get_student_id()

    # Save to database
    database.save_ticket(student_id, result, training_opt_in=training_opt_in)

    response = make_response(jsonify(result))
    set_student_cookie(response, student_id)
    return response


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(metrics_data)


@app.route("/api/history", methods=["GET"])
def api_history():
    student_id = get_student_id()
    tickets = database.get_student_history(student_id, limit=50)
    response = make_response(jsonify({
        "student_id": student_id,
        "total_tickets_logged": len(tickets),
        "tickets": tickets,
    }))
    set_student_cookie(response, student_id)
    return response


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    student_id = get_student_id()
    deleted = database.clear_student_history(student_id)
    return jsonify({"status": "cleared", "deleted_count": deleted})


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Save student's correction for a ticket (for future model retraining)."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Некорректный формат запроса."}), 400
    ticket_id = data.get("ticket_id", "")
    correct_category = data.get("correct_category", "")

    if (
        not isinstance(ticket_id, str)
        or not ticket_id
        or not isinstance(correct_category, str)
        or correct_category not in CATEGORY_INFO
    ):
        return jsonify({"error": "ticket_id and valid correct_category required."}), 400

    student_id = get_student_id()
    if not database.save_feedback(student_id, ticket_id, correct_category):
        return jsonify({"error": "Обращение не найдено в вашей истории."}), 404
    return jsonify({"status": "feedback_saved", "ticket_id": ticket_id, "correct_category": correct_category})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return aggregate statistics (total tickets, category/language distribution)."""
    stats = database.get_stats()
    return jsonify(stats)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "student-support",
        "model_loaded": model_pipeline is not None,
        "database": "sqlite",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting student-support Server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
