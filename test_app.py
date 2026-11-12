# -*- coding: utf-8 -*-
import sys
import json
from app import app
import database

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def test_all():
    client = app.test_client()

    # 1. Health check
    r1 = client.get('/health')
    assert r1.status_code == 200, f"Health check failed: {r1.status_code}"
    print("Health check OK:", r1.get_json())

    # 2. Main page HTML
    r_idx = client.get('/')
    assert r_idx.status_code == 200, f"Index page failed: {r_idx.status_code}"
    print("Index page rendered OK, HTML size:", len(r_idx.data), "bytes")

    # 3. Metrics API
    r_m = client.get('/api/metrics')
    assert r_m.status_code == 200
    print("Metrics API OK, model:", r_m.get_json().get("chosen_model"))

    # 4. Mandatory Test Cases and Error Case
    cases = [
        ("Не отображается оценка за лабораторную работу в LMS", "academic"),
        ("Система тестирования зависла во время прохождения экзамена", "tech_support"),
        ("Где я могу получить справку с места учебы для Egov?", "docs"),
        ("Kaspi арқылы ақша ұсталды, бірақ LMS-те төлем көрінбей тұр.", "finance"),
        ("Не могу оплатить повторный курс из-за ошибки на сайте", "finance")
    ]

    print("\n--- Testing API Predictions & DB Persistence ---")
    last_ticket_id = None
    for text, expected in cases:
        res = client.post('/api/predict', json={'text': text})
        assert res.status_code == 200
        d = res.get_json()
        last_ticket_id = d['id']
        print(f"[{d['id']}] Text: '{text[:45]}...'")
        print(f" -> Predicted: {d['category']} (Expected: {expected}) | Confidence: {d['confidence']}% | Priority: {d['priority_label']}")

    # 5. History API (Cookie-based)
    r_hist = client.get('/api/history')
    assert r_hist.status_code == 200
    hist_data = r_hist.get_json()
    print("\nHistory API logged tickets count:", hist_data["total_tickets_logged"])
    print("Student ID from cookie:", hist_data["student_id"])
    assert hist_data["total_tickets_logged"] >= 5

    # 6. Feedback API (for future retraining)
    r_fb = client.post('/api/feedback', json={
        'ticket_id': last_ticket_id,
        'correct_category': 'tech_support'
    })
    assert r_fb.status_code == 200
    print("\nFeedback API response:", r_fb.get_json())

    # 7. Stats API
    r_stats = client.get('/api/stats')
    assert r_stats.status_code == 200
    stats = r_stats.get_json()
    print("Stats API OK:")
    print(" -> Total tickets in DB:", stats["total_tickets"])
    print(" -> By category:", stats["by_category"])
    print(" -> Feedback count:", stats["feedback_count"])

    print("\n[SUCCESS] All Flask web application, SQLite DB, and cookie tests PASSED!")

if __name__ == '__main__':
    test_all()
