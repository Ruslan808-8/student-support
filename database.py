# -*- coding: utf-8 -*-
"""
database.py — SQLite persistence layer for student-support
Stores all classified tickets and optional student feedback for future model retraining.
Uses Python's built-in sqlite3 module (no extra dependencies).
"""

import os
import json
import sqlite3

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "student_support.db")


def get_connection():
    """Get a thread-local SQLite connection with WAL mode for concurrent reads."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tickets (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id       TEXT    NOT NULL,
                student_id      TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                text            TEXT    NOT NULL,
                category        TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                priority        TEXT    NOT NULL,
                language        TEXT    NOT NULL,
                probabilities   TEXT    NOT NULL,
                latency_ms      REAL    NOT NULL,
                training_opt_in INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_tickets_student
                ON tickets(student_id);

            CREATE INDEX IF NOT EXISTS idx_tickets_category
                ON tickets(category);

            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id       TEXT    NOT NULL,
                student_id      TEXT    NOT NULL,
                correct_category TEXT   NOT NULL,
                created_at      TEXT    DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_ticket
                ON feedback(ticket_id);
        """)
        # Additive migration for an existing database created before consent tracking.
        ticket_columns = {row[1] for row in conn.execute("PRAGMA table_info(tickets)")}
        if "training_opt_in" not in ticket_columns:
            conn.execute(
                "ALTER TABLE tickets ADD COLUMN training_opt_in INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()
        print(f"[DB] Database initialized at {DB_PATH}")
    finally:
        conn.close()


def save_ticket(student_id: str, ticket: dict, training_opt_in: bool = False) -> None:
    """Persist a classified ticket to the database."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO tickets
               (ticket_id, student_id, timestamp, text, category, confidence,
                priority, language, probabilities, latency_ms, training_opt_in)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticket["id"],
                student_id,
                ticket["timestamp"],
                ticket["text"],
                ticket["category"],
                ticket["confidence"],
                ticket["priority"],
                ticket["language"],
                json.dumps(ticket["probabilities"], ensure_ascii=False),
                ticket["latency_ms"],
                int(training_opt_in),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_student_history(student_id: str, limit: int = 50) -> list:
    """Return the most recent tickets for a given student (cookie-scoped)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ticket_id, timestamp, text, category, confidence,
                      priority, language, probabilities, latency_ms
               FROM tickets
               WHERE student_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (student_id, limit),
        ).fetchall()

        tickets = []
        for row in rows:
            tickets.append({
                "id": row["ticket_id"],
                "timestamp": row["timestamp"],
                "text": row["text"],
                "category": row["category"],
                "confidence": row["confidence"],
                "priority": row["priority"],
                "language": row["language"],
                "probabilities": json.loads(row["probabilities"]),
                "latency_ms": row["latency_ms"],
            })
        return tickets
    finally:
        conn.close()


def clear_student_history(student_id: str) -> int:
    """Delete all tickets for a specific student. Returns deleted count."""
    conn = get_connection()
    try:
        conn.execute(
            """DELETE FROM feedback
               WHERE ticket_id IN (
                   SELECT ticket_id FROM tickets WHERE student_id = ?
               )""",
            (student_id,),
        )
        cur = conn.execute(
            "DELETE FROM tickets WHERE student_id = ?", (student_id,)
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def student_has_tickets(student_id: str) -> bool:
    """Check whether a legacy browser id exists before migrating its cookie."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM tickets WHERE student_id = ? LIMIT 1", (student_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def save_feedback(student_id: str, ticket_id: str, correct_category: str) -> bool:
    """Save or update a label only when the ticket belongs to this student."""
    conn = get_connection()
    try:
        ticket = conn.execute(
            "SELECT 1 FROM tickets WHERE ticket_id = ? AND student_id = ?",
            (ticket_id, student_id),
        ).fetchone()
        if ticket is None:
            return False

        existing = conn.execute(
            """SELECT id FROM feedback
               WHERE ticket_id = ? AND student_id = ? ORDER BY id DESC LIMIT 1""",
            (ticket_id, student_id),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE feedback SET correct_category = ?, created_at = datetime('now') WHERE id = ?",
                (correct_category, existing["id"]),
            )
        else:
            conn.execute(
                """INSERT INTO feedback (ticket_id, student_id, correct_category)
                   VALUES (?, ?, ?)""",
                (ticket_id, student_id, correct_category),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def get_stats() -> dict:
    """Return aggregate statistics for admin/analytics."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]

        cat_rows = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM tickets GROUP BY category"
        ).fetchall()
        by_category = {r["category"]: r["cnt"] for r in cat_rows}

        lang_rows = conn.execute(
            "SELECT language, COUNT(*) as cnt FROM tickets GROUP BY language"
        ).fetchall()
        by_language = {r["language"]: r["cnt"] for r in lang_rows}

        feedback_count = conn.execute(
            "SELECT COUNT(DISTINCT ticket_id) FROM feedback"
        ).fetchone()[0]

        unique_students = conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM tickets"
        ).fetchone()[0]

        return {
            "total_tickets": total,
            "unique_students": unique_students,
            "by_category": by_category,
            "by_language": by_language,
            "feedback_count": feedback_count,
        }
    finally:
        conn.close()


def get_training_examples() -> list:
    """Return only opted-in examples with an explicit user-provided label."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT t.ticket_id, t.text, f.correct_category
               FROM tickets t
               JOIN feedback f ON t.ticket_id = f.ticket_id AND t.student_id = f.student_id
               WHERE t.training_opt_in = 1
                 AND f.id = (
                     SELECT MAX(f2.id) FROM feedback f2
                     WHERE f2.ticket_id = t.ticket_id AND f2.student_id = t.student_id
                 )
               ORDER BY t.id"""
        ).fetchall()

        result = []
        for r in rows:
            result.append({
                "ticket_id": r["ticket_id"],
                "text": r["text"],
                "corrected_category": r["correct_category"],
            })
        return result
    finally:
        conn.close()
