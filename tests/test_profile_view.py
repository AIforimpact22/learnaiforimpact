import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

from flask import Flask, g
from psycopg.errors import UndefinedTable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import profile  # noqa: E402
from profile import create_profile_blueprint  # noqa: E402


def test_profile_view_handles_missing_users_table(monkeypatch):
    app = Flask(__name__)
    app.testing = True

    def fake_render(template_name, **context):
        return f"{template_name}::{context.get('display_name')}"

    monkeypatch.setattr(profile, "render_template", fake_render)

    def fake_fetch_one(sql, params=()):
        if "FROM public.users" in sql:
            raise UndefinedTable("relation \"public.users\" does not exist")
        if "FROM public.registrations" in sql:
            return {
                "id": 42,
                "user_email": params[0] if params else "user@example.com",
                "first_name": "Reg",
                "last_name": "User",
            }
        return None

    monkeypatch.setattr(profile, "_fetch_one", fake_fetch_one)
    monkeypatch.setattr(profile, "_fetch_all", lambda *args, **kwargs: [])
    monkeypatch.setattr(profile, "_ensure_enrollment", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile, "_get_progress", lambda *args, **kwargs: {})
    monkeypatch.setattr(profile, "_save_progress", lambda *args, **kwargs: None)

    bp = create_profile_blueprint()
    app.register_blueprint(bp)

    @app.before_request
    def _set_user():
        g.user_email = "user@example.com"

    client = app.test_client()
    resp = client.get("/profile")

    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == "profile.html::Reg, User"


def test_profile_view_handles_db_unavailable(monkeypatch):
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))
    app.testing = True
    app.jinja_env.globals.setdefault("page_allowed", lambda *args, **kwargs: False)

    def boom(*args, **kwargs):
        raise RuntimeError("pool init failed")

    monkeypatch.setattr(profile, "_fetch_one", boom)

    bp = create_profile_blueprint()
    app.register_blueprint(bp)

    @app.before_request
    def _set_user():
        g.user_email = "user@example.com"

    client = app.test_client()
    resp = client.get("/profile")

    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "Profile data is temporarily unavailable" in body


def test_profile_view_casts_textual_score_points(monkeypatch):
    app = Flask(__name__)
    app.testing = True

    sample_created_at = datetime.now(timezone.utc)
    activity_rows = [
        {
            "id": 1,
            "course_id": 7,
            "lesson_uid": "lesson-1",
            "a_type": "lesson",
            "created_at": sample_created_at,
            "score_points": "7.5",
            "passed": True,
            "payload": {},
            "course_title": "Course 7",
        }
    ]

    def fake_render(template_name, **context):
        # ensure the query ran without triggering our guard below
        assert context["stats"]["points_total"] == 7.5
        return f"{template_name}::{context['stats']['points_total']}"

    monkeypatch.setattr(profile, "render_template", fake_render)

    def fake_fetch_one(sql, params=()):
        sql_clean = " ".join(sql.split())
        if "FROM public.users" in sql:
            return {
                "id": 11,
                "email": params[0],
                "full_name": "Test User",
                "role": "student",
                "created_at": sample_created_at,
            }
        if "FROM public.registrations" in sql and "ORDER BY" in sql:
            return {
                "id": 22,
                "user_email": params[0],
                "course_id": 7,
                "first_name": "Reg",
                "last_name": "User",
            }
        if "COUNT(DISTINCT lesson_uid)" in sql:
            if "::numeric" not in sql_clean:
                raise UndefinedTable("function sum(character varying) does not exist")
            return {
                "lessons_seen": 1,
                "last_active_at": sample_created_at,
                "points": Decimal("7.5"),
                "passes": 1,
            }
        if "FROM public.courses" in sql:
            return {
                "id": params[0],
                "title": "Course 7",
                "structure": {
                    "sections": [
                        {
                            "order": 1,
                            "title": "Section",
                            "lessons": [
                                {
                                    "order": 1,
                                    "title": "Lesson",
                                    "lesson_uid": "lesson-1",
                                    "content": {"duration_sec": 60},
                                }
                            ],
                        }
                    ]
                },
            }
        if "SELECT lesson_uid" in sql and "ORDER BY" in sql:
            return {"lesson_uid": "lesson-1"}
        return None

    def fake_fetch_all(sql, params=()):
        if "SELECT DISTINCT course_id" in sql and "public.registrations" in sql:
            return [{"course_id": 7}]
        if "SELECT DISTINCT course_id" in sql and "public.activity_log" in sql:
            return [{"course_id": 7}]
        if "SELECT created_at, score_points, passed, payload" in sql:
            return []
        if "FROM public.activity_log" in sql and "LIMIT 10" in sql:
            return activity_rows
        return []

    monkeypatch.setattr(profile, "_fetch_one", fake_fetch_one)
    monkeypatch.setattr(profile, "_fetch_all", fake_fetch_all)
    monkeypatch.setattr(profile, "_ensure_enrollment", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile, "_get_progress", lambda *args, **kwargs: {})
    monkeypatch.setattr(profile, "_save_progress", lambda *args, **kwargs: None)

    bp = create_profile_blueprint()
    app.register_blueprint(bp)

    @app.before_request
    def _set_user():
        g.user_email = "user@example.com"
        g.user_id = 11

    client = app.test_client()
    resp = client.get("/profile")

    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == "profile.html::7.5"


def test_profile_view_casts_textual_payload_for_exams(monkeypatch):
    app = Flask(__name__)
    app.testing = True

    sample_created_at = datetime.now(timezone.utc)

    def fake_render(template_name, **context):
        stats = context["stats"]
        assert stats["exam_total_weeks"] == 1
        assert stats["exam_graded_weeks"] == 1
        assert stats["exam_avg_score"] == 88.0
        course_ctx = context["courses"][0]
        assert course_ctx["exam_weeks_total"] == 1
        assert course_ctx["exam_weeks_graded"] == 1
        assert course_ctx["exam_passed_weeks"] == 1
        assert course_ctx["exam_avg_score"] == 88.0
        return f"{template_name}::exam"

    monkeypatch.setattr(profile, "render_template", fake_render)

    def fake_fetch_one(sql, params=()):
        sql_clean = " ".join(sql.split())
        if "FROM public.users" in sql:
            return {
                "id": 11,
                "email": params[0],
                "full_name": "Test User",
                "role": "student",
                "created_at": sample_created_at,
            }
        if "SELECT *" in sql_clean and "FROM public.registrations" in sql_clean:
            return {
                "id": 22,
                "user_email": params[0],
                "course_id": 7,
                "first_name": "Reg",
                "last_name": "User",
            }
        if "COUNT(DISTINCT lesson_uid)" in sql:
            return {
                "lessons_seen": 0,
                "last_active_at": sample_created_at,
                "points": Decimal("0"),
                "passes": 0,
            }
        if "FROM public.courses" in sql:
            return {
                "id": params[0],
                "title": "Course 7",
                "structure": {
                    "sections": [
                        {
                            "order": 1,
                            "title": "Section",
                            "exam": {"enabled": True},
                            "lessons": [
                                {
                                    "order": 1,
                                    "title": "Lesson",
                                    "lesson_uid": "lesson-1",
                                    "content": {"duration_sec": 60},
                                }
                            ],
                        }
                    ]
                },
            }
        if "SELECT lesson_uid" in sql and "ORDER BY" in sql:
            return {"lesson_uid": "lesson-1"}
        return None

    def fake_fetch_all(sql, params=()):
        sql_clean = " ".join(sql.split())
        if "SELECT DISTINCT course_id" in sql and "public.registrations" in sql:
            return [{"course_id": 7}]
        if "SELECT DISTINCT course_id" in sql and "public.activity_log" in sql:
            return [{"course_id": 7}]
        if "SELECT created_at, score_points, passed" in sql and "activity_log" in sql:
            assert "payload::jsonb" in sql
            assert "payload::jsonb ->> 'kind'" in sql
            return [
                {
                    "created_at": sample_created_at,
                    "score_points": "9.0",
                    "passed": True,
                    "payload": {
                        "kind": "exam",
                        "event": "graded",
                        "week_index": 1,
                        "score_percent": "88.0",
                        "passed": True,
                    },
                }
            ]
        if "FROM public.activity_log" in sql and "LIMIT 10" in sql:
            return []
        return []

    monkeypatch.setattr(profile, "_fetch_one", fake_fetch_one)
    monkeypatch.setattr(profile, "_fetch_all", fake_fetch_all)
    monkeypatch.setattr(profile, "_ensure_enrollment", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile, "_get_progress", lambda *args, **kwargs: {})
    monkeypatch.setattr(profile, "_save_progress", lambda *args, **kwargs: None)

    bp = create_profile_blueprint()
    app.register_blueprint(bp)

    @app.before_request
    def _set_user():
        g.user_email = "user@example.com"
        g.user_id = 11

    client = app.test_client()
    resp = client.get("/profile")

    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == "profile.html::exam"


def test_profile_view_handles_exam_payload_text(monkeypatch):
    app = Flask(__name__)
    app.testing = True

    sample_created_at = datetime.now(timezone.utc)

    def fake_render(template_name, **context):
        stats = context["stats"]
        assert stats["exam_total_weeks"] == 1
        assert stats["exam_graded_weeks"] == 1
        course_ctx = context["courses"][0]
        assert course_ctx["exam_weeks_total"] == 1
        assert course_ctx["exam_weeks_graded"] == 1
        return f"{template_name}::text-exam"

    monkeypatch.setattr(profile, "render_template", fake_render)

    def fake_fetch_one(sql, params=()):
        sql_clean = " ".join(sql.split())
        if "FROM public.users" in sql:
            return {
                "id": 11,
                "email": params[0],
                "full_name": "Test User",
                "role": "student",
                "created_at": sample_created_at,
            }
        if "SELECT *" in sql_clean and "FROM public.registrations" in sql_clean:
            return {
                "id": 22,
                "user_email": params[0],
                "course_id": 7,
                "first_name": "Reg",
                "last_name": "User",
            }
        if "COUNT(DISTINCT lesson_uid)" in sql:
            return {
                "lessons_seen": 0,
                "last_active_at": sample_created_at,
                "points": Decimal("0"),
                "passes": 0,
            }
        if "FROM public.courses" in sql:
            return {
                "id": params[0],
                "title": "Course 7",
                "structure": {
                    "sections": [
                        {
                            "order": 1,
                            "title": "Section",
                            "exam": {"enabled": True},
                            "lessons": [
                                {
                                    "order": 1,
                                    "title": "Lesson",
                                    "lesson_uid": "lesson-1",
                                    "content": {"duration_sec": 60},
                                }
                            ],
                        }
                    ]
                },
            }
        if "SELECT lesson_uid" in sql and "ORDER BY" in sql:
            return {"lesson_uid": "lesson-1"}
        return None

    def fake_fetch_all(sql, params=()):
        sql_clean = " ".join(sql.split())
        if "SELECT DISTINCT course_id" in sql and "public.registrations" in sql:
            return [{"course_id": 7}]
        if "SELECT DISTINCT course_id" in sql and "public.activity_log" in sql:
            return [{"course_id": 7}]
        if "SELECT created_at, score_points, passed" in sql and "activity_log" in sql:
            rows = [
                {
                    "created_at": sample_created_at,
                    "score_points": "9.5",
                    "passed": True,
                    "payload": json.dumps(
                        {
                            "kind": "exam",
                            "event": "graded",
                            "week_index": "1",
                            "score_percent": "92.0",
                            "passed": True,
                            "attempt_uid": "attempt-1",
                        }
                    ),
                }
            ]
            normalized = profile._normalize_payload_rows(rows)
            assert isinstance(normalized[0]["payload"], dict)
            return normalized
        if "FROM public.activity_log" in sql_clean and "LIMIT 10" in sql_clean:
            return []
        return []

    monkeypatch.setattr(profile, "_fetch_one", fake_fetch_one)
    monkeypatch.setattr(profile, "_fetch_all", fake_fetch_all)
    monkeypatch.setattr(profile, "_ensure_enrollment", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile, "_get_progress", lambda *args, **kwargs: {})
    monkeypatch.setattr(profile, "_save_progress", lambda *args, **kwargs: None)

    bp = create_profile_blueprint()
    app.register_blueprint(bp)

    @app.before_request
    def _set_user():
        g.user_email = "user@example.com"
        g.user_id = 11

    client = app.test_client()
    resp = client.get("/profile")

    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == "profile.html::text-exam"
