import json
import sys
from pathlib import Path

import pytest
from flask import Flask, g


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import course
from course import register_course_routes
from exam import create_exam_blueprint


class FakeDB:
    def __init__(self, structure):
        self._structure_json = json.dumps(structure)
        self.activity_queries = 0
        self.course_queries = 0
        self.executed = []
        self.fetch_all_sql = []

    def fetch_one(self, sql, params=()):
        if "FROM public.courses" in sql or "FROM courses" in sql:
            self.course_queries += 1
            course_id = params[0] if params else 1
            return {"id": course_id, "title": "Test Course", "structure": self._structure_json}
        if "FROM pg_attribute" in sql:
            return {"tname": "text"}
        return None

    def fetch_all(self, sql, params=()):
        self.fetch_all_sql.append(sql)
        if "public.activity_log" in sql:
            self.activity_queries += 1
            return []
        if "FROM pg_enum" in sql:
            return []
        return []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))


def _ensure_structure(raw):
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {"sections": []}


def _flatten_lessons(structure):
    items = []
    for section in structure.get("sections", []):
        for lesson in section.get("lessons", []):
            items.append((section, lesson))
    return items


def _first_lesson_uid(structure):
    lessons = _flatten_lessons(structure)
    if not lessons:
        return None
    return lessons[0][1].get("lesson_uid")


def _find_lesson(structure, lesson_uid):
    for _, lesson in _flatten_lessons(structure):
        if str(lesson.get("lesson_uid")) == str(lesson_uid):
            return lesson
    return None


def _next_prev_uids(structure, lesson_uid):
    lessons = [lesson for _, lesson in _flatten_lessons(structure)]
    ids = [str(lesson.get("lesson_uid")) for lesson in lessons]
    try:
        idx = ids.index(str(lesson_uid))
    except ValueError:
        return None, None
    prev_uid = ids[idx - 1] if idx > 0 else None
    next_uid = ids[idx + 1] if idx + 1 < len(ids) else None
    return prev_uid, next_uid


def _lesson_index_map(structure):
    return {str(lesson.get("lesson_uid")): idx for idx, (_, lesson) in enumerate(_flatten_lessons(structure))}


def _uid_by_index(structure, index):
    lessons = [lesson for _, lesson in _flatten_lessons(structure)]
    if 0 <= index < len(lessons):
        return lessons[index].get("lesson_uid")
    return None


def _num_lessons(structure):
    return len(_flatten_lessons(structure))


def _frontier_from_seen(structure, seen):
    ordered = [str(lesson.get("lesson_uid")) for _, lesson in _flatten_lessons(structure)]
    frontier = -1
    for idx, uid in enumerate(ordered):
        if uid in seen:
            frontier = idx
        else:
            break
    return frontier


def _slugify(value):
    return str(value).lower().replace(" ", "-")


@pytest.fixture
def course_structure_data():
    return {
        "sections": [
            {
                "title": "Module 1",
                "order": 1,
                "lessons": [
                    {
                        "lesson_uid": "lesson-1",
                        "title": "Lesson 1",
                        "order": 1,
                    }
                ],
            }
        ]
    }


def test_learn_lesson_reuses_cached_exam_status(monkeypatch, course_structure_data):
    db = FakeDB(course_structure_data)

    app = Flask(__name__)
    app.testing = True

    def fake_render_template(template_name, **context):
        return f"rendered {template_name}"

    monkeypatch.setattr(course, "render_template", fake_render_template)

    @app.before_request
    def _set_user():
        g.user_id = 7
        g.user_email = "user@example.com"

    exam_deps = {
        "fetch_one": db.fetch_one,
        "fetch_all": db.fetch_all,
        "execute": db.execute,
        "ensure_structure": _ensure_structure,
    }
    exam_bp = create_exam_blueprint("", exam_deps)
    app.register_blueprint(exam_bp)

    course_deps = {
        "fetch_one": db.fetch_one,
        "ensure_structure": _ensure_structure,
        "flatten_lessons": _flatten_lessons,
        "sorted_sections": None,
        "first_lesson_uid": _first_lesson_uid,
        "find_lesson": _find_lesson,
        "next_prev_uids": _next_prev_uids,
        "lesson_index_map": _lesson_index_map,
        "uid_by_index": _uid_by_index,
        "num_lessons": _num_lessons,
        "total_course_duration": lambda structure: 0,
        "format_duration": lambda duration: str(duration),
        "slugify": _slugify,
        "seen_lessons": lambda user_id, course_id: set(),
        "last_seen_uid": lambda user_id, course_id: None,
        "log_activity": lambda *args, **kwargs: None,
        "log_view_once": lambda *args, **kwargs: None,
        "frontier_from_seen": _frontier_from_seen,
        "latest_registration": lambda email, course_id: None,
    }
    register_course_routes(app, "", course_deps)

    client = app.test_client()

    first = client.get("/learn/1/lesson-1")
    assert first.status_code == 200
    assert db.activity_queries == 1, db.fetch_all_sql

    second = client.get("/learn/1/lesson-1")
    assert second.status_code == 200
    assert db.activity_queries == 1

    save_resp = client.post(
        "/learn/1/exam/attempt-123/save",
        json={"module_index": 1, "answers": {}, "progress_percent": 50},
    )
    assert save_resp.status_code == 200
    assert db.activity_queries == 1

    third = client.get("/learn/1/lesson-1")
    assert third.status_code == 200
    assert db.activity_queries == 2
