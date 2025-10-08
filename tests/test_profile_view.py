import sys
from pathlib import Path

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
