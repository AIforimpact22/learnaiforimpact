# main.py — clean, login-ready, BASE_PATH-aware (psycopg3 + pooling)
# Enforces: only users with latest registrations.enrollment_status = 'accepted' can sign in.

import os
import re
import json
from contextlib import contextmanager
from urllib.parse import urlparse, parse_qs, unquote, quote, urlsplit, urlunsplit
from typing import Any, Dict, Optional, Set, List, Tuple

from flask import (
    Flask, render_template, abort, request, redirect, url_for, g, session, flash,
    has_request_context,
)
from markupsafe import Markup, escape

from functools import lru_cache

# Database (psycopg 3)
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

# OAuth (Google via Authlib)
from authlib.integrations.flask_client import OAuth

# External backends / blueprints
from admin import create_admin_blueprint
from profile import create_profile_blueprint
from learn import create_learn_blueprint
from exam import create_exam_blueprint
from home import register_home_routes
from course import register_course_routes
from course_content_loader import load_course_content

# =============================================================================
# BASE_PATH & Flask app
# =============================================================================
BASE_PATH = (os.getenv("BASE_PATH", "") or "").rstrip("/")
STATIC_URL_PATH = (BASE_PATH + "/static") if BASE_PATH else "/static"

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=STATIC_URL_PATH,
    template_folder="templates",
)
app.url_map.strict_slashes = False
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True,  # HTTPS on Render/production
)

# =============================================================================
# Auth mode
# =============================================================================
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "1").lower() in {"1", "true", "yes"}

# =============================================================================
# OAuth (Google) — supports base or full callback in OAUTH_REDIRECT_BASE
# =============================================================================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
OAUTH_REDIRECT_BASE = (os.getenv("OAUTH_REDIRECT_BASE", "") or "").rstrip("/")

oauth: Optional[OAuth] = None
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth = OAuth(app)
    oauth.register(
        "google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

def _require_oauth() -> OAuth:
    if oauth is None:
        abort(503, description="Google OAuth is not configured.")
    return oauth

def _bp(path: str = "") -> str:
    """Prefix a path with BASE_PATH (if set)."""
    p = path or "/"
    if not p.startswith("/"):
        p = "/" + p
    if BASE_PATH and (p == BASE_PATH or p.startswith(BASE_PATH + "/")):
        return p
    return (BASE_PATH + p) if BASE_PATH else p

def _oauth_callback_url() -> str:
    """
    Build the external callback URL:
    - If OAUTH_REDIRECT_BASE is a full callback, use it as-is.
    - Else treat it as a base and append '/auth/google/callback'.
    - If empty, derive from request.url_root + BASE_PATH.
    """
    base = OAUTH_REDIRECT_BASE or (request.url_root.rstrip("/") + (BASE_PATH or ""))
    if base.endswith("/auth/callback") or base.endswith("/auth/google/callback"):
        return base
    return base.rstrip("/") + "/auth/google/callback"

# =============================================================================
# DB configuration
# =============================================================================
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD")  # support either name
DB_NAME = os.getenv("DB_NAME")

ADMIN_MODE = os.getenv("ADMIN_MODE", "open").lower()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
SUPERADMIN_EMAIL = os.getenv("SUPERADMIN_EMAIL", "aiforimpact22@gmail.com")
ADMIN_EMAIL = SUPERADMIN_EMAIL

SIMPLE_LOGIN_PASSWORD = os.getenv("SIMPLE_LOGIN_PASSWORD", "Impact2025")
SIMPLE_LOGIN_USER_EMAIL = (
    os.getenv("SIMPLE_LOGIN_USER_EMAIL") or SUPERADMIN_EMAIL or "impact-user@example.com"
)
_enable_password_login_env = os.getenv("ENABLE_PASSWORD_LOGIN")
if _enable_password_login_env is not None:
    ENABLE_PASSWORD_LOGIN = _enable_password_login_env.lower() in {"1", "true", "yes"}
else:
    ENABLE_PASSWORD_LOGIN = not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
PASSWORD_LOGIN_ENABLED = ENABLE_PASSWORD_LOGIN or oauth is None

COURSE_PASSWORD = os.getenv("COURSE_PASSWORD", SIMPLE_LOGIN_PASSWORD)

if AUTH_REQUIRED:
    if PASSWORD_LOGIN_ENABLED:
        if oauth is None:
            print(
                "[Auth] Google OAuth not configured; falling back to single-password login.",
                flush=True,
            )
        else:
            print(
                "[Auth] Password login enabled; Google OAuth will be bypassed.",
                flush=True,
            )
    elif oauth is not None:
        print("[Auth] Google OAuth configured; password login disabled by default.", flush=True)

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_LOCAL = os.getenv("DATABASE_URL_LOCAL")
DB_HOST_OVERRIDE = os.getenv("DB_HOST")
DB_PORT_OVERRIDE = os.getenv("DB_PORT")
FORCE_TCP = os.getenv("FORCE_TCP", "").lower() in {"1", "true", "yes"}

ALLOW_RAW_HTML = os.getenv("ALLOW_RAW_HTML", "1").lower() in {"1", "true", "yes"}
SANITIZE_HTML = os.getenv("SANITIZE_HTML", "0").lower() in {"1", "true", "yes"}

BLEACH_ALLOWED_TAGS = [
    "a","abbr","acronym","b","blockquote","code","em","i","li","ol","strong","ul",
    "p","h1","h2","h3","h4","h5","h6","pre","hr","br","span","div","img","table",
    "thead","tbody","tr","th","td","caption","figure","figcaption","video","source",
    "iframe"
]
BLEACH_ALLOWED_ATTRS = {
    "*": ["class","id","style","title"],
    "a": ["href","name","target","rel"],
    "img": ["src","alt","width","height","loading"],
    "video": ["src","controls","preload","poster","width","height"],
    "source": ["src","type"],
    "iframe": ["src","width","height","allow","allowfullscreen","frameborder"]
}
BLEACH_ALLOWED_PROTOCOLS = ["http","https","mailto","data"]

def _on_managed_runtime() -> bool:
    # GAE or Cloud Run, etc.
    return os.getenv("GAE_ENV", "").startswith("standard") or bool(os.getenv("K_SERVICE"))

def _log_choice(kwargs: dict, origin: str):
    if "host" in kwargs and isinstance(kwargs["host"], str) and kwargs["host"].startswith("/cloudsql/"):
        print(f"[DB] {origin}: Unix socket -> {kwargs['host']}")
    else:
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 5432)
        print(f"[DB] {origin}: TCP -> {host}:{port}")

def _parse_database_url(url: str) -> dict:
    if not url:
        raise ValueError("Empty DATABASE_URL")
    # Normalize SA-style scheme to plain postgres for psycopg usage
    SA_PREFIXES = (
        "postgresql+psycopg://",
        "postgres+psycopg://",
        "postgresql+psycopg2://",
        "postgres+psycopg2://",
    )
    for pref in SA_PREFIXES:
        if url.startswith(pref):
            url = "postgresql://" + url.split("://", 1)[1]
            break

    p = urlparse(url)
    if p.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"Unsupported scheme '{p.scheme}'")
    user = unquote(p.username or "")
    password = unquote(p.password or "")
    dbname = (p.path or "").lstrip("/")
    qs = parse_qs(p.query or "", keep_blank_values=True)
    host = p.hostname
    port = p.port
    if "host" in qs and qs["host"]:
        host = qs["host"][0]
    if not dbname:
        if "dbname" in qs and qs["dbname"]:
            dbname = qs["dbname"][0]
        else:
            raise ValueError("DATABASE_URL missing dbname")
    kwargs = {
        "dbname": dbname,
        "user": user,
        "password": password,
        "connect_timeout": 10,
        "options": "-c search_path=public",
    }
    if host:
        kwargs["host"] = host
    if port and not (isinstance(host, str) and host.startswith("/")):
        kwargs["port"] = port
    if "sslmode" in qs and qs["sslmode"]:
        kwargs["sslmode"] = qs["sslmode"][0]
    return kwargs

def _tcp_kwargs() -> dict:
    host = DB_HOST_OVERRIDE or "127.0.0.1"
    port = int(DB_PORT_OVERRIDE or "5432")
    if not all([DB_NAME, DB_USER, DB_PASS]):
        raise RuntimeError("DB_NAME, DB_USER, DB_PASS must be set for TCP mode.")
    return {
        "host": host,
        "port": port,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASS,
        "sslmode": "disable",
        "connect_timeout": 10,
        "options": "-c search_path=public",
    }

def _socket_kwargs() -> dict:
    if not all([INSTANCE_CONNECTION_NAME, DB_NAME, DB_USER, DB_PASS]):
        raise RuntimeError("INSTANCE_CONNECTION_NAME, DB_NAME, DB_USER, DB_PASS must be set for socket mode.")
    return {
        "host": f"/cloudsql/{INSTANCE_CONNECTION_NAME}",
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASS,
        "connect_timeout": 10,
        "options": "-c search_path=public",
    }

def _connection_kwargs() -> dict:
    managed = _on_managed_runtime()

    if FORCE_TCP and not managed:
        kwargs = _tcp_kwargs(); _log_choice(kwargs, "FORCE_TCP"); return kwargs

    if not managed and DATABASE_URL_LOCAL:
        try:
            kwargs = _parse_database_url(DATABASE_URL_LOCAL)
            _log_choice(kwargs, "Using DATABASE_URL_LOCAL (parsed)")
            return kwargs
        except Exception as e:
            print(f"[DB] Ignoring DATABASE_URL_LOCAL: {e}")

    if DATABASE_URL:
        try:
            parsed = _parse_database_url(DATABASE_URL)
            host = parsed.get("host")
            if (not managed) and isinstance(host, str) and host.startswith("/cloudsql/"):
                print("[DB] DATABASE_URL targets /cloudsql/ but we are local; ignoring and using TCP.")
            else:
                _log_choice(parsed, "Using DATABASE_URL (parsed)")
                return parsed
        except Exception as e:
            print(f"[DB] Ignoring DATABASE_URL: {e}")

    if managed:
        kwargs = _socket_kwargs(); _log_choice(kwargs, "Managed runtime"); return kwargs

    kwargs = _tcp_kwargs(); _log_choice(kwargs, "Local dev"); return kwargs

# =============================================================================
# psycopg3 Connection Pool + helpers
# =============================================================================
_pg_pool: Optional[ConnectionPool] = None

def _to_conninfo(kwargs: dict) -> str:
    # Build libpq conninfo string from kwargs dict
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        # Quote only if needed
        s = str(v)
        if any(ch.isspace() for ch in s) or "'" in s or '"' in s:
            s = "'" + s.replace("'", r"\'") + "'"
        parts.append(f"{k}={s}")
    return " ".join(parts)

def init_pool():
    global _pg_pool
    if _pg_pool is not None:
        return
    kwargs = _connection_kwargs()
    conninfo = _to_conninfo(kwargs)
    _pg_pool = ConnectionPool(conninfo=conninfo, min_size=1, max_size=6)

@contextmanager
def get_conn():
    if _pg_pool is None:
        init_pool()
    # psycopg_pool returns connection from pool with context manager
    with _pg_pool.connection() as conn:
        yield conn

def fetch_all(q, params=None):
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(q, params or ())
            return cur.fetchall()

def fetch_one(q, params=None):
    rows = fetch_all(q, params)
    return rows[0] if rows else None

def execute(q, params=None):
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(q, params or ())
        conn.commit()

def execute_returning(q, params=None):
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(q, params or ())
            rows = cur.fetchall()
        conn.commit()
        return rows

# =============================================================================
# Activity logging + progress helpers (ONLY source of truth)
# =============================================================================
_ACTIVITY_TYPE_NAME: Optional[str] = None
_ACTIVITY_ENUM_LABELS: List[str] = []

def _detect_activity_type():
    """Detect enum type name of activity_log.a_type (if any) and cache labels."""
    global _ACTIVITY_TYPE_NAME, _ACTIVITY_ENUM_LABELS
    if _ACTIVITY_TYPE_NAME is not None:
        return
    try:
        row = fetch_one("""
            SELECT atttypid::regtype::text AS tname
            FROM pg_attribute
            WHERE attrelid = 'public.activity_log'::regclass
              AND attname = 'a_type';
        """)
        tname = (row or {}).get("tname")
        if tname and tname.lower() not in ("text", "varchar", "character varying", "pg_catalog.text"):
            _ACTIVITY_TYPE_NAME = tname
            labels = fetch_all(f"""
                SELECT enumlabel
                FROM pg_enum
                WHERE enumtypid = '{_ACTIVITY_TYPE_NAME}'::regtype
                ORDER BY enumsortorder;
            """)
            _ACTIVITY_ENUM_LABELS = [r["enumlabel"] for r in labels or []]
        else:
            _ACTIVITY_TYPE_NAME = None
            _ACTIVITY_ENUM_LABELS = []
    except Exception as e:
        print(f"[activity] type detect failed: {e}")
        _ACTIVITY_TYPE_NAME = None
        _ACTIVITY_ENUM_LABELS = []

def log_activity(user_id: int, course_id: int, lesson_uid: Optional[str],
                 a_type: Optional[str], score_points: Optional[int] = None,
                 passed: Optional[bool] = None, payload: Optional[dict] = None):
    _detect_activity_type()
    if not payload or not isinstance(payload, (dict, list)):
        payload = {"kind": "event"}
    payload_json = json.dumps(payload)
    label = (a_type or "event")
    try:
        if _ACTIVITY_TYPE_NAME:
            if _ACTIVITY_ENUM_LABELS and label not in _ACTIVITY_ENUM_LABELS:
                label = _ACTIVITY_ENUM_LABELS[0]
            execute(f"""
                INSERT INTO public.activity_log
                    (user_id, course_id, lesson_uid, a_type, created_at, score_points, passed, payload)
                VALUES (%s, %s, %s, %s::{_ACTIVITY_TYPE_NAME}, now(), %s, %s, %s);
            """, (user_id, course_id, str(lesson_uid), label, score_points, passed, payload_json))
        else:
            execute("""
                INSERT INTO public.activity_log
                    (user_id, course_id, lesson_uid, a_type, created_at, score_points, passed, payload)
                VALUES (%s, %s, %s, %s, now(), %s, %s, %s);
            """, (user_id, course_id, str(lesson_uid), label, score_points, passed, payload_json))
    except Exception as e:
        print(f"[activity] insert failed (safe): {e}")

def log_view_once(user_id: int, course_id: int, lesson_uid: str, window_seconds: int = 120):
    """Debounced 'view' log: at most one 'view' per user/course/lesson within the time window.
       Works whether created_at is TEXT, timestamp, or timestamptz."""
    try:
        row = fetch_one("""
            SELECT id
              FROM public.activity_log
             WHERE user_id = %s
               AND course_id = %s
               AND lesson_uid = %s
               AND (a_type::text = 'view')
               -- Cast created_at to timestamptz so we can compare to now() - interval
               AND (created_at::timestamptz) >= (now() - make_interval(secs => %s))
             LIMIT 1;
        """, (user_id, course_id, str(lesson_uid), int(window_seconds)))
        if not row:
            # Include a small payload so it's never NULL
            log_activity(user_id, course_id, str(lesson_uid), "view", payload={"kind": "view"})
    except Exception as e:
        print(f"[activity] log_view_once failed: {e}")

def seen_lessons(user_id: int, course_id: int) -> Set[str]:
    try:
        rows = fetch_all("""
            SELECT DISTINCT lesson_uid
              FROM public.activity_log
             WHERE user_id = %s AND course_id = %s AND lesson_uid IS NOT NULL;
        """, (user_id, course_id))
        return {str(r["lesson_uid"]) for r in rows if r.get("lesson_uid")}
    except Exception as e:
        print(f"[activity] seen_lessons failed: {e}")
        return set()

def last_seen_uid(user_id: int, course_id: int) -> Optional[str]:
    try:
        row = fetch_one("""
            SELECT lesson_uid
              FROM public.activity_log
             WHERE user_id = %s AND course_id = %s AND lesson_uid IS NOT NULL
             ORDER BY created_at DESC
             LIMIT 1;
        """, (user_id, course_id))
        return str(row["lesson_uid"]) if row and row.get("lesson_uid") else None
    except Exception as e:
        print(f"[activity] last_seen_uid failed: {e}")
        return None

# ---- Structure helpers -------------------------------------------------------
def _structure_signature(structure: Dict[str, Any]):
    if not isinstance(structure, dict):
        return None
    sections = structure.get("sections")
    if isinstance(sections, (list, tuple)):
        lesson_ids = []
        for sec in sections:
            if isinstance(sec, dict):
                lesson_ids.append(id(sec.get("lessons")))
            else:
                lesson_ids.append(id(sec))
        return (id(sections), tuple(lesson_ids))
    return id(sections)

def _section_sort_key(section: Any) -> Tuple[Any, Any]:
    if isinstance(section, dict):
        return (section.get("order") or 0, section.get("title") or "")
    return (0, "")

def _lesson_sort_key(lesson: Any) -> Tuple[Any, Any]:
    if isinstance(lesson, dict):
        return (lesson.get("order") or 0, lesson.get("title") or "")
    return (0, "")

def _structure_cache_data(structure: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(structure, dict):
        return {"sections": tuple(), "flat": tuple(), "uids": tuple()}

    sections_raw = structure.get("sections")
    if isinstance(sections_raw, (list, tuple)):
        sections_iter = list(sections_raw)
    elif sections_raw:
        sections_iter = [sections_raw]
    else:
        sections_iter = []

    sections_sorted: List[Dict[str, Any]] = []
    flat: List[Tuple[dict, dict]] = []
    uids: List[str] = []

    for section in sorted(sections_iter, key=_section_sort_key):
        if isinstance(section, dict):
            lessons_raw = section.get("lessons")
            if isinstance(lessons_raw, (list, tuple)):
                lessons_iter = list(lessons_raw)
            elif lessons_raw:
                lessons_iter = [lessons_raw]
            else:
                lessons_iter = []
            lessons_sorted = sorted(lessons_iter, key=_lesson_sort_key)
            section_copy = dict(section)
        else:
            lessons_sorted = []
            section_copy = {"value": section, "lessons": tuple()}
        section_copy["lessons"] = tuple(lessons_sorted)
        sections_sorted.append(section_copy)
        for lesson in lessons_sorted:
            if not isinstance(lesson, dict):
                continue
            flat.append((section_copy, lesson))
            uid = lesson.get("lesson_uid")
            if uid is not None:
                uids.append(str(uid))

    return {
        "sections": tuple(sections_sorted),
        "flat": tuple(flat),
        "uids": tuple(uids),
    }

def _structure_cache(structure: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(structure, dict):
        return {"version": None, "sections": tuple(), "flat": tuple(), "uids": tuple()}

    version = _structure_signature(structure)

    if has_request_context():
        store: Dict[int, Dict[str, Any]] = getattr(g, "_structure_cache", None)
        if store is None:
            store = g._structure_cache = {}
        cached = store.get(id(structure))
        if cached and cached.get("version") == version:
            return cached
        data = _structure_cache_data(structure)
        cached = {"version": version, **data}
        store[id(structure)] = cached
        return cached

    data = _structure_cache_data(structure)
    return {"version": version, **data}

def sorted_sections(structure: Dict[str, Any]):
    return _structure_cache(structure)["sections"]

def flatten_lessons(structure: Dict[str, Any]):
    return _structure_cache(structure)["flat"]

def _lesson_uids(structure: Dict[str, Any]) -> Tuple[str, ...]:
    return _structure_cache(structure)["uids"]

def _frontier_from_seen(structure: Dict[str, Any], seen: Set[str]) -> int:
    frontier = -1
    for i, uid in enumerate(_lesson_uids(structure)):
        if uid in seen:
            frontier = i
        else:
            break
    return frontier

# =============================================================================
# Rendering helpers (Markdown/HTML)
# =============================================================================
_HTML_PATTERN = re.compile(r"</?\w+[^>]*>")

def _sanitize_if_enabled(html: str) -> str:
    if not SANITIZE_HTML:
        return html
    try:
        import bleach
        return bleach.clean(
            html,
            tags=BLEACH_ALLOWED_TAGS,
            attributes=BLEACH_ALLOWED_ATTRS,
            protocols=BLEACH_ALLOWED_PROTOCOLS,
            strip=False
        )
    except Exception:
        return html

@lru_cache(maxsize=512)
def _render_rich_cached(text: str, allow_raw: bool, sanitize_flag: bool) -> str:
    if not text:
        return ""
    if allow_raw and _HTML_PATTERN.search(text):
        return _sanitize_if_enabled(text)
    try:
        import markdown

        html = markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "sane_lists", "toc", "codehilite", "md_in_html", "attr_list"],
            output_format="html5",
        )
        return _sanitize_if_enabled(html)
    except Exception:
        safe = "<p>" + escape(text).replace("\n\n", "</p><p>").replace("\n", "<br/>") + "</p>"
        return str(safe)

def render_rich(text: Optional[str]) -> Markup:
    if text is None:
        return Markup("")
    text_str = text if isinstance(text, str) else str(text)
    html = _render_rich_cached(text_str, ALLOW_RAW_HTML, SANITIZE_HTML)
    return Markup(html)

app.jinja_env.filters["rich"] = render_rich

def slugify(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum(): out.append(ch)
        elif ch in (" ", "-", "_"): out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "course"

def ensure_structure(structure_raw: Any) -> Dict[str, Any]:
    if not structure_raw: return {"sections": []}
    if isinstance(structure_raw, dict): return structure_raw
    try:
        return json.loads(structure_raw)
    except Exception:
        return {"sections": []}


@lru_cache(maxsize=1)
def _course_structure_from_file() -> Dict[str, Any]:
    return load_course_content() or {"sections": []}


def load_course_structure(
    structure_raw: Any,
    *,
    course_title: Optional[str] = None,
) -> Dict[str, Any]:
    title = (course_title or "").strip()
    if title == COURSE_TITLE:
        return ensure_structure(_course_structure_from_file())
    return ensure_structure(structure_raw)


def course_structure_from_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"sections": []}
    return load_course_structure(
        row.get("structure"),
        course_title=row.get("title"),
    )

def first_lesson_uid(structure: Dict[str, Any]) -> Optional[str]:
    flat = flatten_lessons(structure)
    return str(flat[0][1].get("lesson_uid")) if flat else None

def find_lesson(structure: Dict[str, Any], lesson_uid: str):
    secs = structure.get("sections") or []
    for si, s in enumerate(secs):
        for li, l in enumerate(s.get("lessons") or []):
            if str(l.get("lesson_uid")) == str(lesson_uid):
                return si, li
    return None, None

def next_prev_uids(structure: Dict[str, Any], current_uid: str):
    flat = _lesson_uids(structure)
    if not flat:
        return (None, None)
    try:
        idx = flat.index(str(current_uid))
    except ValueError:
        return (None, None)
    prev_uid = flat[idx - 1] if idx > 0 else None
    next_uid = flat[idx + 1] if idx < len(flat) - 1 else None
    return (prev_uid, next_uid)

def lesson_index_map(structure: Dict[str, Any]) -> Dict[str, int]:
    mapping = {}
    for i, uid in enumerate(_lesson_uids(structure)):
        mapping[uid] = i
    return mapping

def uid_by_index(structure: Dict[str, Any], index: int) -> Optional[str]:
    flat = _lesson_uids(structure)
    if 0 <= index < len(flat):
        return flat[index]
    return None

def num_lessons(structure: Dict[str, Any]) -> int:
    return len(flatten_lessons(structure))

def total_course_duration(structure: Dict[str, Any]) -> int:
    total = 0
    for _, l in flatten_lessons(structure):
        c = l.get("content") or {}
        dur = c.get("duration_sec") or 0
        if isinstance(dur, int): total += max(0, dur)
    return total

def format_duration(total_sec: Optional[int]) -> str:
    if not total_sec: return "—"
    m, _ = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m"
    return f"{m}m"

app.jinja_env.filters["duration"] = format_duration

# =============================================================================
# Course seed
# =============================================================================
COURSE_TITLE = "Advanced AI Utilization and Real-Time Deployment"
COURSE_COVER = "https://i.imgur.com/iIMdWOn.jpeg"
COURSE_DESC = (
    "This course is a master course that offers Participants will develop advanced skills in "
    "coding, database management, machine learning, and real-time application deployment. "
    "This course focuses on practical implementations, enabling learners to create AI-driven solutions, "
    "deploy them in real-world scenarios, and integrate apps with cloud and database systems."
)
WEEKS = [
    "Week 1: Ice Breaker for Coding",
    "Week 2: UI and UX",
    "Week 3: Modularity",
    "Week 4: Advanced SQL and Databases",
    "Week 5: Fundamental of Statistics for Machine Learning",
    "Week 6: Unsupervised Machine Learning",
    "Week 7: Supervised Machine Learning",
    "Week 8: Utilizing AI API",
    "Week 9: Capstone Project",
]

def seed_course_if_missing() -> int:
    row = fetch_one("SELECT id FROM courses WHERE title = %s;", (COURSE_TITLE,))
    if row:
        return row["id"]
    structure = {
        "thumbnail_url": COURSE_COVER,
        "category": "Artificial Intelligence",
        "level": "Intermediate–Advanced",
        "rating": 4.9,
        "description_md": COURSE_DESC,
        "what_you_will_learn": [
            "Design end-to-end AI applications.",
            "Integrate cloud + database with ML pipelines.",
            "Deploy real-time inference and monitoring."
        ],
        "instructors": [
            {"name": "Course Lead", "title": "AI Engineer", "avatar_url": ""}
        ],
        "sections": []
    }
    for i, title in enumerate(WEEKS, start=1):
        structure["sections"].append({"title": title, "order": i, "lessons": []})
    created = execute_returning("""
        WITH admin_user AS (
            INSERT INTO users (email, full_name, role)
            VALUES (%s, %s, 'admin')
            ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING id
        )
        INSERT INTO courses (title, created_by, is_published, published_at, structure)
        SELECT %s, admin_user.id, TRUE, now(), %s
        FROM admin_user
        RETURNING id;
    """, ("aiforimpact22@gmail.com", "Portal Admin", COURSE_TITLE, json.dumps(structure)))
    return created[0]["id"]

# =============================================================================
# Identity helpers
# =============================================================================
def _session_email() -> Optional[str]:
    u = session.get("user") or {}
    e = (u.get("email") or "").strip().lower()
    return e or None

def _iap_email() -> Optional[str]:
    h = (
        request.headers.get("X-Goog-Authenticated-User-Email")
        or request.headers.get("X-Appengine-User-Email")
    )
    if not h:
        return None
    return h.split(":", 1)[-1].strip().lower()

def current_user_email() -> Optional[str]:
    return _session_email() or _iap_email()

def ensure_user_row(email: str) -> int:
    row = fetch_one("SELECT id FROM users WHERE email = %s;", (email,))
    if row:
        return row["id"]
    display = email.split("@", 1)[0].replace(".", " ").title()
    rows = execute_returning("""
        INSERT INTO users (email, full_name, role)
        VALUES (%s, %s, 'learner')
        ON CONFLICT (email) DO UPDATE SET full_name = EXCLUDED.full_name
        RETURNING id;
    """, (email, display))
    return rows[0]["id"]

# ---- Enrollment gate helpers -------------------------------------------------
def _latest_enrollment_status(email: str) -> Optional[str]:
    """
    Returns the 'enrollment_status' of the *latest* registration row for this email,
    or None if no registration exists. Comparison is case-insensitive.
    """
    try:
        r = fetch_one("""
            SELECT enrollment_status
              FROM public.registrations
             WHERE lower(user_email) = lower(%s)
             ORDER BY created_at DESC
             LIMIT 1;
        """, (email,))
        status = (r or {}).get("enrollment_status")
        if isinstance(status, str):
            return status.strip().lower() or None
        return None
    except Exception as e:
        print(f"[auth] enrollment lookup failed for {email}: {e}")
        return None

def _is_superadmin(email: Optional[str]) -> bool:
    return bool(email) and email.strip().lower() == (SUPERADMIN_EMAIL or "").strip().lower()

def _signin_allowed(email: str) -> Tuple[bool, str]:
    """
    Gate: allow sign-in only if latest status == 'accepted', or superadmin.
    Returns (allowed, reason) where reason is one of: 'accepted', 'pending', 'rejected', 'none', 'error', 'superadmin'.
    """
    simple_email = (SIMPLE_LOGIN_USER_EMAIL or "").strip().lower()
    if simple_email and email.strip().lower() == simple_email:
        return True, "password-login"
    if _is_superadmin(email):
        return True, "superadmin"
    status = _latest_enrollment_status(email)
    if status == "accepted":
        return True, "accepted"
    # Normalize some common statuses; unknown becomes 'none'
    if status in {"pending", "applied", "awaiting", "review", "reviewing"}:
        return False, "pending"
    if status in {"rejected", "declined", "denied"}:
        return False, "rejected"
    if status is None:
        return False, "none"
    return False, status  # any non-accepted value blocks

# =============================================================================
# Jinja helpers
# =============================================================================
@app.context_processor
def inject_user_and_base():
    def page_allowed(_name: str) -> bool:
        return True
    return {
        "current_user_email": getattr(g, "user_email", None),
        "base_path": BASE_PATH,
        "bp": _bp,
        "page_allowed": page_allowed
    }

# =============================================================================
# Routes (auth, health, identity)
# =============================================================================
@app.get("/healthz")
def healthz():
    try:
        row = fetch_one("SELECT 1 AS ok;")
        ok = bool(row and row.get("ok") == 1)
        return ("ok" if ok else "db-fail", 200 if ok else 500)
    except Exception as e:
        return (f"error: {e}", 500)

@app.get("/favicon.ico")
def favicon():
    return ("", 204)

def _sanitize_next(next_url: Optional[str]) -> str:
    if not next_url:
        return _bp("/")
    parts = urlsplit(next_url)
    if parts.scheme or parts.netloc:
        return _bp("/")
    path = parts.path or "/"
    blocked_prefixes = {_bp("/login"), _bp("/auth"), "/login", "/auth"}
    if any(path == p or path.startswith(p + "/") for p in blocked_prefixes):
        return _bp("/")
    safe = urlunsplit(("", "", path, parts.query, ""))
    return safe or _bp("/")

# --- LOGIN (root) ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if PASSWORD_LOGIN_ENABLED:
        if request.method == "POST":
            next_url = _sanitize_next(request.form.get("next") or session.get("login_next"))
        else:
            next_url = _sanitize_next(request.args.get("next") or session.get("login_next"))
        session["login_next"] = next_url

        error = None
        if request.method == "POST":
            password = (request.form.get("password") or "").strip()
            if password == SIMPLE_LOGIN_PASSWORD:
                email = (SIMPLE_LOGIN_USER_EMAIL or "").strip().lower()
                if not email:
                    email = "impact-user@example.com"
                allowed, reason = _signin_allowed(email)
                if not allowed:
                    session.pop("user", None)
                    return redirect(
                        f"{_bp('/auth/blocked')}?status={quote(str(reason))}&next={quote(next_url, safe='/:?&=')}"
                    )
                session["user"] = {
                    "email": email,
                    "name": "Portal User",
                    "picture": None,
                    "sub": "password-login",
                }
                next_redirect = _sanitize_next(session.pop("login_next", None))
                return redirect(next_redirect)
            else:
                error = "Incorrect password. Please try again."

        return render_template(
            "password_login.html",
            next_url=next_url,
            error=error,
            base_path=BASE_PATH,
        )

    provider = _require_oauth()
    next_url = _sanitize_next(request.args.get("next"))
    session["login_next"] = next_url
    return provider.google.authorize_redirect(_oauth_callback_url())


@app.route("/course/unlock", methods=["GET", "POST"])
def course_unlock():
    if request.method == "POST":
        next_url = _sanitize_next(request.form.get("next"))
    else:
        next_url = _sanitize_next(request.args.get("next"))
    if (
        not next_url
        or next_url.startswith("/course/unlock")
        or next_url.startswith(_bp("/course/unlock"))
    ):
        next_url = _bp("/learn")

    error = None
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if password == COURSE_PASSWORD:
            session["course_unlocked"] = True
            g.course_unlocked = True
            return redirect(next_url)
        error = "Incorrect password. Please try again."

    return render_template(
        "course_unlock.html",
        next_url=next_url,
        error=error,
        base_path=BASE_PATH,
    )

# --- LOGOUT (root) ---
@app.get("/logout")
def logout():
    session.clear()
    flash("Signed out.", "success")
    return redirect(_bp("/"))

# --- BLOCKED PAGE (root) ---
@app.get("/auth/blocked")
def auth_blocked():
    """
    Public page explaining why access is blocked.
    Query params:
      - status: 'pending' | 'rejected' | 'none' | other
      - next: optional path to return to after acceptance
    """
    status = (request.args.get("status") or "").strip().lower() or "none"
    next_url = _sanitize_next(request.args.get("next"))
    # Minimal, template-free HTML to avoid missing templates in some deployments
    def _msg_for(s: str) -> str:
        if s == "pending":
            return "Your registration is pending. You'll gain access once it's accepted."
        if s == "rejected":
            return "Your registration was not accepted. Contact support if you believe this is an error."
        if s == "none":
            return "No registration found for your account. Please register first."
        return f"Access is restricted (status: {escape(s)})."
    html = f"""
<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>Access Restricted</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;margin:2rem;line-height:1.5">
  <h1>Access Restricted</h1>
  <p>{_msg_for(status)}</p>
  <p>If you need help, contact: <a href="mailto:{escape(ADMIN_EMAIL)}">{escape(ADMIN_EMAIL)}</a></p>
  <p>
    <a href="{_bp('/logout')}">Try a different Google account</a>
    {" | " if next_url else ""}
    {f'<a href="{escape(next_url)}">Go back</a>' if next_url else ""}
  </p>
</body></html>"""
    return (html, 403)

# --- CALLBACK (root: support both /auth/callback and /auth/google/callback) ---
@app.get("/auth/callback")
@app.get("/auth/google/callback")
def auth_callback():
    provider = _require_oauth()
    token = provider.google.authorize_access_token()

    # Prefer ID token; fallback to userinfo
    claims = None
    try:
        claims = provider.google.parse_id_token(token)
    except Exception:
        claims = None
    if not claims:
        try:
            meta = provider.google.load_server_metadata() or {}
        except Exception:
            meta = {}
        userinfo_url = meta.get("userinfo_endpoint") or "https://openidconnect.googleapis.com/v1/userinfo"
        resp = provider.google.get(userinfo_url)
        claims = resp.json()

    email = (claims.get("email") or "").strip().lower()
    if not email:
        abort(400, description="Google authentication failed (no email).")

    # Enrollment gate BEFORE establishing a session
    allowed, reason = _signin_allowed(email)
    if not allowed:
        # Ensure no lingering session is left
        session.pop("user", None)
        next_url = _sanitize_next(session.pop("login_next", None))
        return redirect(f"{_bp('/auth/blocked')}?status={quote(str(reason))}&next={quote(next_url, safe='/:?&=')}")

    # Allowed: persist session and ensure user record
    session["user"] = {
        "email": email,
        "name": claims.get("name"),
        "picture": claims.get("picture"),
        "sub": claims.get("sub"),
    }
    try:
        ensure_user_row(email)
    except Exception as e:
        print(f"[Auth] ensure_user_row failed for {email}: {e}")

    next_url = _sanitize_next(session.pop("login_next", None))
    return redirect(next_url)

# --- Register the SAME routes under BASE_PATH aliases (e.g., /learn/login) ---
if BASE_PATH:
    app.add_url_rule(
        f"{BASE_PATH}/login", endpoint="login_bp", view_func=login, methods=["GET", "POST"]
    )
    app.add_url_rule(f"{BASE_PATH}/logout", endpoint="logout_bp", view_func=logout, methods=["GET"])
    app.add_url_rule(f"{BASE_PATH}/auth/callback", endpoint="auth_callback_bp", view_func=auth_callback, methods=["GET"])
    app.add_url_rule(f"{BASE_PATH}/auth/google/callback", endpoint="auth_callback_google_bp", view_func=auth_callback, methods=["GET"])
    app.add_url_rule(f"{BASE_PATH}/auth/blocked", endpoint="auth_blocked_bp", view_func=auth_blocked, methods=["GET"])
    app.add_url_rule(
        f"{BASE_PATH}/course/unlock",
        endpoint="course_unlock_bp",
        view_func=course_unlock,
        methods=["GET", "POST"],
    )

def _is_public_path(path: str) -> bool:
    if path.startswith(STATIC_URL_PATH):
        return True
    public_exact = {
        "/",
        _bp("/"),
        "/favicon.ico",
        _bp("/favicon.ico"),
        "/healthz",
        _bp("/healthz"),
        "/login",
        _bp("/login"),
        "/logout",
        _bp("/logout"),
        "/auth/callback",
        _bp("/auth/callback"),
        "/auth/google/callback",
        _bp("/auth/google/callback"),
        "/auth/blocked",
        _bp("/auth/blocked"),
        "/course/unlock",
        _bp("/course/unlock"),
        "/admin/whoami",
        _bp("/admin/whoami"),
    }
    if path in public_exact:
        return True

    public_prefixes = {
        "/course/",
        _bp("/course/"),
    }
    return any(path.startswith(prefix) for prefix in public_prefixes)


def _is_course_path(path: str) -> bool:
    course_roots = {
        "/learn",
        "/learn/",
        _bp("/learn"),
        _bp("/learn/")
    }
    if path in course_roots:
        return True
    course_prefixes = {
        "/learn/",
        _bp("/learn/"),
    }
    return any(path.startswith(prefix) for prefix in course_prefixes)

@app.before_request
def enforce_or_attach_identity():
    path = request.path
    if _is_public_path(path):
        return
    email = current_user_email()
    if _is_course_path(path) and not email:
        if session.get("course_unlocked"):
            g.course_unlocked = True
            return
        full = request.full_path if request.query_string else request.path
        next_url = _sanitize_next(full)
        return redirect(f"{_bp('/course/unlock')}?next={quote(next_url, safe='/:?&=')}")
    if email:
        allowed, reason = _signin_allowed(email)
        if not allowed:
            # Drop any session and redirect to blocked
            session.clear()
            full = request.full_path if request.query_string else request.path
            next_url = _sanitize_next(full)
            return redirect(f"{_bp('/auth/blocked')}?status={quote(str(reason))}&next={quote(next_url, safe='/:?&=')}")
        # Allowed -> attach identity and ensure user record
        g.user_email = email
        try:
            g.user_id = ensure_user_row(email)
        except Exception as e:
            print(f"[Auth] ensure_user_row failed for {email}: {e}")
        return
    if _is_course_path(path):
        return
    if AUTH_REQUIRED:
        full = request.full_path if request.query_string else request.path
        next_url = _sanitize_next(full)
        return redirect(f"{_bp('/login')}?next={quote(next_url, safe='/:?&=')}")

# ---- Convenience: latest registration for display name
def _latest_registration(email: str, course_id: int):
    return fetch_one("""
        SELECT *
          FROM public.registrations
         WHERE lower(user_email) = lower(%s) AND course_id = %s
         ORDER BY created_at DESC
         LIMIT 1;
    """, (email, course_id)) or fetch_one("""
        SELECT *
          FROM public.registrations
         WHERE lower(user_email) = lower(%s)
         ORDER BY created_at DESC
         LIMIT 1;
    """, (email,))

# =============================================================================
# Register split backends (home.py & course.py)
# =============================================================================
_home_deps = {
    "COURSE_TITLE": COURSE_TITLE,
    "COURSE_COVER": COURSE_COVER,
    "fetch_one": fetch_one,
    "ensure_structure": ensure_structure,
    "course_structure": course_structure_from_row,
    "load_course_structure": load_course_structure,
    "flatten_lessons": flatten_lessons,
    "sorted_sections": sorted_sections,
    "total_course_duration": total_course_duration,
    "format_duration": format_duration,
    "first_lesson_uid": first_lesson_uid,
    "slugify": slugify,
    "last_seen_uid": last_seen_uid,
    "seed_course_if_missing": seed_course_if_missing,
}
_course_deps = {
    "fetch_one": fetch_one,
    "ensure_structure": ensure_structure,
    "course_structure": course_structure_from_row,
    "load_course_structure": load_course_structure,
    "flatten_lessons": flatten_lessons,
    "sorted_sections": sorted_sections,
    "first_lesson_uid": first_lesson_uid,
    "find_lesson": find_lesson,
    "next_prev_uids": next_prev_uids,
    "lesson_index_map": lesson_index_map,
    "uid_by_index": uid_by_index,
    "num_lessons": num_lessons,
    "total_course_duration": total_course_duration,
    "format_duration": format_duration,
    "slugify": slugify,
    "seen_lessons": seen_lessons,
    "last_seen_uid": last_seen_uid,
    "log_activity": log_activity,
    "log_view_once": log_view_once,
    "frontier_from_seen": _frontier_from_seen,
    "latest_registration": _latest_registration,
}
register_home_routes(app, BASE_PATH, _home_deps)
register_course_routes(app, BASE_PATH, _course_deps)

# =============================================================================
# Admin & other blueprints
# =============================================================================
_admin_deps = {
    "COURSE_TITLE": COURSE_TITLE,
    "fetch_one": fetch_one,
    "execute": execute,
    "execute_returning": execute_returning,
    "ensure_structure": ensure_structure,
    "course_structure": course_structure_from_row,
    "load_course_structure": load_course_structure,
    "seed_course_if_missing": seed_course_if_missing,
}
app.register_blueprint(create_admin_blueprint("", _admin_deps, name="admin"))
if BASE_PATH:
    app.register_blueprint(create_admin_blueprint(BASE_PATH, _admin_deps, name="admin_alias"))

app.register_blueprint(create_profile_blueprint())  # self-contained; handles its own BASE_PATH aliasing

learn_bp = create_learn_blueprint(BASE_PATH, {
    "fetch_one": fetch_one, "fetch_all": fetch_all, "execute": execute,
    "ensure_structure": ensure_structure,
    "course_structure": course_structure_from_row,
    "load_course_structure": load_course_structure,
    "flatten_lessons": flatten_lessons,
    "sorted_sections": sorted_sections,
    "first_lesson_uid": first_lesson_uid, "find_lesson": find_lesson,
    "next_prev_uids": next_prev_uids, "lesson_index_map": lesson_index_map,
    "uid_by_index": uid_by_index, "num_lessons": num_lessons,
    "total_course_duration": total_course_duration, "format_duration": format_duration,
    "slugify": slugify, "latest_registration": _latest_registration,
})
app.register_blueprint(learn_bp)

exam_bp = create_exam_blueprint(BASE_PATH, {
    "fetch_one": fetch_one, "fetch_all": fetch_all, "execute": execute,
    "ensure_structure": ensure_structure,
    "course_structure": course_structure_from_row,
    "load_course_structure": load_course_structure,
    "flatten_lessons": flatten_lessons,
    "sorted_sections": sorted_sections,
})
app.register_blueprint(exam_bp)

# =============================================================================
# Local dev entry
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
