import os
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone

from flask import Blueprint, render_template, g, redirect, url_for, request, session
from urllib.parse import urlsplit, urlunsplit, quote

from psycopg import conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# =============================== Env / BASE PATH ==============================
BASE_PATH = (os.getenv("BASE_PATH", "") or "").rstrip("/")
STATIC_URL_PATH = (BASE_PATH + "/static") if BASE_PATH else "/static"

# Exam attempt cap for display purposes (server is enforced in exam blueprint)
EXAM_MAX_SUBMISSIONS = int(os.getenv("EXAM_MAX_SUBMISSIONS") or 3)

from course_content_loader import load_course_content

PRIMARY_COURSE_TITLE = "Advanced AI Utilization and Real-Time Deployment"


@lru_cache(maxsize=1)
def _course_structure_from_file() -> Dict[str, Any]:
    data = load_course_content()
    if isinstance(data, dict):
        return data
    return {"sections": []}

def _bp(path: str = "") -> str:
    p = path or "/"
    if not p.startswith("/"):
        p = "/" + p
    if BASE_PATH and (p == BASE_PATH or p.startswith(BASE_PATH + "/")):
        return p
    return (BASE_PATH + p) if BASE_PATH else p

# =============================== DB Helpers ==================================
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_LOCAL = os.getenv("DATABASE_URL_LOCAL")
DB_HOST_OVERRIDE = os.getenv("DB_HOST")
DB_PORT_OVERRIDE = os.getenv("DB_PORT")
FORCE_TCP = os.getenv("FORCE_TCP", "").lower() in ("1", "true", "yes")

def _on_managed_runtime() -> bool:
    return os.getenv("GAE_ENV", "").startswith("standard") or bool(os.getenv("K_SERVICE"))

def _parse_database_url(url: str) -> dict:
    if not url:
        raise ValueError("Empty DATABASE_URL")
    if url.startswith("postgresql+psycopg2://"):
        url = "postgresql://" + url.split("postgresql+psycopg2://", 1)[1]
    if url.startswith("postgres+psycopg2://"):
        url = "postgres://" + url.split("postgres+psycopg2://", 1)[1]
    from urllib.parse import urlparse, parse_qs, unquote
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
        return _tcp_kwargs()
    if not managed and DATABASE_URL_LOCAL:
        try:
            return _parse_database_url(DATABASE_URL_LOCAL)
        except Exception:
            pass
    if DATABASE_URL:
        try:
            parsed = _parse_database_url(DATABASE_URL)
            if (not managed) and isinstance(parsed.get("host"), str) and parsed["host"].startswith("/cloudsql/"):
                pass
            else:
                return parsed
        except Exception:
            pass
    if managed:
        return _socket_kwargs()
    return _tcp_kwargs()

_pg_pool: Optional[ConnectionPool] = None  # private to profile.py

def _init_pool():
    global _pg_pool
    if _pg_pool is None:
        kwargs = _connection_kwargs()
        conn_str = conninfo.make_conninfo(**kwargs)
        _pg_pool = ConnectionPool(conn_str, min_size=1, max_size=4)

def _with_conn(fn):
    def wrapper(*args, **kwargs):
        _init_pool()
        assert _pg_pool is not None
        with _pg_pool.connection() as conn:
            return fn(conn, *args, **kwargs)
    return wrapper

@_with_conn
def _fetch_all(conn, q, params=None):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(q, params or ())
        return cur.fetchall()

@_with_conn
def _fetch_one(conn, q, params=None):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(q, params or ())
        row = cur.fetchone()
    return row

@_with_conn
def _execute(conn, q, params=None):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(q, params or ())
    conn.commit()

# =============================== Auth helpers ================================
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

def _current_user_email() -> Optional[str]:
    return getattr(g, "user_email", None) or _session_email() or _iap_email()

def _sanitize_next(next_url: Optional[str]) -> str:
    if not next_url:
        return _bp("/")
    parts = urlsplit(next_url)
    if parts.scheme or parts.netloc:
        return _bp("/")
    path = parts.path or "/"
    blocked = {_bp("/login"), _bp("/auth"), "/login", "/auth"}
    if any(path == p or path.startswith(p + "/") for p in blocked):
        return _bp("/")
    return urlunsplit(("", "", path, parts.query, "")) or _bp("/")

# =============================== Local helpers ===============================
def _ensure_structure(structure_raw: Any) -> Dict[str, Any]:
    if not structure_raw:
        return {"sections": []}
    if isinstance(structure_raw, dict):
        return structure_raw
    try:
        return json.loads(structure_raw)
    except Exception:
        return {"sections": []}


def _structure_from_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"sections": []}
    title = (row.get("title") or "").strip()
    if title == PRIMARY_COURSE_TITLE:
        return _ensure_structure(_course_structure_from_file())
    return _ensure_structure(row.get("structure"))

def _flatten_lessons(structure: Dict[str, Any]):
    out = []
    secs = structure.get("sections") or []
    secs = sorted(secs, key=lambda s: (s.get("order") or 0, s.get("title") or ""))
    for s in secs:
        lessons = s.get("lessons") or []
        lessons = sorted(lessons, key=lambda l: (l.get("order") or 0, l.get("title") or ""))
        for l in lessons:
            out.append((s, l))
    return out

def _num_lessons(structure: Dict[str, Any]) -> int:
    return len(_flatten_lessons(structure))

def _total_course_duration(structure: Dict[str, Any]) -> int:
    total = 0
    for _, l in _flatten_lessons(structure):
        c = l.get("content") or {}
        dur = c.get("duration_sec") or 0
        if isinstance(dur, int):
            total += max(0, dur)
    return total

def _format_duration(total_sec: Optional[int]) -> str:
    if not total_sec:
        return "—"
    m = total_sec // 60
    h = m // 60
    m = m % 60
    if h:
        return f"{h}h {m}m"
    return f"{m}m"

def _as_int(x) -> int:
    if x is None: return 0
    if isinstance(x, Decimal): return int(x)
    try: return int(x)
    except Exception: return 0

def _name_commas(first: Optional[str], middle: Optional[str], last: Optional[str]) -> str:
    parts = [p.strip() for p in [first or "", middle or "", last or ""] if (p and p.strip())]
    return ", ".join(parts) if parts else ""

def _fmt_dt_simple(v) -> Optional[str]:
    if v is None: return None
    try:
        if isinstance(v, str):
            try:
                v2 = v.replace("Z", "+00:00")
                dt = datetime.fromisoformat(v2)
                return dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                s = v.replace("T", " ")
                if "." in s: s = s.split(".", 1)[0]
                if "+" in s: s = s.split("+", 1)[0]
                if "-" in s and s.count(":") >= 2:
                    parts = s.split(":")
                    s = ":".join(parts[:2])
                return s.strip()
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d %H:%M")
        return str(v)
    except Exception:
        return str(v)

def _latest_registration_for(email: str) -> Optional[Dict[str, Any]]:
    return _fetch_one("""
        SELECT *
        FROM public.registrations
        WHERE lower(user_email) = lower(%s)
        ORDER BY created_at DESC
        LIMIT 1;
    """, (email,))

# ---------- Impact Survey helpers (stored in enrollments.progress JSONB) ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _ensure_enrollment(user_id: int, course_id: int):
    row = _fetch_one(
        "SELECT 1 FROM enrollments WHERE user_id = %s AND course_id = %s;",
        (user_id, course_id),
    )
    if not row:
        _execute("""
            INSERT INTO enrollments (user_id, course_id, status, enrolled_at, progress)
            VALUES (%s, %s, 'active', now(), '{}'::jsonb);
        """, (user_id, course_id))

def _get_progress(user_id: int, course_id: int) -> Dict[str, Any]:
    row = _fetch_one(
        "SELECT progress FROM enrollments WHERE user_id = %s AND course_id = %s;",
        (user_id, course_id),
    )
    raw = (row or {}).get("progress")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}

def _save_progress(user_id: int, course_id: int, progress_obj: Dict[str, Any]):
    _execute("""
        UPDATE enrollments
           SET progress = %s::jsonb
         WHERE user_id = %s AND course_id = %s;
    """, (json.dumps(progress_obj), user_id, course_id))

def _modules_from_structure(structure: Dict[str, Any]) -> List[Tuple[int, str]]:
    secs = structure.get("sections") or []
    out: List[Tuple[int, str]] = []
    for i, s in enumerate(sorted(secs, key=lambda x: (x.get("order") or 0, x.get("title") or ""))):
        order = int(s.get("order") or (i + 1))
        title = s.get("title") or f"Module {order}"
        out.append((order, title))
    return out

def _pick_impact_course(email: str, user_id: Optional[int], listed_courses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    reg = _latest_registration_for(email)
    if reg and reg.get("course_id"):
        row = _fetch_one("SELECT id, title, structure FROM public.courses WHERE id = %s;", (reg["course_id"],))
        if row:
            row["structure"] = _structure_from_row(row)
            return row
    if listed_courses:
        cid = listed_courses[0]["course_id"]
        row = _fetch_one("SELECT id, title, structure FROM public.courses WHERE id = %s;", (cid,))
        if row:
            row["structure"] = _structure_from_row(row)
            return row
    row = _fetch_one("""
        SELECT id, title, structure
        FROM public.courses
        ORDER BY published_at DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT 1;
    """, ())
    if row:
        row["structure"] = _structure_from_row(row)
        return row
    return None

# ========================== Blueprint factory (self-contained) =================
def create_profile_blueprint(name: str = "profile") -> Blueprint:
    bp = Blueprint(name, __name__)

    @bp.context_processor
    def inject_profile_utils():
        return {"bp": _bp, "base_path": BASE_PATH}

    # --------------------------- view functions -------------------------------
    def profile_view():
        email = _current_user_email()
        if not email:
            full = request.full_path if request.query_string else request.path
            next_url = _sanitize_next(full)
            return redirect(f"{_bp('/login')}?next={quote(next_url, safe='/:?&=')}")

        saved = (request.args.get("saved") == "1")
        error = request.args.get("error")

        user = _fetch_one("""
            SELECT id, email::text AS email, full_name, role::text AS role, created_at
            FROM public.users
            WHERE lower(email::text) = lower(%s)
            LIMIT 1;
        """, (email,))
        user_id = user["id"] if user else None

        primary_reg = _latest_registration_for(email)

        display_name = None
        if primary_reg:
            display_name = _name_commas(
                primary_reg.get("first_name"),
                primary_reg.get("middle_name"),
                primary_reg.get("last_name"),
            )
        if not display_name:
            display_name = (user or {}).get("full_name") or email

        # gather course ids
        reg_course_rows = _fetch_all("""
            SELECT DISTINCT course_id
            FROM public.registrations
            WHERE lower(user_email) = lower(%s) AND course_id IS NOT NULL;
        """, (email,))
        reg_course_ids = {r["course_id"] for r in reg_course_rows if r.get("course_id") is not None}

        act_course_ids = set()
        if user_id:
            act_rows = _fetch_all("""
                SELECT DISTINCT course_id
                FROM public.activity_log
                WHERE user_id = %s AND course_id IS NOT NULL;
            """, (user_id,))
            act_course_ids = {r["course_id"] for r in act_rows if r.get("course_id") is not None}

        all_course_ids = reg_course_ids | act_course_ids

        # courses + stats
        courses: List[Dict[str, Any]] = []
        total_unlocked_sum = 0
        total_lessons_sum = 0
        points_total = 0.0
        passes_total = 0
        last_active_at_global_str = None

        # overall exam aggregates
        exam_total_weeks_sum = 0
        exam_graded_weeks_sum = 0
        exam_passed_weeks_sum = 0
        exam_score_sum = 0.0
        exam_score_cnt = 0

        for cid in sorted(all_course_ids):
            course = _fetch_one("""
                SELECT id, title, structure
                FROM public.courses
                WHERE id = %s;
            """, (cid,))
            if not course:
                continue

            st = _structure_from_row(course)
            total_lessons = _num_lessons(st)

            # ---------- existing progress over lessons ----------
            if user_id:
                prog = _fetch_one("""
                    SELECT
                      COUNT(DISTINCT lesson_uid) AS lessons_seen,
                      MAX(created_at)           AS last_active_at,
                      COALESCE(SUM(score_points), 0) AS points,
                      COUNT(*) FILTER (WHERE passed IS TRUE) AS passes
                    FROM public.activity_log
                    WHERE user_id = %s AND course_id = %s;
                """, (user_id, cid)) or {}
            else:
                prog = {}

            lessons_seen = int(prog.get("lessons_seen") or 0)
            last_active_at_str = _fmt_dt_simple(prog.get("last_active_at"))
            points_raw = prog.get("points") or 0
            passes = int(prog.get("passes") or 0)

            try:
                points_total += float(points_raw)
            except Exception:
                points_total += _as_int(points_raw)

            last_lesson_uid = None
            if user_id:
                last_lesson_row = _fetch_one("""
                    SELECT lesson_uid
                    FROM public.activity_log
                    WHERE user_id = %s AND course_id = %s AND lesson_uid IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 1;
                """, (user_id, cid))
                last_lesson_uid = (last_lesson_row or {}).get("lesson_uid")

            if total_lessons:
                total_unlocked_sum += min(lessons_seen, total_lessons)
                total_lessons_sum += total_lessons
            passes_total += passes

            if last_active_at_str and (last_active_at_global_str is None or last_active_at_str > last_active_at_global_str):
                last_active_at_global_str = last_active_at_str

            progress_pct = int(round((min(lessons_seen, total_lessons) / total_lessons) * 100)) if total_lessons else 0
            duration_str = _format_duration(_total_course_duration(st))
            thumb = (st or {}).get("thumbnail_url", "")

            # ---------- NEW: exam progress per course ----------
            secs = sorted((st.get("sections") or []), key=lambda s: (s.get("order") or 0, s.get("title") or ""))
            enabled_weeks: List[int] = []
            for i, s in enumerate(secs, start=1):
                exam_cfg = (s.get("exam") or {})
                if bool(exam_cfg.get("enabled", True)):
                    enabled_weeks.append(i)
            exam_weeks_total = len(enabled_weeks)

            exam_weeks_graded = 0
            exam_passed_weeks = 0
            exam_scores_this_course: List[float] = []
            exam_active_week = None
            exam_next_week = None
            exam_attempts_used_current = 0

            if user_id and exam_weeks_total > 0:
                exam_rows = _fetch_all("""
                    SELECT created_at, score_points, passed, payload
                    FROM public.activity_log
                    WHERE user_id = %s
                      AND course_id = %s
                      AND (payload->>'kind') = 'exam';
                """, (user_id, cid))

                by_week: Dict[int, List[Dict[str, Any]]] = {}
                for r in exam_rows:
                    p = r.get("payload") or {}
                    try:
                        w = int(p.get("week_index") or 0)
                    except Exception:
                        w = 0
                    if w >= 1:
                        by_week.setdefault(w, []).append(r)

                for w in enabled_weeks:
                    rows_w = by_week.get(w, []) or []
                    graded_rows = [rw for rw in rows_w if (rw.get("payload") or {}).get("event") == "graded"]
                    graded_rows.sort(key=lambda x: x.get("created_at") or datetime.min)
                    latest_graded = graded_rows[-1] if graded_rows else None

                    if latest_graded:
                        exam_weeks_graded += 1
                        p = latest_graded.get("payload") or {}
                        sp = p.get("score_percent")
                        if sp is None:
                            sp = latest_graded.get("score_points")
                        try:
                            spf = float(sp)
                            exam_scores_this_course.append(spf)
                        except Exception:
                            pass
                        if bool(p.get("passed")) or bool(latest_graded.get("passed")):
                            exam_passed_weeks += 1
                        continue

                    graded_uids = { (rw.get("payload") or {}).get("attempt_uid") for rw in graded_rows }
                    active_started = None
                    for rw in rows_w:
                        pr = rw.get("payload") or {}
                        if pr.get("event") == "started" and pr.get("attempt_uid") not in graded_uids:
                            active_started = rw
                            break

                    if active_started and exam_active_week is None:
                        exam_active_week = w
                        sub_uids = set()
                        for rw in rows_w:
                            pr = rw.get("payload") or {}
                            if pr.get("event") == "submitted" and pr.get("attempt_uid"):
                                sub_uids.add(pr["attempt_uid"])
                        exam_attempts_used_current = len(sub_uids)

                    if (exam_next_week is None) and (latest_graded is None):
                        exam_next_week = w

            exam_progress_pct = int(round(100.0 * exam_weeks_graded / exam_weeks_total)) if exam_weeks_total else 0
            exam_avg_score = round(sum(exam_scores_this_course) / len(exam_scores_this_course), 1) if exam_scores_this_course else None

            exam_total_weeks_sum += exam_weeks_total
            exam_graded_weeks_sum += exam_weeks_graded
            exam_passed_weeks_sum += exam_passed_weeks
            if exam_scores_this_course:
                exam_score_sum += sum(exam_scores_this_course)
                exam_score_cnt += len(exam_scores_this_course)

            courses.append({
                "course_id": course["id"],
                "course_title": course.get("title") or f"Course {course['id']}",
                "thumbnail_url": thumb,
                "total_lessons": total_lessons,
                "lessons_seen": min(lessons_seen, total_lessons),
                "progress_pct": progress_pct,
                "last_lesson_uid": last_lesson_uid,
                "resume_uid": last_lesson_uid,
                "last_active_at_str": last_active_at_str,
                "points": points_raw,
                "passes": passes,
                "duration_total": duration_str,

                # NEW: exam fields
                "exam_weeks_total": exam_weeks_total,
                "exam_weeks_graded": exam_weeks_graded,
                "exam_passed_weeks": exam_passed_weeks,
                "exam_progress_pct": exam_progress_pct,
                "exam_avg_score": exam_avg_score,
                "exam_active_week": exam_active_week,
                "exam_next_week": exam_next_week,
                "exam_attempts_used": exam_attempts_used_current,
                "exam_attempt_cap": EXAM_MAX_SUBMISSIONS,
            })

        courses.sort(key=lambda c: (c["last_active_at_str"] or "", c["course_title"] or ""), reverse=True)

        stats = {
            "courses_count": len(courses),
            "total_unlocked": total_unlocked_sum,
            "total_lessons": total_lessons_sum,
            "progress_overall_pct": int(round((total_unlocked_sum / total_lessons_sum) * 100)) if total_lessons_sum else 0,
            "points_total": points_total,
            "passes_total": passes_total,
            "last_active_at_str": last_active_at_global_str,

            # NEW: overall exam stats
            "exam_total_weeks": exam_total_weeks_sum,
            "exam_graded_weeks": exam_graded_weeks_sum,
            "exam_overall_progress_pct": int(round(100.0 * exam_graded_weeks_sum / exam_total_weeks_sum)) if exam_total_weeks_sum else 0,
            "exam_avg_score": (round(exam_score_sum / exam_score_cnt, 1) if exam_score_cnt else None),
        }

        # Recent activity
        activities = []
        if user_id:
            activities = _fetch_all("""
                SELECT al.id, al.course_id, al.lesson_uid, al.a_type, al.created_at, al.score_points, al.passed, al.payload,
                       c.title AS course_title
                FROM public.activity_log al
                LEFT JOIN public.courses c ON c.id = al.course_id
                WHERE al.user_id = %s
                ORDER BY al.created_at DESC
                LIMIT 10;
            """, (user_id,))
            for a in activities:
                a["created_at_str"] = _fmt_dt_simple(a.get("created_at"))

        # Billing snapshot
        primary_reg = _latest_registration_for(email)
        billing = None
        if primary_reg:
            inv = _fetch_one("""
                SELECT id, invoice_no, issue_date, due_date, currency, gross_total, status
                FROM public.invoices
                WHERE customer_registration_id = %s
                ORDER BY issue_date DESC NULLS LAST, id DESC
                LIMIT 1;
            """, (primary_reg["id"],))
            if inv:
                paid_row = _fetch_one("""
                    SELECT COALESCE(SUM(amount), 0) AS paid
                    FROM public.payments
                    WHERE invoice_id = %s;
                """, (inv["id"],))
                gross = inv.get("gross_total") or 0
                paid = (paid_row or {}).get("paid") or 0
                try:
                    outstanding = float(gross) - float(paid)
                except Exception:
                    outstanding = _as_int(gross) - _as_int(paid)
                billing = {
                    "invoice_no": inv.get("invoice_no"),
                    "status": inv.get("status"),
                    "currency": inv.get("currency"),
                    "gross_total": gross,
                    "paid_total": paid,
                    "outstanding": outstanding,
                    "issue_date": inv.get("issue_date"),
                    "due_date": inv.get("due_date"),
                }

        ai_form = {
            "ai_current_involvement": (primary_reg or {}).get("ai_current_involvement") or "",
            "ai_goals_wish_to_achieve": (primary_reg or {}).get("ai_goals_wish_to_achieve") or "",
            "ai_datasets_available": (primary_reg or {}).get("ai_datasets_available") or "",
        }

        # -------- Impact Survey context --------
        impact_course = None
        impact_modules: List[Tuple[int, str]] = []
        impact_modules_map: Dict[int, str] = {}
        impact_latest = None
        impact_history: List[Dict[str, Any]] = []
        if user_id:
            picked = _pick_impact_course(email, user_id, courses)
            if picked:
                impact_course = {"id": picked["id"], "title": picked.get("title")}
                impact_modules = _modules_from_structure(picked["structure"])
                impact_modules_map = {order: title for order, title in impact_modules}
                _ensure_enrollment(user_id, picked["id"])
                prog = _get_progress(user_id, picked["id"])
                survey = prog.get("impact_survey") or {}
                impact_latest = survey.get("latest")
                impact_history = (survey.get("responses") or [])[::-1]

        return render_template(
            "profile.html",
            display_name=display_name,
            email=email,
            user=user,
            primary_reg=primary_reg,
            courses=courses,
            stats=stats,
            activities=activities,
            billing=billing,
            saved=saved,
            error=error,
            ai_form=ai_form,
            # impact survey context
            impact_course=impact_course,
            impact_modules=impact_modules,
            impact_modules_map=impact_modules_map,
            impact_latest=impact_latest,
            impact_history=impact_history,
        )

    def update_ai_fields():
        email = _current_user_email()
        if not email:
            full = request.full_path if request.query_string else request.path
            next_url = _sanitize_next(full)
            return redirect(f"{_bp('/login')}?next={quote(next_url, safe='/:?&=')}")

        ai_current_involvement   = (request.form.get("ai_current_involvement") or "").strip()
        ai_goals_wish_to_achieve = (request.form.get("ai_goals_wish_to_achieve") or "").strip()
        ai_datasets_available    = (request.form.get("ai_datasets_available") or "").strip()

        MAXLEN = 20000
        ai_current_involvement   = ai_current_involvement[:MAXLEN]
        ai_goals_wish_to_achieve = ai_goals_wish_to_achieve[:MAXLEN]
        ai_datasets_available    = ai_datasets_available[:MAXLEN]

        reg = _latest_registration_for(email)
        try:
            if reg:
                _execute("""
                    UPDATE public.registrations
                       SET ai_current_involvement   = %s,
                           ai_goals_wish_to_achieve = %s,
                           ai_datasets_available    = %s,
                           updated_at               = now()
                     WHERE id = %s;
                """, (ai_current_involvement, ai_goals_wish_to_achieve, ai_datasets_available, reg["id"]))
            else:
                _execute("""
                    INSERT INTO public.registrations
                        (created_at, user_email, ai_current_involvement, ai_goals_wish_to_achieve, ai_datasets_available)
                    VALUES (now(), %s, %s, %s, %s);
                """, (email, ai_current_involvement, ai_goals_wish_to_achieve, ai_datasets_available))

            return redirect(url_for(f"{bp.name}.profile_view", saved=1))
        except Exception as e:
            print("[profile] update_ai_fields failed:", e)
            return redirect(url_for(f"{bp.name}.profile_view", error="update_failed"))

    def submit_impact_survey():
        email = _current_user_email()
        if not email:
            full = request.full_path if request.query_string else request.path
            next_url = _sanitize_next(full)
            return redirect(f"{_bp('/login')}?next={quote(next_url, safe='/:?&=')}")

        user = _fetch_one("""
            SELECT id FROM public.users WHERE lower(email::text)=lower(%s) LIMIT 1;
        """, (email,))
        if not user:
            return redirect(url_for(f"{bp.name}.profile_view", error="no_user"))

        user_id = user["id"]

        try:
            form_course_id = request.form.get("course_id")
            course_id = int(form_course_id) if form_course_id else None
        except Exception:
            course_id = None

        if not course_id:
            picked = _pick_impact_course(email, user_id, [])
            if not picked:
                return redirect(url_for(f"{bp.name}.profile_view", error="no_course"))
            course_id = picked["id"]

        _ensure_enrollment(user_id, course_id)
        prog = _get_progress(user_id, course_id)

        def _as_int_or_none(v):
            try:
                return int(v)
            except Exception:
                return None

        existing = (prog.get("impact_survey") or {}).get("responses") or []
        period = "baseline" if not existing else "update"

        career_stage = (request.form.get("career_stage") or "").strip() or None
        goals        = request.form.getlist("goals") or []
        confidence   = _as_int_or_none(request.form.get("confidence"))
        hours_week   = _as_int_or_none(request.form.get("hours_last_week"))
        work_impact  = (request.form.get("work_impact") or "").strip() or None
        outcomes     = request.form.getlist("outcomes") or []
        best_modules_raw = request.form.getlist("best_modules")
        best_modules: List[int] = []
        for v in best_modules_raw:
            try:
                best_modules.append(int(v))
            except Exception:
                pass
        blockers     = (request.form.get("blockers") or "").strip() or None
        nps          = _as_int_or_none(request.form.get("nps"))
        update_note  = (request.form.get("update_note") or "").strip() or None

        entry = {
            "ts": _now_iso(),
            "period": period,
            "career_stage": career_stage,
            "goals": goals,
            "confidence": confidence,
            "hours_last_week": hours_week,
            "work_impact": work_impact,
            "outcomes": outcomes,
            "best_modules": best_modules,
            "blockers": blockers,
            "nps": nps,
            "update_note": update_note,
        }

        survey = prog.get("impact_survey") or {}
        responses = survey.get("responses") or []
        responses.append(entry)
        survey["responses"] = responses
        survey["latest"] = entry
        survey["last_submitted_at"] = entry["ts"]
        prog["impact_survey"] = survey

        _save_progress(user_id, course_id, prog)
        return redirect(url_for(f"{bp.name}.profile_view", saved=1))

    # --------------------------- route registrations --------------------------
    bp.add_url_rule("/profile", view_func=profile_view, methods=["GET"], endpoint="profile_view")
    bp.add_url_rule("/profile/ai/update", view_func=update_ai_fields, methods=["POST"], endpoint="update_ai_fields")
    bp.add_url_rule("/profile/survey", view_func=submit_impact_survey, methods=["POST"], endpoint="submit_impact_survey")

    if BASE_PATH:
        bp.add_url_rule(f"{BASE_PATH}/profile", view_func=profile_view, methods=["GET"], endpoint="profile_view_alias")
        bp.add_url_rule(f"{BASE_PATH}/profile/ai/update", view_func=update_ai_fields, methods=["POST"], endpoint="update_ai_fields_alias")
        bp.add_url_rule(f"{BASE_PATH}/profile/survey", view_func=submit_impact_survey, methods=["POST"], endpoint="submit_impact_survey_alias")

    return bp
