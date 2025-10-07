import os
import re
import json
import uuid
import urllib.request
import urllib.error
from typing import Any, Dict, List, Tuple, Optional, Set

from flask import (
    Blueprint, render_template, jsonify, abort, request,
    redirect, url_for, g
)

# =========================
# Admin gating / constants
# =========================
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "1").lower() in ("1", "true", "yes")
_ADMIN_EMAILS_RAW = os.getenv("ADMIN_EMAILS", "")
ADMIN_EMAILS = {
    e.strip().lower()
    for part in _ADMIN_EMAILS_RAW.split(";")
    for e in part.split(",")
    if e.strip()
}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

IAP_ROLE = "roles/iap.httpsResourceAccessor"

# Name of THIS App Engine service (for service-level IAP bindings)
SERVICE_NAME = (
    os.getenv("GAE_SERVICE")
    or os.getenv("K_SERVICE")
    or os.getenv("SERVICE")
    or "learningportal"
)

# Pages used by Access UI
PAGES: List[Tuple[str, str]] = [
    ("overview", "Overview"),
    ("profile", "Profile"),
    ("player", "Course Player"),
    ("notes", "Notes"),
    ("assignments", "Assignments"),
    ("analytics", "Analytics"),
    ("admin", "Admin"),
]

# ======================================================
# IAP Admin (REST) helpers — all embedded here
# ======================================================

def _project_id() -> str:
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
    if not pid:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    return pid

def _project_number(timeout_sec: float = 1.5) -> Optional[str]:
    pnum = os.environ.get("PROJECT_NUMBER")
    if pnum:
        return pnum
    try:
        req = urllib.request.Request(
            "http://metadata/computeMetadata/v1/project/numeric-project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            txt = resp.read().decode("utf-8").strip()
            return txt or None
    except Exception:
        return None

def _access_token_from_google_auth() -> Optional[str]:
    try:
        import google.auth  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not creds.valid:
            creds.refresh(Request())
        return creds.token
    except Exception:
        return None

def _access_token_from_metadata(timeout_sec: float = 1.5) -> Optional[str]:
    try:
        req = urllib.request.Request(
            "http://metadata/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("access_token")
    except Exception:
        return None

def _get_access_token() -> str:
    tok = _access_token_from_google_auth()
    if tok:
        return tok
    tok = _access_token_from_metadata()
    if tok:
        return tok
    raise RuntimeError("Could not obtain access token (google-auth and metadata both unavailable)")

def _http_json(url: str, method: str = "POST", body: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    token = _get_access_token()
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        try:
            err_json = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_json = {"error": {"message": e.reason, "status": f"HTTP_{e.code}"}}
        raise RuntimeError(f"IAP API call failed: {err_json}") from None
    except Exception as e:
        raise RuntimeError(f"IAP API call error: {e}") from None

def _iap_resource_names(service: str) -> Tuple[str, str]:
    pid = _project_id()
    pnum = _project_number()
    if not pnum:
        raise RuntimeError("PROJECT_NUMBER unavailable (metadata/API)")
    app_res = f"projects/{pnum}/iap_web/appengine-{pid}"
    svc_res = f"{app_res}/services/{service}"
    return app_res, svc_res

def _iap_get_policy(service: str, level: str = "service") -> Dict[str, Any]:
    app_res, svc_res = _iap_resource_names(service)
    resource = app_res if level == "app" else svc_res
    url = f"https://iap.googleapis.com/v1/{resource}:getIamPolicy"
    return _http_json(url, method="POST", body={})

def _iap_set_policy(service: str, policy: Dict[str, Any], level: str = "service") -> Dict[str, Any]:
    app_res, svc_res = _iap_resource_names(service)
    resource = app_res if level == "app" else svc_res
    url = f"https://iap.googleapis.com/v1/{resource}:setIamPolicy"
    body = {"policy": policy}
    return _http_json(url, method="POST", body=body)

def _policy_add_member(policy: Dict[str, Any], role: str, member: str) -> Tuple[Dict[str, Any], bool]:
    bindings: List[Dict[str, Any]] = policy.get("bindings", []) or []
    for b in bindings:
        if b.get("role") == role:
            members = b.get("members", []) or []
            if member in members:
                return policy, False
            members.append(member)
            b["members"] = members
            policy["bindings"] = bindings
            return policy, True
    bindings.append({"role": role, "members": [member]})
    policy["bindings"] = bindings
    return policy, True

def _policy_remove_member(policy: Dict[str, Any], role: str, member: str) -> Tuple[Dict[str, Any], bool]:
    bindings: List[Dict[str, Any]] = policy.get("bindings", []) or []
    changed = False
    for b in list(bindings):
        if b.get("role") == role:
            members = b.get("members", []) or []
            if member in members:
                members.remove(member)
                changed = True
            if not members:
                bindings.remove(b)
                changed = True
            else:
                b["members"] = members
            break
    policy["bindings"] = bindings
    return policy, changed

def _list_iap_users(service: str, level: str = "service") -> List[str]:
    try:
        pol = _iap_get_policy(service, level=level)
    except Exception:
        return []
    out: List[str] = []
    for b in pol.get("bindings", []) or []:
        if b.get("role") == IAP_ROLE:
            for m in b.get("members", []) or []:
                if m.startswith("user:"):
                    out.append(m.split(":", 1)[1].lower())
    return sorted(set(out))

def _grant_iap(email: str, service: str, level: str = "service") -> Tuple[bool, str]:
    member = f"user:{email}"
    pol = _iap_get_policy(service, level=level)
    pol2, changed = _policy_add_member(pol, IAP_ROLE, member)
    if changed:
        if pol.get("etag"):
            pol2["etag"] = pol.get("etag")
        _iap_set_policy(service, pol2, level=level)
    return changed, ("Added to IAP" if changed else "Already in IAP")

def _revoke_iap(email: str, service: str, level: str = "service") -> Tuple[bool, str]:
    member = f"user:{email}"
    pol = _iap_get_policy(service, level=level)
    pol2, changed = _policy_remove_member(pol, IAP_ROLE, member)
    if changed:
        if pol.get("etag"):
            pol2["etag"] = pol.get("etag")
        _iap_set_policy(service, pol2, level=level)
    return changed, ("Removed from IAP" if changed else "Not present in IAP")


# =========================
# Blueprint factory
# =========================
def create_admin_blueprint(
    url_prefix: str,
    deps: Dict[str, Any],
    name: str = "admin",
) -> Blueprint:
    """
    Admin blueprint, including:
      • Course builder
      • Access control (App users + IAP at SERVICE level)
      • Page-level enforcement helpers (player/admin/profile)
      • NEW: Module-level access control (per user, per course)
    deps:
      - COURSE_TITLE
      - fetch_one(sql, params)
      - execute(sql, params)
      - ensure_structure(json_or_none) -> dict
      - seed_course_if_missing() -> int
    """
    COURSE_TITLE = deps["COURSE_TITLE"]
    fetch_one = deps["fetch_one"]
    execute = deps["execute"]
    ensure_structure = deps["ensure_structure"]
    course_structure = deps.get("course_structure")
    load_course_structure = deps.get("load_course_structure")

    def _structure_from_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if course_structure and row:
            return course_structure(row)
        if load_course_structure and row is not None:
            return load_course_structure(
                row.get("structure"),
                course_title=row.get("title"),
            )
        return ensure_structure((row or {}).get("structure"))
    seed_course_if_missing = deps["seed_course_if_missing"]

    # Mount at /<BASE_PATH>/admin (e.g. /learn/admin) or /admin if url_prefix=""
    mount_prefix = (url_prefix.rstrip("/") + "/admin") if url_prefix else "/admin"
    base_prefix = url_prefix.rstrip("/") if url_prefix else ""  # e.g. "/learn"
    bp = Blueprint(name, __name__, url_prefix=mount_prefix)

    # ---------- Page rules: fetch & check ----------
    def _allowed_set_for(email: str) -> Optional[set]:
        """
        Returns None => ALL pages allowed (empty CSV or no row).
        Else returns a set of allowed slugs.
        """
        if not email:
            return None
        row = fetch_one("SELECT page_access FROM user_page_rules WHERE email = %s;", (email,))
        if not row:
            return None
        csv = (row.get("page_access") or "").strip()
        if not csv:
            return None
        return {s for s in csv.split(",") if s}

    def _user_role(email: str) -> Optional[str]:
        row = fetch_one("SELECT role FROM users WHERE email=%s;", (email,))
        return (row or {}).get("role")

    def page_allowed_for(email: str, slug: str) -> bool:
        allowed = _allowed_set_for(email)
        if allowed is None:
            return True
        return slug in allowed

    def require_page(slug: str):
        email = (getattr(g, "user_email", None) or "").lower().strip()
        if not email:
            return
        if page_allowed_for(email, slug):
            return
        if slug == "admin":
            role = (_user_role(email) or "").lower()
            if role in ("admin", "instructor") or (ADMIN_EMAILS and email in ADMIN_EMAILS):
                return
        abort(403)

    def require_admin():
        require_page("admin")

    # ---------- Module access storage ----------
    def _ensure_user_module_rules_table():
        execute("""
            CREATE TABLE IF NOT EXISTS user_module_rules (
              email         CITEXT NOT NULL,
              course_id     BIGINT NOT NULL,
              allowed_modules TEXT,              -- NULL/'' => ALL, 'NONE' => none, else CSV of section 'order' ints
              updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (email, course_id)
            );
        """, ())

    def _normalize_module_csv(selected_orders: List[str], all_orders: List[int], mode: str) -> str:
        """
        mode: 'all' | 'none' | 'custom'
        Returns storage string for allowed_modules.
        """
        if mode == "all":
            return ""
        if mode == "none":
            return "NONE"
        picked = sorted({int(x) for x in selected_orders if str(x).isdigit()})
        if not picked:
            return "NONE"
        if set(picked) == set(all_orders):
            return ""
        return ",".join(str(n) for n in picked)

    def _get_course_and_modules() -> Tuple[Optional[Dict[str, Any]], List[Tuple[int, str]]]:
        row = fetch_one("""
            SELECT id, title, structure
            FROM courses
            WHERE title = %s
            LIMIT 1;
        """, (COURSE_TITLE,))
        if not row:
            return None, []
        st = _structure_from_row(row)
        secs = st.get("sections") or []
        # modules: list of (order, label)
        modules = []
        for i, s in enumerate(sorted(secs, key=lambda x: (x.get("order") or 0, x.get("title") or ""))):
            order = int(s.get("order") or (i + 1))
            label = s.get("title") or f"Section {order}"
            modules.append((order, label))
        return row, modules

    def _fetch_module_rules_map(course_id: int) -> Dict[str, Any]:
        _ensure_user_module_rules_table()
        rows = fetch_one("""
            SELECT COALESCE(json_agg(x), '[]'::json) AS rows
            FROM (
              SELECT email, allowed_modules
              FROM user_module_rules
              WHERE course_id = %s
            ) x;
        """, (course_id,))
        arr = rows.get("rows") if rows else []
        if isinstance(arr, str):
            try:
                arr = json.loads(arr)
            except Exception:
                arr = []
        m: Dict[str, Any] = {}
        for r in (arr or []):
            em = r.get("email")
            raw = (r.get("allowed_modules") or "").strip()
            if raw == "" or raw is None:
                m[em] = None           # None => ALL
            elif raw == "NONE":
                m[em] = set()          # empty set => NONE
            else:
                picks: Set[int] = set()
                for tok in str(raw).split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        picks.add(int(tok))
                m[em] = picks
        return m

    def _upsert_module_rules(email: str, course_id: int, allowed_modules: str):
        _ensure_user_module_rules_table()
        execute("""
            INSERT INTO user_module_rules (email, course_id, allowed_modules)
            VALUES (%s, %s, %s)
            ON CONFLICT (email, course_id)
            DO UPDATE SET allowed_modules = EXCLUDED.allowed_modules, updated_at = now();
        """, (email, course_id, allowed_modules))

    # ---------- Enforcing module rules on /learn/... ----------
    def _parse_learn_path(p: str) -> Optional[Tuple[int, Optional[str]]]:
        prefix = f"{base_prefix}/learn/" if base_prefix else "/learn/"
        if not p.startswith(prefix):
            return None
        rest = p[len(prefix):]
        parts = rest.strip("/").split("/")
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0]), None
        if len(parts) >= 2 and parts[0].isdigit():
            return int(parts[0]), "/".join(parts[1:])
        return None

    def _lesson_section_order(structure: Dict[str, Any], lesson_uid: str) -> Optional[int]:
        secs = structure.get("sections") or []
        # sort by (order, title) to match UI ordering
        secs_sorted = sorted(secs, key=lambda s: (s.get("order") or 0, s.get("title") or ""))
        for i, s in enumerate(secs_sorted):
            order = int(s.get("order") or (i + 1))
            lessons = s.get("lessons") or []
            for l in lessons:
                if str(l.get("lesson_uid")) == str(lesson_uid):
                    return order
        return None

    def _first_allowed_lesson_uid(structure: Dict[str, Any], allowed_orders: Set[int]) -> Optional[str]:
        secs = structure.get("sections") or []
        secs_sorted = sorted(secs, key=lambda s: (s.get("order") or 0, s.get("title") or ""))
        for i, s in enumerate(secs_sorted):
            order = int(s.get("order") or (i + 1))
            if order in allowed_orders:
                lessons = s.get("lessons") or []
                if lessons:
                    uid = lessons[0].get("lesson_uid")
                    if uid:
                        return str(uid)
        return None

    # ---------- Inject helper for templates ----------
    def _tpl_page_allowed(slug: str) -> bool:
        email = (getattr(g, "user_email", None) or "").lower().strip()
        if not email:
            return False
        return page_allowed_for(email, slug)

    # ---------- Global guard: page + module visibility ----------
    @bp.record_once
    def _install_global_guards(state):
        app = state.app

        @app.before_request
        def _page_gate():
            # Skip static & favicon
            p = request.path or ""
            if p.startswith("/static/") or p.endswith("/favicon.ico"):
                return

            # Skip CSS/JS under base path
            if base_prefix and p.startswith(f"{base_prefix}/static/"):
                return

            # Skip auth endpoints
            if base_prefix and (p.startswith(f"{base_prefix}/login")
                                or p.startswith(f"{base_prefix}/auth")
                                or p.startswith(f"{base_prefix}/whoami")):
                return

            # Profile access
            if p == "/profile" or (base_prefix and p == f"{base_prefix}/profile"):
                require_page("profile")
                return

            # Admin area
            if base_prefix and p.startswith(f"{base_prefix}/admin"):
                require_admin()
                return

            # For any page under base_prefix => requires 'player'
            if base_prefix and p.startswith(f"{base_prefix}/"):
                require_page("player")

                # Module-level enforcement for /learn/...
                parsed = _parse_learn_path(p)
                if not parsed:
                    return
                email = (getattr(g, "user_email", None) or "").lower().strip()
                if not email:
                    return  # upstream auth will handle
                course_id, lesson_uid = parsed

                # Load course & module rules
                course_row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
                if not course_row:
                    return
                st = _structure_from_row(course_row)
                _, modules = _get_course_and_modules()  # for orders list
                all_orders = [int(o) for o, _ in modules]

                rules_row = fetch_one("""
                    SELECT allowed_modules FROM user_module_rules
                    WHERE email = %s AND course_id = %s;
                """, (email, course_id))
                raw = (rules_row or {}).get("allowed_modules")
                if raw is None or str(raw).strip() == "":
                    return  # ALL modules allowed

                if str(raw).strip() == "NONE":
                    # no access to modules at all
                    abort(403)

                # Parse allowed orders
                allowed: Set[int] = set()
                for tok in str(raw).split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        allowed.add(int(tok))

                # If no lesson uid, send to first allowed lesson
                if lesson_uid is None:
                    uid = _first_allowed_lesson_uid(st, allowed)
                    if uid:
                        return redirect((base_prefix or "") + f"/learn/{course_id}/{uid}")
                    abort(403)

                # With lesson uid: ensure its section order is allowed
                order = _lesson_section_order(st, lesson_uid)
                if order is None or order not in allowed:
                    uid = _first_allowed_lesson_uid(st, allowed)
                    if uid:
                        return redirect((base_prefix or "") + f"/learn/{course_id}/{uid}")
                    abort(403)

        @app.context_processor
        def _inject_page_helpers():
            return {"page_allowed": _tpl_page_allowed}

        # Optional route name compat (if needed)
        try:
            app.add_url_rule(f"{mount_prefix}/access", endpoint="admin_access", view_func=admin_access, methods=["GET"])
        except Exception:
            pass
        try:
            app.add_url_rule(f"{mount_prefix}/users", endpoint="admin_users", view_func=users, methods=["GET", "POST"])
        except Exception:
            pass

    # ---------- Diagnostics ----------
    @bp.get("/whoami")
    def admin_whoami():
        return jsonify({
            "auth_required": AUTH_REQUIRED,
            "current_user_email": getattr(g, "user_email", None),
            "admin_emails_enforced": bool(ADMIN_EMAILS),
            "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT"),
            "service_name": SERVICE_NAME,
        })

    # ---------- Admin Home / Builder ----------
    @bp.get("/")
    def admin_home():
        require_admin()
        course = fetch_one("""
            SELECT id, title, is_published, published_at, created_at
            FROM courses WHERE title = %s LIMIT 1;
        """, (COURSE_TITLE,))
        return render_template(
            "admin.html",
            course=course,
            msg=request.args.get("msg"),
            err=request.args.get("err")
        )

    @bp.get("/seed")
    def admin_seed():
        require_admin()
        try:
            cid = seed_course_if_missing()
            return redirect(url_for("admin.admin_builder", course_id=cid, msg="Course seeded"))
        except Exception as e:
            return redirect(url_for("admin.admin_home", err=f"Seed failed: {e}"))

    @bp.get("/course/<int:course_id>/edit")
    def admin_edit_course(course_id: int):
        require_admin()
        row = fetch_one("""
            SELECT id, title, is_published, published_at, created_at, structure
            FROM courses WHERE id = %s;
        """, (course_id,))
        if not row:
            abort(404)
        structure = _structure_from_row(row)
        return render_template(
            "admin_edit_course.html",
            course=row,
            structure_text=json.dumps(structure, indent=2),
            msg=request.args.get("msg"),
            err=request.args.get("err"),
        )

    @bp.post("/course/<int:course_id>/edit")
    def admin_edit_course_post(course_id: int):
        require_admin()
        try:
            title = (request.form.get("title") or "").strip()
            is_published = bool(request.form.get("is_published"))
            structure_text = request.form.get("structure_json") or "{}"
            structure = json.loads(structure_text)
            execute("""
                UPDATE courses
                SET title = %s,
                    is_published = %s,
                    published_at = CASE WHEN %s THEN COALESCE(published_at, now()) ELSE NULL END,
                    structure = %s
                WHERE id = %s;
            """, (title, is_published, is_published, json.dumps(structure), course_id))
            return redirect(url_for("admin.admin_edit_course", course_id=course_id, msg="Saved"))
        except Exception as e:
            return redirect(url_for("admin.admin_edit_course", course_id=course_id, err=f"Save failed: {e}"))

    @bp.get("/course/<int:course_id>/builder")
    def admin_builder(course_id: int):
        require_admin()
        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        if not row:
            abort(404)
        st = _structure_from_row(row)
        return render_template(
            "admin_builder.html",
            course=row,
            sections=st.get("sections") or [],
            msg=request.args.get("msg"),
            err=request.args.get("err")
        )

    # ---------- Structure helpers ----------
    def _save_structure(course_id: int, structure: Dict[str, Any]):
        execute("UPDATE courses SET structure = %s WHERE id = %s;", (json.dumps(structure), course_id))

    @bp.post("/course/<int:course_id>/add-week")
    def admin_add_week(course_id: int):
        require_admin()
        title = (request.form.get("title") or "").strip() or f"Week {uuid.uuid4().hex[:4]}"
        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        st = _structure_from_row(row)
        secs = st.get("sections") or []
        order = (max([s.get("order", 0) for s in secs]) + 1) if secs else 1
        secs.append({"title": title, "order": order, "lessons": []})
        st["sections"] = secs
        _save_structure(course_id, st)
        return redirect(url_for("admin.admin_builder", course_id=course_id, msg="Week added"))

    @bp.post("/course/<int:course_id>/add-lesson")
    def admin_add_lesson(course_id: int):
        require_admin()
        week_index = int(request.form.get("week_index", "0"))
        kind = (request.form.get("kind") or "article").strip()
        title = (request.form.get("title") or kind.title()).strip()
        uid = str(uuid.uuid4())

        if kind == "article":
            content = {"body_md": (request.form.get("body_md") or "<h2>Article</h2>").strip()}
        elif kind == "video":
            url = (request.form.get("video_url") or "").strip()
            duration = request.form.get("duration_sec")
            content = {
                "provider": "url" if url else "upload",
                "url": url,
                "duration_sec": int(duration) if (duration or "").isdigit() else None,
                "notes_md": (request.form.get("notes_md") or "").strip()
            }
        elif kind == "quiz":
            try:
                raw = request.form.get("quiz_json") or ""
                parsed = json.loads(raw) if raw.strip() else {"questions": []}
                content = parsed if isinstance(parsed, dict) else {"questions": parsed if isinstance(parsed, list) else []}
            except Exception:
                content = {"questions": []}
        elif kind == "assignment":
            content = {
                "instructions_md": (request.form.get("instructions_md") or "").strip(),
                "resource_url": (request.form.get("resource_url") or "").strip()
            }
        else:
            content = {}

        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        st = _structure_from_row(row)
        secs = st.get("sections") or []
        if week_index < 0 or week_index >= len(secs):
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Invalid week index"))
        lessons = secs[week_index].get("lessons") or []
        order = (max([l.get("order", 0) for l in lessons]) + 1) if lessons else 1
        lessons.append({"lesson_uid": uid, "title": title, "kind": kind, "order": order, "content": content})
        secs[week_index]["lessons"] = lessons
        st["sections"] = secs
        _save_structure(course_id, st)
        return redirect(url_for("admin.admin_builder", course_id=course_id, msg=f"Added {kind}"))

    @bp.post("/course/<int:course_id>/remove-lesson")
    def admin_remove_lesson(course_id: int):
        require_admin()
        week_index = int(request.form.get("week_index", "0"))
        lesson_uid = request.form.get("lesson_uid") or ""
        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        st = _structure_from_row(row)
        secs = st.get("sections") or []
        if week_index < 0 or week_index >= len(secs):
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Invalid week index"))
        lessons = secs[week_index].get("lessons") or []
        lessons = [l for l in lessons if str(l.get("lesson_uid")) != str(lesson_uid)]
        secs[week_index]["lessons"] = lessons
        st["sections"] = secs
        _save_structure(course_id, st)
        return redirect(url_for("admin.admin_builder", course_id=course_id, msg="Lesson removed"))

    @bp.post("/course/<int:course_id>/update-week")
    def admin_update_week(course_id: int):
        require_admin()
        try:
            week_index = int(request.form.get("week_index", "0"))
        except Exception:
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Invalid week index"))
        title = (request.form.get("title") or "").strip()

        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        if not row:
            abort(404)
        st = _structure_from_row(row)
        secs = st.get("sections") or []
        if week_index < 0 or week_index >= len(secs):
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Invalid week index"))

        secs[week_index]["title"] = title or f"Week {week_index+1}"
        st["sections"] = secs
        _save_structure(course_id, st)
        return redirect(url_for("admin.admin_builder", course_id=course_id, msg="Week updated"))

    @bp.post("/course/<int:course_id>/update-lesson")
    def admin_update_lesson(course_id: int):
        require_admin()
        try:
            week_index = int(request.form.get("week_index", "0"))
        except Exception:
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Invalid week index"))
        lesson_uid = (request.form.get("lesson_uid") or "").strip()
        new_title = (request.form.get("title") or "").strip()
        new_kind = (request.form.get("kind") or "article").strip()

        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        if not row:
            abort(404)
        st = _structure_from_row(row)
        secs = st.get("sections") or []
        if week_index < 0 or week_index >= len(secs):
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Invalid week index"))

        lessons = secs[week_index].get("lessons") or []
        idx = None
        for i, l in enumerate(lessons):
            if str(l.get("lesson_uid")) == str(lesson_uid):
                idx = i
                break
        if idx is None:
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Lesson not found"))

        old = lessons[idx]
        order = old.get("order") or (idx + 1)

        if new_kind == "article":
            content = {"body_md": (request.form.get("body_md") or "").strip()}
        elif new_kind == "video":
            url = (request.form.get("video_url") or "").strip()
            duration = request.form.get("duration_sec")
            try:
                duration_int = int(duration) if duration and str(duration).strip().isdigit() else None
            except Exception:
                duration_int = None
            content = {
                "provider": "url" if url else "upload",
                "url": url,
                "duration_sec": duration_int,
                "notes_md": (request.form.get("notes_md") or "").strip()
            }
        elif new_kind == "quiz":
            raw = request.form.get("quiz_json") or ""
            try:
                parsed = json.loads(raw) if raw.strip() else {"questions": []}
                content = parsed if isinstance(parsed, dict) else {"questions": parsed if isinstance(parsed, list) else []}
            except Exception:
                content = {"questions": []}
        elif new_kind == "assignment":
            content = {
                "instructions_md": (request.form.get("instructions_md") or "").strip(),
                "resource_url": (request.form.get("resource_url") or "").strip()
            }
        else:
            new_kind = old.get("kind") or "article"
            content = old.get("content") or {}

        lessons[idx] = {
            "lesson_uid": str(lesson_uid),
            "title": new_title or old.get("title") or new_kind.title(),
            "kind": new_kind,
            "order": order,
            "content": content
        }
        secs[week_index]["lessons"] = lessons
        st["sections"] = secs
        _save_structure(course_id, st)
        return redirect(url_for("admin.admin_builder", course_id=course_id, msg="Lesson updated"))

    # ==============================
    # Access (App + IAP + Modules)
    # ==============================
    def _ensure_page_rules_table():
        execute("""
            CREATE TABLE IF NOT EXISTS user_page_rules (
              email       CITEXT PRIMARY KEY,
              page_access TEXT,
              updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """, ())

    def _normalize_page_csv(selected_slugs: List[str]) -> str:
        all_slugs = {s for s, _ in PAGES}
        picked = set(selected_slugs or [])
        if not picked or picked == all_slugs:
            return ""  # '' means "all pages"
        return ",".join(sorted(picked & all_slugs))

    def _render_admin_access_page(message: Optional[Dict[str, str]] = None, helper_cmd: Optional[str] = None):
        require_admin()
        _ensure_page_rules_table()
        _ensure_user_module_rules_table()

        # Current course + modules
        course, modules = _get_course_and_modules()
        course_id = (course or {}).get("id")

        # Users list
        row = fetch_one("""
            SELECT COALESCE(json_agg(x), '[]'::json) AS rows
            FROM (
              SELECT email, role, created_at
              FROM users
              ORDER BY created_at DESC
              LIMIT 500
            ) x;
        """, ())
        users_json = row.get("rows") if row else []
        users_list = json.loads(users_json) if isinstance(users_json, str) else (users_json or [])

        # Page rules map
        row2 = fetch_one("""
            SELECT COALESCE(jsonb_object_agg(email, page_access), '{}'::jsonb) AS map
            FROM user_page_rules;
        """, ())
        raw_map = row2.get("map") if row2 else {}
        if isinstance(raw_map, str):
            try:
                raw_map = json.loads(raw_map)
            except Exception:
                raw_map = {}
        page_rules: Dict[str, Any] = {}
        for em, csv in (raw_map or {}).items():
            page_rules[em] = None if not csv else {s for s in str(csv).split(",") if s}

        # Module rules (for this course)
        module_rules = _fetch_module_rules_map(course_id) if course_id else {}

        # IAP listing (service + app)
        iap_list: List[str] = []
        iap_error: Optional[str] = None
        try:
            iap_list_svc = set(_list_iap_users(SERVICE_NAME, level="service"))
            iap_list_app = set(_list_iap_users(SERVICE_NAME, level="app"))
            iap_list = sorted(iap_list_svc | iap_list_app)
        except Exception as e:
            iap_error = str(e)

        # Query param message
        if not message:
            t = request.args.get("t"); m = request.args.get("m")
            if t and m:
                message = {"type": t, "text": m}

        return render_template(
            "admin_access.html",
            message=message,
            helper_cmd=helper_cmd,
            pages=PAGES,
            course=course,
            modules=modules,                # [(order, label), ...]
            users=users_list,
            page_rules=page_rules,          # email -> None(all) | set(slugs)
            module_rules=module_rules,      # email -> None(all) | set(orders) | empty set (NONE)
            iap_list=iap_list,
            iap_error=iap_error,
        )

    @bp.get("/access")
    def admin_access():
        return _render_admin_access_page()

    # Keep endpoint name exactly "users" so url_for('admin.users') works
    @bp.route("/users", endpoint="users", methods=["GET", "POST"])
    def users():
        if request.method == "GET":
            return _render_admin_access_page()

        section = (request.form.get("section") or "").strip()

        # Combined App + IAP (+ Modules)
        if section == "combined":
            action = (request.form.get("action") or "").strip()
            email = (request.form.get("email") or "").strip().lower()
            if not EMAIL_RE.match(email):
                return _render_admin_access_page({"type": "error", "text": "Invalid email."})

            course, modules = _get_course_and_modules()
            course_id = (course or {}).get("id")
            all_orders = [int(o) for o, _ in (modules or [])]

            if action == "add":
                requested_role = (request.form.get("role") or "learner").strip().lower()
                allowed_roles = {"learner", "student", "instructor", "admin", "owner"}
                role = requested_role if requested_role in allowed_roles else "admin"

                selected_pages = request.form.getlist("page_allowed")
                page_csv = _normalize_page_csv(selected_pages)

                # --- Modules ---
                module_mode = (request.form.get("module_mode") or "all").strip()  # all|none|custom
                module_selected = request.form.getlist("module_allowed")           # orders
                allowed_modules = _normalize_module_csv(module_selected, all_orders, module_mode)

                # Upsert user
                full_name = (request.form.get("name") or "")
                full_name = full_name.strip() or email.split("@", 1)[0].replace(".", " ").title()
                try:
                    execute("""
                        INSERT INTO users (email, full_name, role)
                        VALUES (%s, %s, %s::role)
                        ON CONFLICT (email) DO UPDATE SET role = EXCLUDED.role;
                    """, (email, full_name, role))
                except Exception as e:
                    return _render_admin_access_page({"type": "error", "text": f"DB upsert failed: {e}"})

                # Page rules
                try:
                    _ensure_page_rules_table()
                    execute("""
                        INSERT INTO user_page_rules (email, page_access)
                        VALUES (%s, %s)
                        ON CONFLICT (email) DO UPDATE SET page_access = EXCLUDED.page_access, updated_at = now();
                    """, (email, page_csv))
                except Exception as e:
                    return _render_admin_access_page({"type": "error", "text": f"Saving page rules failed: {e}"})

                # Module rules
                try:
                    if course_id:
                        _upsert_module_rules(email, course_id, allowed_modules)
                except Exception as e:
                    return _render_admin_access_page({"type": "error", "text": f"Saving module rules failed: {e}"})

                # IAP grant (service level)
                try:
                    _grant_iap(email, SERVICE_NAME, level="service")
                    return _render_admin_access_page({"type": "ok", "text": f"Added {email} to App + Modules + IAP."})
                except Exception as e:
                    return _render_admin_access_page({"type": "error", "text": f"IAP change failed: {e}"})

            elif action == "delete":
                email = (request.form.get("email") or "").strip().lower()
                if not EMAIL_RE.match(email):
                    return _render_admin_access_page({"type": "error", "text": "Invalid email."})
                # Remove from App
                try:
                    execute("DELETE FROM user_page_rules WHERE email=%s;", (email,))
                    execute("DELETE FROM users WHERE email=%s;", (email,))
                    if course_id:
                        execute("DELETE FROM user_module_rules WHERE email=%s AND course_id=%s;", (email, course_id))
                except Exception as e:
                    return _render_admin_access_page({"type": "error", "text": f"Failed removing from App: {e}"})

                # Remove from IAP (service level)
                try:
                    _revoke_iap(email, SERVICE_NAME, level="service")
                except Exception as e:
                    return _render_admin_access_page({"type": "error", "text": f"IAP removal failed: {e}"})
                return _render_admin_access_page({"type": "ok", "text": f"Removed {email} from App + Modules + IAP."})

            return _render_admin_access_page({"type": "error", "text": "Unknown combined action."})

        # IAP only
        if section == "iap":
            email = (request.form.get("email") or "").strip().lower()
            if not EMAIL_RE.match(email):
                return _render_admin_access_page({"type": "error", "text": "Invalid email."})
            iap_action = (request.form.get("iap_action") or "add").strip()
            try:
                if iap_action == "add":
                    _grant_iap(email, SERVICE_NAME, level="service")
                    return _render_admin_access_page({"type": "ok", "text": f"Added {email} to IAP (service level)."})
                elif iap_action == "remove":
                    _revoke_iap(email, SERVICE_NAME, level="service")
                    return _render_admin_access_page({"type": "ok", "text": f"Removed {email} from IAP (service level)."})
                else:
                    return _render_admin_access_page({"type": "error", "text": "Unknown IAP action."})
            except Exception as e:
                return _render_admin_access_page({"type": "error", "text": f"IAP change failed: {e}"})

        # App roles only
        if section == "amas":
            email = (request.form.get("email") or "").strip().lower()
            if not EMAIL_RE.match(email):
                return _render_admin_access_page({"type": "error", "text": "Invalid email."})
            role = (request.form.get("role") or "learner").strip().lower()
            if role not in {"learner", "student", "instructor", "admin", "owner"}:
                return _render_admin_access_page({"type": "error", "text": "Invalid role."})
            try:
                full_name = email.split("@", 1)[0].replace(".", " ").title()
                execute("""
                    INSERT INTO users (email, full_name, role)
                    VALUES (%s, %s, %s::role)
                    ON CONFLICT (email) DO UPDATE SET role = EXCLUDED.role;
                """, (email, full_name, role))
                return _render_admin_access_page({"type": "ok", "text": f"Saved {email} as {role}."})
            except Exception as e:
                return _render_admin_access_page({"type": "error", "text": f"Saving role failed: {e}"})

        # Page visibility only
        if section == "pages":
            email = (request.form.get("email") or "").strip().lower()
            if not EMAIL_RE.match(email):
                return _render_admin_access_page({"type": "error", "text": "Invalid email for page rules."})
            selected = request.form.getlist("page_allowed")
            page_csv = _normalize_page_csv(selected)
            try:
                _ensure_page_rules_table()
                execute("""
                    INSERT INTO user_page_rules (email, page_access)
                    VALUES (%s, %s)
                    ON CONFLICT (email) DO UPDATE SET page_access = EXCLUDED.page_access, updated_at = now();
                """, (email, page_csv))
                return _render_admin_access_page({"type": "ok", "text": f"Updated page visibility for {email}."})
            except Exception as e:
                return _render_admin_access_page({"type": "error", "text": f"Updating page visibility failed: {e}"})

        # NEW: Modules only
        if section == "modules":
            email = (request.form.get("email") or "").strip().lower()
            if not EMAIL_RE.match(email):
                return _render_admin_access_page({"type": "error", "text": "Invalid email for module rules."})
            course, modules = _get_course_and_modules()
            course_id = (course or {}).get("id")
            if not course_id:
                return _render_admin_access_page({"type": "error", "text": "Course not found."})

            module_mode = (request.form.get("module_mode") or "all").strip()     # all|none|custom
            module_selected = request.form.getlist("module_allowed")             # orders
            all_orders = [int(o) for o, _ in (modules or [])]
            allowed_modules = _normalize_module_csv(module_selected, all_orders, module_mode)
            try:
                _upsert_module_rules(email, course_id, allowed_modules)
                return _render_admin_access_page({"type": "ok", "text": f"Updated module access for {email}."})
            except Exception as e:
                return _render_admin_access_page({"type": "error", "text": f"Updating module access failed: {e}"})

        return _render_admin_access_page({"type": "error", "text": "Unknown section."})

    # ---------- Learner enroll (builder quick add) ----------
    def _enroll_learner(email: str, name: str) -> Tuple[bool, str, str]:
        execute("""
            INSERT INTO users (email, full_name, role)
            VALUES (%s, %s, 'learner')
            ON CONFLICT (email)
            DO UPDATE SET
              full_name = COALESCE(EXCLUDED.full_name, users.full_name),
              role = 'learner';
        """, (email, name))
        try:
            changed, _ = _grant_iap(email, SERVICE_NAME, level="service")
            return changed, f"Enrolled & IAP: {'access granted' if changed else 'already allowed'}", ""
        except Exception as e:
            return False, "Enrolled in app DB", f"IAP not updated: {e}"

    @bp.post("/course/<int:course_id>/add-learner")
    def admin_add_learner(course_id: int):
        require_admin()
        email = (request.form.get("email") or "").strip().lower()
        name = (request.form.get("name") or "").strip()
        if not EMAIL_RE.match(email):
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="Valid email required"))
        if not name:
            name = email.split("@", 1)[0].replace(".", " ").title()
        try:
            _, msg, err = _enroll_learner(email, name)
            return redirect(url_for("admin.admin_builder", course_id=course_id, msg=msg, err=err or ""))
        except Exception as e:
            return redirect(url_for("admin.admin_builder", course_id=course_id, err=f"Enroll failed: {e}"))

    @bp.post("/course/<int:course_id>/add-learners-bulk")
    def admin_add_learners_bulk(course_id: int):
        require_admin()
        raw = request.form.get("emails") or ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return redirect(url_for("admin.admin_builder", course_id=course_id, err="No emails provided"))

        added, iap_applied, failed = 0, 0, []
        for ln in lines:
            if "," in ln:
                email, name = ln.split(",", 1)
                email = email.strip().lower()
                name = name.strip()
            else:
                email = ln.strip().lower()
                name = email.split("@", 1)[0].replace(".", " ").title()

            if not EMAIL_RE.match(email):
                failed.append((ln, "invalid email"))
                continue

            try:
                changed, _, err = _enroll_learner(email, name)
                added += 1
                if changed:
                    iap_applied += 1
                if err:
                    failed.append((email, err))
            except Exception as e:
                failed.append((email, str(e)))

        msg = f"Enrolled {added} learner(s); IAP applied to {iap_applied}"
        if failed:
            details = "; ".join([f"{em}: {why}" for em, why in failed][:5])
            return redirect(url_for("admin.admin_builder", course_id=course_id, msg=msg, err=details))
        return redirect(url_for("admin.admin_builder", course_id=course_id, msg=msg))

    return bp
