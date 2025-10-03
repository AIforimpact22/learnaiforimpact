# learn.py
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, List, Callable, Tuple

from flask import Blueprint, render_template, redirect, url_for, g, request, jsonify

# --------- Local helpers (order-aware sections/lessons) ----------
def _sorted_sections(structure: Dict[str, Any]) -> List[dict]:
    secs = (structure.get("sections") or [])
    secs = sorted(secs, key=lambda s: (s.get("order") or 0, s.get("title") or ""))
    for s in secs:
        lessons = (s.get("lessons") or [])
        lessons = sorted(lessons, key=lambda l: (l.get("order") or 0, l.get("title") or ""))
        s["lessons"] = lessons
    return secs

def _flatten_lessons_ordered(structure: Dict[str, Any]) -> List[Tuple[dict, dict]]:
    out = []
    for s in _sorted_sections(structure):
        for l in (s.get("lessons") or []):
            out.append((s, l))
    return out

def _lesson_index_map_ordered(structure: Dict[str, Any]) -> Dict[str, int]:
    mapping = {}
    for i, (_, l) in enumerate(_flatten_lessons_ordered(structure)):
        uid = l.get("lesson_uid")
        if uid is not None:
            mapping[str(uid)] = i
    return mapping


# ----------------------------- Blueprint -----------------------------
def create_learn_blueprint(base_path: str, deps: Dict[str, Any], name: str = "learn") -> Blueprint:
    """
    Conversation endpoints and page, mounted under /learn (no double-prefix).
    Also reuses your lesson view (rendered by course.py) without changing it.
    """
    mount_prefix = base_path if base_path else "/learn"
    bp = Blueprint(name, __name__, url_prefix=mount_prefix)

    # --- DB helpers from deps
    fetch_one = deps["fetch_one"]; fetch_all = deps["fetch_all"]; execute = deps["execute"]

    # --- Structure helpers (or safe fallbacks)
    ensure_structure      = deps.get("ensure_structure") or (lambda s: (json.loads(s) if isinstance(s, str) else (s or {"sections": []})))
    flatten_lessons       = deps.get("flatten_lessons")  or _flatten_lessons_ordered
    num_lessons           = deps.get("num_lessons")      or (lambda st: len(_flatten_lessons_ordered(st)))
    total_course_duration = deps.get("total_course_duration") or (lambda st: 0)
    format_duration       = deps.get("format_duration")  or (lambda t: "—")
    slugify               = deps.get("slugify")          or (lambda s: "course")
    lesson_index_map_dep  = deps.get("lesson_index_map")
    latest_registration   = deps.get("latest_registration")

    # Robust progress fallbacks (so Conversation page never “forgets” progress)
    def _seen_lessons_db(user_id: int, course_id: int) -> Set[str]:
        try:
            rows = fetch_all("""
                SELECT DISTINCT lesson_uid
                FROM public.activity_log
                WHERE user_id = %s AND course_id = %s AND lesson_uid IS NOT NULL;
            """, (user_id, course_id))
            return {str(r["lesson_uid"]) for r in rows if r.get("lesson_uid")}
        except Exception as e:
            print("[conversation] seen_lessons fallback failed:", e)
            return set()

    def _frontier_from_seen_local(structure: Dict[str, Any], seen: Set[str]) -> int:
        flat_uids = [str(l[1].get("lesson_uid")) for l in flatten_lessons(structure) if l[1].get("lesson_uid") is not None]
        frontier = -1
        for i, uid in enumerate(flat_uids):
            if uid in seen:
                frontier = i
            else:
                break
        return frontier

    seen_lessons = deps.get("seen_lessons") or _seen_lessons_db
    frontier_from_seen = deps.get("frontier_from_seen") or _frontier_from_seen_local

    # ------------------------ Module -> Tag buttons mapping -------------------
    MODULE_TAGS: Dict[int, List[str]] = {
        1: ["Big picture","AI levels","Study habits","Python basics","Script parts","Library picks","Terms explained","Quick wins","Next steps","Mindset shift"],
        2: ["GitHub setup","Codespaces","Streamlit app","CSV viewer","Port forwarding","Commit push","Page layout","Bulk upload","Helper modules","Quality checks"],
        3: ["Small pieces","Clear interfaces","Reusable blocks","Separate pages","Helper files","Add features","Safer changes","Code cleanup","One improvement","Repo health"],
        4: ["SQL basics","Table design","Keys indexes","Data types","Constraints","Joins filters","SQLite setup","DB viewer","Attendance log","Bulk import"],
        5: ["Cloud database","Pooled connection","Secrets file","App forms","Attendance page","Deploy cloud","Public link","Access gate","Error logs","Cost notes"],
        6: ["Chart basics","Color scales","Rolling averages","Outlier flags","Live refresh","Calendar heatmaps","Data stories","D3 embeds","Smart polling","Dashboard flow"],
        7: ["ML basics","Clustering","Cluster plots","Anomaly alerts","Train test","Model metrics","Random forest","Boosted trees","Feature lags","Forecast steps"],
    }
    DEFAULT_TAGS = ["Highlights","Questions","Blockers","Ideas","Next steps","Other"]
    def _tags_for_week(week_index: int) -> List[str]:
        return MODULE_TAGS.get(int(week_index)) or DEFAULT_TAGS

    # ----------------------- Conversation helpers (JSONB) --------------------
    _conversation_column_ready = False
    def _ensure_conversation_column():
        nonlocal _conversation_column_ready
        if _conversation_column_ready:
            return
        try:
            execute("""
                ALTER TABLE public.courses
                ADD COLUMN IF NOT EXISTS conversation JSONB NOT NULL DEFAULT '{}'::jsonb;
            """)
        except Exception as e:
            print("[conversation] ensure column failed:", e)
        _conversation_column_ready = True

    def _ensure_conversation_json(course_id: int) -> Dict[str, Any]:
        _ensure_conversation_column()
        row = fetch_one("SELECT conversation FROM public.courses WHERE id = %s;", (course_id,))
        conv = (row or {}).get("conversation") or {}
        if not isinstance(conv, dict): conv = {}
        if "weeks" not in conv or not isinstance(conv["weeks"], dict):
            conv["weeks"] = {}
        return conv

    def _user_contributed_week(conv: Dict[str, Any], week_index: int, user_id: int) -> bool:
        bucket = (conv.get("weeks") or {}).get(str(week_index)) or []
        try:
            for it in bucket:
                if int(it.get("user_id") or 0) == int(user_id or 0):
                    return True
        except Exception:
            pass
        return False

    def _save_conversation_json(course_id: int, data: Dict[str, Any]) -> bool:
        try:
            execute("UPDATE public.courses SET conversation = %s::jsonb WHERE id = %s;", (json.dumps(data), course_id))
            return True
        except Exception as e:
            print("[conversation] save failed:", e)
            return False

    # -------------------------------- API: load / save -----------------------
    @bp.get("/<int:course_id>/week/<int:week_index>/feedback")
    def get_week_feedback(course_id: int, week_index: int):
        if not getattr(g, "user_id", None):
            return jsonify({"ok": False, "error": "unauthenticated"}), 401

        row = fetch_one("SELECT structure FROM public.courses WHERE id = %s;", (course_id,))
        if not row: return jsonify({"ok": False, "error": "course"}), 404
        st = ensure_structure(row.get("structure"))
        secs = _sorted_sections(st)
        if week_index < 1 or week_index > len(secs):
            return jsonify({"ok": False, "error": "invalid_week"}), 404

        conv = _ensure_conversation_json(course_id)
        items = list((conv.get("weeks") or {}).get(str(week_index), []))
        try:
            items.sort(key=lambda x: x.get("created_at", ""))
        except Exception:
            pass
        return jsonify({"ok": True, "items": items, "tags": _tags_for_week(week_index)}), 200

    @bp.post("/<int:course_id>/week/<int:week_index>/feedback")
    def post_week_feedback(course_id: int, week_index: int):
        if not getattr(g, "user_id", None):
            return jsonify({"ok": False, "error": "unauthenticated"}), 401

        row = fetch_one("SELECT structure FROM public.courses WHERE id = %s;", (course_id,))
        if not row: return jsonify({"ok": False, "error": "course"}), 404
        st = ensure_structure(row.get("structure"))
        secs = _sorted_sections(st)
        if week_index < 1 or week_index > len(secs):
            return jsonify({"ok": False, "error": "invalid_week"}), 404

        user_email = getattr(g, "user_email", None) or ""
        display = user_email
        try:
            if latest_registration:
                reg = latest_registration(user_email, course_id)
                if reg:
                    parts = [reg.get("first_name") or "", reg.get("middle_name") or "", reg.get("last_name") or ""]
                    name = " ".join([p for p in parts if p]).strip()
                    if name: display = name
        except Exception:
            pass

        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        tags_raw = data.get("tags") or []
        if not isinstance(tags_raw, list): tags_raw = []
        allowed = set(_tags_for_week(week_index))
        tags = []
        for t in tags_raw:
            t = (t or "").strip()
            if t and t in allowed and t not in tags:
                tags.append(t)
            if len(tags) >= 10:
                break
        if len(text) > 5000:
            text = text[:5000]

        item = {
            "user_id": int(getattr(g, "user_id", 0) or 0),
            "user_email": user_email,
            "user_display": display,
            "tags": tags,
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }

        conv = _ensure_conversation_json(course_id)
        bucket = list((conv.get("weeks") or {}).get(str(week_index), []))
        bucket.append(item)
        conv.setdefault("weeks", {})[str(week_index)] = bucket
        if not _save_conversation_json(course_id, conv):
            return jsonify({"ok": False, "error": "db_save"}), 500

        # Signal to the client that the next section can now unlock
        return jsonify({"ok": True, "saved": True, "unlocked_next": True, "item": item}), 200

    # -------------------------- Conversation PAGE ----------------------------
    @bp.get("/<int:course_id>/week/<int:week_index>/conversation")
    def conversation_page(course_id: int, week_index: int):
        row = fetch_one("SELECT id, title, structure FROM public.courses WHERE id = %s;", (course_id,))
        if not row:
            return redirect(url_for("course_detail", course_id=course_id))
        st = ensure_structure(row.get("structure"))
        sections_sorted = _sorted_sections(st)

        if not sections_sorted:
            return redirect(url_for("course_detail", course_id=course_id))
        if week_index < 1 or week_index > len(sections_sorted):
            week_index = 1

        # Compute flat list / per-section indices
        flat_uids: List[str] = []
        sec_first_idx: List[Optional[int]] = []
        sec_last_idx: List[Optional[int]] = []
        cursor = 0
        for sec in sections_sorted:
            n = len(sec.get("lessons") or [])
            if n > 0:
                sec_first_idx.append(cursor)
                sec_last_idx.append(cursor + n - 1)
                for l in sec["lessons"]:
                    flat_uids.append(str(l.get("lesson_uid")))
                cursor += n
            else:
                sec_first_idx.append(None)
                sec_last_idx.append(None)

        # Frontier / gating
        try:
            s = seen_lessons(g.user_id, course_id) if getattr(g, "user_id", None) else set()
            frontier_before = frontier_from_seen(st, s)
        except Exception:
            frontier_before = -1

        conv = _ensure_conversation_json(course_id)

        # With gating removed, expose every lesson link in the sidebar.
        max_unlocked_index = (len(flat_uids) - 1) if flat_uids else 0

        # Prev/Next destinations (lessons)
        cur_si0 = week_index - 1

        # Prev → last lesson of previous non-empty section
        conv_prev_href = None
        for j in range(cur_si0 - 1, -1, -1):
            li = sec_last_idx[j]
            if li is not None and 0 <= li < len(flat_uids):
                conv_prev_href = url_for(f"{bp.name}.learn_lesson", course_id=course_id, lesson_uid=flat_uids[li])
                break

        # Next (first lesson of next non-empty section)
        next_first_href = None
        for j in range(cur_si0 + 1, len(sections_sorted)):
            fi = sec_first_idx[j]
            if fi is not None and 0 <= fi < len(flat_uids):
                next_first_href = url_for(f"{bp.name}.learn_lesson", course_id=course_id, lesson_uid=flat_uids[fi])
                break

        # Only expose Next if the learner has contributed for the current week
        conv_next_href = None
        if getattr(g, "user_id", None) and _user_contributed_week(conv, week_index, g.user_id):
            conv_next_href = next_first_href

        # Course meta + learner
        course_meta = {
            "id": row["id"],
            "title": row.get("title"),
            "slug": slugify(row.get("title") or f"course-{course_id}"),
            "duration_total": format_duration(total_course_duration(st)),
            "lessons_count": len(flatten_lessons(st)),
        }

        learner_name = None; reg = None
        try:
            email = getattr(g, "user_email", None)
            if email and latest_registration:
                reg = latest_registration(email, course_id)
                if reg:
                    parts = [reg.get("first_name") or "", reg.get("middle_name") or "", reg.get("last_name") or ""]
                    learner_name = " ".join([p for p in parts if p]).strip() or reg.get("user_email")
        except Exception as e:
            print("[conversation] registration name lookup failed:", e)

        # Build lesson index map (helper if provided, else local)
        idx_map = (lesson_index_map_dep(st) if callable(lesson_index_map_dep) else _lesson_index_map_ordered(st))

        return render_template(
            "learn.html",
            course=course_meta,
            sections=sections_sorted,
            current_section_index=week_index - 1,
            current_lesson_uid=None,
            lesson=None,
            prev_uid=None,
            next_uid=None,
            max_unlocked_index=max_unlocked_index,          # gating disabled – expose all lessons
            global_frontier_index=frontier_before,
            lesson_index_by_uid=idx_map,
            registration=reg,
            learner_name=learner_name,
            conversation_mode=True,
            conversation_week=week_index,
            conversation_tags=_tags_for_week(week_index),
            conv_prev_href=conv_prev_href,
            conv_next_href=conv_next_href,                  # only present if contributed
            conv_next_first_href=next_first_href,           # always known; used to reveal Next after submit
        )

    # ------------------------------ LEARN ROUTES (delegates) -----------------
    @bp.get("/<int:course_id>/<lesson_uid>")
    def learn_lesson(course_id: int, lesson_uid: str):  # delegate to global route
        return redirect(url_for("learn_lesson", course_id=course_id, lesson_uid=lesson_uid))

    @bp.get("/<int:course_id>")
    def learn_redirect_to_first(course_id: int):        # delegate to global route
        return redirect(url_for("learn_redirect_to_first", course_id=course_id))

    return bp
