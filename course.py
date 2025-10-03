# course.py
from typing import Any, Dict, List
from flask import render_template, redirect, url_for, g, current_app

def register_course_routes(app, base_path: str, deps: Dict[str, Any]):
    """
    Registers:
      - GET "/learn/<int:course_id>" -> endpoint 'learn_redirect_to_first'
      - GET "/learn/<int:course_id>/<lesson_uid>" -> endpoint 'learn_lesson'
    Also creates BASE_PATH aliases without changing endpoint names used by templates.
    """
    fetch_one = deps["fetch_one"]
    ensure_structure = deps["ensure_structure"]
    flatten_lessons = deps["flatten_lessons"]
    sorted_sections_dep = deps.get("sorted_sections")
    first_lesson_uid = deps["first_lesson_uid"]
    find_lesson = deps["find_lesson"]
    next_prev_uids = deps["next_prev_uids"]
    lesson_index_map = deps["lesson_index_map"]
    uid_by_index = deps["uid_by_index"]
    num_lessons = deps["num_lessons"]
    total_course_duration = deps["total_course_duration"]
    format_duration = deps["format_duration"]
    slugify = deps["slugify"]
    seen_lessons = deps["seen_lessons"]
    last_seen_uid = deps["last_seen_uid"]
    log_activity = deps["log_activity"]
    log_view_once = deps["log_view_once"]
    frontier_from_seen = deps["frontier_from_seen"]
    latest_registration = deps["latest_registration"]

    def _exam_statuses_for_course(course_id: int) -> Dict[int, Dict[str, Any]]:
        if not getattr(g, "user_id", None):
            return {}
        helper = None
        try:
            helpers = (current_app.extensions.get("exam_helpers", {}) if current_app else {})
            if isinstance(helpers, dict):
                helper = helpers.get("collect_statuses")
        except Exception as e:
            print("[learn] exam helpers lookup failed:", e)
            helper = None
        if not helper:
            return {}
        try:
            data = helper(g.user_id, course_id) or {}
        except Exception as e:
            print("[learn] exam statuses helper failed:", e)
            return {}
        cleaned: Dict[int, Dict[str, Any]] = {}
        for k, v in (data or {}).items():
            try:
                cleaned[int(k)] = v
            except Exception:
                cleaned[k] = v
        return cleaned

    def _alias(rule: str, view_func, methods=None, endpoint_suffix="alias"):
        if not base_path:
            return
        alias_rule = f"{base_path}{rule if rule.startswith('/') else '/' + rule}"
        endpoint = f"{view_func.__name__}_{endpoint_suffix}_{abs(hash(alias_rule))}"
        app.add_url_rule(alias_rule, endpoint=endpoint, view_func=view_func, methods=methods or ["GET"])

    def _has_any_activity(user_id: int, course_id: int) -> bool:
        try:
            row = fetch_one("""
                SELECT 1
                  FROM public.activity_log
                 WHERE user_id = %s AND course_id = %s
                 LIMIT 1;
            """, (user_id, course_id))
            return bool(row)
        except Exception:
            return False

    # --- Display-order utilities (stable section/lesson ordering) ---
    def _sorted_sections_for_viz(structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        if sorted_sections_dep:
            result: List[Dict[str, Any]] = []
            for sec in sorted_sections_dep(structure):
                lessons = list((sec.get("lessons") or ()))
                sec_copy = dict(sec)
                sec_copy["lessons"] = lessons
                result.append(sec_copy)
            return result

        secs_raw = (structure.get("sections") or [])
        secs_sorted = sorted(secs_raw, key=lambda s: (s.get("order") or 0, s.get("title") or ""))
        out = []
        for s in secs_sorted:
            s2 = dict(s)
            lessons = s.get("lessons") or []
            lessons_sorted = sorted(lessons, key=lambda l: (l.get("order") or 0, l.get("title") or ""))
            s2["lessons"] = lessons_sorted
            out.append(s2)
        return out

    def _find_lesson_in_sorted_sections(sections_sorted: List[Dict[str, Any]], lesson_uid: str):
        for si, sec in enumerate(sections_sorted):
            for li, l in enumerate(sec.get("lessons") or []):
                if str(l.get("lesson_uid")) == str(lesson_uid):
                    return si, li, l
        return None, None, None

    # ----- Routes -----
    def learn_redirect_to_first(course_id: int):
        row = fetch_one("SELECT id, structure FROM courses WHERE id = %s;", (course_id,))
        if not row:
            from flask import abort
            abort(404)
        st = ensure_structure(row.get("structure"))

        if num_lessons(st) == 0:
            return redirect(url_for("course_detail", course_id=course_id))

        # Default to first
        target = first_lesson_uid(st)
        try:
            if getattr(g, "user_id", None):
                # On first ever entry, log a 'start' row (enum-safe)
                if not _has_any_activity(g.user_id, course_id):
                    log_activity(g.user_id, course_id, None, "start", payload={"source": "learn_entry"})
                last_uid = last_seen_uid(g.user_id, course_id)
                if last_uid:
                    target = last_uid
        except Exception as e:
            print("[learn_redirect] last_seen/start failed:", e)

        return redirect(url_for("learn_lesson", course_id=course_id, lesson_uid=target))

    def learn_lesson(course_id: int, lesson_uid: str):
        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        if not row:
            from flask import abort
            abort(404)
        st = ensure_structure(row.get("structure"))
        sections_viz = _sorted_sections_for_viz(st)
        idx_map = lesson_index_map(st)

        # Validate lesson exists in structure
        cur_idx = idx_map.get(str(lesson_uid))
        if cur_idx is None:
            fallback_uid = first_lesson_uid(st)
            return redirect(url_for("learn_lesson", course_id=course_id, lesson_uid=fallback_uid))

        # With gating removed, expose all lessons regardless of prior progress.
        seen = seen_lessons(g.user_id, course_id) if getattr(g, "user_id", None) else set()
        frontier_before = frontier_from_seen(st, seen)
        total = num_lessons(st)
        allowed_next = total - 1 if total else 0

        # Log view (debounced) + unlock on advance
        try:
            if getattr(g, "user_id", None):
                log_view_once(g.user_id, course_id, str(lesson_uid), window_seconds=120)
                seen_after = set(seen); seen_after.add(str(lesson_uid))
                frontier_after = frontier_from_seen(st, seen_after)
                if frontier_after > frontier_before:
                    log_activity(
                        g.user_id, course_id, str(lesson_uid), "unlock",
                        payload={"kind": "unlock", "from": frontier_before, "to": frontier_after}
                    )
        except Exception as e:
            print("[activity] logging failed:", e)

        si_sorted, li_sorted, lesson_viz = _find_lesson_in_sorted_sections(sections_viz, lesson_uid)
        if si_sorted is None:
            allowed_uid = uid_by_index(st, allowed_next) or first_lesson_uid(st)
            return redirect(url_for("learn_lesson", course_id=course_id, lesson_uid=allowed_uid))

        lesson = lesson_viz
        prev_uid, next_uid = next_prev_uids(st, lesson_uid)

        course_meta = {
            "id": row["id"],
            "title": row.get("title"),
            "slug": slugify(row.get("title") or f"course-{course_id}"),
            "duration_total": format_duration(total_course_duration(st)),
            "lessons_count": len(flatten_lessons(st)),
        }

        learner_name = None
        reg = None
        try:
            email = getattr(g, "user_email", None)
            if email:
                reg = latest_registration(email, course_id)
                if reg:
                    parts = [reg.get("first_name") or "", reg.get("middle_name") or "", reg.get("last_name") or ""]
                    learner_name = " ".join([p for p in parts if p]).strip() or reg.get("user_email")
        except Exception as e:
            print("[learn] registration name lookup failed:", e)

        global_frontier_index = frontier_before

        exam_statuses = _exam_statuses_for_course(course_id)

        return render_template(
            "learn.html",
            course=course_meta,
            sections=sections_viz,                 # sorted for display and week mapping
            current_section_index=si_sorted,       # sorted index → Week N = +1
            current_lesson_uid=str(lesson_uid),
            lesson=lesson,
            prev_uid=prev_uid,
            next_uid=next_uid,
            max_unlocked_index=allowed_next,            # gating disabled – expose all lessons
            global_frontier_index=global_frontier_index,
            lesson_index_by_uid=idx_map,
            registration=reg,
            learner_name=learner_name,
            exam_statuses=exam_statuses,
        )

    # Register with same endpoint names as before
    app.add_url_rule("/learn/<int:course_id>", view_func=learn_redirect_to_first, methods=["GET"], endpoint="learn_redirect_to_first")
    app.add_url_rule("/learn/<int:course_id>/<lesson_uid>", view_func=learn_lesson, methods=["GET"], endpoint="learn_lesson")

    _alias("/learn/<int:course_id>", learn_redirect_to_first, ["GET"])
    _alias("/learn/<int:course_id>/<lesson_uid>", learn_lesson, ["GET"])
