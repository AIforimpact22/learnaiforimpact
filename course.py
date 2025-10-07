# course.py
from typing import Any, Dict, List
from flask import render_template, redirect, url_for, g, current_app, jsonify

def register_course_routes(app, base_path: str, deps: Dict[str, Any]):
    """
    Registers:
      - GET "/learn/<int:course_id>" -> endpoint 'learn_redirect_to_first'
      - GET "/learn/<int:course_id>/<lesson_uid>" -> endpoint 'learn_lesson'
    Also creates BASE_PATH aliases without changing endpoint names used by templates.
    """
    fetch_one = deps["fetch_one"]
    ensure_structure = deps["ensure_structure"]
    course_structure = deps.get("course_structure") or (lambda row: ensure_structure((row or {}).get("structure")))
    flatten_lessons = deps["flatten_lessons"]
    sorted_sections_dep = deps.get("sorted_sections")
    load_course_sidebar_dep = deps.get("load_course_sidebar")
    load_course_lesson_dep = deps.get("load_course_lesson")
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
    def _sorted_sections_for_viz(structure: Dict[str, Any], row: Dict[str, Any]) -> List[Dict[str, Any]]:
        def _sanitize_sections(sections_source) -> List[Dict[str, Any]]:
            cleaned: List[Dict[str, Any]] = []
            for sec in sections_source or []:
                if not isinstance(sec, dict):
                    continue
                entry = {
                    "title": sec.get("title"),
                    "order": sec.get("order"),
                    "lessons": []
                }
                lessons_raw = sec.get("lessons")
                if isinstance(lessons_raw, (list, tuple)):
                    lessons_iter = list(lessons_raw)
                elif lessons_raw:
                    lessons_iter = [lessons_raw]
                else:
                    lessons_iter = []
                lessons_sorted = sorted(lessons_iter, key=lambda l: (l.get("order") or 0, l.get("title") or ""))
                for lesson in lessons_sorted:
                    if not isinstance(lesson, dict):
                        continue
                    lesson_entry = {
                        "lesson_uid": lesson.get("lesson_uid"),
                        "title": lesson.get("title"),
                        "order": lesson.get("order"),
                        "kind": lesson.get("kind"),
                    }
                    duration = None
                    content = lesson.get("content") if isinstance(lesson.get("content"), dict) else None
                    if isinstance(content, dict):
                        duration = content.get("duration_sec")
                    if isinstance(duration, int):
                        lesson_entry["duration_sec"] = duration
                    entry["lessons"].append(lesson_entry)
                cleaned.append(entry)
            return cleaned

        sections_candidate = None
        if load_course_sidebar_dep:
            try:
                sections_candidate = load_course_sidebar_dep(structure)
            except TypeError:
                sections_candidate = load_course_sidebar_dep(row.get("structure"), course_title=row.get("title"))
            except Exception as e:
                print("[learn] load_course_sidebar failed:", e)
                sections_candidate = None
            if isinstance(sections_candidate, list):
                return _sanitize_sections(sections_candidate)

        if sorted_sections_dep:
            try:
                sections_candidate = sorted_sections_dep(structure)
            except Exception as e:
                print("[learn] sorted_sections failed:", e)
                sections_candidate = None
            if sections_candidate is not None:
                return _sanitize_sections(sections_candidate)

        secs_raw = structure.get("sections")
        if isinstance(secs_raw, (list, tuple)):
            sections_candidate = sorted(secs_raw, key=lambda s: (s.get("order") or 0, s.get("title") or ""))
        elif secs_raw:
            sections_candidate = [secs_raw]
        else:
            sections_candidate = []
        return _sanitize_sections(sections_candidate)

    def _find_lesson_in_sorted_sections(sections_sorted: List[Dict[str, Any]], lesson_uid: str):
        for si, sec in enumerate(sections_sorted):
            for li, l in enumerate(sec.get("lessons") or []):
                if str(l.get("lesson_uid")) == str(lesson_uid):
                    return si, li
        return None, None

    def _lesson_for_uid(structure: Dict[str, Any], lesson_uid: str, row: Dict[str, Any]):
        lesson = None
        if load_course_lesson_dep:
            try:
                lesson = load_course_lesson_dep(structure, lesson_uid)
            except TypeError:
                lesson = load_course_lesson_dep(row.get("structure"), lesson_uid, course_title=row.get("title"))
            except Exception as e:
                print("[learn] load_course_lesson failed:", e)
                lesson = None
        if lesson is None and find_lesson:
            try:
                si, li = find_lesson(structure, lesson_uid)
            except Exception as e:
                print("[learn] find_lesson failed:", e)
                si = li = None
            if si is not None and li is not None:
                try:
                    sections_raw = structure.get("sections") or []
                    section = sections_raw[si] if isinstance(sections_raw, list) else None
                    lessons = section.get("lessons") if isinstance(section, dict) else None
                    if isinstance(lessons, list) and 0 <= li < len(lessons):
                        lesson = lessons[li]
                except Exception as e:
                    print("[learn] lesson lookup failed:", e)
                    lesson = None
        return lesson

    def _lesson_context(course_id: int, row: Dict[str, Any], lesson_uid: str, *, include_sections: bool):
        structure = course_structure(row)
        idx_map = lesson_index_map(structure)
        current_idx = idx_map.get(str(lesson_uid))
        if current_idx is None:
            return {"redirect": first_lesson_uid(structure)}

        sections_viz = _sorted_sections_for_viz(structure, row) if include_sections else None
        seen = seen_lessons(g.user_id, course_id) if getattr(g, "user_id", None) else set()
        frontier_before = frontier_from_seen(structure, seen)
        total = num_lessons(structure)
        allowed_next = total - 1 if total else 0

        try:
            if getattr(g, "user_id", None):
                log_view_once(g.user_id, course_id, str(lesson_uid), window_seconds=120)
                seen_after = set(seen)
                seen_after.add(str(lesson_uid))
                frontier_after = frontier_from_seen(structure, seen_after)
                if frontier_after > frontier_before:
                    log_activity(
                        g.user_id,
                        course_id,
                        str(lesson_uid),
                        "unlock",
                        payload={"kind": "unlock", "from": frontier_before, "to": frontier_after},
                    )
        except Exception as e:
            print("[activity] logging failed:", e)

        prev_uid, next_uid = next_prev_uids(structure, lesson_uid)
        lesson = _lesson_for_uid(structure, lesson_uid, row)
        if lesson is None:
            return {"redirect": first_lesson_uid(structure)}

        section_index = None
        if include_sections and sections_viz is not None:
            si_sorted, _ = _find_lesson_in_sorted_sections(sections_viz, lesson_uid)
            if si_sorted is None:
                allowed_uid = uid_by_index(structure, allowed_next) or first_lesson_uid(structure)
                return {"redirect": allowed_uid}
            section_index = si_sorted

        course_meta = {
            "id": row["id"],
            "title": row.get("title"),
            "slug": slugify(row.get("title") or f"course-{course_id}"),
            "duration_total": format_duration(total_course_duration(structure)),
            "lessons_count": len(flatten_lessons(structure)),
        }

        return {
            "structure": structure,
            "sections": sections_viz,
            "lesson": lesson,
            "prev_uid": prev_uid,
            "next_uid": next_uid,
            "allowed_next": allowed_next,
            "frontier_before": frontier_before,
            "idx_map": idx_map,
            "section_index": section_index,
            "course_meta": course_meta,
        }

    # ----- Routes -----
    def learn_redirect_to_first(course_id: int):
        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        if not row:
            from flask import abort
            abort(404)
        st = course_structure(row)

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
        ctx = _lesson_context(course_id, row, lesson_uid, include_sections=True)
        redirect_uid = ctx.get("redirect")
        if redirect_uid:
            return redirect(url_for("learn_lesson", course_id=course_id, lesson_uid=redirect_uid))

        sections_viz = ctx.get("sections") or []
        lesson = ctx.get("lesson")
        prev_uid = ctx.get("prev_uid")
        next_uid = ctx.get("next_uid")
        allowed_next = ctx.get("allowed_next", 0)
        idx_map = ctx.get("idx_map") or {}
        course_meta = ctx.get("course_meta") or {}
        global_frontier_index = ctx.get("frontier_before", 0)
        current_section_index = ctx.get("section_index")

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

        exam_statuses = _exam_statuses_for_course(course_id)

        return render_template(
            "learn.html",
            course=course_meta,
            sections=sections_viz,                 # sorted for display and week mapping
            current_section_index=current_section_index,       # sorted index → Week N = +1
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

    def learn_lesson_data(course_id: int, lesson_uid: str):
        row = fetch_one("SELECT id, title, structure FROM courses WHERE id = %s;", (course_id,))
        if not row:
            from flask import abort
            abort(404)

        ctx = _lesson_context(course_id, row, lesson_uid, include_sections=False)
        redirect_uid = ctx.get("redirect")
        if redirect_uid:
            redirect_url = url_for("learn_lesson", course_id=course_id, lesson_uid=redirect_uid)
            return jsonify({"ok": False, "redirect_url": redirect_url}), 404

        lesson = ctx.get("lesson")
        if lesson is None:
            return jsonify({"ok": False}), 404

        prev_uid = ctx.get("prev_uid")
        next_uid = ctx.get("next_uid")
        prev_url = url_for("learn_lesson", course_id=course_id, lesson_uid=prev_uid) if prev_uid else None
        next_url = url_for("learn_lesson", course_id=course_id, lesson_uid=next_uid) if next_uid else None

        payload = {
            "ok": True,
            "lesson_uid": str(lesson_uid),
            "lesson_title": lesson.get("title") or "Lesson",
            "lesson_html": render_template("partials/lesson_body.html", lesson=lesson),
            "prev_uid": prev_uid,
            "next_uid": next_uid,
            "prev_url": prev_url,
            "next_url": next_url,
            "course_home_url": url_for("course_detail", course_id=course_id),
            "page_url": url_for("learn_lesson", course_id=course_id, lesson_uid=lesson_uid),
            "prefetch_urls": [u for u in (prev_url, next_url) if u],
        }

        return jsonify(payload)

    # Register with same endpoint names as before
    app.add_url_rule("/learn/<int:course_id>", view_func=learn_redirect_to_first, methods=["GET"], endpoint="learn_redirect_to_first")
    app.add_url_rule("/learn/<int:course_id>/<lesson_uid>", view_func=learn_lesson, methods=["GET"], endpoint="learn_lesson")
    app.add_url_rule(
        "/learn/<int:course_id>/<lesson_uid>/data",
        view_func=learn_lesson_data,
        methods=["GET"],
        endpoint="learn_lesson_data",
    )

    _alias("/learn/<int:course_id>", learn_redirect_to_first, ["GET"])
    _alias("/learn/<int:course_id>/<lesson_uid>", learn_lesson, ["GET"])
    _alias(
        "/learn/<int:course_id>/<lesson_uid>/data",
        learn_lesson_data,
        ["GET"],
        endpoint_suffix="alias_data",
    )
