# home.py
from typing import Any, Dict, Optional, List
from flask import render_template, g

def register_home_routes(app, base_path: str, deps: Dict[str, Any]):
    """
    Registers:
      - GET "/" -> endpoint 'index'
      - GET "/course/<int:course_id>" and "/course/<int:course_id>-<slug>" -> endpoint 'course_detail'
    Also creates BASE_PATH aliases without changing endpoint names used by templates.
    """
    COURSE_TITLE = deps["COURSE_TITLE"]
    COURSE_COVER = deps["COURSE_COVER"]
    fetch_one = deps["fetch_one"]
    ensure_structure = deps["ensure_structure"]
    get_course_structure = deps.get("get_course_structure") or (lambda cid, structure_raw=None: ensure_structure(structure_raw))
    flatten_lessons = deps["flatten_lessons"]
    total_course_duration = deps["total_course_duration"]
    format_duration = deps["format_duration"]
    first_lesson_uid = deps["first_lesson_uid"]
    slugify = deps["slugify"]
    last_seen_uid = deps["last_seen_uid"]
    seed_course_if_missing = deps["seed_course_if_missing"]

    def _alias(rule: str, view_func, methods=None, endpoint_suffix="alias"):
        if not base_path:
            return
        alias_rule = f"{base_path}{rule if rule.startswith('/') else '/' + rule}"
        endpoint = f"{view_func.__name__}_{endpoint_suffix}_{abs(hash(alias_rule))}"
        app.add_url_rule(alias_rule, endpoint=endpoint, view_func=view_func, methods=methods or ["GET"])

    # ----- Routes -----
    def index():
        try:
            seed_course_if_missing()
        except Exception as e:
            print(f"Seed failed: {e}")

        c = fetch_one("""
            SELECT id, title, is_published, published_at, created_at, structure
            FROM courses WHERE title = %s LIMIT 1;
        """, (COURSE_TITLE,))
        if not c:
            return render_template("index.html", course=None, err="Course not found.")

        st = get_course_structure(c.get("id"), structure_raw=c.get("structure"))
        weeks = st.get("sections") or []
        lessons_count = len(flatten_lessons(st))
        duration_total = format_duration(total_course_duration(st))
        first_uid = first_lesson_uid(st)

        c["slug"] = slugify(c.get("title") or f"course-{c['id']}")
        c["thumbnail_url"] = st.get("thumbnail_url") or COURSE_COVER
        c["category"] = st.get("category") or "Artificial Intelligence"
        c["level"] = st.get("level") or "Intermediate–Advanced"
        c["lessons_count"] = lessons_count
        c["duration_total"] = duration_total

        weeks_meta = [{"title": s.get("title") or "", "lessons_count": len(s.get("lessons") or [])} for s in weeks]

        # Continue target from activity_log if available
        continue_uid = first_uid
        try:
            if getattr(g, "user_id", None) and c.get("id"):
                last_uid = last_seen_uid(g.user_id, c["id"])
                if last_uid:
                    continue_uid = last_uid
        except Exception as e:
            print("[index] continue_uid calc failed:", e)

        # AFTER
        return render_template(
            "home.html",
            course=c,
            err=None,
            weeks=weeks_meta,
            modules_count=len(weeks),
            lessons_count=lessons_count,
            duration_total=duration_total,
            first_uid=first_uid,
            continue_uid=continue_uid,
            # Optional (nice to have): provide has_started to the template;
            # home.html can also infer it if it's missing.
            has_started=bool(continue_uid and first_uid and str(continue_uid) != str(first_uid)),
        )

    def course_detail(course_id: int, slug: Optional[str] = None):
        row = fetch_one("""
            SELECT id, title, is_published, published_at, created_at, structure
            FROM courses WHERE id = %s;
        """, (course_id,))
        if not row:
            from flask import abort
            abort(404)
        st = get_course_structure(row.get("id"), structure_raw=row.get("structure"))
        sections = st.get("sections", [])
        row["duration_total"] = format_duration(total_course_duration(st))
        row["lessons_count"] = len(flatten_lessons(st))
        row["level"] = st.get("level") or "All levels"
        row["category"] = st.get("category") or "Artificial Intelligence"
        row["thumbnail_url"] = st.get("thumbnail_url") or COURSE_COVER
        row["slug"] = slugify(row.get("title") or f"course-{course_id}")
        return render_template(
            "course_detail.html",
            course=row,
            sections=sections,
            instructors=st.get("instructors") or [],
            what_learn=st.get("what_you_will_learn") or [],
            description_md=st.get("description_md") or ""
        )

    # Register rules with same endpoint names as before
    app.add_url_rule("/", view_func=index, methods=["GET"], endpoint="index")
    _alias("/", index, ["GET"])

    # Two routes share the same endpoint 'course_detail' (unchanged)
    app.add_url_rule("/course/<int:course_id>", view_func=course_detail, methods=["GET"], endpoint="course_detail")
    app.add_url_rule("/course/<int:course_id>-<slug>", view_func=course_detail, methods=["GET"], endpoint="course_detail")
    _alias("/course/<int:course_id>", course_detail, ["GET"])
    _alias("/course/<int:course_id>-<slug>", course_detail, ["GET"])
