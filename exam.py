# exam.py
# -----------------------------------------------------------------------------
# Module-based exam engine (WRITING-ONLY, STRICT MODULE RELEVANCE) + Learn layout.
# - Sidebar preserved on the exam page (modules, lessons, exam links + status chips)
# - Bottom nav: Previous = last lesson of this module; Next = first lesson of next module
# - Strict module-only context; anchor-bound prompts; signature/version guard
# - Writing-only; no MCQ; no auto-submit; lenient GPT grading; 3 attempts; 500-char answers
# - Back-compat "week" routes preserved; primary routes use "module"
# -----------------------------------------------------------------------------

import os, json, uuid, time, hashlib, re
from typing import Any, Dict, Optional, List, Tuple, Callable

from flask import (
    Blueprint, request, jsonify, render_template, render_template_string,
    redirect, url_for, g
)

# -----------------------------------------------------------------------------
# Blueprint factory
# -----------------------------------------------------------------------------
def create_exam_blueprint(base_path: str, deps: Dict[str, Any], name: str = "exam") -> Blueprint:
    """
    Factory that returns a Blueprint mounted at base_path (e.g. "/learn").
    Required deps: fetch_one, fetch_all, execute
    Optional deps: ensure_structure
    """
    url_prefix = base_path or "/learn"
    bp = Blueprint(name, __name__, url_prefix=url_prefix)

    # ---- Required deps -------------------------------------------------------
    fetch_one: Callable = deps["fetch_one"]
    fetch_all: Callable = deps["fetch_all"]
    execute:   Callable = deps["execute"]
    ensure_structure: Callable = deps.get("ensure_structure") or _ensure_structure_fallback

    # ---- Config --------------------------------------------------------------
    OPENAI_API_KEY      = (os.getenv("OPENAI_API_KEY") or "").strip()
    OPENAI_QGEN_MODEL   = (os.getenv("OPENAI_QGEN_MODEL") or "gpt-4o-mini").strip()
    OPENAI_GRADER_MODEL = (os.getenv("OPENAI_GRADER_MODEL") or "gpt-4o-mini").strip()
    EXAMS_USE_GPT       = (os.getenv("EXAMS_USE_GPT", "1").lower() in ("1", "true", "yes"))

    DEFAULT_Q_COUNT        = int(os.getenv("EXAM_QUESTIONS_PER_MODULE") or 5)
    DEFAULT_TIME_LIMIT_MIN = int(os.getenv("EXAM_TIME_LIMIT_MIN") or 30)
    DEFAULT_PASS_SCORE     = int(os.getenv("EXAM_PASS_SCORE") or 70)

    MAX_SUBMISSIONS        = int(os.getenv("EXAM_MAX_SUBMISSIONS") or 3)
    ANSWER_CHAR_LIMIT      = int(os.getenv("EXAM_ANSWER_CHAR_LIMIT") or 500)

    ENFORCE_TIME_LIMIT     = False  # manual submit only
    QGEN_VERSION           = "module-anchors-v2"  # bump to invalidate legacy attempts

    # ---------------------- activity_log helpers & enums ----------------------
    _ACTIVITY_TYPE_NAME: Optional[str] = None
    _ACTIVITY_ENUM_LABELS: List[str] = []

    def _detect_activity_type():
        nonlocal _ACTIVITY_TYPE_NAME, _ACTIVITY_ENUM_LABELS
        if _ACTIVITY_TYPE_NAME is not None:
            return
        try:
            row = fetch_one("""
                SELECT atttypid::regtype::text AS tname
                  FROM pg_attribute
                 WHERE attrelid = 'public.activity_log'::regclass
                   AND attname  = 'a_type';
            """, ())
            tname = (row or {}).get("tname")
            if tname and tname.lower() not in ("text","varchar","character varying","pg_catalog.text"):
                _ACTIVITY_TYPE_NAME = tname
                labs = fetch_all(f"""
                    SELECT enumlabel
                      FROM pg_enum
                     WHERE enumtypid = '{_ACTIVITY_TYPE_NAME}'::regtype
                     ORDER BY enumsortorder;
                """, ())
                _ACTIVITY_ENUM_LABELS = [r["enumlabel"] for r in (labs or [])]
            else:
                _ACTIVITY_TYPE_NAME = None
                _ACTIVITY_ENUM_LABELS = []
        except Exception:
            _ACTIVITY_TYPE_NAME = None
            _ACTIVITY_ENUM_LABELS = []

    def _log_exam_activity(user_id: int, course_id: int, module_index: int,
                           event: str, attempt_uid: str,
                           extra_payload: Optional[dict] = None,
                           score_points: Optional[int] = None,
                           passed: Optional[bool] = None):
        """Append-only activity row. Synthetic lesson_uid per module: 'EXAM-M{n:02d}'."""
        _detect_activity_type()
        lesson_uid = f"EXAM-M{int(module_index):02d}"
        payload = {
            "kind": "exam",
            "event": event,
            "attempt_uid": attempt_uid,
            # keep both for back-compat
            "module_index": module_index,
            "week_index": module_index
        }
        if extra_payload:
            payload.update(extra_payload)
        payload_json = json.dumps(payload, ensure_ascii=False)

        label_text = f"exam_{event}"
        try:
            if _ACTIVITY_TYPE_NAME:
                label = label_text
                if _ACTIVITY_ENUM_LABELS and label not in _ACTIVITY_ENUM_LABELS:
                    label = _ACTIVITY_ENUM_LABELS[0]
                execute(f"""
                    INSERT INTO public.activity_log
                        (user_id, course_id, lesson_uid, a_type, created_at, score_points, passed, payload)
                    VALUES (%s, %s, %s, %s::{_ACTIVITY_TYPE_NAME}, now(), %s, %s, %s);
                """, (user_id, course_id, lesson_uid, label, score_points, passed, payload_json))
            else:
                execute("""
                    INSERT INTO public.activity_log
                        (user_id, course_id, lesson_uid, a_type, created_at, score_points, passed, payload)
                    VALUES (%s, %s, %s, %s, now(), %s, %s, %s);
                """, (user_id, course_id, lesson_uid, label_text, score_points, passed, payload_json))
        except Exception as e:
            print(f"[exam] activity insert failed: {e}")

    # ------------------------------- structure utils ---------------------------
    def _list_modules(struct: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Accepts structure with either 'modules', 'sections', or a single-module object."""
        if isinstance(struct, dict):
            if isinstance(struct.get("modules"), list):
                return struct["modules"]
            if isinstance(struct.get("sections"), list):
                mods = []
                for s in struct["sections"]:
                    mods.append({
                        "module": s.get("title") or "",
                        "title":  s.get("title") or "",
                        "goal":   s.get("goal"),
                        "outcomes": s.get("outcomes") or [],
                        "lessons": s.get("lessons") or []
                    })
                return mods
            if struct.get("module") or struct.get("lessons"):
                return [struct]
        return []

    def _module_meta(struct: Dict[str, Any], module_index: int) -> Tuple[Dict[str, Any], str]:
        modules = _list_modules(struct)
        if not modules or not (1 <= module_index <= len(modules)):
            return {}, f"Module {module_index}"
        m = modules[module_index - 1] or {}
        title = (m.get("title") or m.get("module") or f"Module {module_index}").strip() or f"Module {module_index}"
        return m, title

    def _text_from_lesson(lesson: Dict[str, Any]) -> str:
        parts = []
        if lesson.get("title"):           parts.append(f"## {lesson['title']}")
        if lesson.get("focus"):           parts.append(f"Focus: {lesson['focus']}")
        if lesson.get("content_preview"): parts.append(f"Preview: {lesson['content_preview']}")
        c = lesson.get("content") or {}
        for k in ("body_md","notes_md","transcript_md","text_md","markdown","summary_md","content"):
            v = c.get(k)
            if v: parts.append(str(v))
        return "\n\n".join(parts)

    def _module_context_text(struct: Dict[str, Any], module_index: int, cap: int = 24000) -> Tuple[str, List[str], Dict[str, Any]]:
        """Returns (context_md, anchors, module_obj) for this module only."""
        module, mod_title = _module_meta(struct, module_index)
        if not module:
            return "", [], {}
        parts: List[str] = []
        anchors: List[str] = []

        title = (module.get("title") or module.get("module") or mod_title).strip()
        if title:
            parts.append(f"# {title}")
            anchors.append(title)

        goal = (module.get("goal") or "").strip()
        if goal:
            parts.append(f"**Goal:** {goal}")
            anchors.append(goal)

        outs = module.get("outcomes") or []
        if isinstance(outs, list) and outs:
            parts.append("**Outcomes:**")
            for o in outs:
                o_str = str(o).strip()
                if o_str:
                    parts.append(f"- {o_str}")
                    anchors.append(o_str)

        lessons = module.get("lessons") or []
        if lessons:
            parts.append("**Lessons:**")
            for l in sorted(lessons, key=lambda x: (x.get("order") or 0, x.get("title") or "")):
                if l.get("title"):           anchors.append(str(l["title"]).strip())
                if l.get("focus"):           anchors.append(str(l["focus"]).strip())
                if l.get("content_preview"): anchors.append(str(l["content_preview"]).strip())
                parts.append(_text_from_lesson(l))

        # Deduplicate short anchors only
        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()
        uniq: List[str] = []
        seen = set()
        for a in anchors:
            a2 = _norm(a)
            if a2 and len(a2) <= 180 and a2.lower() not in seen:
                uniq.append(a2); seen.add(a2.lower())

        md = "\n\n".join(parts).strip()
        return md[:cap], uniq[:120], module

    def _module_context_sig(struct: Dict[str, Any], module_index: int) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """Returns (sig, context_md, anchors, module_obj)."""
        context_md, anchors, module_obj = _module_context_text(struct, module_index)
        basis = json.dumps({"context_md": context_md, "anchors": anchors}, ensure_ascii=False)
        sig = hashlib.sha256(basis.encode("utf-8")).hexdigest()
        return sig, context_md, anchors, module_obj

    # ------------------------------- sidebar & nav ----------------------------
    def _lesson_uid_of(lesson: Dict[str, Any]) -> Optional[str]:
        for k in ("lesson_uid","uid","id"):
            v = lesson.get(k)
            if v:
                return str(v)
        return None

    def _lesson_url(course_id: int, lesson_uid: str) -> str:
        try:
            return url_for("learn_lesson", course_id=course_id, lesson_uid=lesson_uid)
        except Exception:
            # Fallback to /learn/<course_id>/<uid>
            return f"{url_prefix}/{course_id}/{lesson_uid}"

    def _sidebar_sections(course_id: int, struct: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Builds Learn-like sidebar data structure for template."""
        modules = _list_modules(struct)
        out: List[Dict[str, Any]] = []
        for i, m in enumerate(modules, start=1):
            title = (m.get("title") or m.get("module") or f"Module {i}")
            items: List[Dict[str, Any]] = []
            lessons = sorted((m.get("lessons") or []), key=lambda x: (x.get("order") or 0, x.get("title") or ""))
            for l in lessons:
                uid = _lesson_uid_of(l)
                items.append({
                    "title": l.get("title") or "Lesson",
                    "href": (_lesson_url(course_id, uid) if uid else None)
                })
            exam_href = url_for(f"{bp.name}.exam_start_or_resume", course_id=course_id, week_index=i)  # back-compat path
            status_url = url_for(f"{bp.name}.exam_status", course_id=course_id, week_index=i)
            out.append({
                "index": i,
                "title": title,
                "lessons": items,
                "exam_href": exam_href,
                "status_url": status_url
            })
        return out

    def _nav_urls(course_id: int, struct: Dict[str, Any], module_index: int) -> Tuple[Optional[str], Optional[str], str]:
        """Prev = last lesson of this module; Next = first lesson of NEXT module; back_url = course detail."""
        # back_url
        try:
            back_url = url_for("course_detail", course_id=course_id)
        except Exception:
            back_url = f"/course/{course_id}"

        modules = _list_modules(struct)
        prev_url: Optional[str] = None
        next_url: Optional[str] = None

        # prev: last lesson of current module
        if 1 <= module_index <= len(modules):
            cur = modules[module_index-1] or {}
            lessons = sorted((cur.get("lessons") or []), key=lambda x: (x.get("order") or 0, x.get("title") or ""))
            # last with uid
            for l in reversed(lessons):
                uid = _lesson_uid_of(l)
                if uid:
                    prev_url = _lesson_url(course_id, uid)
                    break

        # next: first lesson of NEXT module
        if module_index + 1 <= len(modules):
            nxt = modules[module_index] or {}
            lessons = sorted((nxt.get("lessons") or []), key=lambda x: (x.get("order") or 0, x.get("title") or ""))
            for l in lessons:
                uid = _lesson_uid_of(l)
                if uid:
                    next_url = _lesson_url(course_id, uid)
                    break
            # fallback: if next module has no lessons, point to its exam
            if not next_url:
                try:
                    next_url = url_for(f"{bp.name}.exam_start_or_resume", course_id=course_id, week_index=(module_index+1))
                except Exception:
                    next_url = f"{url_prefix}/{course_id}/week/{module_index+1}/exam"

        return prev_url, next_url, back_url

    # ------------------------------- OpenAI calls -----------------------------
    def _openai_chat_json(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set (env_variables).")
        import requests
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            return json.loads(m.group(1) if m else content)

    # ---- Question generation (strict, writing-only, anchor-bound) -----------
    def _qgen_from_gpt_module(context_md: str,
                              anchors: List[str],
                              course_title: str,
                              module_title: str,
                              num_questions: int) -> List[Dict[str, Any]]:
        sys = (
            "You are an assessment generator. Create ONLY JSON for SHORT-ANSWER questions "
            "(no multiple choice; no T/F). Questions MUST be grounded in the provided Module context. "
            "For each question select exactly ONE 'source_anchor' from the allowed list AND "
            "INCLUDE that anchor verbatim in the prompt in parentheses at the end, e.g. '(anchor: <anchor>)'."
        )
        anchor_block = "\n".join(f"- {a}" for a in anchors)
        usr = f"""
COURSE: {course_title}
MODULE: {module_title}

MODULE CONTEXT (THIS module only):
---
{context_md}
---

ALLOWED ANCHORS (choose exactly ONE per question; must match line verbatim):
{anchor_block}

REQUIREMENTS:
- Exactly {num_questions} questions.
- Each item:
  {{
    "q_type": "short",
    "prompt_md": "short-answer prompt grounded in context and ending with (anchor: <source_anchor>)",
    "points": number,
    "source_anchor": "EXACT string from ALLOWED ANCHORS",
    "rubric": {{
      "criteria": [
        {{"id":"coverage","desc":"Addresses key ideas","weight":0.6}},
        {{"id":"clarity","desc":"Clear, concise writing","weight":0.4}}
      ]
    }}
  }}
- Stay within this module; DO NOT ask about topics absent from context.
- No yes/no prompts; target answers 2–5 sentences.

Return ONLY JSON:
{{
  "questions":[
    {{
      "q_type":"short",
      "prompt_md":"... (anchor: <verbatim anchor>)",
      "points":20,
      "source_anchor":"<verbatim anchor>",
      "rubric":{{"criteria":[...]}}
    }}
  ]
}}
"""
        data = _openai_chat_json(
            [{"role":"system","content":sys},
             {"role":"user","content":usr}],
            model=OPENAI_QGEN_MODEL, temperature=0.15, max_tokens=1800
        )
        raw = (data or {}).get("questions") or []
        qs: List[Dict[str, Any]] = []
        for q in raw[:num_questions]:
            qs.append({
                "q_type": "short",
                "prompt_md": str(q.get("prompt_md") or "").strip(),
                "points": float(q.get("points") or 20.0),
                "source_anchor": str(q.get("source_anchor") or "").strip(),
                "rubric": q.get("rubric") or {
                    "criteria":[
                        {"id":"coverage","desc":"Addresses key ideas","weight":0.6},
                        {"id":"clarity","desc":"Clear and concise writing","weight":0.4}
                    ]
                }
            })
        # rescale to 100
        s = sum(float(x["points"]) for x in qs) or 1.0
        f = 100.0 / s
        for x in qs:
            x["points"] = round(float(x["points"]) * f, 2)
        return qs

    def _validate_qs(qs: List[Dict[str, Any]], context_md: str, anchors: List[str]) -> List[str]:
        errs: List[str] = []
        ctx_lc = context_md.lower()
        set_anchors = set(a.strip() for a in anchors)
        for i, q in enumerate(qs, start=1):
            if (q.get("q_type") or "").lower() != "short":
                errs.append(f"Q{i}: q_type must be 'short'")
            prompt = (q.get("prompt_md") or "").strip()
            if not prompt:
                errs.append(f"Q{i}: missing prompt_md")
            sa = (q.get("source_anchor") or "").strip()
            if not sa:
                errs.append(f"Q{i}: missing source_anchor")
            elif sa not in set_anchors:
                errs.append(f"Q{i}: source_anchor not in allowed anchors")
            else:
                if sa.lower() not in ctx_lc:
                    errs.append(f"Q{i}: source_anchor not present in module context")
                if sa not in prompt:
                    errs.append(f"Q{i}: prompt_md does not include source_anchor verbatim")
        return errs

    def _generate_validated_questions(struct: Dict[str, Any],
                                      course_title: str,
                                      module_index: int,
                                      num_questions: int) -> Tuple[List[Dict[str, Any]], str, str]:
        """Return (questions, context_md, context_sig)."""
        sig, context_md, anchors, module_obj = _module_context_sig(struct, module_index)
        if not context_md:
            raise RuntimeError("This module has no content.")
        if not EXAMS_USE_GPT:
            raise RuntimeError("EXAMS_USE_GPT disabled.")

        module_title = (module_obj.get("title") or module_obj.get("module") or f"Module {module_index}")

        qs = _qgen_from_gpt_module(context_md, anchors, course_title, module_title, num_questions)
        errs = _validate_qs(qs, context_md, anchors)
        if not errs:
            return qs, context_md, sig

        # second try with feedback
        sys = "Fix the questions according to the validation errors. Return ONLY JSON with the same schema."
        fix_user = f"""
MODULE: {module_title}

CONTEXT:
---
{context_md}
---

ALLOWED ANCHORS:
{chr(10).join("- " + a for a in anchors)}

VALIDATION ERRORS:
{chr(10).join("- " + e for e in errs)}

REQUIREMENTS:
- Exactly {num_questions} short-answer items.
- Each includes its source_anchor verbatim in the prompt (e.g., "(anchor: <anchor>)").
"""
        data = _openai_chat_json(
            [{"role":"system","content":sys},{"role":"user","content":fix_user}],
            model=OPENAI_QGEN_MODEL, temperature=0.0, max_tokens=1800
        )
        raw2 = (data or {}).get("questions") or []
        qs2: List[Dict[str, Any]] = []
        for q in raw2[:num_questions]:
            qs2.append({
                "q_type":"short",
                "prompt_md": str(q.get("prompt_md") or "").strip(),
                "points": float(q.get("points") or 20.0),
                "source_anchor": str(q.get("source_anchor") or "").strip(),
                "rubric": q.get("rubric") or {
                    "criteria":[
                        {"id":"coverage","desc":"Addresses key ideas","weight":0.6},
                        {"id":"clarity","desc":"Clear and concise writing","weight":0.4}
                    ]
                }
            })
        s = sum(float(x["points"]) for x in qs2) or 1.0
        f = 100.0 / s
        for x in qs2:
            x["points"] = round(float(x["points"]) * f, 2)

        errs2 = _validate_qs(qs2, context_md, anchors)
        if not errs2:
            return qs2, context_md, sig

        # deterministic fallback tied to module
        derived: List[Dict[str, Any]] = []
        src = anchors[:] or [module_title]
        bases: List[str] = []
        outs = [str(o) for o in (module_obj.get("outcomes") or []) if str(o).strip()]
        for o in outs:
            bases.append(f"In 3–6 sentences, explain: {o} (anchor: {o})")
        for l in (module_obj.get("lessons") or []):
            lt = str(l.get("title") or "").strip()
            if lt:
                bases.append(f"Summarize key ideas from '{lt}' in 3–6 sentences (anchor: {lt})")
        while len(bases) < num_questions:
            bases.append(f"Describe a core concept from '{module_title}' in 3–6 sentences (anchor: {module_title})")
        for i in range(num_questions):
            a = src[i % len(src)]
            derived.append({
                "q_type":"short",
                "prompt_md": bases[i],
                "points": round(100.0/num_questions, 2),
                "source_anchor": a,
                "rubric": {"criteria":[
                    {"id":"coverage","desc":"Addresses key ideas","weight":0.6},
                    {"id":"clarity","desc":"Clear and concise writing","weight":0.4}
                ]}
            })
        return derived, context_md, sig

    # ------------------------------- Grading ----------------------------------
    def _grade_short_with_gpt(answer_text: str, rubric: Dict[str, Any], context_md: str, max_points: float) -> Dict[str, Any]:
        sys = (
            "You are a supportive, LENIENT short-answer grader. Grade ONLY using the provided module context and rubric. "
            "Award partial credit generously. Accept synonyms and minor omissions. Ignore grammar/format. "
            "Return JSON {\"points\": number, \"notes\":\"<=300 chars\"}."
        )
        usr = f"""
MAX_POINTS: {max_points}

RUBRIC:
{json.dumps(rubric, ensure_ascii=False)}

MODULE CONTEXT:
---
{context_md}
---

STUDENT ANSWER:
---
{(answer_text or '').strip()}
---
"""
        data = _openai_chat_json(
            [{"role":"system","content":sys}, {"role":"user","content":usr}],
            model=OPENAI_GRADER_MODEL, temperature=0.0, max_tokens=400
        )
        pts = float((data or {}).get("points") or 0.0)
        notes = str((data or {}).get("notes") or "")
        return {"points": max(0.0, min(round(pts, 2), float(max_points))), "notes": notes[:300]}

    # ------------------------------- attempt I/O ------------------------------
    def _attempt_rows(user_id: int, course_id: int, module_index: int) -> List[dict]:
        # Cast payload to jsonb for WHERE and SELECT; cast created_at for correct ordering
        return fetch_all("""
            SELECT  id,
                    (created_at::timestamptz) AS created_at,
                    a_type,
                    score_points,
                    passed,
                    (payload::jsonb) AS payload
              FROM  public.activity_log
             WHERE  user_id   = %s
               AND  course_id = %s
               AND  ((payload::jsonb)->>'kind') = 'exam'
               AND  ( ((payload::jsonb)->>'module_index') = %s
                   OR ((payload::jsonb)->>'week_index')   = %s )
             ORDER  BY (created_at::timestamptz) DESC
             LIMIT  400;
        """, (user_id, course_id, str(module_index), str(module_index)))

    def _latest_active_started(rows: List[dict]) -> Optional[dict]:
        graded_uids = set()
        for r in rows:
            p = r.get("payload") or {}
            if p.get("event") == "graded":
                graded_uids.add(p.get("attempt_uid"))
        for r in rows:
            p = r.get("payload") or {}
            if p.get("event") == "started" and p.get("attempt_uid") not in graded_uids:
                return r
        return None

    def _existing_grade(rows: List[dict], attempt_uid: str) -> Optional[dict]:
        for r in rows:
            p = r.get("payload") or {}
            if p.get("event") == "graded" and p.get("attempt_uid") == attempt_uid:
                return {
                    "total_points": float(p.get("total_points") or 0.0),
                    "awarded_points": float(p.get("awarded_points") or 0.0),
                    "score_percent": float(p.get("score_percent") or 0.0),
                    "passed": bool(p.get("passed")),
                    "breakdown": p.get("breakdown") or []
                }
        return None

    def _saved_answers(rows: List[dict], attempt_uid: str) -> Dict[str, Any]:
        for r in rows:
            p = r.get("payload") or {}
            if p.get("event") == "saved" and p.get("attempt_uid") == attempt_uid:
                return p.get("answers") or {}
        return {}

    def _submission_count(user_id: int, course_id: int, module_index: int) -> int:
        row = fetch_one("""
            SELECT COUNT(DISTINCT ((payload::jsonb)->>'attempt_uid')) AS n
              FROM public.activity_log
             WHERE user_id   = %s
               AND course_id = %s
               AND ((payload::jsonb)->>'kind') = 'exam'
               AND ( ((payload::jsonb)->>'module_index') = %s
                   OR ((payload::jsonb)->>'week_index')   = %s )
               AND ((payload::jsonb)->>'event') = 'submitted';
        """, (user_id, course_id, str(module_index), str(module_index)))
        return int((row or {}).get("n") or 0)

    # ------------------------------- text clamps ------------------------------
    def _clamp_text(s: str, limit: int) -> str:
        s = (s or "")
        return s[:max(0, int(limit))]

    def _clamp_answers(answers: Dict[str, Any], limit: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (answers or {}).items():
            txt = str(v.get("text") if isinstance(v, dict) else v or "")
            out[str(k)] = {"text": _clamp_text(txt, limit)}
        return out

    def _answers_fingerprint(answers: Dict[str, Any]) -> str:
        norm = json.dumps(answers, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    # ------------------------------- rendering --------------------------------
    def _render_exam_page(context: Dict[str, Any]):
        """Try templates/exam.html first; else render full Learn layout inline (with sidebar + nav)."""
        try:
            return render_template("exam.html", **context)
        except Exception:
            inline = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>{{ course.title if course else 'Module ' ~ module_index }} · Exam</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<style>
  :root{--ink:#111827;--muted:#6b7280;--line:#e5e7eb}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;line-height:1.55;color:var(--ink)}
  a{color:inherit;text-decoration:none}
  .learn-layout{display:grid;grid-template-columns:280px 1fr;min-height:100vh}
  .learn-sidebar{border-right:1px solid var(--line);background:#fff}
  .sidebar-inner{padding:18px}
  .sidebar-title{font-weight:700;margin:6px 0 12px}
  .sidebar-sections{list-style:none;margin:0;padding:0}
  .sidebar-sections>li{margin-bottom:14px}
  .section-title{font-weight:600;margin:6px 0}
  .sidebar-lessons{list-style:none;margin:6px 0 0;padding:0}
  .lesson-link{display:block;padding:8px 10px;border-radius:8px;border:1px solid transparent}
  .lesson-link:hover{border-color:var(--line);background:#f9fafb}
  .lesson-link.locked{opacity:.6;cursor:not-allowed}
  .exam-link{display:flex;align-items:center;gap:8px;margin-top:6px}
  .exam-chip{font-size:11px;line-height:1;border:1px solid #d1d5db;border-radius:999px;padding:4px 8px;color:#374151;background:#f9fafb}
  .exam-chip.ok{border-color:#10b981;color:#065f46;background:#ecfdf5}
  .exam-chip.warn{border-color:#f59e0b;color:#92400e;background:#fffbeb}
  .exam-chip.muted{opacity:.7}
  .learn-content{padding:24px}
  .card{border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0;background:#fff}
  .btn{display:inline-block;padding:10px 16px;border-radius:8px;background:#111827;color:#fff;text-decoration:none}
  .btn.ghost{background:#fff;color:#111827;border:1px solid #111827}
  .muted{color:var(--muted);font-size:12px}
  textarea{width:100%}
  .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid #d1d5db;background:#f9fafb}
  .badge.ok{border-color:#10b981;background:#ecfdf5;color:#065f46}
  .badge.fail{border-color:#ef4444;background:#fef2f2;color:#991b1b}
  .learn-header h1{margin:0 0 8px}
  .learn-nav{display:flex;justify-content:space-between;align-items:center;margin-top:16px}
</style>
</head>
<body>
<div class="learn-layout">
  <aside class="learn-sidebar">
    <div class="sidebar-inner">
      {% if course %}
        <a class="back" href="{{ back_url or '#' }}">← Back to course</a>
        <h2 class="sidebar-title">{{ course.title }}</h2>
      {% endif %}
      <ol class="sidebar-sections">
        {% for s in sidebar_sections %}
          <li>
            <div class="section-title">{{ s.title or ('Module ' ~ s.index) }}</div>
            <ul class="sidebar-lessons">
              {% for item in s.lessons %}
                <li>
                  {% if item.href %}
                    <a class="lesson-link" href="{{ item.href }}">{{ item.title }}</a>
                  {% else %}
                    <span class="lesson-link locked" aria-disabled="true">🔒 {{ item.title }}</span>
                  {% endif %}
                </li>
              {% endfor %}
              <li>
                <a class="lesson-link exam-link"
                   href="{{ s.exam_href }}"
                   data-status-url="{{ s.status_url }}"
                   data-module="{{ s.index }}">
                   📝 Module {{ s.index }} Exam
                   <span class="exam-chip muted" aria-live="polite">checking…</span>
                </a>
              </li>
            </ul>
          </li>
        {% endfor %}
      </ol>
    </div>
  </aside>

  <section class="learn-content">
    <div class="learn-header">
      <h1>Module {{ module_index }} · Exam</h1>
      <div class="muted">Time limit: {{ cfg.time_limit_min }} min · Pass: {{ cfg.pass_score }}% · Attempts used: {{ submissions_used }}/{{ max_submissions }}</div>
      {% if attempt_uid %}<div class="muted">Attempt ID: {{ attempt_uid }}</div>{% endif %}
    </div>

    <div class="learn-body">
      {% if error_msg %}
        <div class="card" style="color:#b91c1c">{{ error_msg }}</div>
      {% else %}
        <div id="exam">
          {% for q in questions %}
            <div class="card">
              <div><strong>Q{{ loop.index }}</strong> <span class="muted">({{ q.points }} pts)</span></div>
              <div class="prose" style="margin:6px 0 8px">{{ q.prompt_md }}</div>
              <textarea name="t{{ loop.index0 }}" rows="6" maxlength="{{ char_limit }}"
                placeholder="Type your answer (max {{ char_limit }} chars)…">{{ (saved_answers.get(loop.index ~ '') or {}).get('text','') }}</textarea>
              <div class="muted"><span id="ch-{{ loop.index }}">{{ ((saved_answers.get(loop.index ~ '') or {}).get('text','')|length) }}</span> / {{ char_limit }}</div>
            </div>
          {% endfor %}
        </div>
        <div style="margin-top:18px">
          <a href="#" class="btn" id="submit-btn">Submit</a>
          <a href="#" class="btn ghost" id="save-btn">Save draft</a>
          <span id="result" class="muted" style="margin-left:8px"></span>
          <div id="final" class="card" style="display:none;margin-top:12px">
            <div><strong>Result:</strong> <span id="score"></span> <span id="pass" class="badge" style="margin-left:6px"></span></div>
            <div id="breakdown" class="muted" style="margin-top:6px"></div>
          </div>
        </div>
      {% endif %}
    </div>

    <div class="learn-nav">
      {% if prev_url %}
        <a class="btn ghost" rel="prev" href="{{ prev_url }}">← Previous</a>
      {% else %}
        <span></span>
      {% endif %}
      {% if next_url %}
        <a class="btn" rel="next" id="next-btn" href="{{ next_url }}">Next →</a>
      {% else %}
        <a class="btn ghost" href="{{ back_url or '#' }}">Back to course</a>
      {% endif %}
    </div>
  </section>
</div>

<script>
(function(){
  const MODULE_INDEX={{ module_index }};
  const SUBMIT_URL="{{ submit_url }}";
  const SAVE_URL="{{ save_url }}";
  const Q_COUNT={{ questions|length }};
  const LIMIT_USED={{ submissions_used }};
  const LIMIT_MAX={{ max_submissions }};
  const LIMIT_REACHED = (LIMIT_USED >= LIMIT_MAX);
  const CHAR_LIMIT={{ char_limit }};

  // Choice counters & clamp
  for(let i=0;i<Q_COUNT;i++){
    const t=document.querySelector('textarea[name="t'+i+'"]');
    const ch=document.getElementById('ch-'+(i+1));
    if(!t || !ch) continue;
    const upd=()=>{ if(t.value.length>CHAR_LIMIT) t.value=t.value.slice(0,CHAR_LIMIT); ch.textContent=String(t.value.length); };
    t.addEventListener('input', upd); upd();
  }

  function collectAnswers(){
    const ans={};
    for(let i=0;i<Q_COUNT;i++){
      const t=document.querySelector('textarea[name="t'+i+'"]');
      let v=(t&&t.value)?t.value:"";
      if(v.length>CHAR_LIMIT) v=v.slice(0,CHAR_LIMIT);
      ans[String(i+1)]={text:v};
    }
    return ans;
  }

  async function saveDraft(){
    try{
      const payload={module_index:MODULE_INDEX,answers:collectAnswers(),progress_percent:0};
      const r=await fetch(SAVE_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
      const j=await r.json();
      document.getElementById('result').textContent = j.ok ? "Draft saved." : (j.error||"Save failed");
    }catch(e){
      document.getElementById('result').textContent="Save error.";
    }
  }
  document.getElementById('save-btn').addEventListener('click', function(e){ e.preventDefault(); saveDraft(); });

  async function submitExam(){
    if(LIMIT_REACHED){
      document.getElementById('result').textContent="Attempt limit reached. You cannot submit.";
      return;
    }
    document.getElementById('result').textContent="Grading…";
    try{
      const payload={module_index:MODULE_INDEX,answers:collectAnswers()};
      const r=await fetch(SUBMIT_URL,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
      const j=await r.json();
      if(!j.ok) throw new Error(j.error||"Error");
      document.getElementById('result').textContent="";
      const final=document.getElementById('final'); final.style.display='';
      document.getElementById('score').textContent=j.score_percent+"%";
      const badge=document.getElementById('pass');
      badge.className="badge "+(j.passed?"ok":"fail");
      badge.textContent=j.passed?"Passed":"Not passed";
      const bd=document.getElementById('breakdown'); bd.innerHTML='';
      (j.breakdown||[]).forEach(function(row){
        const d=document.createElement('div');
        d.textContent="Q"+row.q_index+": "+(row.points_awarded||0)+" / "+(row.points||0)+" pts";
        bd.appendChild(d);
        if(row.notes){
          const n=document.createElement('div'); n.style.fontSize='12px'; n.textContent=row.notes;
          bd.appendChild(n);
        }
      });
    }catch(e){
      document.getElementById('result').textContent="Submit error: "+e.message;
    }
  }
  document.getElementById('submit-btn').addEventListener('click', function(e){ e.preventDefault(); submitExam(); });

  // Keyboard navigation (matches Learn)
  document.addEventListener('keydown', function(e){
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) return;
    if (e.key === 'ArrowRight') {
      var n = document.querySelector('a#next-btn[href]');
      if (n) { window.location.href = n.getAttribute('href'); }
    } else if (e.key === 'ArrowLeft') {
      var p = document.querySelector('.learn-nav a[rel="prev"][href]');
      if (p) { window.location.href = p.getAttribute('href'); }
    }
  });

  // Exam status chips in sidebar
  document.querySelectorAll('.exam-link').forEach(function(a){
    var chip = a.querySelector('.exam-chip');
    var url  = a.getAttribute('data-status-url');
    if (!chip || !url) return;
    fetch(url, {headers: {'Accept':'application/json'}})
      .then(function(r){ return r.json(); })
      .then(function(j){
        if (!j || !j.ok) throw new Error();
        if (!j.enabled) { chip.textContent = 'disabled'; chip.className = 'exam-chip muted'; return; }
        if (j.state === 'graded' && j.result && typeof j.result.score_percent !== 'undefined') {
          chip.textContent = 'score ' + j.result.score_percent + '%';
          chip.className = 'exam-chip ok';
        } else if (j.state === 'started') {
          chip.textContent = 'resume';
          chip.className = 'exam-chip warn';
        } else {
          chip.textContent = 'start';
          chip.className = 'exam-chip';
        }
      })
      .catch(function(){
        chip.textContent = '—';
        chip.className = 'exam-chip muted';
      });
  });
})();
</script>
</body></html>
"""
            return render_template_string(inline, **context)

    def _render_exam_error(module_index: int, msg: str, course: Optional[Dict[str, Any]] = None,
                           sidebar_sections: Optional[List[Dict[str, Any]]] = None,
                           nav: Tuple[Optional[str], Optional[str], str] = (None, None, "#")):
        prev_url, next_url, back_url = nav
        ctx = {
            "course": course or {"title":"Course"},
            "course_id": None,
            "module_index": module_index,
            "attempt_uid": None,
            "questions": [],
            "cfg": {"time_limit_min": DEFAULT_TIME_LIMIT_MIN, "pass_score": DEFAULT_PASS_SCORE},
            "saved_answers": {},
            "submit_url": "#",
            "save_url": "#",
            "error_msg": msg,
            "submissions_used": 0,
            "max_submissions": MAX_SUBMISSIONS,
            "char_limit": ANSWER_CHAR_LIMIT,
            "sidebar_sections": sidebar_sections or [],
            "prev_url": prev_url,
            "next_url": next_url,
            "back_url": back_url,
        }
        return _render_exam_page(ctx)

    # --------------------------------- routes ---------------------------------
    # Status — module path
    @bp.get("/<int:course_id>/module/<int:module_index>/exam/status")
    def exam_status_module(course_id: int, module_index: int):
        return _exam_status(course_id, module_index)

    # Status — back-compat alias used by learn.html
    @bp.get("/<int:course_id>/week/<int:week_index>/exam/status", endpoint="exam_status")
    def exam_status_week(course_id: int, week_index: int):
        return _exam_status(course_id, week_index)

    def _exam_status(course_id: int, module_index: int):
        if not getattr(g, "user_id", None):
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        c = fetch_one("SELECT id, title, structure FROM public.courses WHERE id = %s;", (course_id,))
        if not c:
            return jsonify({"ok": False, "error": "course not found"}), 404
        st = ensure_structure(c.get("structure"))

        ctx_sig, _ctx_md, _anchors, _ = _module_context_sig(st, module_index)
        rows = _attempt_rows(g.user_id, course_id, module_index)
        submissions_used = _submission_count(g.user_id, course_id, module_index)

        # Find valid active start (matching signature + version + module)
        state, attempt_uid, result = "none", None, None
        started = _latest_active_started(rows)
        if started:
            p = started.get("payload") or {}
            if p.get("context_sig") == ctx_sig and p.get("qgen_version") == QGEN_VERSION and int(p.get("module_index") or 0) == int(module_index):
                state = "started"
                attempt_uid = p.get("attempt_uid")

        if state == "none":
            for r in rows:
                p = r.get("payload") or {}
                if p.get("event") == "graded":
                    state = "graded"
                    attempt_uid = p.get("attempt_uid")
                    result = {"score_percent": p.get("score_percent"),
                              "passed": p.get("passed"),
                              "breakdown": p.get("breakdown") or []}
                    break

        return jsonify({
            "ok": True,
            "enabled": True,
            "state": state,
            "attempt_uid": attempt_uid,
            "cfg": {
                "time_limit_min": DEFAULT_TIME_LIMIT_MIN,
                "pass_score": DEFAULT_PASS_SCORE,
                "num_questions": DEFAULT_Q_COUNT,
                "max_submissions": MAX_SUBMISSIONS,
                "submissions_used": submissions_used,
                "char_limit": ANSWER_CHAR_LIMIT
            },
            "result": result
        })

    # Start/resume — module path
    @bp.get("/<int:course_id>/module/<int:module_index>/exam")
    def exam_start_or_resume_module(course_id: int, module_index: int):
        return _start_or_resume(course_id, module_index)

    # Start/resume — back-compat alias used by learn.html
    @bp.get("/<int:course_id>/week/<int:week_index>/exam", endpoint="exam_start_or_resume")
    def exam_start_or_resume_week(course_id: int, week_index: int):
        return _start_or_resume(course_id, week_index)

    def _start_or_resume(course_id: int, module_index: int):
        if not getattr(g, "user_id", None):
            return redirect(url_for("index"))

        row = fetch_one("SELECT id, title, structure FROM public.courses WHERE id = %s;", (course_id,))
        if not row:
            return redirect(url_for("course_detail", course_id=course_id))

        st = ensure_structure(row.get("structure"))
        modules = _list_modules(st)
        if not modules or not (1 <= module_index <= len(modules)):
            sections = _sidebar_sections(course_id, st)
            nav = _nav_urls(course_id, st, module_index)
            return _render_exam_error(module_index, "Invalid module index.", {"title": row.get("title","Course")}, sections, nav)

        # Sidebar + nav context (always present on exam page)
        sections = _sidebar_sections(course_id, st)
        prev_url, next_url, back_url = _nav_urls(course_id, st, module_index)

        ctx_sig, _ctx_md, _anchors, _module_obj = _module_context_sig(st, module_index)

        rows = _attempt_rows(g.user_id, course_id, module_index)
        submissions_used = _submission_count(g.user_id, course_id, module_index)

        # Resume only if signature/version/module matches; else invalidate and start fresh
        started = _latest_active_started(rows)
        if started:
            p = started.get("payload") or {}
            ok = (p.get("context_sig") == ctx_sig and
                  p.get("qgen_version") == QGEN_VERSION and
                  int(p.get("module_index") or 0) == int(module_index))
            if ok:
                attempt_uid = p["attempt_uid"]
                answers = _saved_answers(rows, attempt_uid)
                return _render_exam_page({
                    "course": {"id": course_id, "title": row.get("title","Course")},
                    "course_id": course_id,
                    "module_index": module_index,
                    "attempt_uid": attempt_uid,
                    "questions": p["questions"],
                    "cfg": {"time_limit_min": p.get("time_limit_min", DEFAULT_TIME_LIMIT_MIN),
                            "pass_score": p.get("blueprint", {}).get("pass_score", DEFAULT_PASS_SCORE)},
                    "saved_answers": answers,
                    "submit_url": url_for(f"{bp.name}.exam_submit_and_grade", course_id=course_id, attempt_uid=attempt_uid),
                    "save_url": url_for(f"{bp.name}.exam_save", course_id=course_id, attempt_uid=attempt_uid),
                    "error_msg": None,
                    "submissions_used": submissions_used,
                    "max_submissions": MAX_SUBMISSIONS,
                    "char_limit": ANSWER_CHAR_LIMIT,
                    "sidebar_sections": sections,
                    "prev_url": prev_url,
                    "next_url": next_url,
                    "back_url": back_url,
                })
            else:
                _log_exam_activity(
                    g.user_id, course_id, module_index, "invalidated",
                    p.get("attempt_uid") or uuid.uuid4().hex,
                    extra_payload={
                        "reason": "context/version mismatch",
                        "was_qgen_version": p.get("qgen_version"),
                        "was_context_sig": p.get("context_sig")
                    }
                )

        # New attempt (respect cap)
        if submissions_used >= MAX_SUBMISSIONS:
            return _render_exam_error(
                module_index,
                f"Attempt limit reached ({MAX_SUBMISSIONS}). You cannot start another submission.",
                {"title": row.get("title","Course")}, sections, (prev_url, next_url, back_url)
            )

        if not EXAMS_USE_GPT:
            return _render_exam_error(module_index, "EXAMS_USE_GPT is disabled.",
                                      {"title": row.get("title","Course")}, sections, (prev_url, next_url, back_url))
        if not OPENAI_API_KEY:
            return _render_exam_error(module_index, "OPENAI_API_KEY missing.",
                                      {"title": row.get("title","Course")}, sections, (prev_url, next_url, back_url))

        try:
            questions, ctx_md, sig_now = _generate_validated_questions(
                struct=st,
                course_title=(row.get("title") or f"Course {course_id}"),
                module_index=module_index,
                num_questions=DEFAULT_Q_COUNT
            )
        except Exception as e:
            return _render_exam_error(
                module_index, f"Question generation failed: {e}",
                {"title": row.get("title","Course")}, sections, (prev_url, next_url, back_url)
            )

        attempt_uid = uuid.uuid4().hex
        _log_exam_activity(
            g.user_id, course_id, module_index,
            event="started", attempt_uid=attempt_uid,
            extra_payload={
                "qgen_version": QGEN_VERSION,
                "context_sig": sig_now,
                "questions": questions,
                "time_limit_min": DEFAULT_TIME_LIMIT_MIN,
                "blueprint": {"num_questions": DEFAULT_Q_COUNT, "pass_score": DEFAULT_PASS_SCORE},
                "context_snapshot": ctx_md[:4000],
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )

        return _render_exam_page({
            "course": {"id": course_id, "title": row.get("title","Course")},
            "course_id": course_id,
            "module_index": module_index,
            "attempt_uid": attempt_uid,
            "questions": questions,
            "cfg": {"time_limit_min": DEFAULT_TIME_LIMIT_MIN, "pass_score": DEFAULT_PASS_SCORE},
            "saved_answers": {},
            "submit_url": url_for(f"{bp.name}.exam_submit_and_grade", course_id=course_id, attempt_uid=attempt_uid),
            "save_url": url_for(f"{bp.name}.exam_save", course_id=course_id, attempt_uid=attempt_uid),
            "error_msg": None,
            "submissions_used": submissions_used,
            "max_submissions": MAX_SUBMISSIONS,
            "char_limit": ANSWER_CHAR_LIMIT,
            "sidebar_sections": sections,
            "prev_url": prev_url,
            "next_url": next_url,
            "back_url": back_url,
        })

    @bp.post("/<int:course_id>/exam/<attempt_uid>/save")
    def exam_save(course_id: int, attempt_uid: str):
        if not getattr(g, "user_id", None):
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        data = request.get_json(force=True) or {}
        module_index = int(data.get("module_index") or data.get("week_index") or 1)
        answers = _clamp_answers(data.get("answers") or {}, ANSWER_CHAR_LIMIT)
        progress = float(data.get("progress_percent") or 0.0)
        _log_exam_activity(
            g.user_id, course_id, module_index,
            event="saved", attempt_uid=attempt_uid,
            extra_payload={"answers": answers, "progress_percent": progress, "answers_fingerprint": _answers_fingerprint(answers)}
        )
        return jsonify({"ok": True})

    @bp.post("/<int:course_id>/exam/<attempt_uid>/submit")
    def exam_submit_and_grade(course_id: int, attempt_uid: str):
        if not getattr(g, "user_id", None):
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        data = request.get_json(force=True) or {}
        module_index = int(data.get("module_index") or data.get("week_index") or 1)
        answers = _clamp_answers(data.get("answers") or {}, ANSWER_CHAR_LIMIT)

        rows = _attempt_rows(g.user_id, course_id, module_index)
        started = _latest_active_started(rows)
        if not started or (started.get("payload") or {}).get("attempt_uid") != attempt_uid:
            prior = _existing_grade(rows, attempt_uid)
            if prior:
                return jsonify({"ok": True,
                                "score_percent": round(float(prior["score_percent"]), 2),
                                "passed": bool(prior["passed"]),
                                "breakdown": prior["breakdown"]})
            return jsonify({"ok": False, "error": "attempt not found or not active"}), 400

        # Idempotency
        prior = _existing_grade(rows, attempt_uid)
        if prior:
            return jsonify({"ok": True,
                            "score_percent": round(float(prior["score_percent"]), 2),
                            "passed": bool(prior["passed"]),
                            "breakdown": prior["breakdown"]})

        # Hard cap (counts only submitted)
        used = _submission_count(g.user_id, course_id, module_index)
        if used >= MAX_SUBMISSIONS:
            return jsonify({"ok": False, "error": f"submission limit reached ({MAX_SUBMISSIONS})"}), 403

        sp = started["payload"]
        questions = sp["questions"]
        pass_score = int(((sp.get("blueprint") or {}).get("pass_score")) or DEFAULT_PASS_SCORE)
        time_limit_min = int(sp.get("time_limit_min") or DEFAULT_TIME_LIMIT_MIN)
        started_at_iso = str(sp.get("started_at") or "")

        if ENFORCE_TIME_LIMIT and started_at_iso:
            try:
                t = time.strptime(started_at_iso, "%Y-%m-%dT%H:%M:%SZ")
                started_epoch = int(time.mktime(t))
                now_epoch = int(time.time())
                if now_epoch > started_epoch + time_limit_min*60:
                    return jsonify({"ok": False, "error": "time limit exceeded"}), 400
            except Exception:
                pass

        # Consume attempt
        _log_exam_activity(
            g.user_id, course_id, module_index,
            event="submitted", attempt_uid=attempt_uid,
            extra_payload={"answers": answers, "answers_fingerprint": _answers_fingerprint(answers)}
        )

        # Grade with module context only
        c = fetch_one("SELECT structure FROM public.courses WHERE id = %s;", (course_id,))
        st = ensure_structure(c.get("structure"))
        _sig, ctx_md, _anchors, _ = _module_context_sig(st, module_index)

        total_pts, awarded = 0.0, 0.0
        breakdown: List[Dict[str, Any]] = []

        for idx, q in enumerate(questions, start=1):
            pts_max = float(q.get("points") or 0.0)
            total_pts += pts_max
            ans = answers.get(str(idx)) or answers.get(idx) or {}
            answer_text = (ans.get("text") or "").strip()
            rubric = q.get("rubric") or {"criteria":[{"id":"coverage","desc":"Addresses key ideas","weight":1.0}]}
            try:
                gret = _grade_short_with_gpt(answer_text, rubric, ctx_md, pts_max) if EXAMS_USE_GPT else {"points":0.0,"notes":"grading disabled"}
                pts = float(gret.get("points") or 0.0)
                notes = str(gret.get("notes") or "")
            except Exception as e:
                pts, notes = 0.0, f"Grader error: {e}"

            awarded += pts
            breakdown.append({"q_index": idx, "points": pts_max, "points_awarded": round(pts,2), "notes": notes})

        score_percent = round(100.0 * awarded / (total_pts or 1.0), 2)
        passed = bool(score_percent >= pass_score)

        _log_exam_activity(
            g.user_id, course_id, module_index,
            event="graded", attempt_uid=attempt_uid,
            extra_payload={
                "total_points": total_pts,
                "awarded_points": awarded,
                "score_percent": score_percent,
                "passed": passed,
                "breakdown": breakdown
            },
            score_points=int(round(score_percent)),
            passed=passed
        )

        return jsonify({"ok": True, "score_percent": score_percent, "passed": passed, "breakdown": breakdown})

    return bp

# -----------------------------------------------------------------------------#
# Fallback helpers
# -----------------------------------------------------------------------------#
def _ensure_structure_fallback(structure_raw: Any) -> Dict[str, Any]:
    if not structure_raw: return {"modules": []}
    if isinstance(structure_raw, dict): return structure_raw
    try: return json.loads(structure_raw)
    except Exception: return {"modules": []}
