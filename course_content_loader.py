"""Utilities for loading the course content structure from disk."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

COURSE_CONTENT_DIR = Path(__file__).resolve().parent / "course"
COURSE_INDEX_PATH = COURSE_CONTENT_DIR / "content_index.json"
COURSE_MODULES_DIR = COURSE_CONTENT_DIR / "modules"


def _safe_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[course_content] failed to load '{path}': {exc}")
        return None


def _sorted_modules(modules: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _sort_key(mod: Dict[str, Any]):
        order = mod.get("order")
        try:
            order_val = int(order)
        except Exception:
            order_val = float("inf")
        title = (mod.get("title") or "").lower()
        return (order_val, title)

    return sorted([dict(m) for m in modules if isinstance(m, dict)], key=_sort_key)


@lru_cache(maxsize=1)
def load_course_content() -> Dict[str, Any]:
    """Load and assemble the course content structure from disk."""

    index_data = _safe_load_json(COURSE_INDEX_PATH)
    if not isinstance(index_data, dict):
        return {"sections": []}

    modules_meta = index_data.pop("modules", []) or []
    sections: List[Dict[str, Any]] = []

    for module_meta in _sorted_modules(modules_meta):
        file_name = module_meta.get("file")
        if not file_name:
            continue
        module_path = COURSE_MODULES_DIR / str(file_name)
        module_data = _safe_load_json(module_path)
        if isinstance(module_data, dict):
            # Ensure required metadata is present
            if "order" not in module_data and module_meta.get("order") is not None:
                try:
                    module_data["order"] = int(module_meta["order"])
                except Exception:
                    module_data["order"] = module_meta["order"]
            if "title" not in module_data and module_meta.get("title"):
                module_data["title"] = module_meta["title"]
            sections.append(module_data)

    index_data["sections"] = sections
    return index_data


__all__ = ["load_course_content"]
