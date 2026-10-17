"""Persisted UI preferences for the local settings GUI.

Visibility, font scale, section expansion, and control-use frequency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from file_lock import safe_json_load, safe_json_save

SCREEN_ID = "settings.main"
DEFAULT_FONT_SCALE = 1.0
MIN_FONT_SCALE = 0.9
MAX_FONT_SCALE = 1.6
FONT_SCALE_STEP = 0.1

# Sections needed for a fresh, general-purpose setup. Keep provider selection,
# feature switches, and user-authored review settings visible without requiring
# users to discover the advanced-settings toggle.
MAIN_PATH_SECTIONS = (
    "Reddit",
    "AI providers",
    "Safety",
    "Features",
    "History review",
)

# Fixed-order exceptions (Application Standard 26).
FIXED_ORDER_REASONS = {
    "sections": "semantic-fixed-group",
    "rule_actions": "domain-fixed-order",
    "done_button": "primary-action-fixed",
}


def default_prefs() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "screenId": SCREEN_ID,
        "hiddenSettingIds": [],
        "customizeMode": False,
        "showAdvanced": False,
        "fontScale": DEFAULT_FONT_SCALE,
        "actionUseCounts": {},
        "expandedSections": list(MAIN_PATH_SECTIONS),
    }


def prefs_path(repo_root: Path) -> Path:
    return repo_root / "data" / "settings_gui_prefs.json"


def load_prefs(repo_root: Path) -> dict[str, Any]:
    path = prefs_path(repo_root)
    raw = safe_json_load(str(path), default_prefs())
    if not isinstance(raw, dict):
        return default_prefs()
    merged = default_prefs()
    merged.update({k: v for k, v in raw.items() if k in merged or k == "schemaVersion"})
    # Repair font scale
    try:
        scale = float(merged.get("fontScale", DEFAULT_FONT_SCALE))
    except (TypeError, ValueError):
        scale = DEFAULT_FONT_SCALE
    merged["fontScale"] = max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, round(scale, 2)))
    if not isinstance(merged.get("hiddenSettingIds"), list):
        merged["hiddenSettingIds"] = []
    if not isinstance(merged.get("actionUseCounts"), dict):
        merged["actionUseCounts"] = {}
    if not isinstance(merged.get("expandedSections"), list):
        merged["expandedSections"] = list(MAIN_PATH_SECTIONS)
    return merged


def save_prefs(repo_root: Path, prefs: dict[str, Any]) -> dict[str, Any]:
    path = prefs_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = default_prefs()
    cleaned.update(prefs)
    cleaned["screenId"] = SCREEN_ID
    safe_json_save(str(path), cleaned)
    return cleaned


def record_action_use(prefs: dict[str, Any], action_id: str) -> dict[str, Any]:
    counts = dict(prefs.get("actionUseCounts") or {})
    counts[action_id] = int(counts.get(action_id, 0)) + 1
    prefs["actionUseCounts"] = counts
    return prefs


def ranked_action_ids(action_ids: list[str], prefs: dict[str, Any], *, fixed: set[str] | None = None) -> list[str]:
    """Frequency-rank eligible actions; keep fixed IDs at their relative positions."""
    fixed = fixed or set()
    counts = prefs.get("actionUseCounts") or {}
    eligible = [a for a in action_ids if a not in fixed]
    eligible.sort(key=lambda a: (-int(counts.get(a, 0)), action_ids.index(a)))
    result: list[str] = []
    eligible_iter = iter(eligible)
    for action_id in action_ids:
        if action_id in fixed:
            result.append(action_id)
        else:
            result.append(next(eligible_iter))
    return result
