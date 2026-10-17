"""Load and safely use the operator's local community-rules reference.

The snapshot is local and deterministic during bot cycles. A network refresh is
an explicit maintenance action, not an implicit side effect of moderation. This
prevents a changed Reddit rule from silently changing the bot's behavior while
keeping the relevant excerpts available to prompts.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = REPO_ROOT / "data" / "community_rules.reference.json"
DEFAULT_MAX_AGE_DAYS = 30
class CommunityRulesError(ValueError):
    """Raised when the tracked rules snapshot is missing or malformed."""


def reference_path(path: str | Path | None = None) -> Path:
    raw = str(path or os.environ.get("BOT_COMMUNITY_RULES_REFERENCE_FILE", "")).strip()
    if not raw:
        return DEFAULT_REFERENCE_PATH
    candidate = Path(os.path.expandvars(os.path.expanduser(raw)))
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _without_fingerprint(reference: dict[str, Any]) -> dict[str, Any]:
    payload = dict(reference)
    payload.pop("content_fingerprint", None)
    return payload


def content_fingerprint(reference: dict[str, Any]) -> str:
    canonical = json.dumps(
        _without_fingerprint(reference),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def load_reference(path: str | Path | None = None) -> dict[str, Any]:
    candidate = reference_path(path)
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CommunityRulesError(f"Community rules snapshot not found: {candidate}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CommunityRulesError(f"Community rules snapshot could not be read: {candidate}") from exc
    if not isinstance(raw, dict):
        raise CommunityRulesError("Community rules snapshot must be a JSON object")
    for key in (
        "subreddit",
        "official_community_rules",
        "reddit_wide_categories",
        "moderator_guidance",
        "moderation_model",
    ):
        if key not in raw:
            raise CommunityRulesError(f"Community rules snapshot is missing '{key}'")
    for key in ("official_community_rules", "reddit_wide_categories", "moderator_guidance"):
        if not isinstance(raw.get(key), list):
            raise CommunityRulesError(f"Community rules snapshot field '{key}' must be a list")
    expected_subreddit = os.environ.get("BOT_SUBREDDIT", "").strip().casefold()
    reference_subreddit = str(raw.get("subreddit", "")).strip().casefold()
    if not expected_subreddit:
        raise CommunityRulesError(
            "BOT_SUBREDDIT must be configured before loading a community-rules snapshot"
        )
    if reference_subreddit != expected_subreddit:
        unbound_empty_template = not reference_subreddit and not any(
            raw[key] for key in (
                "official_community_rules", "reddit_wide_categories", "moderator_guidance"
            )
        )
        if not unbound_empty_template:
            raise CommunityRulesError("Community rules snapshot is for the wrong subreddit")
    if not isinstance(raw.get("moderation_model"), dict):
        raise CommunityRulesError("Community rules snapshot field 'moderation_model' must be an object")
    return raw


def _parse_fetched_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def reference_status(reference: dict[str, Any] | None = None, *, now: datetime | None = None) -> dict[str, Any]:
    """Return diagnostics-safe snapshot status without fetching Reddit."""
    reference = reference or load_reference()
    fetched = _parse_fetched_at(reference.get("fetched_at"))
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_days = None if fetched is None else max(0.0, (current - fetched).total_seconds() / 86400)
    try:
        max_age_days = max(1, int(os.environ.get("BOT_COMMUNITY_RULES_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS)))
    except ValueError:
        max_age_days = DEFAULT_MAX_AGE_DAYS
    stored = str(reference.get("content_fingerprint", ""))
    computed = content_fingerprint(reference)
    fingerprint_matches = bool(stored and stored == computed)
    stale = fetched is None or (age_days is not None and age_days > max_age_days)
    warnings: list[str] = []
    if stale:
        warnings.append("community rules snapshot is stale or has no valid fetch date")
    if not fingerprint_matches:
        warnings.append("community rules snapshot fingerprint does not match its contents")
    return {
        "subreddit": reference.get("subreddit"),
        "sourceUrls": reference.get("source_urls", {}),
        "fetchedAt": reference.get("fetched_at"),
        "redditRevision": reference.get("reddit_revision"),
        "contentFingerprint": stored,
        "computedFingerprint": computed,
        "fingerprintMatches": fingerprint_matches,
        "ageDays": None if age_days is None else round(age_days, 2),
        "maxAgeDays": max_age_days,
        "stale": stale,
        "warnings": warnings,
    }


def _all_rule_entries(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for section in ("official_community_rules", "reddit_wide_categories", "moderator_guidance"):
        for entry in reference.get(section, []):
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                entries[entry["id"]] = entry
    return entries


def rule_ids(reference: dict[str, Any] | None = None) -> set[str]:
    return set(_all_rule_entries(reference or load_reference()))


def _live_rule_value(rule: Any, *names: str) -> str:
    for name in names:
        if isinstance(rule, dict):
            value = rule.get(name)
        else:
            value = getattr(rule, name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def compare_live_rules(
    reference: dict[str, Any],
    live_rules: Iterable[Any],
) -> dict[str, Any]:
    """Compare a read-only live Reddit rule listing with the tracked snapshot.

    This function never updates the snapshot.  A changed result is a warning
    for operator review, so moderation behavior cannot change merely because a
    subreddit rule was edited on Reddit.
    """
    stored = [
        {
            "name": str(entry.get("name", "")).strip(),
            "text": str(entry.get("reddit_text") or entry.get("description", "")).strip(),
        }
        for entry in reference.get("official_community_rules", [])
        if isinstance(entry, dict)
    ]
    current = [
        {
            "name": _live_rule_value(rule, "short_name", "name", "kind"),
            "text": _live_rule_value(rule, "description", "violation_reason", "text"),
        }
        for rule in live_rules
    ]
    stored_normalized = sorted(
        (entry["name"].casefold(), entry["text"].casefold()) for entry in stored
    )
    current_normalized = sorted(
        (entry["name"].casefold(), entry["text"].casefold()) for entry in current
    )
    changed = stored_normalized != current_normalized
    differences: list[str] = []
    if len(stored) != len(current):
        differences.append(f"rule count changed from {len(stored)} to {len(current)}")
    if stored_normalized != current_normalized:
        differences.append("one or more official rule names or descriptions changed")
    return {
        "changed": changed,
        "storedCount": len(stored),
        "liveCount": len(current),
        "differences": differences,
        "storedRules": stored,
        "liveRules": current,
    }


def relevant_context(
    ids: Iterable[str],
    *,
    reference: dict[str, Any] | None = None,
    include_guidance: bool = True,
    max_chars: int = 6000,
) -> str:
    """Build a small, labelled prompt excerpt for the requested source IDs."""
    reference = reference or load_reference()
    entries = _all_rule_entries(reference)
    wanted = [str(value) for value in ids if str(value) in entries]
    if include_guidance:
        wanted.extend(
            entry_id
            for entry_id in entries
            if entry_id.startswith("guidance.") and entry_id not in wanted
        )
    seen: set[str] = set()
    official: list[dict[str, Any]] = []
    guidance: list[dict[str, Any]] = []
    for entry_id in wanted:
        if entry_id in seen:
            continue
        seen.add(entry_id)
        entry = dict(entries[entry_id])
        if entry_id.startswith("guidance."):
            guidance.append(entry)
        else:
            official.append(entry)
    payload = {
        "official_rules": official,
        "moderator_guidance_not_official_rules": guidance,
        "moderation_model": reference.get("moderation_model", {}),
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text[:max(0, int(max_chars))]


def context_for_rules(
    rules: Iterable[Any],
    *,
    reference: dict[str, Any] | None = None,
    max_chars: int = 6000,
) -> str:
    """Return the smallest labelled rules excerpt useful for these rules.

    An empty list deliberately includes the official community rules and
    guidance, because moderation prompts still need the baseline context even
    when local decision guides have no optional source notes.
    """
    reference = reference or load_reference()
    ids = {
        rule_id
        for rule in rules
        for rule_id in (getattr(rule, "community_rule_ids", ()) or ())
    }
    if not ids:
        ids = {
            entry.get("id")
            for section in ("official_community_rules", "reddit_wide_categories")
            for entry in reference.get(section, [])
            if isinstance(entry, dict) and entry.get("id")
        }
    return relevant_context(ids, reference=reference, include_guidance=True, max_chars=max_chars)


__all__ = [
    "CommunityRulesError",
    "DEFAULT_REFERENCE_PATH",
    "content_fingerprint",
    "context_for_rules",
    "load_reference",
    "compare_live_rules",
    "reference_path",
    "reference_status",
    "relevant_context",
    "rule_ids",
]
