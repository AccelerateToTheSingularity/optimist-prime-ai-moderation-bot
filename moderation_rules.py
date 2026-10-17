"""
Discrete rule-based moderation system.

Rules are independent yes/no questions evaluated by the LLM against each
post/comment. Each rule carries its own action, conditions, and target
scope. Rules are loaded from a JSON file with an optional env-var override.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


# These are the actions the Python runtime can execute.  Keeping this list in
# the rule module gives the validator, GUI, and executor one source of truth.
EXECUTABLE_ACTIONS = frozenset({
    "remove",
    "report",
    "modmail",
    "spam",
    "lock",
    "unlock",
    "approve",
    "ban",
    "log",
})


class RuleLoadError(Exception):
    """Raised when rules or operator-authored history criteria are invalid."""


# Kept as an empty compatibility set for older state-processing callers. New
# installations define their own history criteria in a local file.
DISABLED_RULE_IDS: frozenset[str] = frozenset()

MAX_HISTORY_CRITERIA = 30
DEFAULT_HISTORY_CRITERIA_PATH = Path(__file__).resolve().parent / "data" / "history_criteria.json"
_HISTORY_CRITERIA_FIELDS = frozenset({
    "title",
    "description",
    "high_confidence_yes",
    "not_sufficient_by_itself",
    "closest_boundaries",
    "active",
    "revision",
})


def history_criteria_path(path: str | Path | None = None) -> Path:
    """Resolve the operator-owned history criteria file."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("BOT_HISTORY_CRITERIA_FILE", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_HISTORY_CRITERIA_PATH


def load_history_criteria_definitions(
    path: str | Path | None = None,
    *,
    include_inactive: bool = False,
) -> dict[str, dict]:
    """Load validated operator-authored criteria; a missing default is empty."""
    source = history_criteria_path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if path is None and not os.environ.get("BOT_HISTORY_CRITERIA_FILE", "").strip():
            return {}
        raise RuleLoadError(f"History criteria file not found: {source}")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleLoadError(f"History criteria file could not be read: {source}") from exc

    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("guides"), dict)
        or set(payload) - {"schema_version", "guides"}
    ):
        raise RuleLoadError(
            "History criteria must be an object with schema_version 1 and a guides object"
        )
    guides = payload["guides"]
    if len(guides) > MAX_HISTORY_CRITERIA:
        raise RuleLoadError(
            f"History criteria may define at most {MAX_HISTORY_CRITERIA} guides"
        )

    required_text = {
        "title",
        "description",
        "high_confidence_yes",
        "not_sufficient_by_itself",
        "closest_boundaries",
    }
    normalized: dict[str, dict] = {}
    for name, raw_guide in guides.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
            or not isinstance(raw_guide, dict)
            or set(raw_guide) - _HISTORY_CRITERIA_FIELDS
            or not required_text.issubset(raw_guide)
            or any(
                not isinstance(raw_guide[field], str) or not raw_guide[field].strip()
                for field in required_text
            )
        ):
            raise RuleLoadError(
                "Each history guide needs a valid id and non-empty title, description, "
                "high_confidence_yes, not_sufficient_by_itself, and closest_boundaries"
            )
        active = raw_guide.get("active", False)
        revision = raw_guide.get("revision", 1)
        if not isinstance(active, bool):
            raise RuleLoadError(f"History guide '{name}' has a non-boolean active value")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise RuleLoadError(f"History guide '{name}' has an invalid revision")
        normalized[name] = {
            **{field: raw_guide[field].strip() for field in required_text},
            "active": active,
            "revision": revision,
        }

    if include_inactive:
        return normalized
    return {name: guide for name, guide in normalized.items() if guide["active"]}


# A saved score is reusable only while its operator-authored criterion revision matches.
def _load_history_criterion_revisions() -> dict[str, int]:
    try:
        guides = load_history_criteria_definitions(include_inactive=True)
    except RuleLoadError:
        return {}
    return {name: guide["revision"] for name, guide in guides.items()}


HISTORY_CRITERIA_DEFINITION_REVISIONS = _load_history_criterion_revisions()


def history_criterion_score_is_current(name: str, revisions: dict | None) -> bool:
    recorded = revisions.get(name, 1) if isinstance(revisions, dict) else 1
    # Resolve the local catalog at use time. The runner loads .env after its
    # import-safe module setup, so an import-time snapshot can point at the
    # wrong file when BOT_HISTORY_CRITERIA_FILE is supplied there.
    try:
        expected = load_history_criteria_definitions(include_inactive=True).get(name, {}).get("revision")
    except RuleLoadError:
        expected = None
    return expected is not None and not isinstance(recorded, bool) and recorded == expected


@dataclass
class ModerationRule:
    """A single moderation rule evaluated against Reddit content."""

    name: str
    description: str
    active: bool = True
    order: int = 100
    target: str = "both"  # "posts", "comments", or "both"
    actions: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
    flairs: list[str] = field(default_factory=list)
    use_vision: bool = False
    # Optional IDs from the operator's local community-rules reference. These are
    # explanatory source notes only; a local detector does not need an
    # official-rule mapping in order to evaluate content or be eligible for
    # the separately gated history reviewer.
    community_rule_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def stop_on_match(self) -> bool:
        return self.conditions.get("stop_on_match", False)

    @property
    def skip_mods(self) -> bool:
        return self.conditions.get("skip_mods", False)

    @property
    def skip_approved(self) -> bool:
        return self.conditions.get("skip_approved", False)

    def applies_to(self, content_type: str) -> bool:
        """Return True if this rule applies to the given content type."""
        if self.target == "both":
            return True
        return self.target == content_type


def _validate_rule(rule_dict: dict, index: int) -> ModerationRule:
    """Validate a single rule dict and return a ModerationRule."""
    if "name" not in rule_dict:
        raise RuleLoadError(f"Rule at index {index} is missing 'name'")
    if "description" not in rule_dict:
        raise RuleLoadError(f"Rule '{rule_dict.get('name', '?')}' is missing 'description'")

    name = rule_dict["name"]
    if not isinstance(name, str) or not name.strip():
        raise RuleLoadError(f"Rule at index {index} has empty 'name'")
    if " " in name:
        raise RuleLoadError(
            f"Rule name '{name}' contains spaces; use underscores"
        )

    valid_targets = {"posts", "comments", "both"}
    target = rule_dict.get("target", "both")
    if target not in valid_targets:
        raise RuleLoadError(
            f"Rule '{name}' has invalid target '{target}'; "
            f"must be one of {valid_targets}"
        )

    active = rule_dict.get("active", False)
    if not isinstance(active, bool):
        raise RuleLoadError(f"Rule '{name}' has non-boolean 'active'")

    actions = rule_dict.get("actions", [])
    if not isinstance(actions, list):
        raise RuleLoadError(f"Rule '{name}' actions must be a list")
    if active and not actions:
        raise RuleLoadError(
            f"Active rule '{name}' needs at least one action. Choose an action or make the rule inactive."
        )

    for action in actions:
        if action not in EXECUTABLE_ACTIONS:
            raise RuleLoadError(
                f"Rule '{name}' has invalid action '{action}'; "
                f"must be one of {sorted(EXECUTABLE_ACTIONS)}"
            )

    if name in DISABLED_RULE_IDS:
        active = False

    community_rule_ids = rule_dict.get("community_rule_ids", [])
    if not isinstance(community_rule_ids, list) or not all(
        isinstance(value, str) and value.strip() for value in community_rule_ids
    ):
        raise RuleLoadError(
            f"Rule '{name}' must have a list of non-empty 'community_rule_ids'"
        )

    order = rule_dict.get("order", 100)
    if not isinstance(order, int):
        raise RuleLoadError(f"Rule '{name}' has non-integer order")

    return ModerationRule(
        name=name,
        description=rule_dict["description"],
        active=active,
        order=order,
        target=target,
        actions=actions,
        conditions=rule_dict.get("conditions", {}),
        flairs=rule_dict.get("flairs", []),
        use_vision=rule_dict.get("use_vision", False),
        community_rule_ids=tuple(dict.fromkeys(value.strip() for value in community_rule_ids)),
    )


def load_content_rules(path, *, include_local=True):
    """Load only rules supplied by the operator; no built-in policy family exists."""
    rules = load_rules(path) if include_local else []
    if len({r.name for r in rules}) != len(rules):
        raise RuleLoadError("Duplicate detector names")
    return sorted(rules, key=lambda r: r.order)


def load_rules(path: str | Path | None = None) -> list[ModerationRule]:
    """
    Load moderation rules from a JSON file or the env-var override.

    Resolution order:
      1. BOT_MODERATION_RULES_JSON env var (JSON string)
      2. path argument (file path)
      3. data/rules.json (default)

    Returns a list sorted by order (ascending).
    Raises RuleLoadError on validation failure or missing file.
    """
    # 1. Check env var override
    env_json = os.environ.get("BOT_MODERATION_RULES_JSON", "").strip()
    if env_json:
        try:
            raw = json.loads(env_json)
        except json.JSONDecodeError as e:
            raise RuleLoadError(
                f"BOT_MODERATION_RULES_JSON contains invalid JSON: {e}"
            )
        if not isinstance(raw, list):
            raise RuleLoadError(
                "BOT_MODERATION_RULES_JSON must be a JSON array of rule objects"
            )
        rules = [_validate_rule(r, i) for i, r in enumerate(raw)]
        return sorted(rules, key=lambda r: r.order)

    # 2. Resolve file path
    if path is None:
        path = Path("data/rules.json")
    else:
        path = Path(path)

    if not path.exists():
        raise RuleLoadError(f"Rules file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise RuleLoadError(f"Rules file contains invalid JSON: {e}")

    if not isinstance(raw, list):
        raise RuleLoadError("Rules file must contain a JSON array of rule objects")

    rules = [_validate_rule(r, i) for i, r in enumerate(raw)]
    return sorted(rules, key=lambda r: r.order)


def filter_rules(
    rules: list[ModerationRule],
    content_type: str,
    *,
    active_only: bool = True,
) -> list[ModerationRule]:
    """
    Filter rules by content type and active status.

    Args:
        rules: All loaded rules
        content_type: "posts" or "comments"
        active_only: If True, skip inactive rules

    Returns:
        Filtered list sorted by order
    """
    result = []
    for rule in rules:
        if active_only and not rule.active:
            continue
        if not rule.applies_to(content_type):
            continue
        result.append(rule)
    return sorted(result, key=lambda r: r.order)
