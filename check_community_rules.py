"""Explicit, read-only check for changes to a community's local rules reference.

The normal bot cycle deliberately does not fetch subreddit rules. Run this
maintenance command when an operator wants to compare the local snapshot with
Reddit's current public rule listing. A difference is a warning for human
review; this command never updates the snapshot or local rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from community_rules import CommunityRulesError, compare_live_rules, load_reference, reference_path
from comment_feed import FeedError, create_read_only_reddit
from env_file import clear_inherited_runtime_environment, load_local_env


def _safe_error(exc: BaseException, values: dict[str, str]) -> str:
    """Return a useful error without echoing configured credentials."""
    message = f"{type(exc).__name__}: {exc}"
    for key, value in values.items():
        if value and any(part in key for part in ("SECRET", "TOKEN", "PASSWORD", "KEY")):
            message = message.replace(value, "[redacted]")
    return message


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the configured community's local rules reference with Reddit without changing anything."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="local environment file containing the read-only Reddit credentials (default: .env)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="override the local community-rules snapshot path",
    )
    parser.add_argument(
        "--expected-account",
        default=None,
        help="require Reddit to authenticate as this username",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of a short human-readable report",
    )
    return parser


def _human_report(result: dict[str, Any]) -> str:
    if "error" in result:
        return f"Rules check could not complete. {result['error']}"
    if result.get("changed"):
        differences = "; ".join(result.get("differences", [])) or "the rule listing changed"
        subreddit = result.get("subreddit")
        community = f"r/{subreddit}" if subreddit else "the configured community"
        return (
            f"WARNING: Reddit's current rules for {community} differ from the local snapshot. "
            f"{differences}. The bot did not change its moderation behavior."
        )
    subreddit = result.get("subreddit")
    community = f"r/{subreddit}" if subreddit else "the configured community"
    return (
        f"No change detected in the official rules for {community}. "
        "The local snapshot was not modified."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    clear_inherited_runtime_environment()
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = Path(__file__).resolve().parent / env_path
    values = load_local_env(env_path, override=True)

    result: dict[str, Any]
    try:
        reference = load_reference(args.reference)
        reddit, account_name = create_read_only_reddit(
            environment=os.environ,
            expected_username=args.expected_account,
        )
        live_rules = list(reddit.subreddit(reference["subreddit"]).rules)
        result = compare_live_rules(reference, live_rules)
        result["account"] = account_name
        result["subreddit"] = reference["subreddit"]
        result["referencePath"] = str(reference_path(args.reference))
        result["readOnly"] = True
    except (CommunityRulesError, FeedError, OSError, KeyError, TypeError, ValueError) as exc:
        result = {"error": _safe_error(exc, values)}
    except Exception as exc:  # pragma: no cover - final boundary for PRAW failures
        result = {"error": _safe_error(exc, values)}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_human_report(result))
        if "error" not in result:
            print(f"Saved rules: {result['storedCount']}; current rules: {result['liveCount']}")
            print("Read-only check: no snapshot or moderation setting was changed.")

    if "error" in result:
        return 1
    return 2 if result.get("changed") else 0


if __name__ == "__main__":
    sys.exit(main())
