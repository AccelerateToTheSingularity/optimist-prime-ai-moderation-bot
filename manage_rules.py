#!/usr/bin/env python3
"""
Local CLI for validating moderation rules and optionally syncing them with a subreddit wiki.

Edit data/rules.json on your machine — no in-Reddit config posts required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from env_file import load_local_env

load_local_env()

import config
from moderation_rules import RuleLoadError, load_rules
from reddit_observability import observable_requestor_settings


def _rules_path(path: str | None = None) -> Path:
    return Path(path or config.MODERATION_RULES_FILE)


def _load_raw_rules(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuleLoadError("Rules file must be a JSON array")
    return data


def _save_raw_rules(path: Path, rules: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
        f.write("\n")


def cmd_validate(args: argparse.Namespace) -> int:
    path = _rules_path(args.file)
    try:
        rules = load_rules(path)
        print(f"OK: {len(rules)} rule(s) in {path}")
        return 0
    except (RuleLoadError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    path = _rules_path(args.file)
    rules = _load_raw_rules(path)
    load_rules(path)  # validate
    for r in sorted(rules, key=lambda x: x.get("order", 100)):
        active = r.get("active", True)
        flag = "on " if active else "off"
        print(f"  [{flag}] {r.get('order', '?'):>3}  {r['name']}")
    print(f"\n{len(rules)} rule(s) in {path}")
    return 0


def cmd_set_active(args: argparse.Namespace, active: bool) -> int:
    path = _rules_path(args.file)
    rules = _load_raw_rules(path)
    found = False
    for r in rules:
        if r.get("name") == args.name:
            r["active"] = active
            found = True
            break
    if not found:
        print(f"Rule not found: {args.name}", file=sys.stderr)
        return 1
    _save_raw_rules(path, rules)
    load_rules(path)
    state = "enabled" if active else "disabled"
    print(f"OK: {args.name} {state}")
    return 0


def _reddit_from_env():
    import os
    import praw

    developer_username = os.environ.get("REDDIT_DEVELOPER_USERNAME", "").strip()
    if not developer_username:
        raise ValueError("Set REDDIT_DEVELOPER_USERNAME to the Reddit account used in the app User-Agent")
    if not config.SUBREDDIT:
        raise ValueError("Set BOT_SUBREDDIT before using Reddit wiki commands")
    if not os.environ.get("REDDIT_CLIENT_ID") or not os.environ.get("REDDIT_CLIENT_SECRET"):
        raise ValueError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET before using Reddit wiki commands")
    refresh = os.environ.get("REDDIT_REFRESH_TOKEN", "").strip()
    if not refresh:
        raise ValueError("Set REDDIT_REFRESH_TOKEN before using Reddit wiki commands")
    user_agent = f"script:OptimistPrimeRulesCLI:v1 (by /u/{developer_username})"
    request_settings = observable_requestor_settings(
        config.REDDIT_API_TIMEOUT_SECONDS,
        request_budget_path=config.REDDIT_REQUEST_BUDGET_FILE,
        request_budget_limit=config.REDDIT_REQUEST_BUDGET_MAX_REQUESTS,
        request_budget_window_seconds=config.REDDIT_REQUEST_BUDGET_WINDOW_SECONDS,
        request_budget_per_minute=config.REDDIT_REQUEST_BUDGET_PER_MINUTE,
    )
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        refresh_token=refresh,
        user_agent=user_agent,
        **request_settings,
    )


def cmd_pull_wiki(args: argparse.Namespace) -> int:
    path = _rules_path(args.file)
    page = args.wiki_page or config.MODERATION_RULES_WIKI_PAGE
    reddit = _reddit_from_env()
    sub = reddit.subreddit(config.SUBREDDIT)
    content = sub.wiki[page].content_md
    # wiki stores markdown; expect JSON in fenced block or raw JSON
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    rules = json.loads(text)
    if not isinstance(rules, list):
        raise RuleLoadError("Wiki content must be a JSON array of rules")
    _save_raw_rules(path, rules)
    load_rules(path)
    print(f"Pulled {len(rules)} rule(s) from wiki:{page} -> {path}")
    return 0


def cmd_push_wiki(args: argparse.Namespace) -> int:
    path = _rules_path(args.file)
    rules = _load_raw_rules(path)
    load_rules(path)
    page = args.wiki_page or config.MODERATION_RULES_WIKI_PAGE
    payload = json.dumps(rules, indent=2, ensure_ascii=False)
    body = f"Moderation rules (managed via manage_rules.py).\n\n```json\n{payload}\n```"
    reddit = _reddit_from_env()
    sub = reddit.subreddit(config.SUBREDDIT)
    try:
        sub.wiki[page].edit(body)
    except Exception:
        sub.wiki.create(page, body)
    print(f"Pushed {len(rules)} rule(s) from {path} -> wiki:{page}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage moderation rules locally")
    parser.add_argument("--file", help=f"Rules JSON path (default: {config.MODERATION_RULES_FILE})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Validate rules file")
    sub.add_parser("list", help="List rules")

    p_en = sub.add_parser("enable", help="Enable a rule by name")
    p_en.add_argument("name")

    p_dis = sub.add_parser("disable", help="Disable a rule by name")
    p_dis.add_argument("name")

    p_pull = sub.add_parser("pull-wiki", help="Pull rules from subreddit wiki into local file")
    p_pull.add_argument("--wiki-page", default=None)

    p_push = sub.add_parser("push-wiki", help="Push local rules to subreddit wiki")
    p_push.add_argument("--wiki-page", default=None)

    args = parser.parse_args()
    handlers = {
        "validate": cmd_validate,
        "list": cmd_list,
        "enable": lambda a: cmd_set_active(a, True),
        "disable": lambda a: cmd_set_active(a, False),
        "pull-wiki": cmd_pull_wiki,
        "push-wiki": cmd_push_wiki,
    }
    try:
        return handlers[args.command](args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
