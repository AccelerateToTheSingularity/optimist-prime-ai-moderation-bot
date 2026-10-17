"""Explicitly migrate local plaintext runtime state to encrypted storage."""

from __future__ import annotations

import argparse

from state_store import (
    StateStoreError,
    migrate_plaintext_state,
    migrate_plaintext_state_to_local,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move local bot_state.json to a non-overwriting private runtime path"
    )
    parser.add_argument("--source", default="data/bot_state.json")
    parser.add_argument("--target", default="data/bot_state.enc")
    parser.add_argument(
        "--plaintext-local",
        action="store_true",
        help="Copy to the target as local plaintext instead of encrypted state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source and report its size without writing encrypted state",
    )
    args = parser.parse_args(argv)

    try:
        if args.dry_run:
            import json
            from pathlib import Path

            source = Path(args.source)
            loaded = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise StateStoreError("Plaintext runtime state must contain a JSON object")
            print(
                f"Would migrate {len(loaded)} top-level state field(s) from "
                f"{source} to {args.target}; no file was written."
            )
        else:
            if args.plaintext_local:
                loaded = migrate_plaintext_state_to_local(args.source, args.target)
            else:
                loaded = migrate_plaintext_state(args.source, args.target)
            print(
                f"Migrated {len(loaded)} top-level state field(s) to {args.target}. "
                "The plaintext source was not deleted."
            )
        return 0
    except (OSError, StateStoreError) as exc:
        print(f"State migration failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
