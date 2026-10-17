"""
Load and save local .env files (no python-dotenv dependency).

Writes are atomic (temp + replace) with a last-known-good `.env.bak` backup.
"""

from __future__ import annotations

import os
from pathlib import Path


_LOCAL_RUNTIME_ENV_PREFIXES = (
    "BOT_",
    "REDDIT_",
    "LLM_",
    "OPENAI_",
    "ANTHROPIC_",
    "GEMINI_",
    "DEEPSEEK_",
    "OPENROUTER_",
    "GLM_",
    "ZHIPU_",
    "GROQ_",
    "MISTRAL_",
    "TOGETHER_",
    "XAI_",
)


def clear_inherited_runtime_environment() -> None:
    """Remove inherited bot/provider settings before loading an operator .env.

    A shell, IDE, or launcher can provide stale values. Removing known runtime keys
    first makes the gitignored operator .env the source of truth without
    touching unrelated Windows settings.
    """
    for key in list(os.environ):
        if key == "AUDIT_LOG_PATH" or key.startswith(_LOCAL_RUNTIME_ENV_PREFIXES):
            os.environ.pop(key, None)


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a .env file into key -> value (no quotes unescaping beyond strip)."""
    path = Path(path)
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        result[key] = value
    return result


def _format_env_contents(merged: dict[str, str]) -> str:
    lines = [
        "# Optimist Prime bot — local settings (gitignored; do not commit)",
        "# Edit via: py settings_gui.py — changes auto-save",
        "",
    ]
    for key in sorted(merged.keys()):
        val = merged[key]
        if val == "":
            lines.append(f"{key}=")
        else:
            if any(c in val for c in " #\t"):
                safe = val.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{key}="{safe}"')
            else:
                lines.append(f"{key}={val}")
    return "\n".join(lines) + "\n"


def save_env_file(path: str | Path, values: dict[str, str]) -> None:
    """Atomically write .env and keep a `.env.bak` last-known-good copy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_env_file(path) if path.exists() else {}
    merged = {**existing, **{k: v for k, v in values.items() if v is not None}}
    contents = _format_env_contents(merged)

    tmp_path = Path(str(path) + ".tmp")
    bak_path = Path(str(path) + ".bak")
    tmp_path.write_text(contents, encoding="utf-8")
    if path.exists():
        try:
            # Refresh last-known-good before replace
            bak_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass
    os.replace(tmp_path, path)


def load_local_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """
    Load .env from the repo root into os.environ.

    By default, existing environment variables are not overwritten (CI/shell wins).
    """
    import os

    repo_root = Path(__file__).resolve().parent
    env_path = Path(path) if path else repo_root / ".env"
    values = load_env_file(env_path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def apply_env_to_os(values: dict[str, str]) -> None:
    """Apply dict to os.environ (for preview in same process)."""
    import os
    for k, v in values.items():
        if v == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
