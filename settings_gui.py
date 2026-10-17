"""
Local settings GUI for Optimist Prime (runs in your browser).

Usage:
    py settings_gui.py
    pyw settings_gui.py          (no console)

Writes to .env and data/rules.json (both gitignored locally).
Changes auto-save. Close the tab or click Done when finished.
"""

from __future__ import annotations

import argparse
import json
import secrets
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bot_logging import configure_logging, log_event
from diagnostics import dump_diagnostics
from env_file import load_env_file, save_env_file
from llm_client import PROVIDER_PRESETS
from local_http_security import is_local_http_request
from moderation_rules import RuleLoadError, _validate_rule
from settings_gui_prefs import (
    FIXED_ORDER_REASONS,
    MAIN_PATH_SECTIONS,
    MAX_FONT_SCALE,
    MIN_FONT_SCALE,
    load_prefs,
    ranked_action_ids,
    record_action_use,
    save_prefs,
)
from settings_registry import SETTING_FIELDS

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_PORT = 8765
RULE_ACTIONS = [
    "log", "report", "remove", "modmail", "ban", "lock", "unlock",
    "spam", "approve",
]

# Rankable secondary toolbar actions (Done stays fixed — Standard 26).
RANKABLE_TOOLBAR_ACTIONS = [
    "settings.customize.toggle",
    "settings.advanced.toggle",
    "diagnostics.dump",
    "rules.add",
]


def _bool_to_env(value: str) -> str:
    return "true" if value.lower() in ("1", "true", "yes", "on") else "false"


def _env_to_bool(value: str, default: bool = False) -> bool:
    """Parse a boolean setting without making invalid values less safe."""
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default


def _current_values() -> dict[str, str]:
    env = load_env_file(ENV_PATH)
    out: dict[str, str] = {}
    for field in SETTING_FIELDS:
        raw = env.get(field.key, field.default)
        # Never send credential values to the browser, even though this GUI
        # is bound to loopback. A blank password field is intentionally
        # interpreted as "keep the existing value" by _merge_save_payload.
        if field.field_type == "password":
            out[field.key] = ""
        elif field.field_type == "bool":
            default_value = _env_to_bool(field.default)
            out[field.key] = "true" if _env_to_bool(raw, default_value) else "false"
        else:
            out[field.key] = raw
    return out


def _configured_values() -> dict[str, bool]:
    """Return password-presence flags without exposing password values."""
    env = load_env_file(ENV_PATH)
    return {
        field.key: bool(env.get(field.key, "").strip())
        for field in SETTING_FIELDS
        if field.field_type == "password"
    }


def _merge_save_payload(
    payload: dict[str, str],
    *,
    env_path: str | Path | None = None,
) -> dict[str, str]:
    """Merge editable values while preserving blank password fields.

    ``env_path`` keeps this shared persistence helper usable by the native
    Qt console without duplicating the browser settings GUI's secret-safe
    merge rules.  The existing browser editor continues to use ``ENV_PATH``.
    """
    target_path = Path(env_path) if env_path is not None else ENV_PATH
    existing = load_env_file(target_path)
    merged: dict[str, str] = {}
    for field in SETTING_FIELDS:
        if field.key not in payload:
            continue
        value = payload[field.key].strip()
        if field.field_type == "password" and not value:
            if field.key in existing:
                merged[field.key] = existing[field.key]
            continue
        if field.field_type == "bool":
            merged[field.key] = _bool_to_env(value)
        elif field.field_type == "percent":
            try:
                numeric = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field.label} must be a whole percentage") from exc
            low = field.minimum if field.minimum is not None else 0
            high = field.maximum if field.maximum is not None else 100
            merged[field.key] = str(min(high, max(low, numeric)))
        elif field.field_type in {"int", "float"}:
            # Blank numeric settings use the application's documented default
            # (or leave an optional setting unset). Validate bounded values so
            # the UI cannot silently save a value the runtime later changes.
            if not value:
                merged[field.key] = ""
                continue
            try:
                numeric = int(value) if field.field_type == "int" else float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field.label} must be a number") from exc
            if field.minimum is not None and numeric < field.minimum:
                raise ValueError(f"{field.label} must be at least {field.minimum}")
            if field.maximum is not None and numeric > field.maximum:
                raise ValueError(f"{field.label} must be at most {field.maximum}")
            merged[field.key] = str(numeric) if field.field_type == "float" else str(int(numeric))
        else:
            merged[field.key] = value
    return merged


def _rules_path() -> Path:
    rel = load_env_file(ENV_PATH).get("BOT_MODERATION_RULES_FILE", "data/rules.json")
    return REPO_ROOT / rel


def _load_rules_data() -> dict[str, Any]:
    path = _rules_path()
    example = REPO_ROOT / "data" / "rules.example.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rules = json.load(f)
        source = "file"
    elif example.exists():
        with open(example, encoding="utf-8") as f:
            rules = json.load(f)
        source = "example"
    else:
        rules = []
        source = "empty"
    if not isinstance(rules, list):
        raise RuleLoadError("Rules file must be a JSON array")
    return {"path": str(path), "rules": rules, "source": source}


def _save_rules_data(rules: list[dict]) -> None:
    if not isinstance(rules, list):
        raise RuleLoadError("Rules must be a JSON array")
    for index, rule in enumerate(rules):
        _validate_rule(rule, index)
    path = _rules_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _persist_shutdown_payload(payload: dict[str, Any]) -> None:
    """Save the latest browser state before the local settings server exits."""
    values = payload.get("values")
    if isinstance(values, dict):
        save_env_file(ENV_PATH, _merge_save_payload(values, env_path=ENV_PATH))
    rules = payload.get("rules")
    if isinstance(rules, list):
        _save_rules_data(rules)


def _provider_info() -> dict[str, dict[str, str]]:
    return {
        name: {
            "base_url": preset.get("base_url", ""),
            "model": preset.get("model", ""),
            "api_key_envs": preset.get("api_key_envs", ""),
        }
        for name, preset in PROVIDER_PRESETS.items()
    }


def _field_payload(field) -> dict[str, Any]:
    return {
        "key": field.key,
        "label": field.label,
        "section": field.section,
        "field_type": field.field_type,
        "choices": field.choices,
        "help_text": field.help_text,
        "hideable": field.hideable,
        "advanced": field.advanced,
        "minimum": field.minimum,
        "maximum": field.maximum,
        "step": field.step,
    }


def _list_actions(prefs: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = ranked_action_ids(
        RANKABLE_TOOLBAR_ACTIONS + ["settings.done", "settings.visibility.reset", "settings.persist"],
        prefs,
        fixed={"settings.done"},
    )
    catalog = {
        "settings.persist": ("Persist settings now", "Flush pending .env writes", "local-state"),
        "settings.done": ("Done", "Shortcut: Ctrl+Enter — Flush pending changes and close", "local-state"),
        "settings.customize.toggle": ("Customize settings", "Shortcut: C — Hide or restore optional settings", "local-state"),
        "settings.advanced.toggle": ("Show advanced", "Shortcut: A — Reveal uncommon / advanced fields", "local-state"),
        "settings.visibility.reset": ("Reset visibility", "Show all hideable settings again", "local-state"),
        "settings.fontScale.set": ("Set font scale", "Change shared UI font scale", "local-state"),
        "rules.add": ("Add rule", "Shortcut: + — Append a new moderation rule", "local-state"),
        "diagnostics.dump": ("Diagnostics", "Shortcut: D — Dump health snapshot", "read"),
        "actions.list": ("List actions", "List AI-triggerable actions", "read"),
    }
    out = []
    for action_id in ranked:
        if action_id not in catalog:
            continue
        label, description, side = catalog[action_id]
        out.append({
            "id": action_id,
            "label": label,
            "description": description,
            "sideEffect": side,
            "enabled": True,
        })
    for action_id, (label, description, side) in catalog.items():
        if any(a["id"] == action_id for a in out):
            continue
        out.append({
            "id": action_id,
            "label": label,
            "description": description,
            "sideEffect": side,
            "enabled": True,
        })
    return out


def _invoke_action(action_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    correlation_id = params.get("correlationId") or str(uuid.uuid4())
    prefs = load_prefs(REPO_ROOT)
    log_event(
        "ui.action.started",
        f"action {action_id}",
        action_id=action_id,
        correlation_id=correlation_id,
        component="settings_gui",
    )
    try:
        if action_id == "actions.list":
            result = {"ok": True, "actions": _list_actions(prefs)}
        elif action_id == "diagnostics.dump":
            result = dump_diagnostics(load_environment=True)
        elif action_id == "settings.persist":
            # Client sends values; server-side flush helper for agents
            payload = params.get("values") or {}
            if payload:
                save_env_file(ENV_PATH, _merge_save_payload(payload))
            result = {"ok": True, "message": "Settings persisted"}
        elif action_id == "settings.visibility.reset":
            prefs["hiddenSettingIds"] = []
            save_prefs(REPO_ROOT, prefs)
            result = {"ok": True, "prefs": prefs, "message": "Visibility reset"}
        elif action_id == "settings.fontScale.set":
            scale = float(params.get("fontScale", prefs.get("fontScale", 1.0)))
            prefs["fontScale"] = max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, round(scale, 2)))
            save_prefs(REPO_ROOT, prefs)
            result = {"ok": True, "prefs": prefs}
        elif action_id in (
            "settings.done",
            "settings.customize.toggle",
            "settings.advanced.toggle",
            "rules.add",
        ):
            # Client-side primary; agents get acknowledgment + updated prefs bookkeeping
            record_action_use(prefs, action_id)
            save_prefs(REPO_ROOT, prefs)
            result = {"ok": True, "message": f"Client should run {action_id}", "prefs": prefs}
        else:
            result = {"ok": False, "message": f"Unknown action: {action_id}"}
        if result.get("ok") and action_id in RANKABLE_TOOLBAR_ACTIONS:
            prefs = load_prefs(REPO_ROOT)
            record_action_use(prefs, action_id)
            save_prefs(REPO_ROOT, prefs)
        log_event(
            "ui.action.finished",
            f"action {action_id} ok={result.get('ok')}",
            action_id=action_id,
            correlation_id=correlation_id,
            component="settings_gui",
        )
        result["correlationId"] = correlation_id
        return result
    except Exception as exc:
        log_event(
            "ui.action.rejected",
            str(exc),
            action_id=action_id,
            correlation_id=correlation_id,
            component="settings_gui",
        )
        return {"ok": False, "message": str(exc), "correlationId": correlation_id}


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Optimist Prime — Settings</title>
  <style>
    :root {
      --font-scale: 1;
      --bg: #e8eef4;
      --panel: #ffffff;
      --border: #1c2836;
      --text: #1c2836;
      --muted: #5a6b7d;
      --accent: #0b63ce;
      --ok: #15803d;
      --err: #b91c1c;
      --inset-text: 16px;
      --inset-ctrl: 8px;
      --base: calc(15px * var(--font-scale));
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      font-size: var(--base);
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    header {
      padding: var(--inset-text);
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    header h1 { margin: 0 0 0.25rem; font-size: calc(1.25rem * var(--font-scale)); font-weight: 650; }
    header .sub { margin: 0; color: var(--muted); font-size: 0.9em; }
    main { max-width: 960px; margin: 0 auto; padding: var(--inset-text); }
    .action-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
      gap: 0;
      margin: 0 0 1rem;
      border: 1px solid var(--border);
      background: var(--panel);
    }
    .action-row button {
      margin-block-start: -1px;
      margin-inline-start: -1px;
      min-height: 2.75rem;
      border: 1px solid var(--border);
      border-radius: 0;
      background: #fff;
      color: #000;
      font: inherit;
      font-weight: 600;
      padding: 0.65rem 0.75rem;
      cursor: pointer;
      line-height: 1.25;
      white-space: normal;
    }
    .action-row button:hover { background: #e5e7eb; z-index: 1; position: relative; }
    .action-row button.primary { background: var(--accent); color: #fff; }
    .action-row button.primary:hover { filter: brightness(0.95); }
    #status { min-height: 1.4em; margin: 0.35rem 0 0.85rem; color: var(--muted); font-size: 0.92em; }
    #status.ok { color: var(--ok); }
    #status.err { color: var(--err); }
    .font-row {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-top: 0.65rem;
      flex-wrap: wrap;
    }
    .font-row label { font-weight: 600; }
    .font-row input[type="range"] { flex: 1; min-width: 140px; }
    section.panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0;
      margin: 0 0 -1px;
      padding: var(--inset-text);
    }
    section.panel + section.panel { border-top: none; }
    section.panel h2 {
      margin: 0 0 0.65rem;
      font-size: 1.05em;
      font-weight: 650;
      display: flex;
      justify-content: space-between;
      gap: 0.5rem;
      align-items: center;
    }
    .field {
      margin-bottom: 0.65rem;
      padding: var(--inset-ctrl);
    }
    .field.hidden { display: none; }
    .field label { display: block; font-weight: 600; margin-bottom: 0.2rem; }
    .field .label-row { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
    .help-btn {
      border: 1px solid var(--border);
      background: #fff;
      color: #000;
      border-radius: 0;
      width: 1.6rem;
      height: 1.6rem;
      padding: 0;
      cursor: pointer;
      font-weight: 700;
    }
    .help-pop {
      display: none;
      margin-top: 0.25rem;
      color: var(--muted);
      font-size: 0.88em;
    }
    .help-pop.open { display: block; }
    input[type="text"], input[type="password"], input[type="number"], select, textarea {
      width: 100%;
      padding: 0.45rem 0.55rem;
      border: 1px solid var(--border);
      border-radius: 0;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    .bool-row { display: flex; align-items: center; gap: 0.45rem; }
    .bool-row input { width: auto; }
    .bool-row label { margin: 0; font-weight: 600; }
    .provider-hint {
      font-size: 0.88em;
      color: var(--muted);
      margin-top: 0.35rem;
      padding: var(--inset-ctrl);
      border: 1px dashed var(--border);
    }
    .rules-toolbar { margin-bottom: 0.65rem; }
    .rule-card {
      border: 1px solid var(--border);
      padding: var(--inset-text);
      margin: 0 0 -1px;
      background: #fafbfd;
    }
    .rule-card.inactive { opacity: 0.72; }
    .rule-head { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.5rem; }
    .rule-head .name-input { flex: 1; min-width: 160px; font-weight: 600; }
    .actions-grid { display: flex; flex-wrap: wrap; gap: 0.35rem 0.75rem; margin-top: 0.35rem; }
    .conds-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.35rem; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
    @media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr; } }
    .hidden-box {
      display: none;
      margin-top: 0.75rem;
      padding: var(--inset-text);
      border: 1px solid var(--border);
      background: #f7f9fc;
    }
    .hidden-box.open { display: block; }
    .kbd { font-size: 0.82em; color: var(--muted); }
    .diag {
      display: none;
      white-space: pre-wrap;
      font-family: ui-monospace, Consolas, monospace;
      font-size: 0.82em;
      border: 1px solid var(--border);
      padding: var(--inset-text);
      background: #0f172a;
      color: #e2e8f0;
      margin-bottom: 1rem;
      max-height: 280px;
      overflow: auto;
    }
    .diag.open { display: block; }
    .resume-note { color: var(--muted); font-size: 0.9em; margin: 0 0 0.75rem; }
  </style>
</head>
<body>
  <header>
    <h1>Optimist Prime — Settings</h1>
    <p class="sub">Changes save automatically. Close this tab or press <span class="kbd">Ctrl+Enter</span> / <strong>Done</strong> when finished — you can reopen anytime to resume.</p>
    <div class="font-row">
      <label for="fontScale">Text size</label>
      <input type="range" id="fontScale" min="0.9" max="1.6" step="0.1" value="1" />
      <span id="fontScaleLabel">100%</span>
    </div>
  </header>
  <main>
    <div class="action-row" id="actionRow"></div>
    <div id="status" role="status"></div>
    <p class="resume-note">Start with Reddit access and an AI provider, then choose the features and community policies you want to configure.</p>
    <pre class="diag" id="diagPanel"></pre>
    <div id="sections"></div>
    <section class="panel" id="rulesSection">
      <h2>AI moderation rules</h2>
      <p class="resume-note">Each rule is a yes/no question the AI evaluates on posts and comments.</p>
      <div class="rules-toolbar" id="rulesToolbar"></div>
      <div class="rules-path resume-note" id="rulesPath"></div>
      <div id="rulesMetaFields" class="grid-2"></div>
      <div id="rulesList"></div>
    </section>
    <div class="hidden-box" id="hiddenBox">
      <strong>Hidden settings</strong>
      <div id="hiddenList"></div>
    </div>
  </main>
  <script>
    let providers = {};
    let fields = [];
    let rules = [];
    let rulesPath = "";
    let rulesSource = "";
    let prefs = {};
    let persistRevision = 0;
    let lastPersistedRevision = 0;
    let settingsTimer = null;
    let rulesTimer = null;
    let saving = false;
    const RULE_ACTIONS = """ + json.dumps(RULE_ACTIONS) + r""";
    const MAIN_PATH = """ + json.dumps(list(MAIN_PATH_SECTIONS)) + r""";
    const FIXED_REASONS = """ + json.dumps(FIXED_ORDER_REASONS) + r""";
    const UI_TOKEN = "__OPTIMIST_PRIME_UI_TOKEN__";

    function writeHeaders() {
      return {
        "Content-Type": "application/json",
        "X-Optimist-Prime-UI-Token": UI_TOKEN,
      };
    }

    let shutdownStarted = false;
    function shutdown() {
      if (shutdownStarted) return;
      shutdownStarted = true;
      try {
        fetch("/api/shutdown", {
          method: "POST",
          headers: writeHeaders(),
          keepalive: true,
          body: JSON.stringify({ values: collectSettings(), rules: collectRules() }),
        }).catch(() => {});
      } catch (e) {}
    }
    window.addEventListener("pagehide", shutdown);

    function setStatus(msg, ok) {
      const el = document.getElementById("status");
      el.textContent = msg || "";
      el.className = ok === true ? "ok" : (ok === false ? "err" : "");
    }

    function applyFontScale(scale) {
      document.documentElement.style.setProperty("--font-scale", String(scale));
      document.getElementById("fontScale").value = scale;
      document.getElementById("fontScaleLabel").textContent = Math.round(scale * 100) + "%";
    }

    async function load() {
      const [settingsRes, metaRes, rulesRes, prefsRes, actionsRes] = await Promise.all([
        fetch("/api/settings"),
        fetch("/api/meta"),
        fetch("/api/rules"),
        fetch("/api/prefs"),
        fetch("/api/actions"),
      ]);
      const settings = await settingsRes.json();
      providers = (await metaRes.json()).providers;
      const rulesData = await rulesRes.json();
      prefs = (await prefsRes.json()).prefs || {};
      fields = settings.fields;
      rules = rulesData.rules || [];
      rulesPath = rulesData.path || "";
      rulesSource = rulesData.source || "";
      applyFontScale(prefs.fontScale || 1);
      renderActions(await actionsRes.json());
      render(settings.values);
      renderRules();
      updateProviderHint(settings.values.BOT_LLM_PROVIDER);
      if (rulesSource === "example") {
        setStatus("Showing example rules — edits auto-create your rules file.", true);
      } else {
        setStatus("Ready — edits auto-save.", true);
      }
    }

    function renderActions(data) {
      const actions = (data && data.actions) || [];
      const row = document.getElementById("actionRow");
      row.innerHTML = "";
      const wanted = [
        "settings.done",
        "settings.customize.toggle",
        "settings.advanced.toggle",
        "diagnostics.dump",
      ];
      const byId = Object.fromEntries(actions.map(a => [a.id, a]));
      // Frequency-ranked order from server, Done forced first
      const ordered = ["settings.done"].concat(
        (data.rankedToolbar || wanted.filter(id => id !== "settings.done"))
      );
      const seen = new Set();
      for (const id of ordered) {
        if (seen.has(id) || !byId[id]) continue;
        seen.add(id);
        const a = byId[id];
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.action = a.id;
        if (a.id === "settings.done") btn.className = "primary";
        btn.textContent = a.label;
        btn.title = a.description;
        btn.addEventListener("click", () => runAction(a.id));
        row.appendChild(btn);
      }
      const rulesTb = document.getElementById("rulesToolbar");
      rulesTb.innerHTML = "";
      const add = document.createElement("button");
      add.type = "button";
      add.dataset.action = "rules.add";
      add.textContent = "Add rule";
      add.title = (byId["rules.add"] && byId["rules.add"].description) ||
        "Shortcut: + — Append a new moderation rule";
      add.addEventListener("click", () => runAction("rules.add"));
      rulesTb.appendChild(add);
    }

    async function runAction(actionId, params) {
      if (actionId === "settings.done") {
        const ok = await flushNow();
        if (!ok) return;
        shutdown();
        setStatus("Closing…", true);
        setTimeout(() => window.close(), 250);
        return;
      }
      if (actionId === "settings.customize.toggle") {
        prefs.customizeMode = !prefs.customizeMode;
        await savePrefs();
        render(collectSettings());
        return;
      }
      if (actionId === "settings.advanced.toggle") {
        prefs.showAdvanced = !prefs.showAdvanced;
        await savePrefs();
        render(collectSettings());
        setStatus(prefs.showAdvanced ? "Advanced settings shown." : "Advanced settings hidden.", true);
        return;
      }
      if (actionId === "rules.add") {
        rules.push(defaultRule());
        renderRules();
        scheduleRulesPersist();
        await fetch("/api/actions/rules.add", {
          method: "POST",
          headers: writeHeaders(),
          body: JSON.stringify({}),
        });
        return;
      }
      if (actionId === "diagnostics.dump") {
        const res = await fetch("/api/diagnostics");
        const data = await res.json();
        const panel = document.getElementById("diagPanel");
        panel.textContent = JSON.stringify(data, null, 2);
        panel.classList.add("open");
        setStatus("Diagnostics loaded.", true);
        await fetch("/api/actions/diagnostics.dump", {
          method: "POST",
          headers: writeHeaders(),
          body: JSON.stringify({}),
        });
        return;
      }
      if (actionId === "settings.visibility.reset") {
        const res = await fetch("/api/actions/settings.visibility.reset", {
          method: "POST",
          headers: writeHeaders(),
          body: "{}",
        });
        const data = await res.json();
        prefs = data.prefs || prefs;
        render(collectSettings());
        setStatus("Visibility reset.", true);
      }
    }

    async function savePrefs() {
      await fetch("/api/prefs", {
        method: "POST",
        headers: writeHeaders(),
        body: JSON.stringify(prefs),
      });
    }

    function isFieldVisible(f) {
      const hidden = new Set(prefs.hiddenSettingIds || []);
      if (hidden.has(f.key) && !prefs.customizeMode) return false;
      if (f.advanced && !prefs.showAdvanced && !prefs.customizeMode) return false;
      return true;
    }

    function render(values) {
      const bySection = {};
      for (const f of fields) {
        if (f.section === "Moderation rules") continue;
        (bySection[f.section] ||= []).push(f);
      }
      const root = document.getElementById("sections");
      root.innerHTML = "";
      const sectionOrder = Object.keys(bySection);
      for (const section of sectionOrder) {
        const sectionFields = bySection[section];
        const isMain = MAIN_PATH.includes(section);
        if (!isMain && !prefs.showAdvanced && !prefs.customizeMode) continue;
        const visibleFields = sectionFields.filter(isFieldVisible);
        if (!visibleFields.length && !prefs.customizeMode) continue;
        const panel = document.createElement("section");
        panel.className = "panel";
        panel.dataset.section = section;
        const h2 = document.createElement("h2");
        h2.innerHTML = "<span>" + section + "</span>" +
          (isMain ? "" : '<span class="kbd">advanced</span>');
        panel.appendChild(h2);
        for (const f of sectionFields) {
          if (!isFieldVisible(f) && !(prefs.customizeMode && (prefs.hiddenSettingIds || []).includes(f.key))) {
            if (!(prefs.customizeMode && f.hideable)) continue;
          }
          if (!isFieldVisible(f) && !prefs.customizeMode) continue;
          panel.appendChild(renderField(f, values[f.key] || ""));
        }
        if (section === "LLM") {
          const hint = document.createElement("div");
          hint.className = "provider-hint";
          hint.id = "providerHint";
          panel.appendChild(hint);
        }
        root.appendChild(panel);
      }
      const rulesMeta = document.getElementById("rulesMetaFields");
      rulesMeta.innerHTML = "";
      for (const f of fields) {
        if (f.section === "Moderation rules" && isFieldVisible(f)) {
          rulesMeta.appendChild(renderField(f, values[f.key] || ""));
        }
      }
      renderHiddenBox(values);
      updateProviderHint(values.BOT_LLM_PROVIDER);
      wireFieldListeners();
    }

    function renderHiddenBox(values) {
      const box = document.getElementById("hiddenBox");
      const list = document.getElementById("hiddenList");
      list.innerHTML = "";
      const hidden = prefs.hiddenSettingIds || [];
      if (!prefs.customizeMode) {
        box.classList.remove("open");
        return;
      }
      box.classList.add("open");
      if (!hidden.length) {
        list.textContent = "No hidden settings.";
        return;
      }
      for (const key of hidden) {
        const f = fields.find(x => x.key === key);
        const row = document.createElement("div");
        row.className = "field";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = "Show: " + (f ? f.label : key);
        btn.addEventListener("click", async () => {
          prefs.hiddenSettingIds = hidden.filter(k => k !== key);
          await savePrefs();
          render(collectSettings());
        });
        row.appendChild(btn);
        list.appendChild(row);
      }
      const reset = document.createElement("button");
      reset.type = "button";
      reset.textContent = "Reset visibility";
      reset.addEventListener("click", () => runAction("settings.visibility.reset"));
      list.appendChild(reset);
    }

    function renderField(f, value) {
      const wrap = document.createElement("div");
      wrap.className = "field";
      wrap.dataset.key = f.key;
      const labelRow = document.createElement("div");
      labelRow.className = "label-row";
      if (f.field_type === "bool") {
        const checked = value === "true";
        wrap.innerHTML = '<div class="bool-row"><input type="checkbox" id="' + f.key + '"' +
          (checked ? " checked" : "") + ' /><label for="' + f.key + '">' + f.label + "</label></div>";
      } else if (f.field_type === "choice") {
        const opts = (f.choices || []).map(c =>
          '<option value="' + c + '"' + (c === value ? " selected" : "") + ">" + (c || "(choose)") + "</option>"
        ).join("");
        wrap.innerHTML = '<div class="label-row"><label for="' + f.key + '">' + f.label + '</label></div>' +
          '<select id="' + f.key + '">' + opts + "</select>";
      } else if (f.field_type === "text") {
        wrap.innerHTML = '<div class="label-row"><label for="' + f.key + '">' + f.label + '</label></div>' +
          '<textarea id="' + f.key + '">' + esc(value) + "</textarea>";
      } else if (f.field_type === "percent") {
        const low = Number.isFinite(Number(f.minimum)) ? Number(f.minimum) : 0;
        const high = Number.isFinite(Number(f.maximum)) ? Number(f.maximum) : 100;
        const step = Number.isFinite(Number(f.step)) ? Number(f.step) : 1;
        const current = Math.min(high, Math.max(low, Number.parseInt(value || f.default || low, 10)));
        wrap.innerHTML = '<div class="label-row"><label for="' + f.key + '">' + f.label +
          '</label><output id="' + f.key + 'Value">' + current + '%</output></div>' +
          '<input type="range" id="' + f.key + '" min="' + low + '" max="' + high +
          '" step="' + step + '" value="' + current + '" />';
      } else {
        const type = f.field_type === "password" ? "password" : ((f.field_type === "int" || f.field_type === "float") ? "number" : "text");
        const ph = f.field_type === "password" ? "Leave blank to keep existing" : "";
        const min = f.minimum == null ? "" : ' min="' + f.minimum + '"';
        const max = f.maximum == null ? "" : ' max="' + f.maximum + '"';
        const step = f.step == null ? "" : ' step="' + f.step + '"';
        wrap.innerHTML = '<div class="label-row"><label for="' + f.key + '">' + f.label + '</label></div>' +
          '<input type="' + type + '" id="' + f.key + '" value="' + escAttr(value) + '" placeholder="' + ph + '"' + min + max + step + ' />';
      }
      if (f.help_text) {
        const row = wrap.querySelector(".label-row") || wrap.querySelector(".bool-row");
        const helpBtn = document.createElement("button");
        helpBtn.type = "button";
        helpBtn.className = "help-btn";
        helpBtn.textContent = "?";
        helpBtn.title = f.help_text;
        helpBtn.setAttribute("aria-label", "Help for " + f.label);
        const pop = document.createElement("div");
        pop.className = "help-pop";
        pop.textContent = f.help_text;
        helpBtn.addEventListener("click", () => pop.classList.toggle("open"));
        if (row) row.appendChild(helpBtn);
        wrap.appendChild(pop);
      }
      if (prefs.customizeMode && f.hideable) {
        const hideBtn = document.createElement("button");
        hideBtn.type = "button";
        hideBtn.textContent = "Hide";
        hideBtn.style.marginTop = "0.25rem";
        hideBtn.addEventListener("click", async () => {
          const set = new Set(prefs.hiddenSettingIds || []);
          set.add(f.key);
          prefs.hiddenSettingIds = [...set];
          await savePrefs();
          render(collectSettings());
        });
        wrap.appendChild(hideBtn);
      }
      return wrap;
    }

    function esc(s) { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
    function escAttr(s) { return esc(s).replace(/"/g, "&quot;"); }

    function defaultRule() {
      return {
        name: "new_rule",
        description: "Describe what this rule should detect.",
        active: false,
        order: (rules.length + 1) * 10,
        target: "both",
        actions: [],
        conditions: { stop_on_match: false, skip_mods: true, skip_approved: false },
      };
    }

    function renderRules() {
      document.getElementById("rulesPath").textContent = "File: " + (document.getElementById("BOT_MODERATION_RULES_FILE")?.value || rulesPath);
      const list = document.getElementById("rulesList");
      list.innerHTML = "";
      rules.forEach((rule, idx) => list.appendChild(buildRuleCard(rule, idx)));
    }

    function buildRuleCard(rule, idx) {
      const card = document.createElement("div");
      card.className = "rule-card" + (rule.active ? "" : " inactive");
      const cond = rule.conditions || {};
      const actions = new Set(rule.actions || []);
      const actionBoxes = RULE_ACTIONS.map(a =>
        '<label><input type="checkbox" data-action="' + a + '"' + (actions.has(a) ? " checked" : "") + " /> " + a + "</label>"
      ).join("");
      card.innerHTML =
        '<div class="rule-head">' +
          '<input class="name-input" data-f="name" value="' + escAttr(rule.name || "") + '" />' +
          '<label class="bool-row"><input type="checkbox" data-f="active"' + (rule.active === true ? " checked" : "") + " /> Active</label>" +
          '<button type="button" data-del="' + idx + '">Remove</button>' +
        "</div>" +
        '<div class="field"><label>Question for the AI</label><textarea data-f="description">' + esc(rule.description || "") + "</textarea></div>" +
        '<div class="grid-2">' +
          '<div class="field"><label>Order</label><input type="number" data-f="order" value="' + (rule.order ?? 100) + '" /></div>' +
          '<div class="field"><label>Applies to</label><select data-f="target">' +
            ["posts","comments","both"].map(t => '<option value="' + t + '"' + ((rule.target||"both")===t?" selected":"") + ">" + t + "</option>").join("") +
          "</select></div>" +
        "</div>" +
        '<div class="field"><label>Actions when matched</label><div class="actions-grid" data-actions>' + actionBoxes + "</div></div>" +
        '<div class="conds-row">' +
          '<label><input type="checkbox" data-cond="stop_on_match"' + (cond.stop_on_match ? " checked" : "") + " /> Stop after match</label>" +
          '<label><input type="checkbox" data-cond="skip_mods"' + (cond.skip_mods !== false ? " checked" : "") + " /> Skip moderators</label>" +
          '<label><input type="checkbox" data-cond="skip_approved"' + (cond.skip_approved ? " checked" : "") + " /> Skip approved</label>' +
        "</div>";
      card.querySelector("[data-del]").addEventListener("click", () => {
        // Inline three-activation confirmation for local destructive remove
        const btn = card.querySelector("[data-del]");
        const n = Number(btn.dataset.confirm || 0) + 1;
        btn.dataset.confirm = String(n);
        if (n < 3) {
          btn.textContent = "Confirm remove (" + (3 - n) + ")";
          return;
        }
        rules.splice(idx, 1);
        renderRules();
        scheduleRulesPersist();
      });
      card.querySelectorAll("[data-f]").forEach(el => {
        el.addEventListener("change", () => { syncRuleFromCard(card, idx); scheduleRulesPersist(); });
        el.addEventListener("input", () => { syncRuleFromCard(card, idx); scheduleRulesPersist(); });
      });
      card.querySelectorAll("[data-actions] input, [data-cond]").forEach(el => {
        el.addEventListener("change", () => { syncRuleFromCard(card, idx); scheduleRulesPersist(); });
      });
      return card;
    }

    function syncRuleFromCard(card, idx) {
      const r = rules[idx];
      r.name = card.querySelector('[data-f="name"]').value.trim();
      r.description = card.querySelector('[data-f="description"]').value.trim();
      r.active = card.querySelector('[data-f="active"]').checked;
      r.order = parseInt(card.querySelector('[data-f="order"]').value, 10) || 100;
      r.target = card.querySelector('[data-f="target"]').value;
      r.actions = [...card.querySelectorAll("[data-actions] input:checked")].map(cb => cb.dataset.action);
      r.conditions = {
        stop_on_match: card.querySelector('[data-cond="stop_on_match"]').checked,
        skip_mods: card.querySelector('[data-cond="skip_mods"]').checked,
        skip_approved: card.querySelector('[data-cond="skip_approved"]').checked,
      };
      card.classList.toggle("inactive", !r.active);
    }

    function collectSettings() {
      const values = {};
      for (const f of fields) {
        const el = document.getElementById(f.key);
        if (!el) continue;
        values[f.key] = f.field_type === "bool" ? (el.checked ? "true" : "false") : el.value;
      }
      return values;
    }

    function collectRules() {
      document.querySelectorAll(".rule-card").forEach((card, idx) => syncRuleFromCard(card, idx));
      return rules;
    }

    function markDirty(mode) {
      persistRevision += 1;
      if (mode === "immediate") scheduleSettingsPersist(0);
      else scheduleSettingsPersist(750);
    }

    function scheduleSettingsPersist(ms) {
      clearTimeout(settingsTimer);
      settingsTimer = setTimeout(() => persistSettings(), ms);
    }

    function scheduleRulesPersist() {
      clearTimeout(rulesTimer);
      setStatus("Saving…");
      rulesTimer = setTimeout(() => persistRules(), 750);
    }

    async function persistSettings() {
      const rev = persistRevision;
      saving = true;
      setStatus("Saving…");
      try {
        const settingsRes = await fetch("/api/settings", {
          method: "POST",
          headers: writeHeaders(),
          body: JSON.stringify(collectSettings()),
        });
        const settingsData = await settingsRes.json();
        if (!settingsData.ok) { setStatus(settingsData.message, false); return false; }
        if (rev === persistRevision) lastPersistedRevision = rev;
        setStatus("Saved.", true);
        return true;
      } catch (e) {
        setStatus(String(e), false);
        return false;
      } finally {
        saving = false;
      }
    }

    async function persistRules() {
      try {
        const rulesRes = await fetch("/api/rules", {
          method: "POST",
          headers: writeHeaders(),
          body: JSON.stringify({ rules: collectRules() }),
        });
        const rulesData = await rulesRes.json();
        if (!rulesData.ok) { setStatus(rulesData.message, false); return false; }
        setStatus("Saved.", true);
        return true;
      } catch (e) {
        setStatus(String(e), false);
        return false;
      }
    }

    async function flushNow() {
      clearTimeout(settingsTimer);
      clearTimeout(rulesTimer);
      const a = await persistSettings();
      const b = await persistRules();
      return a && b;
    }

    function wireFieldListeners() {
      for (const f of fields) {
        const el = document.getElementById(f.key);
        if (!el) continue;
        const mode = (f.field_type === "bool" || f.field_type === "choice") ? "immediate" : "debounced";
        const handler = () => {
          if (f.key === "BOT_LLM_PROVIDER") updateProviderHint(el.value);
          if (f.field_type === "percent") {
            const output = document.getElementById(f.key + "Value");
            if (output) output.value = output.textContent = el.value + "%";
          }
          markDirty(mode === "immediate" ? "immediate" : "debounced");
        };
        el.addEventListener("change", handler);
        el.addEventListener("input", handler);
      }
    }

    function updateProviderHint(name) {
      const hint = document.getElementById("providerHint");
      if (!hint) return;
      const p = providers[name];
      if (!p) { hint.textContent = ""; return; }
      hint.textContent = "Endpoint preset: " + (p.base_url || "(set OPENAI_BASE_URL)") +
        ". Enter an LLM_MODEL supported by this provider and configure its API key.";
    }

    document.getElementById("fontScale").addEventListener("input", async (e) => {
      const scale = Number(e.target.value);
      applyFontScale(scale);
      prefs.fontScale = scale;
        await fetch("/api/actions/settings.fontScale.set", {
          method: "POST",
          headers: writeHeaders(),
          body: JSON.stringify({ fontScale: scale }),
        });
    });

    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey && e.key === "Enter") {
        e.preventDefault();
        runAction("settings.done");
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        const tag = (e.target && e.target.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (e.key === "c" || e.key === "C") runAction("settings.customize.toggle");
        if (e.key === "a" || e.key === "A") runAction("settings.advanced.toggle");
        if (e.key === "d" || e.key === "D") runAction("diagnostics.dump");
        if (e.key === "+") runAction("rules.add");
      }
    });

    load().catch(err => setStatus("Failed to load: " + err, false));
  </script>
</body>
</html>
"""


class SettingsHandler(BaseHTTPRequestHandler):
    http_server: ThreadingHTTPServer | None = None
    # Tests that construct the HTTP server directly set this explicitly.
    # The server replaces it with a fresh unpredictable token.
    ui_token: str | None = None

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _write_authorized(self) -> bool:
        expected = type(self).ui_token
        supplied = self.headers.get("X-Optimist-Prime-UI-Token", "")
        return bool(expected and supplied) and secrets.compare_digest(supplied, expected)

    def _local_request_allowed(self, *, require_origin: bool = False) -> bool:
        if is_local_http_request(
            self.headers,
            self.server.server_address,
            require_origin=require_origin,
        ):
            return True
        self.send_error(403, "This settings interface accepts local requests only.")
        return False

    def _schedule_shutdown(self) -> None:
        server = self.http_server
        if server is None:
            return
        threading.Thread(target=server.shutdown, daemon=True).start()

    def do_GET(self) -> None:
        if not self._local_request_allowed():
            return
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            token = type(self).ui_token or ""
            body = HTML_PAGE.replace("__OPTIMIST_PRIME_UI_TOKEN__", token).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/settings":
            self._send_json({
                "ok": True,
                "values": _current_values(),
                "configured": _configured_values(),
                "fields": [_field_payload(f) for f in SETTING_FIELDS],
            })
            return
        if path == "/api/meta":
            self._send_json({
                "ok": True,
                "providers": _provider_info(),
                "env_path": str(ENV_PATH),
                "fixedOrderReasons": FIXED_ORDER_REASONS,
                "mainPathSections": list(MAIN_PATH_SECTIONS),
            })
            return
        if path == "/api/rules":
            try:
                data = _load_rules_data()
                self._send_json({"ok": True, **data})
            except (RuleLoadError, json.JSONDecodeError, OSError) as exc:
                self._send_json({"ok": False, "message": str(exc)}, status=400)
            return
        if path == "/api/prefs":
            self._send_json({"ok": True, "prefs": load_prefs(REPO_ROOT)})
            return
        if path == "/api/diagnostics":
            self._send_json(dump_diagnostics(load_environment=True))
            return
        if path == "/api/actions":
            prefs = load_prefs(REPO_ROOT)
            ranked = ranked_action_ids(RANKABLE_TOOLBAR_ACTIONS, prefs, fixed=set())
            self._send_json({
                "ok": True,
                "actions": _list_actions(prefs),
                "rankedToolbar": ranked,
            })
            return
        self._send_json({"ok": False, "message": "Not found"}, status=404)

    def do_POST(self) -> None:
        if not self._local_request_allowed(require_origin=True):
            return
        path = urlparse(self.path).path
        if path.startswith("/api/") and not self._write_authorized():
            self._send_json(
                {"ok": False, "message": "This local settings session is not authorized."},
                status=403,
            )
            return
        if path == "/api/settings":
            try:
                payload = self._read_json()
                merged = _merge_save_payload(payload)
                save_env_file(ENV_PATH, merged)
                log_event("settings.persisted", "settings saved", component="settings_gui")
                self._send_json({"ok": True, "message": "Settings saved"})
            except Exception as exc:
                self._send_json({"ok": False, "message": str(exc)}, status=400)
            return
        if path == "/api/rules":
            try:
                payload = self._read_json()
                _save_rules_data(payload.get("rules", []))
                log_event("rules.persisted", "rules saved", component="settings_gui")
                self._send_json({"ok": True, "message": "Rules saved"})
            except (RuleLoadError, json.JSONDecodeError, TypeError, OSError) as exc:
                self._send_json({"ok": False, "message": str(exc)}, status=400)
            return
        if path == "/api/prefs":
            try:
                payload = self._read_json()
                prefs = save_prefs(REPO_ROOT, payload)
                self._send_json({"ok": True, "prefs": prefs})
            except Exception as exc:
                self._send_json({"ok": False, "message": str(exc)}, status=400)
            return
        if path.startswith("/api/actions/"):
            action_id = path[len("/api/actions/"):]
            try:
                params = self._read_json()
            except json.JSONDecodeError:
                params = {}
            result = _invoke_action(action_id, params)
            self._send_json(result, status=200 if result.get("ok") else 400)
            return
        if path == "/api/shutdown":
            try:
                _persist_shutdown_payload(self._read_json())
            except Exception as exc:
                self._send_json({"ok": False, "message": str(exc)}, status=400)
                return
            self._send_json({"ok": True})
            self._schedule_shutdown()
            return
        self._send_json({"ok": False, "message": "Not found"}, status=404)


def run_server(port: int, open_browser: bool) -> None:
    configure_logging(also_stderr=False)
    log_event("settings_gui.startup", f"listening on port {port}", component="settings_gui")
    server = ThreadingHTTPServer(("127.0.0.1", port), SettingsHandler)
    SettingsHandler.http_server = server
    SettingsHandler.ui_token = secrets.token_urlsafe(32)
    url = f"http://127.0.0.1:{port}/"
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local settings GUI for Optimist Prime")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port (default 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()
    run_server(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
