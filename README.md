# Reddit Bot (r/accelerate) — Optimist Prime

Multi-function AI community assistant for [r/accelerate](https://www.reddit.com/r/accelerate/). See **[FEATURES.md](FEATURES.md)** for a full capability overview you can share with mods and members.

## How it works

Runs on GitHub Actions on a schedule (when enabled), polling Reddit and performing configured tasks each cycle.

## Stats dashboard

**Live stats:** [https://acceleratetothesingularity.github.io/optimist-prime-ai-moderation-bot/](https://acceleratetothesingularity.github.io/optimist-prime-ai-moderation-bot/)

## Public repository policy

This repo is **public** to use unlimited free GitHub Actions minutes. Do not commit:

- API keys, tokens, or `.env`
- Live `data/rules.json` (use `data/rules.example.json` + GitHub secret `BOT_MODERATION_RULES_JSON`)
- `data/bot_state.json`, `data/audit_log.json` (runtime / PII-adjacent; gitignored locally)

Edit moderation rules locally with `py manage_rules.py validate` / `list` / `enable` / `disable`.

## Setup

### 1. Fork this repository

### 2. Register a Reddit App

Go to https://www.reddit.com/prefs/apps/ and create a **web app**:
- Name: `OptimistPrimeModBot`
- Redirect URI: `http://localhost:8080`

Note the **client ID** (under the app name) and **client secret**.

### 3. Get a Refresh Token

```bash
set REDDIT_CLIENT_ID=your_client_id
set REDDIT_CLIENT_SECRET=your_client_secret
py obtain_refresh_token.py
```

Log in as the bot account and click "Allow". Copy the refresh token.

### 4. Add GitHub Secrets

| Secret | Description |
|--------|-------------|
| `REDDIT_CLIENT_ID` | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Reddit app client secret |
| `REDDIT_REFRESH_TOKEN` | Refresh token from step 3 |
| `REDDIT_APP_NAME` | User-Agent app name (default: OptimistPrimeModBot) |
| `OPENAI_API_KEY` or `LLM_API_KEY` | LLM API key |
| `BOT_MODERATION_RULES_JSON` | Optional: production moderation rules (JSON array) |
| `BOT_LLM_PROVIDER` | Optional: `minimax`, `openai`, `claude`, `gemini`, `deepseek`, `glm`, `groq`, `mistral`, `together`, `xai`, `custom` |

Optional email secrets for failure notifications: `EMAIL_USERNAME`, `EMAIL_PASSWORD`, `NOTIFICATION_EMAIL`.

### 5. Enable Actions (when ready)

**Master switch:** repository variable `BOT_ENABLED` must be `true` for the bot to run. Keep it **`false`** until you are ready (current setting).

The workflow runs on a **3-minute schedule** automatically, but the bot job is skipped while `BOT_ENABLED=false`. When enabling:

1. Set `BOT_ENABLED=true` in repo **Settings → Secrets and variables → Actions → Variables**
2. Default profile is **`post_tldr_only`** (post TLDRs only). Turn on other features one at a time via `BOT_*` variables or the local settings GUI.

## Configuration

Runtime toggles live in `config.py` and can be overridden with `BOT_*` environment variables. Use `BOT_PROFILE=minimax_starter` for a fuller feature preset.

### Local settings GUI

For local development, use the browser-based settings editor (writes `.env`, gitignored):

```bash
py settings_gui.py
```

**Start Menu (Windows):** run once from the repo folder:

```powershell
powershell -ExecutionPolicy Bypass -File install_start_menu_shortcut.ps1
```

Then open **Start → Optimist Prime Settings** — no terminal window, light-themed UI, closes when you close the browser tab or click **Done**. Changes **auto-save** (no Save button). Edit AI moderation rules directly in the **AI moderation rules** section (not raw JSON).

Opens `http://127.0.0.1:8765/` with toggles for LLM provider, Reddit credentials, moderation, TLDR, flair, and more. Password fields left blank keep existing secrets. Use **Show advanced** for uncommon fields; **Customize settings** to hide/restore optional controls.

After editing, run `py bot_runner.py --dry-run` to test. For production (GitHub Actions), mirror the same `BOT_*` vars as repository secrets/variables.

Local health check:

```bash
py diagnostics.py --pretty
```

Supported `BOT_LLM_PROVIDER` presets locally: `minimax`, `openai`, `claude`, `gemini`, `deepseek`, `glm`, `groq`, `mistral`, `together`, `xai`, `custom`.

## Rule management (local)

```bash
py manage_rules.py validate
py manage_rules.py list
py manage_rules.py enable spam_detector
py manage_rules.py push-wiki   # optional wiki sync
```

## API compliance (2026)

- User-Agent: `script:<app>:v<version> (by /u/<developer>)`
- Bot posts use the standard footer identifying automated content

Dry-run: `py bot_runner.py --dry-run`  
Safe mode: `BOT_SAFE_MODE=true`

## Costs

- **GitHub Actions**: Free for public repositories
- **LLM**: Billed by your chosen provider (OpenAI, Gemini, etc.)

## License

MIT
