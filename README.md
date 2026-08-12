# Optimist Prime AI Moderation Bot

**Optimist Prime is an actively developed, open-source AI moderation and community-management platform.**
Originally built for [r/accelerate](https://www.reddit.com/r/accelerate/), it combines AI-assisted moderation with community utilities such as summaries, conversations, flairs, and crossposts.
It is designed around the practical demands of a large, high-traffic community today, with a longer-term goal of becoming a configurable platform other online communities can deploy for their own moderation and community-management workflows.

**[Explore current features](FEATURES.md)** · **[View the live statistics dashboard](https://acceleratetothesingularity.github.io/optimist-prime-ai-moderation-bot/)** · **[MIT License](LICENSE)**

## Real-world deployment context

Optimist Prime is being developed and tested for the operational realities of r/accelerate—not as a toy moderation demo.
The community environment has approximately:

| Measure | Scale |
| --- | ---: |
| Members | ~72,800 |
| Views, last 12 months | 34.9 million |
| Views, most recent 30 days | 4.5 million |
| Published comments, last 12 months | ~508,000 |
| Published posts, last 12 months | ~19,500 |

That scale informs the platform's emphasis on feature gates, rate limits, duplicate-action prevention, auditability, and conservative controls around moderation actions.

## What it does today

- **AI-assisted moderation:** evaluates content through discrete, configurable rules; produces contextual moderator alerts; and supports logging, reporting, removal, and modmail actions.
- **Community intelligence:** generates post and comment TLDRs, thread digests, and summaries for linked Reddit posts.
- **Conversation and participation:** handles direct replies and summons, recognizes contributors, and manages opt-in acceleration flairs.
- **Community operations:** provides troll early-warning signals, curated crossposts, a local settings UI, diagnostics, and a live statistics dashboard.

See [FEATURES.md](FEATURES.md) for the complete user- and moderator-facing capability overview.

## AI architecture and safeguards

The bot separates discrete moderation rules from the optional legacy broad moderation prompt, rather than relying only on a single undifferentiated model instruction.
Each major capability is independently feature-toggled and can be controlled with `BOT_*` environment variables or the local settings UI.

- **Safe execution:** `--dry-run` logs intended actions without posting; `BOT_SAFE_MODE=true` forces the same behavior.
- **Controlled moderation:** content moderation defaults to `log`; reporting, removal, modmail, and auto-ban behavior require explicit configuration and feature toggles.
- **Operational guardrails:** daily and per-run limits, user cooldowns, LLM-call caps, duplicate-action prevention, and audit logging help constrain automation at scale.
- **Configurable rules:** runtime moderation rules can be supplied as local JSON or a GitHub secret; local rule sets can also be synchronized with a subreddit wiki page.

## Model providers and cost

Optimist Prime uses an OpenAI-compatible client and supports OpenAI plus MiniMax, Claude, Gemini, DeepSeek, GLM, Groq, Mistral, Together, xAI, and compatible custom endpoints.
Select a provider with `BOT_LLM_PROVIDER`; provider-specific credentials and endpoint overrides are resolved through environment variables.

The project is **free and open source under the [MIT License](LICENSE)**.
GitHub Actions is free for this public repository; LLM usage is billed by the provider you choose.

## How it works

When enabled, the bot runs on GitHub Actions on a schedule and polls Reddit to perform the configured phases.
The current runner covers content moderation, post and comment summaries, inbox replies, summons, auto-ban checks, crossposts, acceleration flairs, and background scan work.

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

Log in as the bot account and click "Allow".
Copy the refresh token.

### 4. Add GitHub Secrets

| Secret | Description |
| --- | --- |
| `REDDIT_CLIENT_ID` | Reddit app client ID |
| `REDDIT_CLIENT_SECRET` | Reddit app client secret |
| `REDDIT_REFRESH_TOKEN` | Refresh token |
| `REDDIT_APP_NAME` | User-Agent app name (default: OptimistPrimeModBot) |
| `OPENAI_API_KEY` or `LLM_API_KEY` | LLM API key |
| `BOT_MODERATION_RULES_JSON` | Optional: production moderation rules (JSON array) |
| `BOT_LLM_PROVIDER` | Optional: `minimax`, `openai`, `claude`, `gemini`, `deepseek`, `glm`, `groq`, `mistral`, `together`, `xai`, or `custom` |

Optional email secrets for failure notifications: `EMAIL_USERNAME`, `EMAIL_PASSWORD`, and `NOTIFICATION_EMAIL`.

### 5. Enable Actions when ready

**Master switch:** repository variable `BOT_ENABLED` must be `true` for the bot to run.
Keep it **`false`** until the configuration is ready.

The workflow runs on a **3-minute schedule**, but the bot job is skipped while `BOT_ENABLED=false`.
When enabling:

1. Set `BOT_ENABLED=true` in **Settings → Secrets and variables → Actions → Variables**.
2. The default profile is **`post_tldr_only`**.
   Turn on other features one at a time through `BOT_*` variables or the local settings UI.

## Configuration and local development

Runtime toggles live in `config.py` and can be overridden with `BOT_*` environment variables.
Use `BOT_PROFILE=minimax_starter` for a fuller feature preset.

### Local settings UI

For local development, use the browser-based settings editor (it writes the gitignored `.env` file):

```bash
py settings_gui.py
```

**Start Menu (Windows):** run once from the repository folder:

```powershell
powershell -ExecutionPolicy Bypass -File install_start_menu_shortcut.ps1
```

Then open **Start → Optimist Prime Settings**.
The UI autosaves changes, lets you edit AI moderation rules directly, and exposes controls for the LLM provider, Reddit credentials, moderation, TLDRs, flairs, and more.
It opens at `http://127.0.0.1:8765/`.

After editing, validate safely:

```bash
py bot_runner.py --dry-run
py diagnostics.py --pretty
```

For GitHub Actions production use, mirror the same `BOT_*` values as repository secrets or variables.

## Rule management

```bash
py manage_rules.py validate
py manage_rules.py list
py manage_rules.py enable spam_detector
py manage_rules.py push-wiki   # optional wiki sync
```

## Security, privacy, and API compliance

This public repository must never contain API keys, tokens, or `.env` files.
Keep live `data/rules.json` and runtime state (`data/bot_state.json`, `data/audit_log.json`) private; use `data/rules.example.json` and the `BOT_MODERATION_RULES_JSON` GitHub secret for production rules.

- User-Agent format: `script:<app>:v<version> (by /u/<developer>)`
- Bot posts include a footer identifying automated content.
- Prefer refresh-token authentication (`REDDIT_REFRESH_TOKEN`); password authentication is a legacy fallback.

## License

Optimist Prime AI Moderation Bot is released under the [MIT License](LICENSE).
