# Optimist Prime

Optimist Prime is a Python bot for Reddit communities. It can summarize discussions, respond to direct requests, and run moderation checks based on rules supplied by the operator.

The project combines ordinary code for permissions and safety checks with AI services for selected decisions and text generation. Features are individually configurable. A feature being present in the source does not mean it is enabled or proven in a live community.

## Features

| Area | What the source provides | Fresh-install behavior |
| --- | --- | --- |
| Discussion summaries | Summaries for eligible posts, comments, and linked Reddit discussions | Off until configured |
| Conversations | Direct requests, contextual replies, and supported private moderator or user conversations | Reply features are off until configured |
| Content moderation | Ordered checks using rules written by the operator | Moderation is off; sample rules are inactive |
| Public-history review | Report-triggered checks, explicit moderator requests, and local operator preview or recovery commands against criteria written by the operator | No criteria are supplied; review stays unavailable until configured and enabled. The bot does not select users for unsolicited scans. |
| Scheduled crossposting | Selects and crossposts source-community posts using guidance written by the operator | Off; requires explicit source, destination, cutoff, rate, age, schedule, and duplicate limits. |
| User flair requests | Self-service requests for exact flair values on an operator allowlist | Off; requires an explicit allowlist |
| Moderator tools | A local desktop console, settings, diagnostics, and review tools | Requires local setup and Reddit permissions |
| Runtime safeguards | Safe Mode, dry-run support, request pacing, retries, and duplicate controls | Safe Mode is on; write features are off |

See [the feature guide](FEATURES.md) for more detail and [the configuration guide](docs/configuration.md) to set up your own community.

## Quick start

This repository is source code, not a hosted bot service. Use a Reddit account and AI-provider accounts that you control.

1. Install Python 3.11 or newer and the project dependencies. In PowerShell:

   ```powershell
   py -m pip install -r requirements.txt
   ```

   On macOS or Linux, use `python3 -m pip install -r requirements.txt`.

2. Create a local `.env` file from `.env.example`. Set the Reddit account, OAuth credentials, target subreddit, and provider settings for your own installation. Keep `.env` private. Public reply features remain off; their default delivery mode records actions locally until you deliberately choose another mode.

3. Copy `data/community_rules.reference.example.json` to `data/community_rules.reference.json`. Add only the community rules and moderator guidance you want the bot to use. This local file is ignored by Git because moderator guidance may be private.

4. Copy `data/rules.example.json` to `data/rules.json` and replace its inactive examples with rules for your community. History-review criteria are not preloaded; copy `data/history_criteria.example.json` to `data/history_criteria.json` and define your own before enabling a history-review feature. These local policy files are ignored by Git.

5. Validate the rule file and start with a dry run:

   ```powershell
   py manage_rules.py validate
   py bot_runner.py --dry-run
   ```

   On macOS or Linux, use `python3` in place of `py`.

To run repeated cycles in the current terminal, start the optional local worker with `py local_worker.py`.
Stop it with `Ctrl+C`; the project does not install a background service or scheduled task for you.

A dry run is not offline. With valid credentials, the bot can read Reddit content and send eligible text to the configured AI services, which may incur provider usage. Safe Mode prevents Reddit writes. Review [testing and safe operation](docs/testing.md) before enabling any write action.

## Important limits

Operators define the moderation rules and any history-review criteria. The repository does not prescribe a community policy. History review can run for reported content, an explicit moderator request, or a specific local operator preview or recovery command; the bot does not select community members for unsolicited review. Results are AI-generated assessments of a bounded sample, not proof about a person or a substitute for moderator judgment. Automatic bans are off by default and require a separate global opt-in, a user-defined threshold and criteria, an enabled review route, and Safe Mode turned off.

Reddit content may be sent to the configured AI services. Check each service's data-handling terms before enabling features that send it content. Automated moderation can make mistakes; keep write actions disabled while configuring and evaluating the bot.

## License

Optimist Prime is released under the MIT License. See [LICENSE](LICENSE).
