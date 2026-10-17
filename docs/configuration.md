# Configuration guide

Configure Optimist Prime for a community and accounts that you control. The sample settings do not contain moderation policy for your community. Keep all write-capable features off while you set up and review the bot.

## 1. Install the project

Use Python 3.11 or newer and install dependencies from the repository root:

```powershell
py -m pip install -r requirements.txt
```

These commands use PowerShell. On macOS or Linux, use `python3` instead of `py` and `cp SOURCE DESTINATION` instead of `Copy-Item SOURCE DESTINATION`.

## 2. Add Reddit credentials

Create a traditional Reddit OAuth web application for the bot account, using `http://localhost:8080` as its redirect URL, and copy `.env.example` to `.env`. Enter the client ID, client secret, refresh token, expected bot account name, and the subreddit name without the `r/` prefix. The account name must match the account authorized by the refresh token; live runs stop if it is missing or does not match.

The included `obtain_refresh_token.py` helper completes Reddit's browser authorization flow and saves the refresh token to the local `.env` file. It requests only `identity` and `read` by default. If you enable a feature that needs additional Reddit API access, add only the required names to `REDDIT_OAUTH_SCOPES`, then authorize the app again. Check Reddit's current [OAuth scope documentation](https://www.reddit.com/dev/api/oauth) and grant moderator access only for features you intend to use.

History review uses the target community by default. Set `BOT_HISTORY_SUBREDDITS` only if your own policy explicitly allows reviews to include additional communities; leave it blank to keep the sample limited to the target community.

Never commit `.env` or share its credentials. The `.env` file is ignored by Git.

## 3. Choose AI providers

The structured-decision features use the OpenRouter service and need `OPENROUTER_API_KEY`. Text generation uses the provider selected by `BOT_LLM_PROVIDER`; set `LLM_MODEL` and the API key required for that provider. A custom OpenAI-compatible endpoint can be selected with `BOT_LLM_PROVIDER=custom` and configured with `OPENAI_BASE_URL` or `LLM_BASE_URL`.

Reddit text may be sent to those services when an enabled feature needs an AI decision or response. Check provider terms and data handling before turning on those features.

## 4. Keep community reference data local

Copy the empty reference example to the local runtime path:

```powershell
Copy-Item data/community_rules.reference.example.json data/community_rules.reference.json
```

On macOS or Linux:

```sh
cp data/community_rules.reference.example.json data/community_rules.reference.json
```

You may add your community's official rules and optional moderator guidance to this file. Set its `subreddit` field to the name in your `.env` before using it. The file is ignored by Git because moderator guidance may contain private information; keep the tracked example empty. `py check_community_rules.py` compares the saved official-rule snapshot with Reddit in read-only mode, but it does not create or update the snapshot.

## 5. Define content rules

Copy the example rule file to the local file used by the bot:

```powershell
Copy-Item data/rules.example.json data/rules.json
```

On macOS or Linux: `cp data/rules.example.json data/rules.json`.

Edit the rules for your community. Give each rule a clear name and description, choose whether it is active, set its order and target, and select only the action you intend. The example entries are inactive and must not be treated as a ready-made community policy.

Ban actions require the separate `BOT_AUTOMATED_BAN_ENABLED=true` setting and Safe Mode turned off. They are not enabled by activating a rule alone.

Validate the local rules before running the bot:

```powershell
py manage_rules.py validate
```

Keep `data/rules.json` local. It is ignored by Git so your community's policy is not included in commits.

## 6. Configure history review only when needed

The fresh installation supplies no active history-review criteria. History review is limited to reported users when report review is enabled, explicit requests from authorized moderators, or a specific local operator preview or recovery command. The bot does not select users for unsolicited history scans. To configure these review paths, copy the empty example to its ignored local path:

```powershell
Copy-Item data/history_criteria.example.json data/history_criteria.json
```

On macOS or Linux: `cp data/history_criteria.example.json data/history_criteria.json`.

Add only criteria that you define for your community. Each guide should describe the criterion, the evidence that would satisfy it, evidence that is not enough by itself, and important boundaries the review should respect. For example, the local file can use this schema:

```json
{
  "schema_version": 1,
  "guides": {
    "your_criterion": {
      "title": "Your criterion name",
      "description": "Describe what this criterion means.",
      "high_confidence_yes": "Describe the evidence that would satisfy it.",
      "not_sufficient_by_itself": "Describe evidence that is not enough alone.",
      "closest_boundaries": "Describe distinctions the review must preserve.",
      "active": true,
      "revision": 1
    }
  }
}
```

Set `BOT_HISTORY_CRITERIA_FILE` only if you want to store that local file elsewhere. Report-triggered reviews remain disabled until you set `BOT_REPORTED_USER_REVIEW_ENABLED=true`; moderator-requested reviews require an authorized moderator request. Both remain unavailable until you add criteria and set the match threshold.

Set `BOT_HISTORY_REVIEW_MATCH_THRESHOLD_PERCENT` to the cutoff you choose, `BOT_HISTORY_REVIEW_LOOKBACK_DAYS` to the oldest public history the bot may inspect, and `BOT_HISTORY_REVIEW_MAX_ITEMS` to the largest sample that may be sent to the AI provider (1–250). All three are required because they define the review policy and privacy scope; none has an inherited public default. The match cutoff labels which configured criteria count as possible matches; it does not authorize a ban.

Automatic bans require all of these settings and conditions: enable `BOT_AUTOMATED_BAN_ENABLED=true`, set `BOT_HISTORY_REVIEW_BAN_THRESHOLD_PERCENT`, provide at least one user-authored criterion, set the match cutoff, lookback window, and sample limit, enable report-triggered reviews or receive an explicit moderator request, and turn Safe Mode off. These settings are absent or off in a fresh installation. Review the criterion and sample evidence before enabling an automatic action; model scores are not item-level proof.

Local history-review records are retained for 30 days by default. Set `BOT_REPORTED_USER_REVIEW_STATE_RETENTION_DAYS` to choose a different retention period.

Keep `data/history_criteria.json` and any alternate criteria file local and out of Git unless you have reviewed it for public release.

## 7. Optional self-service flair

Self-service flair is off by default. To enable it, set `BOT_USER_CUSTOM_FLAIR_ENABLED=true` and provide a comma-separated list of exact values in `BOT_USER_CUSTOM_FLAIR_ALLOWED_VALUES`. Requests outside that allowlist are not accepted. This setting does not enable changes to another user's flair.

## 8. Optional crossposting

Crossposting is off by default. To enable it, provide the source and target communities, write your own selection guidance, and choose a match cutoff from 1 to 100. You must also choose a daily post limit (1–100), an eligible post-age range (0–8,760 hours), a UTC posting window (0–23 hours), and a duplicate-search period (1–3,650 days). The minimum age must not exceed the maximum, and the earliest posting hour must not exceed the latest. The bot will not fetch posts or call an AI provider until all of those choices are valid. Review the destination community, permissions, and public posting behavior before disabling Safe Mode.

## 9. Start safely

Keep `BOT_SAFE_MODE=true`. In a fresh configuration, moderation, history review, replies, and other actions that can write to Reddit are off. A misspelled Safe Mode value keeps Safe Mode enabled. Public reply features default to local logging; choose `comment`, `report`, or `modmail` only if that is the behavior you want. Before any live run, set `BOT_EXPECTED_USERNAME` to the exact account authorized by the refresh token. Do not disable Safe Mode until you have reviewed the specific feature, required permissions, rules, and safeguards.

Start with a dry run:

```powershell
py bot_runner.py --dry-run
```

A dry run may read Reddit content and call AI providers, but it is intended to prevent Reddit writes. It may incur provider costs. Use fake data for tests and keep real moderator actions out of routine validation.

## Local data

Keep `.env`, `data/rules.json`, `data/community_rules.reference.json`, history criteria, runtime state, review ledgers, and logs on the operator's machine. The standard paths are excluded by `.gitignore`; inspect the files staged for Git before every commit.
