# Testing and safe operation

The test suite is designed to use synthetic examples and fake Reddit or provider objects. These checks are intended to avoid posting, moderating, sending messages, or changing flair on Reddit.

## Local checks

From the repository root, run:

```powershell
py -m unittest discover -s tests
py -m compileall .
py manage_rules.py validate
```

On macOS or Linux, use `python3` instead of `py`.

The rule validator checks `data/rules.json`. Create it from `data/rules.example.json` and configure it for your own community before validating.

## Safe dry run

With valid Reddit and provider credentials, `py bot_runner.py --dry-run` runs the worker in Safe Mode. It may make read-only Reddit requests and send eligible text to the configured AI services, which can incur provider usage. It is not an offline test.

The fresh configuration keeps Safe Mode on and leaves moderation, replies, history review, and other write-capable features disabled. The worker does not select community members for unsolicited history reviews; reviews run only for reported content, authorized moderator requests, or a specific local operator command.

`py bot_runner.py --preview-history-user USERNAME` previews one user's bounded public history using your configured criteria. It does not save bot state or perform Reddit writes, but it does read Reddit and may send selected text to AI providers. `py bot_runner.py --review-user USERNAME` is a targeted recovery path for a report that was no longer available when the worker checked; it follows your configured review and Safe Mode settings. Treat both commands as deliberate review actions and use only where you have appropriate authorization.

Keep Safe Mode on while reviewing your settings and rules. Enable only features you understand and have configured; do not use live Reddit actions as routine tests.

## Test data and local records

Tests should use synthetic comments, usernames, IDs, and subreddit names with fake Reddit/provider objects. Do not add copied Reddit content or real user records to fixtures.

Keep `.env`, `data/rules.json`, `data/community_rules.reference.json`, history-review criteria, runtime state, review ledgers, and logs out of commits. The repository's `.gitignore` excludes the standard local files, but review the staged files before committing.
