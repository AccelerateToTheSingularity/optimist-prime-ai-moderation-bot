# Architecture overview

Optimist Prime is a local Python bot for Reddit. A bounded worker cycle reads eligible Reddit activity, applies local checks, and calls an AI service only when an enabled feature needs it.

## Decision and response flow

1. Local checks apply feature switches, moderator permissions, eligibility rules, request limits, and duplicate controls.
2. The structured-decision client handles supported routing and moderation checks.
3. The configured text provider generates a reply, summary, or explanation when needed.
4. Local code checks permissions and Safe Mode before a Reddit write.

Structured decisions and generated text have separate roles. Text generation does not grant authority to take an action.

## Main components

- `bot_runner.py` runs a bounded cycle and supported preview paths.
- `moderation_rules.py`, `ordered_gate.py`, and `content_moderation.py` load and evaluate operator-authored content rules.
- `user_history.py`, `history_gate_jev.py`, and the review handlers support bounded history collection for moderator reports, explicit moderator requests, and deliberate local operator preview or recovery commands, when the operator has supplied criteria and enabled the relevant feature.
- `jev_client.py` and `llm_client.py` contain the structured-decision and text-generation transports.
- `reddit_observability.py` and `reddit_request_budget.py` apply shared request pacing to Reddit API calls.
- `state_store.py`, review ledgers, and runtime activity modules manage local runtime state.
- `moderator_console.py` and `settings_gui.py` provide local operator controls. `comment_feed.py` is a separate, read-only local observer.

## Configuration and data boundaries

`.env.example` documents environment settings. The operator's `.env`, live moderation rules, history-review criteria, credentials, runtime state, and logs belong on the operator's machine and are excluded from version control.

The shipped moderation rules and community-reference file are empty or inactive examples, not a community policy. Local community rules, moderator guidance, and history criteria are excluded from Git. History review has no active criteria in a fresh installation; it must fail closed until the operator supplies criteria and explicitly enables it. The worker does not select community members for unsolicited history review.

The moderator Modmail assistant authorizes requests and pending replies only against the moderator roster verified during the current bot cycle. A cached roster cannot authorize a request when the fresh check fails.

Reddit text may be sent to the configured AI services when an enabled feature makes a provider request. Review those services' data-handling terms before enabling such a feature.
