# Optimist Prime feature guide

This guide describes general-purpose capabilities represented in the Python source. It does not claim that every path has been exercised against a live Reddit community.

## Status terms

- **In source** means relevant implementation is present.
- **Off by default** means the fresh-install configuration does not enable the feature.
- **Operator-configured** means the operator supplies the settings or rules that define its behavior.
- **Needs moderator review** means an AI result is information for a moderator, not an instruction to take action.

## Decision and response flow

1. Local checks apply feature switches, permissions, eligibility rules, limits, and duplicate controls.
2. The structured decision client handles supported routing and classification tasks.
3. The selected text-generation provider writes summaries or replies when the task needs prose.
4. Local code checks permissions and safety settings before a Reddit write.

The decision and text-generation stages have separate roles. A generated reply does not grant permission to perform a moderation action.

## Reading and conversations

| Capability | Status | Description |
| --- | --- | --- |
| Post and comment summaries | In source; off by default | Creates short summaries for eligible posts or comments. |
| Linked-Reddit summaries | In source; off by default | Can summarize eligible Reddit content linked from a discussion. |
| Discussion summaries | In source; off by default | Can create or update a summary when a discussion reaches configured activity limits. |
| Direct requests and contextual replies | In source; off by default | Responds to supported direct requests and eligible replies, subject to configured limits. |
| Private conversations | In source; off by default | Supports selected one-to-one user and moderator interactions through Reddit's private messaging features. |
| Moderator request routing | In source; requires moderator permissions | Routes supported moderator requests to enabled tools and checks authority before an action. |

## Configurable moderation and history review

| Capability | Status | Description |
| --- | --- | --- |
| Content checks | In source; off by default; operator-configured | Evaluates eligible posts or comments against rules the operator supplies. Rules can select supported outcomes such as logging, reporting, removal, or moderator notification. |
| Public-history review | In source; off by default; operator-configured | A bounded sample of public posts and comments can be reviewed after a moderator report, an explicit moderator request, or a local operator preview or recovery command. The bot does not select community members for unsolicited review. No active criteria are included in a fresh installation. Automatic bans require additional explicit settings described in the configuration guide. |
| Incomplete-review handling | In source | Empty, failed, or incomplete evidence must not be presented as a clear review. Model results describe a bounded sample and are not item-level proof. |

The project does not supply a community review policy. Review history criteria and sample evidence before enabling any automatic action; a model score is not item-level proof.

Scheduled crossposting is off by default. It requires operator-written selection guidance and an explicit score cutoff, plus operator-selected rate, age, UTC schedule, and duplicate-search limits before it will fetch candidates or call an AI provider.

## Optional user flair

| Capability | Status | Description |
| --- | --- | --- |
| Self-service flair requests | Optional; off by default; operator-configured | A user can request a flair value from the exact allowlist supplied by the operator. The bot does not generate descriptions from a user's history or change another user's flair. |

## Moderator and operator tools

| Capability | Status | Description |
| --- | --- | --- |
| Local moderator console | In source; requires local setup | A desktop application presents settings, diagnostics, and supported local controls. It is not a hosted service. |
| Safe Mode and dry run | Safe Mode is on by default; dry run is available by command | Lets operators inspect behavior without Reddit writes. A dry run can still read Reddit and call AI services. |
| Request pacing and retries | In source | Shared request controls and retry state help limit repeated or excessive work. |
| Duplicate and overlap controls | In source | Local state and locks help avoid repeating actions or overlapping worker cycles. |
| Diagnostics and local records | In source | Local tools can report configuration readiness and runtime status. Records stay on the operator's machine unless the operator moves them. |
| Comment observer | Optional local utility; read-only | A separate local view can display recent comments for troubleshooting and does not moderate or post. |

## Operating limits

The bot depends on Reddit permissions, provider availability, and settings supplied by each operator. AI moderation can misread context, and Reddit history may be incomplete or unavailable. Test with Safe Mode and fake data first, and keep a human responsible for policy decisions.

See [configuration](docs/configuration.md), [testing](docs/testing.md), and [dependencies](docs/dependencies.md) for setup details.

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
