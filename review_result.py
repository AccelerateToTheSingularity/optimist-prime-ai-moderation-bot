"""Plain-language results shared by moderator-requested and report-queue reviews."""
from dataclasses import dataclass
from urllib.parse import quote

from moderation_rules import DISABLED_RULE_IDS, history_criterion_score_is_current


@dataclass(frozen=True)
class ReviewResult:
    text: str
    completed: bool


def format_review_result(
    history,
    assessment,
    *,
    display_name=lambda name: name,
    username: str | None = None,
):
    """Summarize which operator-configured history criteria matched."""
    username = str(username or getattr(history, "username", "") or "").strip()
    user_link = (
        f"[u/{username}](https://www.reddit.com/user/{quote(username, safe='_-')}/)"
        if username
        else ""
    )
    comments = len(history.comments)
    posts = len(history.posts)
    counts = [f"{comments} comment{'s' if comments != 1 else ''}"]
    if posts:
        counts.append(f"{posts} post{'s' if posts != 1 else ''}")
    count_text = " and ".join(counts) + " scanned"
    if user_link:
        count_text += f" for {user_link}"
    count_text += "."
    if not history.items:
        return ReviewResult(
            "0 comments scanned"
            + (f" for {user_link}" if user_link else "")
            + ". Review incomplete: no eligible history was available.",
            False,
        )
    if assessment is None:
        return ReviewResult(count_text + " Review incomplete: the configured history review did not finish.", False)

    criterion_probabilities = getattr(assessment, "criterion_probabilities", {})
    if criterion_probabilities:
        flagged = [
            (name, criterion_probabilities[name])
            for name in getattr(assessment, "matched_rules", ())
            if name in criterion_probabilities
            and name not in DISABLED_RULE_IDS
            and history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
        ]
        complete = (
            not history.errors
            and bool(getattr(assessment, "scores_complete", True))
            and all(
                history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
                for name in criterion_probabilities
            )
        )
        if flagged:
            rule_order = {
                name: index for index, name in enumerate(getattr(assessment, "rule_names", ()))
            }
            flagged.sort(
                key=lambda row: (-row[1], rule_order.get(row[0], len(rule_order)))
            )
            names = [str(display_name(name)).strip() for name, _score in flagged]
            decision = f"Matched configured criteria: {_join_names(names)}."
            headline = count_text + " " + decision
            if not complete:
                headline = count_text + " Review incomplete; possible configured match: " + _join_names(names) + "."
        else:
            headline = (
                count_text + " No configured criterion matched."
                if complete
                else count_text + " Review incomplete; no configured criterion matched in the available results."
            )
        lines = [headline]
        truncated = getattr(assessment, "evidence_truncated_items", 0)
        if truncated:
            lines.append(_truncation_note(truncated))
        return ReviewResult("\n".join(lines), complete)

    if (
        history.errors
        or not bool(getattr(assessment, "scores_complete", True))
        or any(
            not history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
            for name in getattr(assessment, "matched_rules", ())
        )
    ):
        lines = [count_text + " Review incomplete; there is no final criterion result."]
        truncated = getattr(assessment, "evidence_truncated_items", 0)
        if truncated:
            lines.append(_truncation_note(truncated))
        return ReviewResult("\n".join(lines), False)

    # Compatibility for historical assessments without the current Noul scores.
    active_matches = [
        name for name in assessment.matched_rules
        if name not in DISABLED_RULE_IDS
        and history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
    ]
    if active_matches:
        names = [str(display_name(name)).strip() for name in active_matches]
        return ReviewResult(
            count_text + f" Matched configured criteria: {_join_names(names)}.",
            True,
        )
    return ReviewResult(count_text + " No configured criterion matched.", True)


def format_no_new_history_result(
    username: str,
    previous_review=None,
    *,
    display_name=lambda name: name,
) -> str:
    """Explain that a request reused an earlier result without another Jev call."""
    username = str(username or "").strip()
    link = (
        f"[u/{username}](https://www.reddit.com/user/{quote(username, safe='_-')}/)"
        if username
        else "that user"
    )
    message = f"No new posts or comments were available for {link}, so I didn't run another review."
    if not previous_review:
        return message + " The earlier review result is unavailable."

    assessment, _history_count, history_incomplete, updated_at = previous_review
    if getattr(assessment, "outdated_criterion_ids", ()) or any(
        not history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
        for name in getattr(assessment, "matched_rules", ())
    ):
        return (
            message
            + " The earlier review used an older criterion definition, so it isn't complete under the current settings."
        )
    names = [
        str(display_name(name)).strip()
        for name in getattr(assessment, "matched_rules", ())
        if name not in DISABLED_RULE_IDS
        and history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
    ]
    review_complete = (
        bool(getattr(assessment, "scores_complete", True))
        and not bool(history_incomplete)
        and all(
            history_criterion_score_is_current(name, getattr(assessment, "criterion_score_revisions", {}))
            for name in getattr(assessment, "matched_rules", ())
        )
    )
    if not review_complete:
        if names:
            earlier = "The earlier review found possible configured matches for " + _join_names(names) + ", but did not finish."
        else:
            earlier = "The earlier review did not finish, so it established neither a match nor a no-match result."
    elif names:
        earlier = "The earlier review matched configured criteria for " + _join_names(names) + "."
    else:
        earlier = "The earlier review found no configured criterion match."
    date = _short_date(updated_at)
    if date:
        if not review_complete:
            summary = (
                "found possible configured matches for " + _join_names(names) + ", but did not finish."
                if names
                else "did not finish, so it established neither a match nor a no-match result."
            )
        else:
            summary = (
                "matched configured criteria for " + _join_names(names) + "."
                if names
                else "found no configured criterion match."
            )
        earlier = f"The earlier review from {date} {summary}"
    return message + " " + earlier


def _short_date(value: str) -> str:
    from datetime import datetime

    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return parsed.strftime("%d %b %Y")


def _join_names(names: list[str]) -> str:
    if len(names) < 2:
        return names[0] if names else "the listed criteria"
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _truncation_note(item_count: int) -> str:
    return (
        f"Some text in {item_count} long item(s) was shortened for the review."
    )
