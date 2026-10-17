"""
Moderator attention summaries and modmail body builders.
"""

import os


def _community_rule_context(context: dict) -> str:
    """Return relevant official rules without adding another provider call."""
    try:
        from community_rules import relevant_context

        ids = context.get("community_rule_ids", ()) or ()
        return relevant_context(ids, include_guidance=True, max_chars=4000)
    except Exception as exc:
        return '{"reference_unavailable": "' + str(exc).replace('"', "'") + '"}'


def generate_mod_attention_summary(context: dict, llm_model) -> str | None:
    """
    Generate an AI summary for moderator attention using the LLM.

    Args:
        context: Dict with keys: kind, username, content_excerpt, rule_reason, metrics
        llm_model: Initialized LLM model

    Returns:
        Summary text or None on error
    """
    default_prompt = (
        "Write a brief, neutral summary for subreddit moderators: what happened, "
        "why it may need attention, and what to verify. Use plain sentences only. "
        "Stay factual; do not invent facts beyond the provided context."
    )
    custom_prompt = os.environ.get("BOT_PROMPT_MOD_ATTENTION_SUMMARY", "").strip()
    prompt_text = custom_prompt or default_prompt

    lines = [
        prompt_text,
        "",
        "COMMUNITY RULE REFERENCE (official rules are authoritative; moderator guidance is not an additional rule):",
        _community_rule_context(context),
        "",
        f"Alert type: {context.get('kind', 'unknown')}",
        f"User: u/{context.get('username', 'unknown')}",
    ]

    if context.get("rule_reason"):
        lines.extend(["", "<moderation_evidence>", "Rule evaluation reason:", context["rule_reason"], "</moderation_evidence>"])

    if context.get("content_excerpt"):
        lines.extend(["", "<moderation_evidence>", "Content excerpt:", context["content_excerpt"][:800], "</moderation_evidence>"])

    if context.get("metrics"):
        m = context["metrics"]
        lines.extend([
            "",
            "Participation metrics:",
            f"- Average comment score: {m.get('average_score', 0):.2f}",
            f"- Local comment count: {m.get('comment_count', 0)}",
        ])

    lines.extend(["", "Output only the summary (2-4 sentences):"])

    full_prompt = "\n".join(lines)

    try:
        response = llm_model.generate_content(
            [{"role": "user", "parts": [full_prompt]}],
            generation_config={"temperature": 0.2, "max_output_tokens": 300},
        )
        return response.text.strip() or None
    except Exception as e:
        print(f"    Error generating mod attention summary: {e}")
        return None


def build_violation_modmail_body(author: str, is_post: bool, reason: str, summary: str | None, link: str) -> str:
    """Build modmail body for a content violation alert."""
    content_type = "Post" if is_post else "Comment"
    lines = [
        "**AI Moderation -- Moderation Alert**",
        "",
        f"**Type:** {content_type} by u/{author}",
    ]
    if summary:
        lines.extend(["", "**AI review summary:**", summary])
    lines.extend(["", "**Rule match reason:**", reason, "", f"**Link:** {link}"])
    return "\n".join(lines)


def build_removal_modmail_body(author: str, is_post: bool, reason: str, summary: str | None, link: str) -> str:
    """Build modmail body for a content removal."""
    content_type = "Post" if is_post else "Comment"
    lines = [
        "**AI Moderation -- Content Removed**",
        "",
        f"**Type:** {content_type} by u/{author}",
        "**Action:** Content was removed by AI moderation.",
    ]
    if summary:
        lines.extend(["", "**AI review summary:**", summary])
    lines.extend(["", "**Rule match reason:**", reason, "", f"**Link:** {link}"])
    return "\n".join(lines)


def build_removal_public_reply(is_post: bool, reason: str, summary: str | None) -> str:
    """Build public reply for removed content."""
    content_type = "post" if is_post else "comment"
    body = (
        f"Your {content_type} was removed because it violates community guidelines.\n\n"
        f"**Reason:** {reason}"
    )
    if summary:
        body += f"\n\n**Details:** {summary}"
    return body


def build_audit_message(action: str, reason: str, summary: str | None = None, extra: str | None = None) -> str:
    """Build a concise audit message."""
    parts = []
    if extra:
        parts.append(extra)
    parts.append(f"Reason: {reason}")
    if summary:
        parts.append(f"AI review summary: {summary}")
    return " ".join(parts)
