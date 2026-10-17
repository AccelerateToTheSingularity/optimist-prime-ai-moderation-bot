"""Content moderation evaluation and action dispatch.

Jev evaluates configured moderation rules. A configured text model may then
provide an explanation and supporting excerpts for selected rules.
"""



import re

import config



from moderation_rules import EXECUTABLE_ACTIONS, ModerationRule



IMPLEMENTED_ACTIONS = EXECUTABLE_ACTIONS





# Discrete rule-based evaluator

# ---------------------------------------------------------------------------



def parse_rule_response(response_text: str, rules: list[ModerationRule]) -> dict[str, dict]:

    """

    Parse the LLM's RULE_NAME: YES/NO response into matched rules.



    Returns:

        Dict mapping rule name -> {"matched": bool, "reason": str}

    """

    rule_map = {r.name: r for r in rules}

    results = {}



    for line in response_text.strip().splitlines():

        line = line.strip()

        if not line:

            continue



        # Match "rule_name: YES" or "rule_name: NO" (with optional reasoning after)

        match = re.match(r"^(.+?):\s*(YES|NO)(?:\s*[-–:]\s*(.+))?$", line, re.IGNORECASE)

        if not match:

            continue



        raw_name = match.group(1).strip()

        decision = match.group(2).strip().upper()

        reason = match.group(3).strip() if match.group(3) else ""



        # Fuzzy match: try exact name, then lowercase with underscores normalized

        normalized = raw_name.lower().replace("-", "_").replace(" ", "_")

        matched_name = None

        for rule_name in rule_map:

            if rule_name == raw_name or rule_name.lower() == normalized:

                matched_name = rule_name

                break



        if matched_name is not None:

            results[matched_name] = {

                "matched": decision == "YES",

                "reason": reason,

            }



    # Fill in any rules that weren't mentioned in the response

    for rule in rules:

        if rule.name not in results:

            results[rule.name] = {"matched": False, "reason": ""}



    return results





def _rule_eval_max_output_tokens(rule_count: int) -> int:

    """Scale output budget so every rule can get a response line."""

    return max(500, rule_count * 40)





def evaluate_rules(
    text: str,
    rules: list[ModerationRule],
    llm_model,
    content_type: str = "posts",
    context: dict | None = None,
    jev_client=None,
    pending_selection: dict | None = None,
) -> dict | None:
    """Evaluate a discrete rule set with Jev as its sole classifier.

    Jev returns the authoritative ordered match set in one request. The
    configured LLM is called only after positive selections, to provide a
    verified explanation and exact content excerpts. For a stop-on-match rule,
    later decisions remain unknown.
    """
    if not rules:
        return {"matches": [], "decisions": {}, "token_info": {}, "raw_response": ""}

    from ordered_gate import evaluate_first_match, evaluate_mixed_rules

    evaluator = (
        evaluate_first_match
        if all(rule.stop_on_match for rule in rules)
        else evaluate_mixed_rules
    )
    return evaluator(
        text,
        rules,
        llm_model,
        context=context,
        jev_client=jev_client,
        pending_selection=pending_selection,
    )


# Action dispatch

# ---------------------------------------------------------------------------



def handle_moderation_action(

    subreddit,

    content_obj,

    author_name: str,

    reason: str,

    action: str,

    dry_run: bool = False,

    llm_model=None,
    *,
    state: dict | None = None,
    bot_username: str | None = None,
    persist_state=None,

) -> bool:

    """

    Take a single moderation action on content.



    Args:

        subreddit: PRAW Subreddit object

        content_obj: PRAW Comment or Submission object

        author_name: Username of the author

        reason: The violation reason from the LLM

        action: One of 'report', 'remove', 'modmail', 'log', etc.

        dry_run: If True, only log what would happen

        llm_model: Optional LLM for optional AI modmail summary generation



    Returns:

        True if action was taken (or would be in dry run)

    """

    if action not in IMPLEMENTED_ACTIONS:

        print(f"    Unsupported moderation action: {action}")

        return False

    if action == "ban" and not config.AUTOMATED_BAN_ENABLED:
        print("    Automated bans are disabled; the configured action was not applied")
        return False



    is_post = hasattr(content_obj, "title")

    content_type = "post" if is_post else "comment"

    link = f"https://reddit.com{content_obj.permalink}"



    if dry_run:

        print(f"    [DRY RUN] Would {action} {content_type} by u/{author_name}: {reason}")

        return True



    try:

        if action == "remove":

            content_obj.mod.remove()

            print(f"    Removed {content_type} by u/{author_name}: {reason}")

            if config.CONTENT_MODERATION_PUBLIC_EXPLANATION_ENABLED:
                from bot_comment_format import format_bot_comment
                from bot_utils import validate_reply_response
                from mod_attention import build_removal_public_reply
                from public_response_safety import post_public_comment

                explanation = build_removal_public_reply(
                    is_post, reason, summary=None
                )
                try:
                    if state is None or not callable(persist_state) or not bot_username:
                        raise RuntimeError("removal explanation requires durable public-response state")
                    reply = post_public_comment(
                        content_obj,
                        format_bot_comment(explanation),
                        state,
                        source_id=str(getattr(content_obj, "id", "")),
                        # This is an automatic explanation, not a response
                        # requested by the content author.
                        author_name=None,
                        bot_username=bot_username,
                        persist_state=persist_state,
                        source_type="post" if is_post else "comment",
                        action="removal_explanation",
                        subreddit=subreddit,
                        validate=validate_reply_response,
                        item_type="removal explanation",
                    )
                    if reply is not None:
                        print(f"    Posted public removal explanation for {content_type} by u/{author_name}")
                except Exception as explanation_error:
                    # Removal is the primary moderation action.  Do not
                    # repeat it merely because an optional explanation reply
                    # failed after the removal succeeded.
                    print(f"    ⚠️ Removed content, but public explanation failed: {explanation_error}")



        elif action == "report":

            content_obj.report(reason=f"AI Moderation: {reason[:95]}")

            print(f"    Reported {content_type} by u/{author_name}: {reason}")



        elif action == "modmail":

            subject = f"AI Moderation Alert - {content_type.title()} by u/{author_name}"

            summary = None

            if llm_model is not None and config.MODERATION_AI_SUMMARY_ENABLED:

                from mod_attention import (

                    generate_mod_attention_summary,

                    build_violation_modmail_body,

                )

                excerpt = ""

                if is_post:

                    excerpt = f"{content_obj.title}\n\n{getattr(content_obj, 'selftext', '')}"

                else:

                    excerpt = getattr(content_obj, "body", "")

                summary = generate_mod_attention_summary(

                    {

                        "kind": "moderation",

                        "username": author_name,

                        "content_excerpt": excerpt[:800],

                        "rule_reason": reason,

                    },

                    llm_model,

                )

                body = build_violation_modmail_body(

                    author_name, is_post, reason, summary, link

                )

            else:

                body = (

                    f"**AI Moderation Alert**\n\n"

                    f"**Type:** {content_type.title()} by u/{author_name}\n"

                    f"**Reason:** {reason}\n\n"

                    f"**Link:** {link}"

                )

            subreddit.message(subject=subject, message=body)

            print(f"    Sent modmail for {content_type} by u/{author_name}: {reason}")



        elif action == "spam":

            content_obj.mod.remove(spam=True)

            print(f"    Marked as spam {content_type} by u/{author_name}: {reason}")



        elif action == "lock":

            content_obj.mod.lock()

            print(f"    Locked {content_type} by u/{author_name}: {reason}")



        elif action == "unlock":

            content_obj.mod.unlock()

            print(f"    Unlocked {content_type} by u/{author_name}: {reason}")



        elif action == "approve":

            content_obj.mod.approve()

            print(f"    Approved {content_type} by u/{author_name}: {reason}")



        elif action == "ban":

            subreddit.banned.add(

                author_name,
                ban_reason=f"Automated moderation: {reason[:100]}",
                ban_message=(
                    f"Your {content_type} was actioned under a moderation rule "
                    "configured by this community's moderators. Contact the "
                    "moderators if you think this was a mistake."
                ),

            )

            print(f"    Banned u/{author_name}: {reason}")



        elif action == "log":

            print(f"    Logged violation by u/{author_name}: {reason}")



        return True



    except Exception as e:

        print(f"    Error taking moderation action ({action}): {e}")

        return False





def execute_rule_actions(

    subreddit,

    content_obj,

    author_name: str,

    match: dict,

    dry_run: bool = False,

    llm_model=None,
    *,
    state: dict | None = None,
    bot_username: str | None = None,
    persist_state=None,

) -> bool:

    """

    Execute all actions for a single matched rule.



    Args:

        subreddit: PRAW Subreddit object

        content_obj: PRAW Comment or Submission object

        author_name: Username of the author

        match: Dict with "rule" (ModerationRule) and "reason" (str)

        dry_run: If True, only log what would happen

        llm_model: Optional LLM for modmail summary generation

    """

    rule: ModerationRule = match["rule"]

    reason = match["reason"]



    succeeded = True
    for action in rule.actions:

        if not handle_moderation_action(

            subreddit, content_obj, author_name, reason, action, dry_run,

            llm_model=llm_model,
            state=state,
            bot_username=bot_username,
            persist_state=persist_state,

        ):
            succeeded = False

    return succeeded


