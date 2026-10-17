"""
Summon detection and response for the Optimist Prime bot.
Handles responding when users explicitly summon the bot in the configured subreddit.
"""

import re
import time
from copy import deepcopy
from datetime import datetime, timezone

import config
from bot_utils import (
    is_likely_bot,
    is_bot_owned_author,
    is_too_old,
    check_user_cooldown,
    get_cached_moderators,
    is_moderator,
    get_daily_non_moderator_reply_count,
    record_reply_success,
    is_current_moderator,
)
from llm_client import model_without_per_run_cap
from persona import (
    build_full_context,
    generate_conversational_response,
    generate_post_summon_response,
)
from bot_comment_format import strip_bot_footer
from self_flair_requests import (
    handle_self_custom_flair_request,
)
from engagement_actions import execute_engagement_action
from catch_up import remember_retryable_item
from moderator_action_router import engagement_event_type, process_moderator_turn
from jev_interaction_router import route_public_interaction
from reddit_conversation_transport import (
    remove_processed_instruction,
)
from private_chat_handler import reply_to_latest_private_chat
from public_response_safety import (
    notify_public_response_limit,
    public_response_already_handled,
    record_public_abstention,
    response_capacity_available,
)


def _public_route_actions(reddit) -> list[str]:
    """Expose only public actions enabled by both configuration and dependencies."""
    allowed = []
    if config.USER_CUSTOM_FLAIR_ENABLED:
        allowed.append("self_custom_flair")
    return allowed


def _route_usage(route_result: dict) -> tuple[int, float]:
    usage = route_result.get("token_info", {})
    if not isinstance(usage, dict):
        return 0, 0.0
    try:
        tokens = int(usage.get("total_tokens", usage.get("input_tokens", 0)) or 0)
    except (TypeError, ValueError):
        tokens = 0
    try:
        cost = float(usage.get("cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return max(0, tokens), max(0.0, cost)


def is_summon(text: str, bot_username: str = "") -> bool:
    """Require a direct address before treating text as a bot request."""
    cleaned_text = strip_bot_footer(text)
    return _has_direct_bot_address(cleaned_text, bot_username)


def is_bot_mentioned(text: str, bot_username: str = "") -> bool:
    """Find an unquoted, visible reference that can be judged by the AI."""
    body = strip_bot_footer(str(text or ""))
    body = re.sub(r"(?m)^\s*>.*$", " ", body)
    body = re.sub(r"[\"“][^\"”\n]*[\"”]", " ", body)
    if not body.strip():
        return False
    names = {bot_username, *config.BOT_OWNED_USERNAMES}
    if any(
        re.search(rf"(?<![\w/])(?:u/|@)?{re.escape(name)}\b", body, re.IGNORECASE)
        for name in names if name
    ):
        return True
    return bool(re.search(r"\boptimist\s*prime\b", body, re.IGNORECASE))


def is_jev_interaction_candidate(text: str, bot_username: str = "") -> bool:
    """Use a broad address cue to decide what Jev should judge, not the outcome.

    Jev makes the final call about whether the author is addressing Optimist,
    asking for an action, or only mentioning the bot. Direct replies are also
    eligible through their parent relationship.
    """
    body = strip_bot_footer(str(text or ""))
    body = re.sub(r"(?m)^\s*>.*$", " ", body)
    body = re.sub(r'["“][^"”\n]*["”]', " ", body)
    if not body.strip():
        return False
    if is_bot_mentioned(body, bot_username):
        return True
    # Generic forms such as "mod bot, review" vary too much for a command
    # phrase list. Send bot references to Jev, which rejects incidental and
    # quoted mentions using the message and thread context.
    return bool(re.search(r"\b(?:optimist|bot)\b", body, re.IGNORECASE))


def _has_direct_bot_address(text: str, bot_username: str) -> bool:
    """Recognize a real address, not a discussion that merely mentions the bot."""
    body = re.sub(r"(?m)^\s*>.*$", " ", str(text or ""))
    body = re.sub(r"[\"“][^\"”\n]*[\"”]", " ", body)
    # Reddit often renders account mentions as a Markdown profile link. Unwrap
    # the visible label and undo Markdown-escaped underscores before checking
    # whether the address is followed by an actual request.
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return False

    bot_terms = r"(?:optimist\s*prime|optimist|mod\s*bot|tldr\s*bot|bot)"
    direct_patterns = (
        rf"^\s*(?:u/|@){re.escape(bot_username)}\b.*\b(?:can|could|would|please|help|tell|check|look|do|what|how|turn\s+on|activate|enable|set)\b",
        rf"\b(?:hey|hi|hello|yo|sup)\s*,?\s*{bot_terms}\b",
        rf"^\s*{bot_terms}\s*[,!:—-]",
        rf"^\s*{bot_terms}\s+(?:can|could|would|will|please|review|scan|summarize|summarise|help|tell|give|check|look)\b",
        rf"(?:^|[,!?]\s*)optimist\s*prime\s+(?:get\s+in\s+here|jump\s+in|weigh\s+in|help\b)",
        r"\bI(?:'m| am)?\s*(?:summoning|summon|calling|paging)\s+(?:the\s+)?(?:bot|optimist\s*prime)\b",
    )
    return any(re.search(pattern, body, re.IGNORECASE) for pattern in direct_patterns)


def check_for_summons(
    subreddit,
    gemini_model,
    state: dict,
    bot_username: str,
    reddit=None,
    dry_run: bool = False,
    recent_posts=None,
    recent_comments=None,
    persist_state=None,
    jev_client=None,
    interaction_cache: dict | None = None,
) -> tuple[int, int, float, dict]:
    """
    Scan recent comments and posts for summon phrases and respond.
    
    Args:
        subreddit: PRAW Subreddit object
        gemini_model: Initialized Gemini model
        state: Current bot state dict
        bot_username: The bot's Reddit username
        reddit: PRAW Reddit instance used for Reddit API access
        dry_run: If True, don't actually post replies
        recent_posts: Optional posts already fetched by the runner.  Reusing
            them avoids a second subreddit-new request in the same cycle.
        recent_comments: Optional comments already fetched by the runner.
            Reusing them keeps summon detection on the persistent catch-up
            path instead of issuing a second comment listing request.
    
    Returns:
        Tuple of (summons_handled, tokens_used, cost, updated_state)
    """
    summons_handled = 0
    non_moderator_replies_sent = max(0, int(state.get("_cycle_non_moderator_replies", 0) or 0))
    total_tokens = 0
    total_cost = 0.0
    daily_limit_notice_printed = False
    run_limit_notice_printed = False

    if dry_run:
        state = deepcopy(state)

    def execute_response(*args, **kwargs):
        kwargs.setdefault("response_state", state)
        kwargs.setdefault("persist_state", persist_state)
        kwargs.setdefault("bot_username", bot_username)
        return execute_engagement_action(*args, **kwargs)

    # Get tracking sets from state
    summon_responses = set(state.get("summon_responses", []))
    recent_user_replies = state.get("recent_user_replies", {})

    def finish_public_history_request(comment, author_name: str, response_text: str) -> bool:
        """Send a public moderator history result only to its existing Chat.

        The request is claimed before attempting delivery, then removed even
        when its saved Chat anchor is unavailable or Reddit rejects the reply.
        """
        if dry_run:
            print(
                "    [DRY RUN] Would send the history result in the moderator's "
                "existing Chat and remove the public request."
            )
            return False
        source_id = str(getattr(comment, "id", "") or "")
        if not source_id:
            return False
        summon_responses.add(source_id)
        state["summon_responses"] = list(summon_responses)[-2000:]
        persisted = True
        if callable(persist_state):
            try:
                persist_state(state)
            except Exception as exc:
                persisted = False
                print(
                    "    ⚠️ Could not save the handled history request "
                    f"({type(exc).__name__}); dropping the private reply."
                )

        delivered = False
        try:
            if persisted and not dry_run and reddit is not None:
                try:
                    delivered = reply_to_latest_private_chat(
                        reddit,
                        state,
                        username=author_name,
                        bot_username=bot_username,
                        text=response_text,
                        request_text=str(getattr(comment, "body", "") or ""),
                    )
                except Exception as exc:
                    print(
                        "    ⚠️ Existing Chat reply failed "
                        f"({type(exc).__name__}); dropping the private result."
                    )
        finally:
            try:
                remove_processed_instruction(comment)
            except Exception as exc:
                print(
                    "    ⚠️ Could not remove the public history request "
                    f"({type(exc).__name__}); it remains marked handled."
                )
        if callable(persist_state):
            try:
                persist_state(state)
            except Exception as exc:
                print(
                    "    ⚠️ Could not save the private Chat transcript "
                    f"({type(exc).__name__})."
                )
        return delivered
    
    print(f"  🔔 Scanning for bot summons in r/{subreddit.display_name}...")
    
    # Check comments
    try:
        comments = (
            list(recent_comments)
            if recent_comments is not None
            else list(subreddit.comments(limit=config.COMMENT_SCAN_LIMIT))
        )
        
        # Reuse the parent-author cache populated by the reply phase. It lets
        # us recognize direct replies to Optimist without looking up every
        # ordinary comment's parent.
        parent_author_cache = state.get("reply_parent_authors", {})
        if not isinstance(parent_author_cache, dict):
            parent_author_cache = {}
        captured_parents = {}
        for captured in comments:
            captured_author = getattr(getattr(captured, "author", None), "name", None)
            captured_id = str(getattr(captured, "id", "") or "")
            if captured_id and is_bot_owned_author(captured_author, bot_username):
                fullname = str(getattr(captured, "fullname", "") or f"t1_{captured_id}")
                captured_parents[fullname] = captured
                parent_author_cache.setdefault(fullname, captured_author)

        # Batch fetch parent authors only for visible bot mentions and direct
        # summons. Direct replies are already covered by the captured/cache
        # lookup above.
        parent_objects = {}
        # Use passed reddit instance or try to get it from subreddit object
        reddit_instance = reddit if reddit else getattr(subreddit, "_reddit", None)

        if reddit_instance:
            parent_ids_to_fetch = set()
            for comment in comments:
                # Pre-filter to reduce fetch count
                if comment.id in summon_responses: continue
                if is_too_old(comment.created_utc): continue
                if comment.author and is_bot_owned_author(comment.author.name, bot_username): continue
                body = getattr(comment, "body", "")
                if not is_jev_interaction_candidate(body, bot_username):
                    continue

                # Check parents for replies
                parent_id = getattr(comment, "parent_id", None)
                if parent_id and parent_id not in parent_author_cache:
                    parent_ids_to_fetch.add(parent_id)

            if parent_ids_to_fetch:
                try:
                    # Fetch all parents in one go
                    parents = reddit_instance.info(fullnames=list(parent_ids_to_fetch))
                    for parent in parents:
                        fullname = str(getattr(parent, "fullname", "") or "")
                        parent_id = str(getattr(parent, "id", "") or "")
                        if not fullname and parent_id:
                            fullname = f"t1_{parent_id}"
                        author_name = getattr(getattr(parent, "author", None), "name", None)
                        if fullname:
                            parent_author_cache[fullname] = author_name or ""
                            parent_objects[fullname] = parent
                except Exception as e:
                    print(f"    ⚠️ Could not batch fetch parents: {e}")

        for comment in comments:
            if public_response_already_handled(
                comment,
                state,
                source_id=str(comment.id),
                bot_username=bot_username,
                persist_state=persist_state,
            ):
                summon_responses.add(comment.id)
                continue
            # Skip already processed
            if comment.id in summon_responses:
                continue
            
            # Skip too old
            if is_too_old(comment.created_utc):
                continue
            
            # Skip if it's our own comment
            if comment.author and is_bot_owned_author(comment.author.name, bot_username):
                continue
            
            # Skip deleted
            if not comment.body or comment.body == '[deleted]':
                continue
            
            author_name = comment.author.name if comment.author else None
            
            # Only direct summons, explicit bot mentions, and replies to a
            # known Optimist comment are candidates for Jev routing.
            direct_summon = is_summon(comment.body, bot_username)
            parent_id = getattr(comment, "parent_id", None)
            parent_author_name = parent_author_cache.get(parent_id)
            parent_object = parent_objects.get(parent_id) or captured_parents.get(parent_id)
            if not parent_author_name and parent_object is not None:
                parent_author_name = getattr(
                    getattr(parent_object, "author", None), "name", None
                )
            direct_reply = bool(
                parent_author_name
                and is_bot_owned_author(parent_author_name, bot_username)
            )
            if not (is_jev_interaction_candidate(comment.body, bot_username) or direct_reply):
                continue

            # Resolve context only for an eligible message. Bot-parent replies
            # stay eligible here; the earlier reply phase may already have
            # handled them, and the public response ledger prevents duplicates.
            try:
                if parent_id and parent_id not in parent_author_cache and not direct_reply:
                    parent = comment.parent()
                    parent_object = parent
                    parent_author_name = getattr(
                        getattr(parent, "author", None), "name", None
                    )
            except Exception:
                pass  # If we can't get parent, proceed normally
            
            # A direct summon is addressed to Optimist. Suppress only Optimist
            # accounts; human names that happen to end in "bot" are valid.
            if is_bot_owned_author(author_name, bot_username):
                summon_responses.add(comment.id)
                continue
            
            moderator_exempt = is_current_moderator(author_name, state, subreddit)
            requester_is_moderator = bool(
                author_name and is_current_moderator(author_name, state, subreddit)
            )
            conversation_model = (
                model_without_per_run_cap(gemini_model) if moderator_exempt else gemini_model
            )

            post_id = str(getattr(comment, "link_id", "") or "").removeprefix("t3_")

            if not moderator_exempt:
                if get_daily_non_moderator_reply_count(state) >= config.MAX_REPLIES_PER_DAY:
                    if not daily_limit_notice_printed:
                        print(
                            "  ⏸️ Ordinary-user daily reply limit reached "
                            f"({config.MAX_REPLIES_PER_DAY}); moderator replies remain eligible"
                        )
                        daily_limit_notice_printed = True
                    continue
                if non_moderator_replies_sent >= config.MAX_REPLIES_PER_RUN:
                    if not run_limit_notice_printed:
                        print(
                            "  ⏸️ Ordinary-user reply limit reached for this run "
                            f"({config.MAX_REPLIES_PER_RUN}); moderator replies remain eligible"
                        )
                        run_limit_notice_printed = True
                    continue
            
            # Check user cooldown (moderators bypass this)
            if not moderator_exempt and check_user_cooldown(author_name, recent_user_replies):
                print(f"    ⏭️ Skipping u/{author_name} (cooldown active)")
                continue
            
            print(f"    🔔 Summon detected from u/{author_name}: {comment.body[:60]}...")
            
            if dry_run:
                print(
                    f"       [DRY RUN] Would {config.ENGAGEMENT_ACTION_MODE} "
                    f"summon in comment {comment.id}"
                )
                continue

            if not moderator_exempt and config.ENGAGEMENT_ACTION_MODE == "comment":
                capacity, reason = response_capacity_available(
                    state,
                    post_id or "unknown",
                    author_name=author_name,
                )
                if not capacity:
                    notify_public_response_limit(
                        state,
                        subreddit,
                        persist_state,
                        author_name=author_name,
                        reason=reason or "global",
                    )
                    continue
            
            turn = None
            try:
                if moderator_exempt:
                    submission = comment.submission
                    context_text = build_full_context(
                        comment, submission, recent_comments
                    )
                    turn = process_moderator_turn(
                        reddit,
                        subreddit,
                        state,
                        conversation_model,
                        author_name=author_name,
                        text=comment.body,
                        channel="comment",
                        source_id=str(comment.id),
                        conversation_key=f"comment:{author_name.casefold()}:{post_id or 'unknown'}",
                        bot_username=bot_username,
                        verified_moderator=True,
                        interaction=comment,
                        reply_context=context_text,
                        parent_author=(
                            None if parent_author_name and is_bot_owned_author(parent_author_name, bot_username)
                            else parent_author_name
                        ),
                        is_summon=direct_summon,
                        persist_state=persist_state,
                        jev_client=jev_client,
                    )
                    if turn is None:
                        raise RuntimeError("moderator summon was not interpreted")
                    turn_route = str(turn.get("route", ""))
                    pending = turn.get("pending_action")
                    private_history_request = (
                        turn_route == "history_review"
                        or (
                            turn_route == "clarify"
                            and isinstance(pending, dict)
                            and pending.get("route") == "history_review"
                        )
                    )
                    if private_history_request:
                        usage = turn.get("token_info", {})
                        total_tokens += int(usage.get("total_tokens", 0) or 0)
                        total_cost += float(usage.get("cost", 0.0) or 0.0)
                        response_text = str(
                            turn.get("response")
                            or turn.get("conversation_response")
                            or ""
                        )
                        delivered = finish_public_history_request(
                            comment, author_name or "", response_text
                        )
                        if delivered:
                            summons_handled += 1
                        continue
                    if turn.get("_response_delivered"):
                        summon_responses.add(comment.id)
                        continue
                    usage = turn.get("token_info", {})
                    total_tokens += int(usage.get("total_tokens", 0) or 0)
                    total_cost += float(usage.get("cost", 0.0) or 0.0)
                    response_text = (
                        turn.get("response") if turn.get("handled")
                        else turn.get("conversation_response")
                    )
                    if response_text == "[[NO_REPLY]]":
                        record_public_abstention(
                            state,
                            source_id=str(comment.id),
                            thread_id=post_id or "unknown",
                            author_name=author_name,
                            persist_state=persist_state,
                        )
                        summon_responses.add(comment.id)
                        continue
                    if not response_text:
                        raise ValueError("moderator action selection produced no reply")
                    if not execute_response(
                        comment,
                        subreddit,
                        author_name or "[deleted]",
                        comment.body,
                        response_text,
                        event_type=engagement_event_type(
                            str(turn.get("route", "conversation")), is_summon=True
                        ),
                        action_mode=config.ENGAGEMENT_ACTION_MODE,
                        dry_run=dry_run,
                    ):
                        remember_retryable_item(
                            state, "comments", comment,
                            "moderator conversation response was not accepted",
                        )
                        continue
                    turn["_response_delivered"] = True
                    if callable(persist_state):
                        persist_state(state)
                    summon_responses.add(comment.id)
                    summons_handled += 1
                    record_reply_success(state, moderator=True)
                    continue

                allowed_actions = _public_route_actions(reddit)
                route_result = route_public_interaction(
                    jev_client,
                    text=comment.body,
                    channel="comment",
                    summoned=direct_summon,
                    parent_author=parent_author_name,
                    thread_title=(
                        getattr(parent_object, "title", None)
                        or getattr(comment, "link_title", None)
                    ),
                    parent_text=(
                        getattr(parent_object, "body", None)
                        or getattr(parent_object, "selftext", None)
                    ),
                    allowed_actions=allowed_actions,
                    interaction_cache=interaction_cache,
                    source_id=str(comment.id),
                )
                route_tokens, route_cost = _route_usage(route_result)
                total_tokens += route_tokens
                total_cost += route_cost

                def abstain_from_public_comment() -> None:
                    record_public_abstention(
                        state,
                        source_id=str(comment.id),
                        thread_id=post_id or "unknown",
                        author_name=author_name,
                        persist_state=persist_state,
                    )
                    summon_responses.add(comment.id)

                if not route_result.get("complete"):
                    print(
                        "       Jev could not choose a public response route; "
                        f"leaving it retryable ({route_result.get('error') or 'decision unavailable'})."
                    )
                    remember_retryable_item(
                        state,
                        "comments",
                        comment,
                        f"Jev public route incomplete: {route_result.get('error') or 'decision unavailable'}",
                    )
                    continue

                selected_route = str(route_result.get("route", "") or "")
                selected_action = str(route_result.get("action", "") or "")
                if selected_route in {"no_reply", "not_addressed"}:
                    abstain_from_public_comment()
                    continue

                supported_routes = {"conversation", "clarify", "unsupported_mod_action"}
                if "self_custom_flair" in allowed_actions:
                    supported_routes.add("self_custom_flair")
                if selected_route not in supported_routes:
                    print(
                        "       Jev selected an unavailable public action; "
                        "leaving it retryable without a fallback classifier."
                    )
                    remember_retryable_item(
                        state,
                        "comments",
                        comment,
                        f"Jev selected unavailable public route: {selected_route}",
                    )
                    continue

                event_type = "summon"
                response_text = ""
                token_info = {"total_tokens": 0, "cost": 0.0}
                action_tokens = 0
                action_cost = 0.0

                if selected_route == "conversation":
                    submission = comment.submission
                    response_text, token_info = generate_conversational_response(
                        comment,
                        submission,
                        conversation_model,
                        is_summon=direct_summon,
                        captured_comments=recent_comments,
                        allow_abstain=False,
                    )
                    token_info = token_info if isinstance(token_info, dict) else {}
                    total_tokens += int(token_info.get("total_tokens", 0) or 0)
                    total_cost += float(token_info.get("cost", 0.0) or 0.0)
                    if str(response_text or "").strip().casefold() in {
                        "[[no_reply]]", "no reply", "abstain"
                    }:
                        print(
                            "       Reply writer abstained after Jev selected a conversation; "
                            "recording the source as handled."
                        )
                        abstain_from_public_comment()
                        continue
                elif selected_route == "clarify":
                    supported = ["answering questions"]
                    if "self_custom_flair" in allowed_actions:
                        supported.append("custom flair requests")
                    response_text = (
                        "Could you clarify what you’d like me to do? I can help with "
                        + ", ".join(supported)
                        + "."
                    )
                elif selected_route == "unsupported_mod_action":
                    response_text = (
                        "I can’t take moderator actions from a public comment. "
                        "Please contact this subreddit’s moderators for help."
                    )
                elif selected_route == "self_custom_flair":
                    if selected_action != "set":
                        raise ValueError("Jev selected an invalid custom flair action")
                    response_text, action_tokens, action_cost, state = (
                        handle_self_custom_flair_request(
                            comment,
                            subreddit,
                            reddit,
                            conversation_model,
                            state,
                            dry_run=dry_run,
                            jev_client=jev_client,
                            intent_override={
                                "action": selected_action,
                                "label": route_result.get("requested_flair", ""),
                            },
                        )
                    )
                    event_type = "user_custom_flair"
                response_text = str(response_text or "").strip()
                if not response_text:
                    raise ValueError(
                        f"selected public route {selected_route!r} produced no response"
                    )
                total_tokens += int(action_tokens or 0)
                total_cost += float(action_cost or 0.0)

                if not execute_response(
                    comment,
                    subreddit,
                    author_name or "[deleted]",
                    comment.body,
                    response_text,
                    event_type=event_type,
                    action_mode=config.ENGAGEMENT_ACTION_MODE,
                    dry_run=dry_run,
                ):
                    remember_retryable_item(
                        state,
                        "comments",
                        comment,
                        "selected public interaction response was not accepted",
                    )
                    continue

                summon_responses.add(comment.id)
                if not moderator_exempt:
                    now = datetime.now(timezone.utc).timestamp()
                    if author_name not in recent_user_replies:
                        recent_user_replies[author_name] = {
                            "count": 1, "first_reply_time": now
                        }
                    else:
                        hours_since = (
                            now - recent_user_replies[author_name].get("first_reply_time", 0)
                        ) / 3600
                        if hours_since >= config.SAME_USER_COOLDOWN_HOURS:
                            recent_user_replies[author_name] = {
                                "count": 1, "first_reply_time": now
                            }
                        else:
                            recent_user_replies[author_name]["count"] = (
                                recent_user_replies[author_name].get("count", 0) + 1
                            )
                    non_moderator_replies_sent += 1
                summons_handled += 1
                record_reply_success(state, moderator=moderator_exempt)
            except Exception as e:
                print(f"       ❌ Error responding to summon: {e}")
                pending = (turn or {}).get("pending_action") if isinstance(turn, dict) else None
                selected_history = (
                    isinstance(turn, dict) and turn.get("route") == "history_review"
                ) or (
                    isinstance(pending, dict) and pending.get("route") == "history_review"
                )
                if requester_is_moderator and selected_history:
                    finish_public_history_request(
                        comment,
                        author_name or "",
                        "I couldn't complete that history review. Send a new request if you'd like me to try again.",
                    )
                    continue
                remember_retryable_item(state, "comments", comment, str(e))
                # Leave the summon unclaimed so a transient failure can be retried.
    
    except Exception as e:
        print(f"  ❌ Error scanning comments for summons: {e}")
    
    # Check posts for summons (in title or body)
    # Scan posts even after ordinary-user capacity is used so a moderator's
    # summon is still eligible for a response.
    if summons_handled < config.MAX_REPLIES_PER_RUN or config.MODERATOR_REPLIES_BYPASS_LIMITS:
        try:
            # Rate limiting: wait before making another API call
            # The runner normally passes the post listing it already fetched.
            # Do not add an artificial delay when no second Reddit request is
            # needed; retain the small pause for the standalone fallback path.
            if recent_posts is None:
                time.sleep(0.5)
            
            posts = (
                list(recent_posts)
                if recent_posts is not None
                else list(subreddit.new(limit=25))
            )
            
            for post in posts:
                post_id = f"post_{post.id}"
                
                # Skip already processed
                if post_id in summon_responses:
                    continue
                
                # Skip too old
                if is_too_old(post.created_utc):
                    continue
                
                # Skip if it's our own post (unlikely but possible)
                if post.author and is_bot_owned_author(post.author.name, bot_username):
                    continue
                
                # Check for summon in title or body
                combined_text = f"{post.title} {post.selftext or ''}"
                # A new post needs an explicit address before it can receive a
                # reply, but the phrase patterns above are only candidate
                # cues. Let Jev judge whether an unusual mention is a real
                # summon or merely discusses Optimist.
                if not is_jev_interaction_candidate(combined_text, bot_username):
                    continue
                
                author_name = post.author.name if post.author else None
                
                # A direct post summon is addressed to Optimist. Suppress only
                # Optimist accounts, not human usernames ending in "bot".
                if is_bot_owned_author(author_name, bot_username):
                    summon_responses.add(post_id)
                    continue
                
                moderator_exempt = is_current_moderator(author_name, state, subreddit)
                requester_is_moderator = bool(
                    author_name and is_current_moderator(author_name, state, subreddit)
                )
                conversation_model = (
                    model_without_per_run_cap(gemini_model) if moderator_exempt else gemini_model
                )
                if not moderator_exempt:
                    if get_daily_non_moderator_reply_count(state) >= config.MAX_REPLIES_PER_DAY:
                        if not daily_limit_notice_printed:
                            print(
                                "  ⏸️ Ordinary-user daily reply limit reached "
                                f"({config.MAX_REPLIES_PER_DAY}); moderator replies remain eligible"
                            )
                            daily_limit_notice_printed = True
                        continue
                    if non_moderator_replies_sent >= config.MAX_REPLIES_PER_RUN:
                        if not run_limit_notice_printed:
                            print(
                                "  ⏸️ Ordinary-user reply limit reached for this run "
                                f"({config.MAX_REPLIES_PER_RUN}); moderator replies remain eligible"
                            )
                            run_limit_notice_printed = True
                        continue
                
                # Check user cooldown (moderators bypass this)
                if not moderator_exempt and check_user_cooldown(author_name, recent_user_replies):
                    print(f"    ⏭️ Skipping u/{author_name} (cooldown active)")
                    continue
                
                print(f"    🔔 Summon in post by u/{author_name}: {post.title[:50]}...")
                
                if dry_run:
                    print(
                        f"       [DRY RUN] Would {config.ENGAGEMENT_ACTION_MODE} "
                        f"summon in post {post.id}"
                    )
                    continue

                if not moderator_exempt and config.ENGAGEMENT_ACTION_MODE == "comment":
                    capacity, reason = response_capacity_available(
                        state,
                        str(post.id),
                        author_name=author_name,
                    )
                    if not capacity:
                        notify_public_response_limit(
                            state,
                            subreddit,
                            persist_state,
                            author_name=author_name,
                            reason=reason or "global",
                        )
                        continue
                
                try:
                    if moderator_exempt:
                        turn = process_moderator_turn(
                            reddit,
                            subreddit,
                            state,
                            conversation_model,
                            author_name=author_name,
                            text=combined_text,
                            channel="post",
                            source_id=str(post.id),
                            conversation_key=f"post:{author_name.casefold()}:{post.id}",
                            bot_username=bot_username,
                            verified_moderator=True,
                            interaction=post,
                            reply_context=combined_text,
                            is_summon=is_summon(combined_text, bot_username),
                            jev_client=jev_client,
                            persist_state=persist_state,
                        )
                        if turn is None:
                            raise RuntimeError("moderator post summon was not interpreted")
                        usage = turn.get("token_info", {})
                        total_tokens += int(usage.get("total_tokens", 0) or 0)
                        total_cost += float(usage.get("cost", 0.0) or 0.0)
                        turn_route = str(turn.get("route", "") or "")
                        if turn.get("_response_delivered"):
                            summon_responses.add(post_id)
                            continue
                        if turn_route == "history_review":
                            # History results are sent through the moderator
                            # workflow, never published under a public post.
                            summon_responses.add(post_id)
                            if callable(persist_state):
                                persist_state(state)
                            if turn.get("success"):
                                summons_handled += 1
                            continue
                        response_text = (
                            turn.get("response") if turn.get("handled")
                            else turn.get("conversation_response")
                        )
                        if response_text == "[[NO_REPLY]]":
                            record_public_abstention(
                                state,
                                source_id=str(post.id),
                                thread_id=str(post.id),
                                author_name=author_name,
                                source_type="post",
                                action="post_summon",
                                persist_state=persist_state,
                            )
                            summon_responses.add(post_id)
                            continue
                        if not response_text:
                            print(
                                "       Moderator route did not complete; "
                                "leaving the post retryable."
                            )
                            remember_retryable_item(
                                state,
                                "posts",
                                post,
                                "moderator route did not produce a response",
                            )
                            continue
                        event_type = engagement_event_type(
                            turn_route or "conversation", is_summon=True
                        )
                    else:
                        allowed_actions = _public_route_actions(reddit)
                        route_result = route_public_interaction(
                            jev_client,
                            text=combined_text,
                            channel="post",
                            summoned=is_summon(combined_text, bot_username),
                            thread_title=str(getattr(post, "title", "") or ""),
                            parent_text=str(getattr(post, "selftext", "") or ""),
                            allowed_actions=allowed_actions,
                            interaction_cache=interaction_cache,
                            source_id=str(post.id),
                        )
                        route_tokens, route_cost = _route_usage(route_result)
                        total_tokens += route_tokens
                        total_cost += route_cost

                        def abstain_from_public_post() -> None:
                            record_public_abstention(
                                state,
                                source_id=str(post.id),
                                thread_id=str(post.id),
                                author_name=author_name,
                                source_type="post",
                                action="post_summon",
                                persist_state=persist_state,
                            )
                            summon_responses.add(post_id)

                        if not route_result.get("complete"):
                            print(
                                "       Jev could not choose a public post route; "
                                f"leaving it retryable ({route_result.get('error') or 'decision unavailable'})."
                            )
                            remember_retryable_item(
                                state,
                                "posts",
                                post,
                                f"Jev public route incomplete: {route_result.get('error') or 'decision unavailable'}",
                            )
                            continue

                        selected_route = str(route_result.get("route", "") or "")
                        selected_action = str(route_result.get("action", "") or "")
                        if selected_route in {"no_reply", "not_addressed"}:
                            abstain_from_public_post()
                            continue
                        supported_routes = {
                            "conversation", "clarify", "unsupported_mod_action"
                        }
                        if "self_custom_flair" in allowed_actions:
                            supported_routes.add("self_custom_flair")
                        if selected_route not in supported_routes:
                            print(
                                "       Jev selected an unavailable public post action; "
                                "leaving it retryable without a fallback classifier."
                            )
                            remember_retryable_item(
                                state,
                                "posts",
                                post,
                                f"Jev selected unavailable public route: {selected_route}",
                            )
                            continue

                        event_type = "post_summon"
                        token_info = {"total_tokens": 0, "cost": 0.0}
                        action_tokens = 0
                        action_cost = 0.0
                        response_text = ""
                        if selected_route == "conversation":
                            response_text, token_info = generate_post_summon_response(
                                post,
                                conversation_model,
                                captured_comments=recent_comments,
                            )
                            token_info = token_info if isinstance(token_info, dict) else {}
                            total_tokens += int(token_info.get("total_tokens", 0) or 0)
                            total_cost += float(token_info.get("cost", 0.0) or 0.0)
                        elif selected_route == "clarify":
                            supported = ["answering questions"]
                            if "self_custom_flair" in allowed_actions:
                                supported.append("custom flair requests")
                            response_text = (
                                "Could you clarify what you’d like me to do? I can help with "
                                + ", ".join(supported)
                                + "."
                            )
                        elif selected_route == "unsupported_mod_action":
                            response_text = (
                                "I can’t take moderator actions from a public post. "
                                "Please contact this subreddit’s moderators for help."
                            )
                        elif selected_route == "self_custom_flair":
                            if selected_action != "set":
                                raise ValueError("Jev selected an invalid custom flair action")
                            response_text, action_tokens, action_cost, state = (
                                handle_self_custom_flair_request(
                                    post,
                                    subreddit,
                                    reddit,
                                    conversation_model,
                                    state,
                                    dry_run=dry_run,
                                    jev_client=jev_client,
                                    intent_override={
                                        "action": selected_action,
                                        "label": route_result.get("requested_flair", ""),
                                    },
                                )
                            )
                            event_type = "user_custom_flair"
                        total_tokens += int(action_tokens or 0)
                        total_cost += float(action_cost or 0.0)

                    response_text = str(response_text or "").strip()
                    if not response_text:
                        raise ValueError("selected post route produced no response")
                    if not execute_response(
                        post,
                        subreddit,
                        author_name or "[deleted]",
                        combined_text,
                        response_text,
                        event_type=event_type,
                        action_mode=config.ENGAGEMENT_ACTION_MODE,
                        dry_run=dry_run,
                    ):
                        remember_retryable_item(
                            state,
                            "posts",
                            post,
                            "selected post response was not accepted",
                        )
                        continue

                    print(
                        f"       ✅ Handled post summon ({config.ENGAGEMENT_ACTION_MODE}; "
                        f"{len(response_text.split())} words)"
                    )
                    summon_responses.add(post_id)
                    if not moderator_exempt:
                        now = datetime.now(timezone.utc).timestamp()
                        if author_name not in recent_user_replies:
                            recent_user_replies[author_name] = {
                                "count": 1, "first_reply_time": now
                            }
                        else:
                            hours_since = (
                                now - recent_user_replies[author_name].get("first_reply_time", 0)
                            ) / 3600
                            if hours_since >= config.SAME_USER_COOLDOWN_HOURS:
                                recent_user_replies[author_name] = {
                                    "count": 1, "first_reply_time": now
                                }
                            else:
                                recent_user_replies[author_name]["count"] = (
                                    recent_user_replies[author_name].get("count", 0) + 1
                                )
                        non_moderator_replies_sent += 1
                    summons_handled += 1
                    record_reply_success(state, moderator=moderator_exempt)
                    if moderator_exempt and callable(persist_state):
                        persist_state(state)
                except Exception as e:
                    print(f"       ❌ Error responding to post summon: {e}")
                    remember_retryable_item(state, "posts", post, str(e))
                    # Leave the summon unclaimed so a transient failure can be retried.
        
        except Exception as e:
            print(f"  ❌ Error scanning posts for summons: {e}")
    
# Clean up old user reply tracking (remove entries older than cooldown window)
    now = datetime.now(timezone.utc).timestamp()
    cutoff_seconds = config.SAME_USER_COOLDOWN_HOURS * 3600
    recent_user_replies = {
        user: data for user, data in recent_user_replies.items()
        if (now - data.get("first_reply_time", 0)) < cutoff_seconds
    }

    # Update state
    state["summon_responses"] = list(summon_responses)[-2000:]  # Keep last 2000
    state["recent_user_replies"] = recent_user_replies

    return summons_handled, total_tokens, total_cost, state
