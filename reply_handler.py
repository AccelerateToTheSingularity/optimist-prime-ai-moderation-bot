"""
Inbox reply monitoring for the Optimist Prime bot.
Handles responding to users who reply to the bot's comments.
"""

import re
from copy import deepcopy
from datetime import datetime, timezone

import config
from bot_utils import (
    is_bot_owned_author,
    is_too_old,
    check_user_cooldown,
    get_daily_non_moderator_reply_count,
    record_reply_success,
    is_current_moderator,
)
from llm_client import model_without_per_run_cap
from persona import build_full_context, generate_conversational_response
from self_flair_requests import handle_self_custom_flair_request
from engagement_actions import execute_engagement_action
from moderator_action_router import engagement_event_type, process_moderator_turn
from jev_interaction_router import route_public_interaction
from catch_up import remember_retryable_item
from public_response_safety import (
    notify_public_response_limit,
    public_response_already_handled,
    record_public_abstention,
    response_capacity_available,
)


_REPLY_PARENT_CACHE_KEY = "reply_parent_authors"
_REPLY_PARENT_LOOKUP_BATCH_SIZE = 100
_REPLY_PARENT_CACHE_LIMIT = 2000


def _normalise_comment_fullname(value) -> str | None:
    """Return a Reddit comment fullname for a bare id or an existing fullname."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("t1_"):
        return text
    if text.startswith("t3_"):
        return None
    return f"t1_{text}"


def _author_name(item) -> str | None:
    author = getattr(item, "author", None)
    name = getattr(author, "name", None) if author else None
    return name if isinstance(name, str) and name.strip() else None


def _recent_direct_replies_to_bot(
    reddit,
    state: dict,
    bot_username: str,
    recent_comments,
    already_seen: set[str],
) -> list:
    """Find direct bot replies missed by the inbox notification listing.

    The normal path remains ``inbox.comment_replies``.  Reddit can omit a valid
    reply from that notification listing, however, so the runner also gives us
    the bounded recent-comment catch-up batch.  Parent authors are resolved in
    batches and cached in runtime state; this avoids one Reddit request per
    comment and avoids repeating the same lookups on every worker cycle.
    """
    if not recent_comments:
        return []

    processed_ids = {
        item_id
        for item_id in state.get("replied_to_comments", [])
        if isinstance(item_id, str) and item_id
    }
    parent_to_items: dict[str, list] = {}
    candidate_ids: set[str] = set()
    for item in recent_comments:
        item_id = getattr(item, "id", None)
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in already_seen
            or item_id in processed_ids
            or item_id in candidate_ids
        ):
            continue
        candidate_ids.add(item_id)
        parent_id = _normalise_comment_fullname(getattr(item, "parent_id", None))
        if parent_id is None:
            continue
        created_utc = getattr(item, "created_utc", None)
        if not isinstance(created_utc, (int, float)) or is_too_old(created_utc):
            continue
        parent_to_items.setdefault(parent_id, []).append(item)

    if not parent_to_items:
        return []

    cache = state.get(_REPLY_PARENT_CACHE_KEY, {})
    if not isinstance(cache, dict):
        cache = {}

    missing_parent_ids = [parent_id for parent_id in parent_to_items if parent_id not in cache]
    if missing_parent_ids:
        info_method = getattr(reddit, "info", None)
        if not callable(info_method):
            print("  ⚠️ Reply fallback unavailable: Reddit parent lookup is not supported")
        else:
            for start in range(0, len(missing_parent_ids), _REPLY_PARENT_LOOKUP_BATCH_SIZE):
                batch = missing_parent_ids[start:start + _REPLY_PARENT_LOOKUP_BATCH_SIZE]
                try:
                    parents = list(info_method(fullnames=batch))
                except Exception as exc:
                    # Do not cache a failed lookup.  The affected comments stay
                    # eligible for retry on a later cycle.
                    print(
                        "  ⚠️ Reply fallback parent lookup failed "
                        f"({type(exc).__name__}); will retry later"
                    )
                    continue

                for parent in parents:
                    fullname = _normalise_comment_fullname(
                        getattr(parent, "fullname", None)
                    )
                    if fullname is None:
                        parent_id = getattr(parent, "id", None)
                        fullname = _normalise_comment_fullname(parent_id)
                    if fullname is None:
                        continue
                    cache[fullname] = _author_name(parent) or ""

    if len(cache) > _REPLY_PARENT_CACHE_LIMIT:
        cache = dict(list(cache.items())[-_REPLY_PARENT_CACHE_LIMIT:])
    state[_REPLY_PARENT_CACHE_KEY] = cache

    fallback_items = []
    for parent_id, items in parent_to_items.items():
        parent_author = cache.get(parent_id)
        # This lookup answers "is this Optimist's comment?", not "does this
        # username look automated?". Human names such as example_human_bot must not
        # create false bot-parent matches.
        if parent_author and is_bot_owned_author(parent_author, bot_username):
            fallback_items.extend(items)

    return fallback_items


def _collect_reply_candidates(
    reddit,
    state: dict,
    bot_username: str,
    recent_comments=None,
) -> list:
    """Collect inbox replies plus bounded catch-up replies, without duplicates."""
    try:
        inbox_items = list(reddit.inbox.comment_replies(limit=50))
    except Exception as exc:
        print(
            "  ⚠️ Could not read Reddit inbox replies "
            f"({type(exc).__name__}); using catch-up comments instead"
        )
        inbox_items = []

    seen_ids = {
        item.id for item in inbox_items
        if isinstance(getattr(item, "id", None), str)
    }
    fallback_items = _recent_direct_replies_to_bot(
        reddit,
        state,
        bot_username,
        recent_comments,
        seen_ids,
    )
    if fallback_items:
        print(
            "  📡 Found "
            f"{len(fallback_items)} direct bot reply(s) in the recent-comment catch-up "
            "that were not in the inbox listing"
        )
    return inbox_items + fallback_items


def _parent_flair_context(item, bot_username: str, recent_comments=None) -> str | None:
    """Return a captured, bot-authored flair parent without making another Reddit read."""
    expected_parent = _normalise_comment_fullname(getattr(item, "parent_id", None))
    if expected_parent is not None:
        for parent in recent_comments or ():
            captured_id = _normalise_comment_fullname(getattr(parent, "id", None))
            if captured_id != expected_parent:
                continue
            parent_author_name = _author_name(parent)
            if not parent_author_name or not is_bot_owned_author(parent_author_name, bot_username):
                return None
            parent_body = str(getattr(parent, "body", "") or "").strip()
            if not re.search(r"\bflair\b", parent_body, re.IGNORECASE):
                return None
            return parent_body[:1200] or None
    return None


def _captured_bot_parent_text(item, bot_username: str, recent_comments=None) -> str | None:
    """Return the captured bot-authored parent without another Reddit read."""
    expected_parent = _normalise_comment_fullname(getattr(item, "parent_id", None))
    if expected_parent is None:
        return None
    for parent in recent_comments or ():
        if _normalise_comment_fullname(getattr(parent, "id", None)) != expected_parent:
            continue
        if not is_bot_owned_author(_author_name(parent), bot_username):
            return None
        body = str(getattr(parent, "body", "") or "").strip()
        return body[:2500] or None
    return None


def check_inbox_replies(
    reddit,
    gemini_model,
    state: dict,
    bot_username: str,
    dry_run: bool = False,
    recent_comments=None,
    jev_client=None,
    persist_state=None,
    interaction_cache: dict | None = None,
) -> tuple[int, int, float, dict]:
    """
    Check bot's inbox for replies to our comments and respond.
    
    Args:
        reddit: Authenticated PRAW Reddit instance
        gemini_model: Initialized Gemini model
        state: Current bot state dict
        bot_username: The bot's Reddit username
        dry_run: If True, don't actually post replies
        recent_comments: Optional bounded catch-up comments used to recover
            direct replies missing from Reddit's inbox notification listing.
    
    Returns:
        Tuple of (replies_sent, tokens_used, cost, updated_state)
    """
    replies_sent = 0
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
    replied_to = set(state.get("replied_to_comments", []))
    recent_user_replies = state.get("recent_user_replies", {})
    
    print(f"  📬 Checking inbox for replies to bot comments...")
    
    try:
        # Get comment replies from the inbox, then use the already-fetched
        # catch-up batch to recover direct replies Reddit omitted from inbox.
        inbox_items = _collect_reply_candidates(
            reddit,
            state,
            bot_username,
            recent_comments=recent_comments,
        )
        
        for item in inbox_items:
            if public_response_already_handled(
                item,
                state,
                source_id=str(item.id),
                bot_username=bot_username,
                persist_state=persist_state,
            ):
                replied_to.add(item.id)
                continue
            # Skip if already replied to
            if item.id in replied_to:
                continue
            
            # Skip if too old
            if is_too_old(item.created_utc):
                continue
            
            # Skip if not from our subreddit
            if item.subreddit.display_name.lower() != config.SUBREDDIT.lower():
                continue
            
            # Skip deleted comments
            if not item.body or item.body == '[deleted]':
                replied_to.add(item.id)  # Mark as processed
                continue
            
            # Skip Optimist's own accounts. A username ending in "bot" is not
            # enough evidence to suppress a direct reply to Optimist.
            author_name = item.author.name if item.author else None
            if is_bot_owned_author(author_name, bot_username):
                replied_to.add(item.id)
                continue
            
            moderator_exempt = is_current_moderator(author_name, state, item.subreddit)
            conversation_model = (
                model_without_per_run_cap(gemini_model) if moderator_exempt else gemini_model
            )

            post_id = str(getattr(item, "link_id", "") or "").removeprefix("t3_")
            if not post_id:
                post_id = str(getattr(getattr(item, "submission", None), "id", "") or "")

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
            # NOTE: Cooldown-skipped comments are NOT added to replied_to — they will be
            # re-fetched and re-evaluated on every subsequent run until they age out
            # according to the configured MAX_AGE_HOURS window.
            # This is intentional: delay-not-prevent. The overhead is acceptable given the
            # low volume and the benefit of retrying after cooldown expires.
            if not moderator_exempt and check_user_cooldown(author_name, recent_user_replies):
                print(f"    ⏭️ Skipping u/{author_name} (cooldown active)")
                continue
            
            print(f"    💬 Reply from u/{author_name}: {item.body[:50]}...")
            
            if dry_run:
                print(
                    f"       [DRY RUN] Would {config.ENGAGEMENT_ACTION_MODE} "
                    f"to comment {item.id}"
                )
                continue
            
            try:
                # Get the submission for context
                submission = item.submission
                subreddit = item.subreddit

                if moderator_exempt:
                    reply_context = build_full_context(
                        item, submission, recent_comments
                    )
                    turn = process_moderator_turn(
                        reddit,
                        subreddit,
                        state,
                        conversation_model,
                        author_name=author_name,
                        text=item.body,
                        channel="comment",
                        source_id=str(item.id),
                        conversation_key=f"comment:{author_name.casefold()}:{post_id or 'unknown'}",
                        bot_username=bot_username,
                        verified_moderator=True,
                        interaction=item,
                        reply_context=reply_context,
                        jev_client=jev_client,
                        persist_state=persist_state,
                    )
                    if turn is None:
                        raise RuntimeError("moderator reply was not interpreted")
                    if turn.get("_response_delivered"):
                        replied_to.add(item.id)
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
                            source_id=str(item.id),
                            thread_id=post_id or "unknown",
                            author_name=author_name,
                            persist_state=persist_state,
                        )
                        replied_to.add(item.id)
                        continue
                    if not response_text:
                        raise ValueError("moderator action selection produced no reply")
                    if not execute_response(
                        item,
                        subreddit,
                        author_name or "[deleted]",
                        item.body,
                        response_text,
                        event_type=engagement_event_type(
                            str(turn.get("route", "conversation")), is_summon=False
                        ),
                        action_mode=config.ENGAGEMENT_ACTION_MODE,
                    ):
                        continue
                    turn["_response_delivered"] = True
                    if callable(persist_state):
                        persist_state(state)
                    replied_to.add(item.id)
                    replies_sent += 1
                    record_reply_success(state, moderator=True)
                    continue

                # Do not spend a public route decision when the ordinary-user
                # comment budget already prevents every possible reply.
                if config.ENGAGEMENT_ACTION_MODE == "comment":
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
                
                allowed_actions = []
                if config.USER_CUSTOM_FLAIR_ENABLED:
                    allowed_actions.append("self_custom_flair")

                reply_context = build_full_context(
                    item, submission, recent_comments
                )
                interaction = route_public_interaction(
                    jev_client,
                    text=item.body,
                    channel="comment",
                    summoned=False,
                    parent_author=bot_username,
                    thread_title=(
                        getattr(submission, "title", None)
                        or getattr(item, "link_title", None)
                    ),
                    parent_text=_captured_bot_parent_text(
                        item, bot_username, recent_comments=recent_comments
                    ),
                    conversation_context=reply_context,
                    allowed_actions=allowed_actions,
                    interaction_cache=interaction_cache,
                    source_id=str(item.id),
                )
                route_usage = interaction.get("token_info", {})
                if not isinstance(route_usage, dict):
                    route_usage = {}
                route_tokens = int(
                    route_usage.get(
                        "total_tokens", route_usage.get("input_tokens", 0)
                    ) or 0
                )
                route_cost = float(route_usage.get("cost", 0.0) or 0.0)
                total_tokens += route_tokens
                total_cost += route_cost
                if not interaction.get("complete"):
                    reason = str(interaction.get("error") or "decision unavailable")
                    remember_retryable_item(
                        state,
                        "comments",
                        item,
                        f"Jev public route incomplete: {reason}",
                    )
                    print("       Jev could not choose a reply; leaving this comment for another try.")
                    continue

                selected_route = interaction.get("route")
                selected_action = str(interaction.get("action", "") or "")
                if (
                    selected_route == "self_custom_flair"
                    and not config.USER_CUSTOM_FLAIR_ENABLED
                ):
                    raise RuntimeError(
                        "Jev selected a custom flair action while the feature is disabled"
                    )
                if selected_route in {"no_reply", "not_addressed"}:
                    record_public_abstention(
                        state,
                        source_id=str(item.id),
                        thread_id=post_id or "unknown",
                        author_name=author_name,
                        persist_state=persist_state,
                    )
                    replied_to.add(item.id)
                    continue

                # Jev selects whether this direct reply needs a response. The
                # configured model writes prose only after that positive route.
                event_type = "reply"
                token_info = {
                    "total_tokens": route_tokens,
                    "cost": route_cost,
                }
                response_text = None

                if selected_route == "conversation":
                    response_text, token_info = generate_conversational_response(
                        item,
                        submission,
                        conversation_model,
                        is_summon=False,
                        captured_comments=recent_comments,
                        allow_abstain=False,
                    )
                    if str(response_text or "").strip().casefold() in {
                        "[[no_reply]]", "no reply", "abstain"
                    }:
                        raise ValueError(
                            "reply writer returned an abstention after Jev selected conversation"
                        )
                    generation_tokens = int(token_info.get("total_tokens", 0) or 0)
                    generation_cost = float(token_info.get("cost", 0.0) or 0.0)
                    total_tokens += generation_tokens
                    total_cost += generation_cost
                    token_info = {
                        "total_tokens": route_tokens + generation_tokens,
                        "cost": route_cost + generation_cost,
                    }
                elif selected_route == "clarify":
                    supported = ["answer questions"]
                    if config.USER_CUSTOM_FLAIR_ENABLED:
                        supported.append("help with custom flair requests")
                    if len(supported) == 1:
                        available = supported[0]
                    elif len(supported) == 2:
                        available = f"{supported[0]} or {supported[1]}"
                    else:
                        available = ", ".join(supported[:-1]) + f", or {supported[-1]}"
                    response_text = (
                        f"Could you clarify what you’d like me to do? I can {available}."
                    )
                elif selected_route == "unsupported_mod_action":
                    response_text = (
                        "I can’t ban users or remove posts or comments. "
                        "Please contact the moderators for help."
                    )
                elif selected_route == "self_custom_flair":
                    if selected_action != "set":
                        raise ValueError("Jev selected an invalid custom flair action")
                    response_text, action_tokens, action_cost, state = (
                        handle_self_custom_flair_request(
                            item,
                            subreddit,
                            reddit,
                            conversation_model,
                            state,
                            dry_run=dry_run,
                            jev_client=jev_client,
                            intent_override={
                                "action": selected_action,
                                "label": interaction.get("requested_flair", ""),
                            },
                        )
                    )
                    action_tokens = int(action_tokens or 0)
                    action_cost = float(action_cost or 0.0)
                    total_tokens += action_tokens
                    total_cost += action_cost
                    token_info = {
                        "total_tokens": route_tokens + action_tokens,
                        "cost": route_cost + action_cost,
                    }
                    event_type = "self_custom_flair"
                    if not response_text:
                        continue
                else:
                    raise ValueError(
                        f"Jev selected an unsupported public reply route: {selected_route!r}"
                    )

                if not str(response_text or "").strip():
                    raise ValueError("selected public reply route produced no response")

                if not execute_response(
                    item,
                    subreddit,
                    author_name or "[deleted]",
                    item.body,
                    response_text,
                    event_type=event_type,
                    action_mode=config.ENGAGEMENT_ACTION_MODE,
                ):
                    remember_retryable_item(
                        state,
                        "comments",
                        item,
                        "selected public reply response was not accepted",
                    )
                    continue

                print(
                    f"       ✅ Handled conversational reply ({config.ENGAGEMENT_ACTION_MODE}; "
                    f"{len(response_text.split())} words, "
                    f"{token_info.get('total_tokens', 0)} tokens)"
                )
                
                # Update tracking
                replied_to.add(item.id)
                # Track ordinary-user reply windows only; moderator replies
                # bypass this throttle and do not fill its bookkeeping.
                if not moderator_exempt:
                    now = datetime.now(timezone.utc).timestamp()
                    if author_name not in recent_user_replies:
                        recent_user_replies[author_name] = {"count": 1, "first_reply_time": now}
                    else:
                        hours_since = (now - recent_user_replies[author_name].get("first_reply_time", 0)) / 3600
                        if hours_since >= config.SAME_USER_COOLDOWN_HOURS:
                            recent_user_replies[author_name] = {"count": 1, "first_reply_time": now}
                        else:
                            recent_user_replies[author_name]["count"] = recent_user_replies[author_name].get("count", 0) + 1
                    non_moderator_replies_sent += 1
                replies_sent += 1
                record_reply_success(state, moderator=moderator_exempt)
                
            except Exception as e:
                print(f"       ❌ Error replying: {e}")
                remember_retryable_item(state, "comments", item, str(e))
    
    except Exception as e:
        print(f"  ❌ Error checking inbox: {e}")
    
# Clean up old user reply tracking (remove entries older than cooldown window)
    now = datetime.now(timezone.utc).timestamp()
    cutoff_seconds = config.SAME_USER_COOLDOWN_HOURS * 3600
    recent_user_replies = {
        user: data for user, data in recent_user_replies.items()
        if (now - data.get("first_reply_time", 0)) < cutoff_seconds
    }

    # Update state
    state["replied_to_comments"] = list(replied_to)[-2000:]  # Keep last 2000
    state["recent_user_replies"] = recent_user_replies

    return replies_sent, total_tokens, total_cost, state
