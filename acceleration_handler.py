"""
Acceleration Factor handler for the Optimist Prime bot.
Manages opt-in flair showing user's karma from pro-AI subreddits.
"""

import re
from collections import deque
from datetime import datetime, timezone

import config
from config import (
    ACCELERATION_PRO_AI_SUBS,
    ACCELERATION_SCAN_LIMIT,
    ACCELERATION_BACKGROUND_SCAN_LIMIT,
    ACCELERATION_REFRESH_DAYS,
    ACCELERATION_BACKGROUND_REFRESH_DAYS,
    ACCELERATION_MAX_SCANS_PER_RUN,
    ACCELERATION_TIERS,
    ACCELERATION_ZERO_TIER,
    ACCELERATION_MODMAIL_THRESHOLD,
    ACCELERATION_AUTOBAN_ENABLED,
    ACCELERATION_AUTOBAN_THRESHOLD,
)

PRO_AI_SUBS_LOWER = {sub.lower() for sub in ACCELERATION_PRO_AI_SUBS}


def calculate_pro_ai_karma(redditor, reddit, scan_limit: int = None) -> tuple[int, int]:
    """
    Calculate karma from pro-AI subreddits and total karma from scanned items.
    Only counts posts/comments from the last year.
    
    Args:
        redditor: PRAW Redditor object
        reddit: PRAW Reddit instance (for accessing subreddits)
        scan_limit: Max items to scan (defaults to ACCELERATION_SCAN_LIMIT)
    
    Returns:
        Tuple of (pro_ai_karma, total_karma)
    """
    if scan_limit is None:
        scan_limit = ACCELERATION_SCAN_LIMIT
    
    pro_ai_karma = 0
    total_karma = 0
    items_scanned = 0
    
    # Only count items from last year
    one_year_ago = datetime.now(timezone.utc).timestamp() - (365 * 24 * 3600)
    
    try:
        # Scan comments
        for comment in redditor.comments.new(limit=scan_limit):
            if items_scanned >= scan_limit:
                break
            
            # Skip if older than 1 year
            if comment.created_utc < one_year_ago:
                break  # Posts are chronological, so we can stop here
            
            sub_name = comment.subreddit.display_name.lower()
            total_karma += comment.score
            if sub_name in PRO_AI_SUBS_LOWER:
                pro_ai_karma += comment.score
            items_scanned += 1
        
        # Scan submissions (posts)
        items_scanned = 0
        for submission in redditor.submissions.new(limit=scan_limit):
            if items_scanned >= scan_limit:
                break
            
            # Skip if older than 1 year
            if submission.created_utc < one_year_ago:
                break  # Posts are chronological, so we can stop here
            
            sub_name = submission.subreddit.display_name.lower()
            total_karma += submission.score
            if sub_name in PRO_AI_SUBS_LOWER:
                pro_ai_karma += submission.score
            items_scanned += 1
    
    except Exception as e:
        print(f"    ⚠️ Error scanning karma for u/{redditor.name}: {e}")
        return 0, 0
    
    return pro_ai_karma, total_karma


def get_acceleration_tier(ratio: float) -> str:
    """
    Get tier name based on ratio of pro-AI karma to total karma.
    
    Args:
        ratio: Pro-AI karma / total karma (0.0 to 1.0)
    
    Returns:
        Tier name string
    """
    if ratio <= 0:
        return ACCELERATION_ZERO_TIER
    
    for threshold, tier_name in ACCELERATION_TIERS:
        if ratio >= threshold:
            return tier_name
    
    # Fallback (shouldn't happen but safety)
    return ACCELERATION_TIERS[-1][1]


def get_first_flair_template(subreddit) -> str | None:
    """
    Get the first flair template ID (for consistent styling).
    Returns the template ID if found, None on error.
    """
    try:
        for template in subreddit.flair.templates:
            return template["id"]
    except Exception as e:
        print(f"    ⚠️ Could not get flair template: {e}")
    return None


def update_user_flair(subreddit, username: str, tier: str | None, remove: bool = False) -> bool:
    """
    Update user's flair to include/update/remove Acceleration tier.
    
    Args:
        subreddit: PRAW Subreddit object
        username: Reddit username
        tier: Tier name to set, or None if removing
        remove: If True, remove acceleration flair entirely
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get current flair
        current_flair = None
        for flair in subreddit.flair(redditor=username):
            current_flair = flair.get("flair_text", "") or ""
            break
        
        if current_flair is None:
            current_flair = ""
        
        # Pattern to match existing acceleration flair
        accel_pattern = r"\s*\|\s*Acceleration:\s*\S+|\s*Acceleration:\s*\S+"
        
        if remove:
            # Remove acceleration portion
            new_flair = re.sub(accel_pattern, "", current_flair).strip()
            # Clean up any leading/trailing pipes
            new_flair = re.sub(r"^\s*\|\s*", "", new_flair)
            new_flair = re.sub(r"\s*\|\s*$", "", new_flair)
            # Just set text, no template needed for removal
            subreddit.flair.set(username, text=new_flair if new_flair else None)
        else:
            accel_text = f"Acceleration: {tier}"
            
            if re.search(r"Acceleration:\s*\S+", current_flair):
                # Update existing acceleration flair
                new_flair = re.sub(r"Acceleration:\s*\S+", accel_text, current_flair)
            elif current_flair.strip():
                # Append to existing flair
                new_flair = f"{current_flair.strip()} | {accel_text}"
            else:
                # No existing flair
                new_flair = accel_text
            
            # Get the first template for consistent styling
            template_id = get_first_flair_template(subreddit)
            
            if template_id:
                # Use template for colors
                subreddit.flair.set(username, text=new_flair, flair_template_id=template_id)
            else:
                # Fallback to plain text if template fails
                subreddit.flair.set(username, text=new_flair)
        
        return True
    
    except Exception as e:
        print(f"    ❌ Error updating flair for u/{username}: {e}")
        return False


def alert_mods_negative_karma(subreddit, username: str, score: int) -> bool:
    """
    Send modmail alerting mods about a user with negative pro-AI karma.
    
    Args:
        subreddit: PRAW Subreddit object
        username: Reddit username
        score: Their negative karma score
    
    Returns:
        True if successful, False otherwise
    """
    try:
        subject = f"⚠️ Negative Acceleration Alert: u/{username}"
        message = (
            f"User **u/{username}** has **{score}** net karma across pro-AI subreddits.\n\n"
            f"**Subreddits scanned:** {', '.join(ACCELERATION_PRO_AI_SUBS)}\n\n"
            f"This user may be a bad actor. Consider reviewing their activity.\n\n"
            f"---\n*Automated alert from Optimist Prime*"
        )
        
        subreddit.message(subject, message)
        print(f"    📧 Sent modmail alert for u/{username} (score: {score})")
        return True
    
    except Exception as e:
        print(f"    ❌ Error sending modmail for u/{username}: {e}")
        return False


def classify_acceleration_intent(comment_body: str, gemini_model) -> dict | None:
    """
    Use Gemini to classify if a comment is requesting acceleration flair action.
    
    Args:
        comment_body: The comment text
        gemini_model: Initialized Gemini model
    
    Returns:
        Dict with 'action' ('on', 'off', 'check', None) or None if not acceleration-related
    """
    from prompts import get_acceleration_intent_prompt
    prompt = get_acceleration_intent_prompt(comment_body)

    try:
        response = gemini_model.generate_content(prompt)
        result = response.text.strip().upper()
        
        if result == "ON":
            return {"action": "on"}
        elif result == "OFF":
            return {"action": "off"}
        elif result == "CHECK":
            return {"action": "check"}
        else:
            return None
    
    except Exception as e:
        print(f"    ⚠️ Error classifying acceleration intent: {e}")
        return None


def handle_acceleration_command(
    comment,
    subreddit,
    reddit,
    gemini_model,
    state: dict,
    dry_run: bool = False
) -> tuple[str | None, dict]:
    """
    Handle an acceleration flair command from a user.
    
    Args:
        comment: PRAW Comment object
        subreddit: PRAW Subreddit object
        reddit: PRAW Reddit instance
        gemini_model: Initialized Gemini model
        state: Current bot state dict
        dry_run: If True, don't actually modify flairs
    
    Returns:
        Tuple of (response_text or None, updated_state)
    """
    if not config.ACCELERATION_ENABLED:
        return None, state
    
    author_name = comment.author.name if comment.author else None
    if not author_name:
        return None, state
    
    # Classify intent using Gemini
    intent = classify_acceleration_intent(comment.body, gemini_model)
    if not intent:
        return None, state
    
    action = intent["action"]
    
    # Initialize acceleration state if needed
    if "acceleration" not in state:
        state["acceleration"] = {
            "high_score": 100,  # Start with reasonable default
            "opted_in_users": {},
            "alerted_users": [],
            "scanned_users": {}  # For background scanning non-opted-in
        }
    
    accel_state = state["acceleration"]
    user_data = accel_state["opted_in_users"].get(author_name, {})
    
    if action == "off":
        # Disable acceleration flair
        if author_name in accel_state["opted_in_users"]:
            del accel_state["opted_in_users"][author_name]
        
        if not dry_run:
            update_user_flair(subreddit, author_name, None, remove=True)
        
        response = (
            f"Done! I've removed your Acceleration flair. "
            f"You can turn it back on anytime by asking me. 🚀"
        )
        return response, state
    
    elif action in ("on", "check"):
        # Check if we have a recent calculation (within 1 week) to prevent fluctuation
        now = datetime.now(timezone.utc).timestamp()
        cached_data = user_data if user_data else accel_state.get("scanned_users", {}).get(author_name, {})
        last_calc = cached_data.get("last_calculated", cached_data.get("last_scanned", 0))
        
        # Use cached score if calculated within the last week (matches flair refresh cycle)
        CACHE_DURATION_SECONDS = 7 * 24 * 3600  # 1 week
        if (now - last_calc) < CACHE_DURATION_SECONDS and cached_data.get("total_karma", 0) > 0:
            # Reuse cached values
            pro_ai_karma = cached_data.get("pro_ai_karma", 0)
            total_karma = cached_data.get("total_karma", 0)
            print(f"    📋 Using cached score for u/{author_name} (calculated {int((now - last_calc) / 60)}m ago)")
        else:
            # Calculate fresh
            try:
                redditor = reddit.redditor(author_name)
                pro_ai_karma, total_karma = calculate_pro_ai_karma(redditor, reddit)
            except Exception as e:
                print(f"    ❌ Error getting redditor u/{author_name}: {e}")
                return "Sorry, I couldn't calculate your score right now. Please try again later!", state
        
        # Calculate ratio
        if total_karma > 0:
            ratio = pro_ai_karma / total_karma
        else:
            ratio = 0.0
        
        tier = get_acceleration_tier(ratio)
        ratio_percent = int(ratio * 100)
        
        if action == "on":
            # Enable and set flair (now is already calculated above for caching)
            accel_state["opted_in_users"][author_name] = {
                "enabled": True,
                "last_calculated": now,
                "pro_ai_karma": pro_ai_karma,
                "total_karma": total_karma,
                "ratio": ratio,
                "tier": tier
            }
            
            if not dry_run:
                from contributor_flair import apply_combined_user_flair
                apply_combined_user_flair(
                    subreddit, reddit, author_name, state, gemini_model,
                    acceleration_tier=tier, dry_run=dry_run,
                )
            
            response = (
                f"Your Acceleration flair is now active! 🚀\n\n"
                f"**Focus:** {ratio_percent}% of your karma is from pro-AI subs\n"
                f"**Tier:** {tier}\n\n"
                f"Your flair will update weekly. To turn it off, just ask me!"
            )
        else:
            # Just checking, don't modify flair
            response = (
                f"Here's your Acceleration status:\n\n"
                f"**Focus:** {ratio_percent}% of your karma is from pro-AI subs\n"
                f"**Tier:** {tier}\n\n"
                f"{'Your flair is active!' if author_name in accel_state['opted_in_users'] else 'Your flair is not active. Ask me to turn it on!'}"
            )
        
        # Check for negative karma alert (still uses raw pro_ai_karma for background scanning)
        if pro_ai_karma < ACCELERATION_MODMAIL_THRESHOLD:
            if author_name not in accel_state.get("alerted_users", []):
                if not dry_run:
                    alert_mods_negative_karma(subreddit, author_name, pro_ai_karma)
                accel_state.setdefault("alerted_users", []).append(author_name)
        
        return response, state
    
    return None, state


def refresh_opted_in_users(
    subreddit,
    reddit,
    state: dict,
    dry_run: bool = False,
    llm_model=None,
) -> tuple[int, dict]:
    """
    Refresh acceleration scores for opted-in users (weekly job).
    
    Args:
        subreddit: PRAW Subreddit object
        reddit: PRAW Reddit instance
        state: Current bot state dict
        dry_run: If True, don't actually modify flairs
    
    Returns:
        Tuple of (users_updated, updated_state)
    """
    if not config.ACCELERATION_ENABLED:
        return 0, state
    
    accel_state = state.get("acceleration", {})
    opted_in = accel_state.get("opted_in_users", {})
    
    if not opted_in:
        return 0, state
    
    now = datetime.now(timezone.utc).timestamp()
    refresh_threshold = ACCELERATION_REFRESH_DAYS * 24 * 3600
    users_updated = 0
    
    print(f"  🔄 Refreshing acceleration scores for {len(opted_in)} opted-in users...")
    
    for username, user_data in list(opted_in.items()):
        last_calc = user_data.get("last_calculated", 0)
        
        if (now - last_calc) < refresh_threshold:
            continue  # Not due for refresh yet
        
        try:
            redditor = reddit.redditor(username)
            pro_ai_karma, total_karma = calculate_pro_ai_karma(redditor, reddit)
            
            # Calculate ratio
            if total_karma > 0:
                ratio = pro_ai_karma / total_karma
            else:
                ratio = 0.0
            
            tier = get_acceleration_tier(ratio)
            ratio_percent = int(ratio * 100)
            
            # Update user data
            opted_in[username] = {
                "enabled": True,
                "last_calculated": now,
                "pro_ai_karma": pro_ai_karma,
                "total_karma": total_karma,
                "ratio": ratio,
                "tier": tier
            }
            
            if not dry_run:
                from contributor_flair import apply_combined_user_flair
                apply_combined_user_flair(
                    subreddit, reddit, username, state, llm_model,
                    acceleration_tier=tier, dry_run=dry_run,
                )
            else:
                print(f"    [DRY RUN] Would refresh flair for u/{username}: {tier}")
            print(f"    ✅ Refreshed u/{username}: {ratio_percent}% → {tier}")
            users_updated += 1
            
        except Exception as e:
            print(f"    ⚠️ Error refreshing u/{username}: {e}")
    
    state["acceleration"] = accel_state
    return users_updated, state


def queue_background_scan(username: str, state: dict) -> dict:
    """
    Add a user to the background scan queue (doesn't scan immediately).
    
    Args:
        username: Reddit username to queue for scanning
        state: Current bot state dict
    
    Returns:
        Updated state
    """
    if not config.ACCELERATION_ENABLED:
        return state
    
    # Initialize acceleration state if needed
    if "acceleration" not in state:
        state["acceleration"] = {
            "high_score": 100,
            "opted_in_users": {},
            "alerted_users": [],
            "scanned_users": {},
            "scan_queue": []
        }
    
    accel_state = state["acceleration"]
    
    # Skip if already opted-in (they get weekly refreshes)
    if username in accel_state.get("opted_in_users", {}):
        return state
    
    # Check if recently scanned
    scanned = accel_state.get("scanned_users", {})
    now = datetime.now(timezone.utc).timestamp()
    refresh_threshold = ACCELERATION_BACKGROUND_REFRESH_DAYS * 24 * 3600
    
    last_scan = scanned.get(username, {}).get("last_scanned", 0)
    if (now - last_scan) < refresh_threshold:
        return state  # Recently scanned, skip
    
    # Add to queue if not already queued
    queue = accel_state.setdefault("scan_queue", [])
    if username not in queue:
        queue.append(username)
        # Keep queue manageable (max 500 pending)
        if len(queue) > 500:
            queue = queue[-500:]
            accel_state["scan_queue"] = queue
    
    state["acceleration"] = accel_state
    return state


def process_scan_queue(
    subreddit,
    reddit,
    state: dict,
    dry_run: bool = False
) -> tuple[int, dict]:
    """
    Process users from the background scan queue (rate-limited).
    Only processes ACCELERATION_MAX_SCANS_PER_RUN users per cycle.
    
    Args:
        subreddit: PRAW Subreddit object
        reddit: PRAW Reddit instance
        state: Current bot state dict
        dry_run: If True, don't send modmail
    
    Returns:
        Tuple of (users_scanned, updated_state)
    """
    if not config.ACCELERATION_ENABLED:
        return 0, state
    
    accel_state = state.get("acceleration", {})
    queue = deque(accel_state.get("scan_queue", []))
    
    if not queue:
        return 0, state
    
    scanned_count = 0
    now = datetime.now(timezone.utc).timestamp()
    scanned_users = accel_state.setdefault("scanned_users", {})
    
    while queue and scanned_count < ACCELERATION_MAX_SCANS_PER_RUN:
        username = queue.popleft()  # Take from front of queue
        
        # Skip if already opted-in
        if username in accel_state.get("opted_in_users", {}):
            continue
        
        try:
            redditor = reddit.redditor(username)
            pro_ai_karma, _ = calculate_pro_ai_karma(redditor, reddit, scan_limit=ACCELERATION_BACKGROUND_SCAN_LIMIT)
            
            # Record the scan (background scanner tracks raw pro-AI karma only)
            scanned_users[username] = {
                "last_scanned": now,
                "pro_ai_karma": pro_ai_karma
            }
            
            # Check for negative karma alert
            if pro_ai_karma < ACCELERATION_MODMAIL_THRESHOLD:
                if username not in accel_state.get("alerted_users", []):
                    if not dry_run:
                        alert_mods_negative_karma(subreddit, username, pro_ai_karma)
                    accel_state.setdefault("alerted_users", []).append(username)
                    print(f"    ⚠️ Background scan: u/{username} has {pro_ai_karma} pro-AI karma")

            from contributor_flair import try_acceleration_autoban
            try_acceleration_autoban(subreddit, username, pro_ai_karma, state, dry_run=dry_run)
            
            scanned_count += 1
            print(f"    📊 Scanned u/{username}: {pro_ai_karma} karma (queue: {len(queue)} remaining)")
            
        except Exception as e:
            print(f"    ⚠️ Error scanning u/{username}: {e}")
    
    # Cleanup
    accel_state["alerted_users"] = accel_state.get("alerted_users", [])[-500:]
    if len(scanned_users) > 2000:
        sorted_users = sorted(scanned_users.items(), key=lambda x: x[1].get("last_scanned", 0))
        scanned_users = dict(sorted_users[-2000:])
        accel_state["scanned_users"] = scanned_users
    
    accel_state["scan_queue"] = queue
    state["acceleration"] = accel_state
    return scanned_count, state
