"""
Reddit reference content resolution for TLDR.
Resolves crosspost parents and reddit.com permalinks to generate TLDRs
when local selftext is short.

Note: This module is not yet integrated into the main bot flow.
Future enhancement: Wire resolve_reddit_reference into bot_runner.py Phase 1
to resolve crosspost parents or linked reddit URLs when posts have short selftext.
"""

import re


REDDIT_HOSTS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "np.reddit.com",
    "new.reddit.com",
}

REDDIT_POST_PATH = re.compile(
    r"/r/[^/]+/comments/([a-z0-9]+)(?:/[^/]*)?(?:/([a-z0-9]+))?/?", re.IGNORECASE
)

REDDIT_URL_IN_TEXT = re.compile(
    r"https?://(?:www\.|old\.|np\.|new\.)?reddit\.com/r/[^/\s]+/comments/[a-z0-9]+(?:/[^/\s]*)?(?:/[a-z0-9]+)?/?",
    re.IGNORECASE,
)


def parse_reddit_url(url_string: str) -> dict | None:
    """
    Parse a reddit.com URL to extract post/comment ID.

    Returns:
        Dict with 'type' ('post' or 'comment') and 'id', or None if not a valid reddit URL
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url_string)
    except Exception:
        return None

    host = parsed.hostname
    if not host:
        return None
    host = host.lower().rstrip(".")
    if host not in REDDIT_HOSTS and not host.endswith(".reddit.com"):
        return None

    match = REDDIT_POST_PATH.search(parsed.path)
    if not match:
        return None

    post_id = match.group(1)
    comment_id = match.group(2)
    if comment_id:
        return {"type": "comment", "id": f"t1_{comment_id}"}
    return {"type": "post", "id": f"t3_{post_id}"}


def find_reddit_urls_in_text(text: str) -> str | None:
    """Find the first reddit.com URL in text."""
    match = REDDIT_URL_IN_TEXT.search(text)
    return match.group(0) if match else None


def resolve_reddit_reference(reddit, post_id: str, title: str, selftext: str, crosspost_parent_id: str | None = None, url: str | None = None) -> dict | None:
    """
    Resolve referenced Reddit content for TLDR when local selftext is short.

    Priority: crosspostParentId -> post.url -> URLs in title/selftext -> fetch submitted post

    Args:
        reddit: Authenticated PRAW Reddit instance
        post_id: The submission ID
        title: Post title
        selftext: Post body text
        crosspost_parent_id: Crosspost parent ID if applicable
        url: Post URL if applicable

    Returns:
        Dict with 'kind', 'source_id', 'subreddit_name', 'title', 'body' or None
    """
    # 1. Check crosspost parent
    if crosspost_parent_id:
        parent_id = crosspost_parent_id
        if not parent_id.startswith("t3_"):
            parent_id = f"t3_{parent_id}"
        try:
            post = reddit.submission(id=parent_id.split("_")[1])
            if post and post.title:
                return {
                    "kind": "crosspost",
                    "source_id": parent_id,
                    "subreddit_name": str(post.subreddit),
                    "title": post.title,
                    "body": post.selftext or "",
                }
        except Exception:
            pass

    # 2. Check URL
    if url:
        parsed = parse_reddit_url(url)
        if parsed:
            ref = _load_content(reddit, parsed)
            if ref:
                return ref

    # 3. Check URLs in title/selftext
    combined = f"{title}\n{selftext}"
    link_in_text = find_reddit_urls_in_text(combined)
    if link_in_text:
        parsed = parse_reddit_url(link_in_text)
        if parsed:
            ref = _load_content(reddit, parsed)
            if ref:
                return ref

    # 4. Fallback: fetch the submitted post itself
    if post_id:
        try:
            post = reddit.submission(id=post_id)
            if post:
                parent_id = getattr(post, "crosspost_parent_id", None)
                if parent_id:
                    if not parent_id.startswith("t3_"):
                        parent_id = f"t3_{parent_id}"
                    ref = _load_content(reddit, {"type": "post", "id": parent_id})
                    if ref:
                        return ref
        except Exception:
            pass

    return None


def _load_content(reddit, parsed: dict) -> dict | None:
    """Load post or comment content from Reddit."""
    try:
        if parsed["type"] == "comment":
            comment = reddit.comment(id=parsed["id"].split("_")[1])
            if comment and comment.body:
                return {
                    "kind": "link",
                    "source_id": parsed["id"],
                    "subreddit_name": str(comment.subreddit),
                    "title": "(linked comment)",
                    "body": comment.body,
                }
        else:
            post = reddit.submission(id=parsed["id"].split("_")[1])
            if post and post.title:
                return {
                    "kind": "link",
                    "source_id": parsed["id"],
                    "subreddit_name": str(post.subreddit),
                    "title": post.title,
                    "body": post.selftext or "",
                }
    except Exception:
        pass
    return None
