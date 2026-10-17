"""Resolve which Reddit communities an operator permits history reviews to inspect."""


def _normalize(name: str) -> str:
    return str(name or "").strip().casefold().removeprefix("r/").rstrip("/")


def load_history_subreddits(*, subreddit=None, configured_subreddits=None):
    """Use an explicit history scope, or the configured target subreddit alone.

    An empty scope stays empty; it never means all of Reddit. Arguments are
    injectable for tests and small local integrations.
    """
    if subreddit is None or configured_subreddits is None:
        import config

        if subreddit is None:
            subreddit = getattr(config, "SUBREDDIT", "")
        if configured_subreddits is None:
            configured_subreddits = getattr(config, "HISTORY_SUBREDDITS", ())

    explicit = tuple(
        normalized
        for value in (configured_subreddits or ())
        if (normalized := _normalize(value))
    )
    if explicit:
        return frozenset(explicit)

    target = _normalize(subreddit)
    return frozenset({target}) if target else frozenset()
