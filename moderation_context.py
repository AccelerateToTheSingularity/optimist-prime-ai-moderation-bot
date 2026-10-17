"""Read only listing data already captured; never hydrate a lazy PRAW object."""


def cached_comment_context(comment, comments, posts):
    data = vars(comment)
    parent = comments.get(data.get("parent_id"))
    post = posts.get(data.get("link_id"))
    result = {}
    title = vars(post).get("title", "") if post is not None else data.get("link_title", "")
    if title:
        result["post_title"] = str(title)[:300]
    if parent is not None:
        body = vars(parent).get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            result["preceding_comment_other_speaker"] = str(body)[:1000]
    return result


def listing_index(items, prefix):
    return {prefix + str(vars(item)["id"]): item for item in items if "id" in vars(item)}
