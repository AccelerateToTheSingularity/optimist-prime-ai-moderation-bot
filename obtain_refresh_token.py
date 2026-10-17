#!/usr/bin/env python3
"""
Obtain and save a Reddit refresh token for the local PRAW runtime.

The Reddit credential used here must be a traditional Reddit web application
from reddit.com/prefs/apps. This helper uses PRAW's browser approval/code flow;
choose a web app rather than a script app in Reddit's app settings.

This is a one-time interactive helper. It opens Reddit's authorization page
in the browser, waits for the local redirect, verifies which Reddit account
approved the access, and saves the resulting refresh token to the local
gitignored ``.env`` file. The token is never printed to the terminal and is
never sent to GitHub.
"""

from __future__ import annotations

import os
import re
import secrets
import socket
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from env_file import clear_inherited_runtime_environment, load_local_env, save_env_file
from reddit_observability import observable_requestor_settings
from reddit_request_budget import resolve_request_budget_settings
from flair_permission import clear_flair_permission_pause


REDIRECT_HOST = "localhost"
REDIRECT_PORT = 8080
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}"
AUTH_TIMEOUT_SECONDS = 300


def _parse_redirect_request(data: str) -> dict[str, str]:
    """Extract the first value for each query parameter from an HTTP request."""
    request_line = data.splitlines()[0]
    parts = request_line.split(" ", 2)
    if len(parts) < 3 or not parts[1].startswith("/"):
        raise ValueError("redirect request did not contain an HTTP target")

    query = parse_qs(urlsplit(parts[1]).query, keep_blank_values=True)
    return {key: values[0] for key, values in query.items() if values}


def _new_oauth_state() -> str:
    """Generate a high-entropy, URL-safe value to bind the OAuth callback."""
    return secrets.token_urlsafe(32)


def _send_browser_response(client: socket.socket, status: str, body: str) -> None:
    payload = body.encode("utf-8")
    response = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii") + payload
    try:
        client.sendall(response)
    except OSError:
        pass


def _expected_bot_username() -> str:
    """Return the account that this local setup is intended to authorize."""
    return (
        os.environ.get("BOT_EXPECTED_USERNAME", "").strip()
    )


# Every installation needs to identify its account and read Reddit content.
# Additional access is requested only when the operator adds scopes locally.
_REQUIRED_REDDIT_SCOPES = frozenset(
    {
        "identity",
        "read",
    }
)


def _parse_reddit_scopes(raw: str | None) -> frozenset[str]:
    """Parse least-privilege local OAuth scopes and require account/read access."""
    configured = raw if raw is not None else "identity read"
    scopes = frozenset(
        part.casefold()
        for part in re.split(r"[\s,]+", str(configured).strip())
        if part
    )
    if not scopes or "*" in scopes or any(not re.fullmatch(r"[a-z][a-z0-9_]*", scope) for scope in scopes):
        raise ValueError("REDDIT_OAUTH_SCOPES must contain valid individual scope names, not '*'")
    if not _REQUIRED_REDDIT_SCOPES.issubset(scopes):
        raise ValueError("REDDIT_OAUTH_SCOPES must include identity and read")
    return scopes


def _has_required_scopes(
    granted_scopes, required_scopes: frozenset[str] = _REQUIRED_REDDIT_SCOPES
) -> bool:
    """Return whether Reddit granted the scopes needed by enabled bot features."""
    scopes = {str(scope).strip().casefold() for scope in (granted_scopes or ())}
    return required_scopes.issubset(scopes)


def main() -> int:
    # Keep the parser and configuration helpers importable instantly for tests
    # and maintainer tooling.  PRAW imports a large model tree; it is needed
    # only when this executable actually starts the authorization flow.
    import praw

    repo_root = Path(__file__).resolve().parent
    env_path = repo_root / ".env"

    # The helper is intended for the operator's local machine. Clear inherited
    # values so an IDE or shell cannot silently authorize a different app.
    clear_inherited_runtime_environment()
    load_local_env(env_path, override=True)
    import config

    client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        print("The local .env file is missing the Reddit client ID or client secret.")
        print(f"Expected local configuration at: {env_path}")
        return 1

    bot_username = _expected_bot_username()
    if not bot_username:
        print("Set BOT_EXPECTED_USERNAME to the Reddit account that should authorize the bot.")
        return 1
    developer_username = os.environ.get("REDDIT_DEVELOPER_USERNAME", "").strip()
    if not developer_username:
        print("Set REDDIT_DEVELOPER_USERNAME to the account named in the Reddit app User-Agent.")
        return 1
    app_name = os.environ.get("REDDIT_APP_NAME", "Optimist Prime").strip() or "Optimist Prime"
    request_budget = resolve_request_budget_settings(
        os.environ,
        default_root=repo_root,
    )

    try:
        scopes = sorted(_parse_reddit_scopes(os.environ.get("REDDIT_OAUTH_SCOPES")))
    except ValueError as exc:
        print(f"Invalid Reddit permission settings: {exc}")
        return 1
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        user_agent=f"{app_name} obtain_refresh_token by u/{developer_username}",
        **observable_requestor_settings(
            config.REDDIT_API_TIMEOUT_SECONDS,
            request_budget_path=request_budget.path,
            request_budget_limit=request_budget.max_requests,
            request_budget_window_seconds=request_budget.window_seconds,
            request_budget_per_minute=request_budget.requests_per_minute,
        ),
    )

    state = _new_oauth_state()
    url = reddit.auth.url(duration="permanent", scopes=scopes, state=state)

    # The URL contains the public client identifier and a one-time state value,
    # not the client secret or refresh token. Show it so the operator can open the
    # exact authorization request manually if the browser launch is hidden.
    print("Authorization URL (open this in the bot account's browser):")
    print(url)

    # Bind before opening the browser. This removes the race where Reddit
    # returns to localhost before the old helper had started listening.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((REDIRECT_HOST, REDIRECT_PORT))
        server.listen(5)
        server.settimeout(AUTH_TIMEOUT_SECONDS)
    except OSError as exc:
        server.close()
        print(
            f"Could not listen on {REDIRECT_URI}. Close any older authorization "
            f"helper and try again ({type(exc).__name__})."
        )
        return 1

    print("Reddit authorization is ready.")
    print(f"A browser window should open for u/{bot_username}.")
    print("Approve the request only if Reddit shows the intended Optimist Prime app.")
    print(f"Waiting up to {AUTH_TIMEOUT_SECONDS // 60} minutes for the browser approval...")
    if not webbrowser.open(url):
        print("The browser did not open automatically. Open this URL manually:")
        print(url)

    client: socket.socket | None = None
    try:
        deadline = time.monotonic() + AUTH_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print("Timed out waiting for the Reddit authorization.")
                return 1
            server.settimeout(remaining)
            try:
                client, _ = server.accept()
            except socket.timeout:
                print("Timed out waiting for the Reddit authorization.")
                return 1

            try:
                data = client.recv(4096).decode("utf-8", errors="replace")
                params = _parse_redirect_request(data)
            except (IndexError, OSError, ValueError):
                _send_browser_response(client, "400 Bad Request", "Could not read Reddit's redirect.")
                client.close()
                client = None
                print("Ignored an invalid browser callback; still waiting for the current authorization.")
                continue

            if params.get("state") != state:
                _send_browser_response(client, "400 Bad Request", "This was not the current authorization request.")
                client.close()
                client = None
                print("Ignored an older or unrelated browser callback; still waiting for the current authorization.")
                continue
            break

        if params.get("error"):
            error_name = params["error"]
            _send_browser_response(client, "400 Bad Request", f"Reddit returned: {error_name}")
            print(f"Reddit did not grant authorization ({error_name}); no token was saved.")
            return 1

        code = params.get("code")
        if not code:
            _send_browser_response(client, "400 Bad Request", "No authorization code was returned.")
            print("Reddit returned no authorization code; no token was saved.")
            return 1

        try:
            refresh_token = reddit.auth.authorize(code)
            authorized_username = str(reddit.user.me().name).strip()
        except Exception as exc:  # noqa: BLE001 - keep the helper safe and actionable
            _send_browser_response(
                client,
                "502 Bad Gateway",
                "Reddit authorization could not be completed. No token was saved.",
            )
            print(
                "Reddit accepted the browser step, but the local token exchange failed "
                f"({type(exc).__name__}). No token was saved."
            )
            return 1

        if not refresh_token:
            _send_browser_response(client, "502 Bad Gateway", "No refresh token was returned.")
            print("Reddit returned no refresh token; no token was saved.")
            return 1

        if authorized_username.casefold() != bot_username.casefold():
            _send_browser_response(
                client,
                "409 Conflict",
                "The approved Reddit account was not the intended bot account.",
            )
            print(
                f"The authorization was for u/{authorized_username}, not u/{bot_username}. "
                "No token was saved."
            )
            return 1

        try:
            granted_scopes = reddit.auth.scopes()
        except Exception as exc:  # noqa: BLE001 - fail closed on unverifiable access
            _send_browser_response(
                client,
                "502 Bad Gateway",
                "Reddit access could not be verified. No token was saved.",
            )
            print(
                "The approved account is correct, but Reddit's granted permissions "
                f"could not be checked ({type(exc).__name__}). No token was saved."
            )
            return 1

        if not _has_required_scopes(granted_scopes, frozenset(scopes)):
            _send_browser_response(
                client,
                "403 Forbidden",
                "Reddit did not grant every required bot permission. No token was saved.",
            )
            print(
                "Reddit did not grant every permission requested in REDDIT_OAUTH_SCOPES. "
                "No token was saved; approve the request again with those permissions."
            )
            return 1

        # save_env_file preserves every existing local setting and updates only
        # the refresh token. .env and .env.bak are gitignored by this project.
        save_env_file(env_path, {"REDDIT_REFRESH_TOKEN": refresh_token})
        pause_cleared = False
        if {"flair", "modflair"}.issubset(scopes):
            try:
                from state_store import load_state, save_state

                runtime_state = load_state()
                pause_cleared = clear_flair_permission_pause(runtime_state)
                if pause_cleared:
                    save_state(runtime_state)
            except Exception as exc:  # noqa: BLE001 - preserve fail-closed pause
                print(
                    "The new token was saved, but the bot's saved flair pause could not "
                    f"be cleared ({type(exc).__name__}). It will not retry flair changes yet."
                )
        _send_browser_response(
            client,
            "200 OK",
            "Authorization complete. The local token was saved; you can close this tab.",
        )
        print(f"Authorization succeeded for u/{authorized_username}.")
        print("Reddit granted the permissions requested in REDDIT_OAUTH_SCOPES.")
        print("The refresh token was saved to the local gitignored .env file.")
        if pause_cleared:
            print("Reddit granted flair access, and the saved retry pause was cleared.")
        print("No GitHub secret was changed and no token was printed.")
        return 0
    finally:
        if client is not None:
            client.close()
        server.close()


if __name__ == "__main__":
    sys.exit(main())
