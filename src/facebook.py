"""Facebook Messenger API client for sending messages and verifying webhooks."""

from __future__ import annotations

import hashlib
import hmac
import httpx

from .config import FB_APP_SECRET, FB_PAGE_ACCESS_TOKEN

_GRAPH_API_BASE = "https://graph.facebook.com/v21.0"


def send_message(psid: str, text: str) -> bool:
    """
    Send a text message to a Facebook user via the Send API.

    Returns True on success, False on failure.
    """
    if not FB_PAGE_ACCESS_TOKEN:
        return False
    url = f"{_GRAPH_API_BASE}/me/messages"
    payload = {
        "recipient": {"id": psid},
        "message": {"text": text},
    }
    headers = {"Content-Type": "application/json"}
    params = {"access_token": FB_PAGE_ACCESS_TOKEN}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers=headers, params=params)
        return resp.status_code == 200
    except Exception:
        return False


def verify_signature(payload: bytes, signature: str | None) -> bool:
    """
    Verify the X-Hub-Signature-256 header from Facebook webhooks.

    Returns True if signature is valid or if FB_APP_SECRET is not set (skip verification).
    """
    if not FB_APP_SECRET or not signature:
        return True
    if not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        FB_APP_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
