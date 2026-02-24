"""FastAPI webhook server for Facebook Messenger."""

from __future__ import annotations

import json
import logging
import threading

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse
from langchain_chroma import Chroma

from src.config import FB_VERIFY_TOKEN
from src.db import get_contact_for_psid, init_db, save_message
from src.facebook import send_message, verify_signature
from src.ingestion import load_vectorstore
from src.retrieval import answer_for_contact

logger = logging.getLogger(__name__)
app = FastAPI(title="Facebook Messenger Webhook")

# Initialize DB on startup
init_db()

# Module-level vectorstore cache — loaded once, reused across all messages.
# A threading lock prevents duplicate loads when concurrent messages arrive.
_vectorstore_cache: Chroma | None = None
_vectorstore_lock = threading.Lock()


def _get_vectorstore() -> Chroma | None:
    """Return the cached vectorstore, loading it on first call."""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache
    with _vectorstore_lock:
        # Double-checked locking: another thread may have loaded it while we waited
        if _vectorstore_cache is None:
            logger.info("Loading vectorstore into memory (first call)…")
            _vectorstore_cache = load_vectorstore()
            if _vectorstore_cache is None:
                logger.warning("Vectorstore not available (not yet indexed or corrupt).")
            else:
                logger.info("Vectorstore loaded and cached.")
    return _vectorstore_cache


def _invalidate_vectorstore_cache() -> None:
    """Drop the cached vectorstore so the next call to _get_vectorstore reloads it."""
    global _vectorstore_cache
    with _vectorstore_lock:
        _vectorstore_cache = None


def _process_message(psid: str, text: str) -> None:
    """Blocking message processing (runs in thread pool)."""
    global _vectorstore_cache
    try:
        save_message(psid, "in", text)
        mapping = get_contact_for_psid(psid)
        if mapping is None:
            reply = "Thanks for reaching out! An agent will connect your account shortly."
        else:
            vectorstore = _get_vectorstore()
            if vectorstore is None:
                reply = "Your account is connected, but our knowledge base is not ready yet. Please try again later."
            else:
                try:
                    reply = answer_for_contact(vectorstore, mapping["hubspot_contact_id"], text)
                except Exception as e:
                    logger.exception("RAG error for psid=%s: %s", psid, e)
                    # If it looks like a corrupt index, drop the cache so it will
                    # attempt to reload (or auto-repair) on the next message
                    if "hnsw" in str(e).lower() or "compactor" in str(e).lower():
                        logger.warning("Possible corrupt index — clearing vectorstore cache.")
                        _invalidate_vectorstore_cache()
                    reply = "Sorry, I encountered an error. Please try again."
        send_message(psid, reply)
        save_message(psid, "out", reply)
    except Exception as e:
        logger.exception("Webhook processing error for psid=%s: %s", psid, e)
        try:
            send_message(psid, "Sorry, something went wrong. Please try again later.")
        except Exception:
            pass


@app.get("/health")
async def health() -> PlainTextResponse:
    """Healthcheck endpoint for Railway - returns 200 when the service is up."""
    return PlainTextResponse("OK")


@app.get("/webhook")
async def verify_webhook(request: Request) -> PlainTextResponse:
    """Facebook verification: return hub.challenge if verify_token matches."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == FB_VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def handle_webhook(request: Request, background: BackgroundTasks) -> PlainTextResponse:
    """Receive incoming Messenger events and dispatch replies."""
    try:
        body = await request.body()
    except Exception as e:
        logger.exception("Failed to read body: %s", e)
        return PlainTextResponse("Bad request", status_code=400)

    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(body, signature):
        return PlainTextResponse("Invalid signature", status_code=401)

    try:
        data = json.loads(body)
    except Exception as e:
        logger.warning("Invalid JSON: %s", e)
        return PlainTextResponse("Bad request", status_code=400)

    if data.get("object") != "page":
        return PlainTextResponse("OK")

    # Return 200 immediately (Facebook requires response within 20s)
    # Process messages in background to avoid 502 from Railway proxy timeout
    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender = event.get("sender", {})
            psid = sender.get("id")
            if not psid:
                continue

            message = event.get("message", {})
            text = (message.get("text") or "").strip()
            if not text:
                continue

            background.add_task(
                lambda p=psid, t=text: _process_message(p, t)
            )

    return PlainTextResponse("OK")
