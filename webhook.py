"""FastAPI webhook server for Facebook Messenger."""

from __future__ import annotations

import json
import logging

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import PlainTextResponse

from src.config import FB_VERIFY_TOKEN
from src.db import get_contact_for_psid, init_db, save_message
from src.facebook import send_message, verify_signature
from src.ingestion import load_vectorstore
from src.retrieval import answer_for_contact

logger = logging.getLogger(__name__)
app = FastAPI(title="Facebook Messenger Webhook")

# Initialize DB on startup
init_db()


def _process_message(psid: str, text: str) -> None:
    """Blocking message processing (runs in thread pool)."""
    try:
        save_message(psid, "in", text)
        mapping = get_contact_for_psid(psid)
        if mapping is None:
            reply = "Thanks for reaching out! An agent will connect your account shortly."
        else:
            vectorstore = load_vectorstore()
            if vectorstore is None:
                reply = "Your account is connected, but our knowledge base is not ready yet. Please try again later."
            else:
                try:
                    reply = answer_for_contact(vectorstore, mapping["hubspot_contact_id"], text)
                except Exception as e:
                    logger.exception("RAG error for psid=%s: %s", psid, e)
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
