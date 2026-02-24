"""FastAPI webhook server for Facebook Messenger."""

from __future__ import annotations

import json

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from src.config import FB_VERIFY_TOKEN
from src.db import get_contact_for_psid, init_db, save_message
from src.facebook import send_message, verify_signature
from src.ingestion import load_vectorstore
from src.retrieval import answer_for_contact

app = FastAPI(title="Facebook Messenger Webhook")

# Initialize DB on startup
init_db()


@app.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    """Facebook verification: return hub.challenge if verify_token matches."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == FB_VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def handle_webhook(request: Request) -> Response:
    """Receive incoming Messenger events and dispatch replies."""
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(body, signature):
        return PlainTextResponse("Invalid signature", status_code=401)

    try:
        data = json.loads(body)
    except Exception:
        return PlainTextResponse("Bad request", status_code=400)

    if data.get("object") != "page":
        return PlainTextResponse("OK")

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender = event.get("sender", {})
            psid = sender.get("id")
            if not psid:
                continue

            message = event.get("message", {})
            text = message.get("text", "").strip()
            if not text:
                continue

            save_message(psid, "in", text)

            mapping = get_contact_for_psid(psid)
            if mapping is None:
                reply = "Thanks for reaching out! An agent will connect your account shortly."
                send_message(psid, reply)
                save_message(psid, "out", reply)
                continue

            contact_id = mapping["hubspot_contact_id"]
            vectorstore = load_vectorstore()
            if vectorstore is None:
                reply = "Your account is connected, but our knowledge base is not ready yet. Please try again later."
                send_message(psid, reply)
                save_message(psid, "out", reply)
                continue

            try:
                reply = answer_for_contact(vectorstore, contact_id, text)
            except Exception:
                reply = "Sorry, I encountered an error. Please try again."
            send_message(psid, reply)
            save_message(psid, "out", reply)

    return PlainTextResponse("OK")
