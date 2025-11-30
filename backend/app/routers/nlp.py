# backend/app/routers/nlp.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from app.nlp.summarizer import summarize_text_mock
from app.routers.uploads import _upload_store

router = APIRouter(prefix="/nlp", tags=["nlp"])


class SummarizePayload(BaseModel):
    document_id: Optional[str] = None
    text: Optional[str] = None
    length: Optional[str] = "short"  # short | medium | long


@router.post("/summarize")
def summarize(payload: SummarizePayload):
    # get text either directly or from upload store
    text = ""
    if payload.text:
        text = payload.text
    elif payload.document_id:
        item = _upload_store.get(payload.document_id)
        if not item:
            raise HTTPException(status_code=404, detail="Document not found")
        text = item.get("text", "")

    summary = summarize_text_mock(text, payload.length or "short")

    # store summary inside upload store (optional)
    if payload.document_id and payload.document_id in _upload_store:
        _upload_store[payload.document_id]["summary"] = summary

        # update status if you want
        _upload_store[payload.document_id]["status"] = "summarized"

    return {
        "summary": summary,
        "length": payload.length,
    }


class AskPayload(BaseModel):
    document_id: Optional[str] = None
    query: str


@router.post("/ask")
def ask(payload: AskPayload):
    if payload.document_id:
        item = _upload_store.get(payload.document_id)
        if not item:
            raise HTTPException(status_code=404, detail="Document not found")
        text = item.get("summary") or item.get("text") or ""
    else:
        raise HTTPException(status_code=400, detail="document_id required")

    if not text:
        raise HTTPException(status_code=400, detail="No text/summary available")

    # naive keyword search
    q_tokens = payload.query.lower().split()
    for sent in text.split("."):
        lower = sent.lower()
        if any(tok in lower for tok in q_tokens):
            return {"answer": sent.strip()}

    return {"answer": "No direct answer found. Try rephrasing the question."}
