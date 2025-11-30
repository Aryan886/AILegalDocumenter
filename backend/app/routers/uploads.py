# backend/app/routers/uploads.py
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from uuid import uuid4
from typing import Dict
from app.routers.documents import documents_db  # simple integration with documents in-memory store

router = APIRouter(prefix="/uploads", tags=["uploads"])

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", Path(__file__).resolve().parents[2] / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# small in-memory mapping: doc_id -> filepath/text_status
_upload_store: Dict[str, Dict] = {}

def save_upload_file(upload_file: UploadFile, dest: Path) -> None:
    with dest.open("wb") as buffer:
        for chunk in iter(lambda: upload_file.file.read(1024*1024), b""):
            buffer.write(chunk)
    upload_file.file.close()

# placeholder text extractor (replace with pdfminer / PyMuPDF logic)
def extract_text_from_file(file_path: Path) -> str:
    # TODO: replace this with real PDF/text extraction
    # simple fallback: try to read as text
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        return ""

def background_extract_text(doc_id: str, file_path: str):
    path = Path(file_path)
    text = extract_text_from_file(path)
    item = _upload_store.get(doc_id, {})
    item["text"] = text
    item["status"] = "parsed"
    _upload_store[doc_id] = item
    # also update document metadata status if present
    if doc_id in documents_db:
        documents_db[doc_id]["status"] = "parsed"

@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # validate content type / extension (basic)
    filename = Path(file.filename).name
    if not filename:
        raise HTTPException(status_code=400, detail="Filename required")

    doc_id = str(uuid4())
    dest = STORAGE_DIR / f"{doc_id}__{filename}"
    save_upload_file(file, dest)

    _upload_store[doc_id] = {"filename": filename, "path": str(dest), "status": "uploaded", "text": ""}

    # create a lightweight document metadata entry so other endpoints can reference it
    documents_db[doc_id] = {
        "id": doc_id,
        "title": filename,
        "filename": filename,
        "status": "uploaded",
    }

    # schedule background extraction
    background_tasks.add_task(background_extract_text, doc_id, str(dest))

    return {"document_id": doc_id, "filename": filename, "status": "uploaded"}

@router.get("/{doc_id}/text")
def get_extracted_text(doc_id: str):
    item = _upload_store.get(doc_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found or not uploaded")
    return {"document_id": doc_id, "status": item.get("status"), "text": item.get("text", "")}

@router.get("/{doc_id}/file")
def download_file(doc_id: str):
    item = _upload_store.get(doc_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"path": item["path"], "filename": item["filename"]}

@router.delete("/{doc_id}", status_code=204)
def delete_upload(doc_id: str):
    item = _upload_store.get(doc_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        p = Path(item["path"])
        if p.exists():
            p.unlink()
    finally:
        _upload_store.pop(doc_id, None)
        documents_db.pop(doc_id, None)
    return None
