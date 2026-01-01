# backend/app/routers/uploads.py
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from uuid import uuid4
from typing import Dict

from fastapi.params import Depends
from app.routers.documents import documents_db
from sqlmodel import Session    

from app.db.models import Upload
from app.db.session import get_session

# PDF extraction
from PyPDF2 import PdfReader

router = APIRouter(prefix="/uploads", tags=["uploads"])

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", Path(__file__).resolve().parents[2] / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# small in-memory mapping: doc_id -> filepath/text_status
_upload_store: Dict[str, Dict] = {}

def save_upload_file(upload_file: UploadFile, dest: Path) -> None:
    """Save uploaded file to disk"""
    with dest.open("wb") as buffer:
        for chunk in iter(lambda: upload_file.file.read(1024*1024), b""):
            buffer.write(chunk)
    upload_file.file.close()


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text from various file formats.
    Currently supports: PDF, TXT, DOCX
    """
    suffix = file_path.suffix.lower()
    
    try:
        # PDF files
        if suffix == '.pdf':
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if not text.strip():
                return "⚠️ PDF appears to be empty or contains only images."
            return text.strip()
        
        # Text files
        elif suffix == '.txt':
            return file_path.read_text(encoding="utf-8")
        
        # DOCX files (optional - uncomment if you install python-docx)
        # elif suffix == '.docx':
        #     import docx
        #     doc = docx.Document(str(file_path))
        #     text = "\n".join([para.text for para in doc.paragraphs])
        #     return text
        
        else:
            return f"❌ Unsupported file format: {suffix}. Please upload PDF or TXT files."
    
    except Exception as e:
        return f"❌ Error extracting text: {str(e)}"


def background_extract_text(doc_id: str, file_path: str):
    """Background task for text extraction (currently unused but kept for future)"""
    path = Path(file_path)
    text = extract_text_from_file(path)
    item = _upload_store.get(doc_id, {})
    item["text"] = text
    item["status"] = "parsed"
    _upload_store[doc_id] = item
    # also update document metadata status if present
    if doc_id in documents_db:
        documents_db[doc_id]["status"] = "parsed"


@router.post("/")
async def upload_file(
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    """
    Upload a file, save it to storage, and extract text content.
    Returns the upload record with extracted text.
    """
    # Validate file size (10MB limit)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    # Validate file extension
    allowed_extensions = {'.pdf', '.txt', '.doc', '.docx'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save the file to storage
    file_path = STORAGE_DIR / file.filename
    save_upload_file(file, file_path)
    
    # Extract text from the file
    extracted_text = extract_text_from_file(file_path)
    
    # Create upload record in database
    upload = Upload(
        filename=file.filename,
        file_path=str(file_path),
        extracted_text=extracted_text
    )

    session.add(upload)
    session.commit()
    session.refresh(upload)

    return upload


@router.get("/{doc_id}/text")
def get_extracted_text(doc_id: str):
    """Get extracted text for a specific upload"""
    item = _upload_store.get(doc_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found or not uploaded")
    return {"document_id": doc_id, "status": item.get("status"), "text": item.get("text", "")}


@router.get("/{doc_id}/file")
def download_file(doc_id: str):
    """Get file path for download"""
    item = _upload_store.get(doc_id)
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"path": item["path"], "filename": item["filename"]}


@router.delete("/{doc_id}", status_code=204)
def delete_upload(doc_id: str):
    """Delete an upload and its associated file"""
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