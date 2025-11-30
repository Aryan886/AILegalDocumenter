from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from uuid import UUID, uuid4
from typing import List, Optional

router = APIRouter(prefix="/documents", tags=["documents"])

class DocumentCreate(BaseModel):
    title: str
    filename : Optional[str] = None

class DocumentOut(DocumentCreate):
    id : UUID
    status : str # e.g., "processing", "completed"


#very small in-memory "database" for now - replace with real DB later
documents_db = {}

@router.post("/", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def create_document(payload: DocumentCreate):
    doc_id = uuid4()
    document = {"id": doc_id, "title": payload.title, "filename": payload.filename or "", "status": "created"}
    documents_db[str(doc_id)] = document
    return document

@router.get("/", response_model=List[DocumentOut])
def list_documents():
    return list(documents_db.values())

@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: UUID):
    document = documents_db.get(str(document_id))
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: UUID):
    if str(document_id) in documents_db:
        del documents_db[str(document_id)]
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")