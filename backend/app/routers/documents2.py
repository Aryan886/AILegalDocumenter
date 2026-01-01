from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from uuid import UUID, uuid4
from typing import List, Optional
from app.db.models import Document, Upload
from app.db.session import get_session
from sqlmodel import Session, select

router = APIRouter(prefix="/documents", tags=["documents"])

class DocumentCreate(BaseModel):
    title: str
    content: str  # Added content field
    filename : Optional[str] = None

class DocumentOut(DocumentCreate):
    id : int  # Changed from UUID to int to match your DB
    summary : Optional[str] = None


#very small in-memory "database" for now - replace with real DB later
documents_db = {}

@router.post("/", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def create_document(doc_in: DocumentCreate, session: Session = Depends(get_session)):
    # Create Document object from the input
    doc = Document(
        title=doc_in.title,
        content=doc_in.content,
        filename=doc_in.filename
    )
    
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


@router.get("/")
def list_documents(session: Session = Depends(get_session)):
    statement = select(Document)
    results = session.exec(statement).all()
    return results


@router.get("/{doc_id}")
def get_document(doc_id: int, session: Session = Depends(get_session)):
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, session: Session = Depends(get_session)):
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    session.delete(doc)
    session.commit()
    

@router.patch("/{doc_id}/summary")
def update_summary(
    doc_id: int,
    summary: str,
    session: Session = Depends(get_session)
):
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.summary = summary
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


# NEW ENDPOINT FOR SUMMARIZATION
@router.post("/{doc_id}/summarize")
def summarize_document(
    doc_id: int,
    session: Session = Depends(get_session)
):
    """
    Summarize a document using its stored content.
    This endpoint generates a summary and stores it in the document.
    """
    # Get the document
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get text content from the document
    text_to_summarize = doc.content or ""
    
    if not text_to_summarize:
        raise HTTPException(
            status_code=400, 
            detail="Document has no content to summarize"
        )
    
    # Use the mock summarizer (import here to avoid circular imports)
    from app.nlp.summarizer import summarize_text_mock
    summary = summarize_text_mock(text_to_summarize, "medium")
    
    # Update document with the summary
    doc.summary = summary
    session.add(doc)
    session.commit()
    session.refresh(doc)
    
    return {
        "id": doc.id,
        "title": doc.title,
        "summary": summary
    }