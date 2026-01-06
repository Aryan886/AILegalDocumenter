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
    content: str
    filename : Optional[str] = None

class DocumentOut(DocumentCreate):
    id : int
    summary : Optional[str] = None

class ChatQuery(BaseModel):
    query: str


documents_db = {}

@router.post("/", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def create_document(doc_in: DocumentCreate, session: Session = Depends(get_session)):
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


@router.post("/{doc_id}/summarize")
def summarize_document(
    doc_id: int,
    session: Session = Depends(get_session)
):
    """
    Summarize a document using its stored content.
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    text_to_summarize = doc.content or ""
    
    if not text_to_summarize:
        raise HTTPException(
            status_code=400, 
            detail="Document has no content to summarize"
        )
    
    #from app.nlp.summarizer import summarize_text_smart
    from app.nlp.summarizer import summarize_text_ai
    summary = summarize_text_ai(text_to_summarize, "medium")
    
    doc.summary = summary
    session.add(doc)
    session.commit()
    session.refresh(doc)
    
    return {
        "id": doc.id,
        "title": doc.title,
        "summary": summary
    }


@router.post("/{doc_id}/chat")
def chat_with_document(
    doc_id: int,
    chat: ChatQuery,
    session: Session = Depends(get_session)
):
    """
    Chat with the document - answer questions based on content
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    text = (doc.summary or doc.content or "").lower()
    query_lower = chat.query.lower()
    
    if not text:
        return {"response": "I don't have any content to answer questions about."}
    
    query_words = [w for w in query_lower.split() if len(w) > 3]
    
    sentences = []
    for line in text.split('\n'):
        sentences.extend([s.strip() + '.' for s in line.split('.') if s.strip()])
    
    matches = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for word in query_words if word in sentence_lower)
        if score > 0:
            matches.append((score, sentence))
    
    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        best_match = matches[0][1]
        return {"response": best_match.capitalize()}
    
    return {"response": "I couldn't find specific information about that in the document. Try rephrasing your question or asking about different aspects of the document."}