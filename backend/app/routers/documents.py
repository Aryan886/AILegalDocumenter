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
    Chat with the document - answer questions based on content and summary
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Use both content and summary for better responses
    full_text = (doc.content or "") + "\n" + (doc.summary or "")
    
    if not full_text.strip():
        return {"response": "I don't have any content to answer questions about yet. Please wait for the document to be processed."}
    
    query = chat.query.strip()
    query_lower = query.lower()
    
    # Handle common greetings
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if query_lower in greetings:
        return {"response": "Hello! I'm here to help you understand this legal document. What would you like to know?"}
    
    # Handle thank you
    if any(word in query_lower for word in ["thank", "thanks", "thx"]):
        return {"response": "You're welcome! Feel free to ask if you have more questions about the document."}
    
    # Handle general document questions
    if any(phrase in query_lower for phrase in ["what is this about", "what's this about", "summarize", "summary", "overview"]):
        if doc.summary:
            # Return first 500 chars of summary
            summary_preview = doc.summary[:500] + "..." if len(doc.summary) > 500 else doc.summary
            return {"response": f"This document is about: {summary_preview}"}
        return {"response": "This is a legal document. The full summary is still being processed."}
    
    # Handle "who", "what", "when", "where", "why", "how" questions
    question_keywords = {
        "who": ["party", "parties", "plaintiff", "defendant", "petitioner", "respondent", "appellant", "court"],
        "what": ["issue", "matter", "case", "claim", "relief", "prayer", "subject"],
        "when": ["date", "dated", "filed", "passed", "delivered", "year", "month"],
        "where": ["court", "tribunal", "location", "place", "jurisdiction"],
        "why": ["reason", "ground", "basis", "cause", "therefore", "because"],
        "how": ["procedure", "process", "method", "manner"]
    }
    
    # Detect question type
    question_type = None
    for qtype in question_keywords.keys():
        if query_lower.startswith(qtype):
            question_type = qtype
            break
    
    # Extract relevant sentences
    sentences = []
    for line in full_text.split('\n'):
        line = line.strip()
        if len(line) > 20:  # Ignore very short lines
            sentences.append(line)
    
    # Score sentences based on query relevance
    query_words = [w.lower() for w in query_lower.split() if len(w) > 3]
    
    if question_type:
        # Add question-specific keywords to search
        query_words.extend(question_keywords[question_type])
    
    matches = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Score based on keyword matches
        score = sum(2 if word in sentence_lower else 0 for word in query_words)
        
        # Boost score for question-specific keywords
        if question_type:
            for keyword in question_keywords[question_type]:
                if keyword in sentence_lower:
                    score += 3
        
        if score > 0:
            matches.append((score, sentence))
    
    # Return best matches
    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 2-3 relevant sentences
        top_matches = matches[:3]
        response_parts = []
        
        for _, sentence in top_matches:
            # Clean up the sentence
            clean_sentence = sentence.strip()
            if clean_sentence and not clean_sentence.startswith('---'):
                response_parts.append(clean_sentence)
        
        if response_parts:
            response = " ".join(response_parts[:2])  # Return max 2 sentences
            # Limit response length
            if len(response) > 400:
                response = response[:400] + "..."
            return {"response": response}
    
    # Fallback responses based on common legal terms in query
    legal_terms = {
        "direction": "The court directions are outlined in the summary. Please check the 'Court Directions' section.",
        "appeal": "Information about the appeal can be found in the document summary.",
        "judgment": "The judgment details are available in the document summary above.",
        "order": "The court order is documented in the summary section.",
        "penalty": "Penalty or punishment details should be in the court's final directions.",
        "compensation": "Compensation details, if any, are mentioned in the court directions.",
        "damages": "Damage amounts, if specified, are in the court's order section."
    }
    
    for term, response in legal_terms.items():
        if term in query_lower:
            return {"response": response}
    
    # Final fallback
    return {
        "response": "I couldn't find specific information about that in the document. Try asking about the parties involved, court directions, or the main issues of the case."
    }