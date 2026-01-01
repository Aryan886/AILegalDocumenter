from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Upload(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    file_path: str
    extracted_text: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

