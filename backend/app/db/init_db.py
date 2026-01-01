from sqlmodel import SQLModel
from app.db.session import engine
from app.db.models import Document, Upload  # IMPORTANT: import models

def init_db():
    SQLModel.metadata.create_all(engine)
