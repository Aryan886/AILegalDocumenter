from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = "sqlite:///./ai_legal_documenter.db"

engine = create_engine(
    DATABASE_URL,
    echo=True,  # logs SQL (good for dev)
    connect_args={"check_same_thread": False}
)

def get_session():
    with Session(engine) as session:
        yield session
