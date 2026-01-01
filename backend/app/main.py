from fastapi import FastAPI
from app.routers import documents, health, uploads, nlp
from app.db.init_db import init_db
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Legal Documenter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(uploads.router)
app.include_router(nlp.router) 

@app.get("/")
def root():
    return {"message": "Welcome to AI Legal Documenter!"}

@app.on_event("startup")
def on_startup():
    init_db()