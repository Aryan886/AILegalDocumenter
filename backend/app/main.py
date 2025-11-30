from fastapi import FastAPI
from app.routers import documents, health, uploads, nlp

app = FastAPI(title="AI Legal Dcoumenter")

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(uploads.router)
app.include_router(nlp.router) 

@app.get("/")
def root():
    return {"message": "Welcome to AI Legal Documenter!"}

