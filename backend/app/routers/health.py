from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health", summary="Service Health")
def health_check():
    return {"status": "ok", "service": "AI Legal Documenter"}