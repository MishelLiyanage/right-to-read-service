from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "Healthy", "message": "The TTS service is healthy."}