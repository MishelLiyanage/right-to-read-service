import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from src.main.services.tts_service import process_tts_request

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/tts_service")
async def tts_service(pdf_file: UploadFile = File(...)):
    try:
        response_data = await process_tts_request(pdf_file)
        logger.info("TTS processing completed successfully for file: %s", pdf_file.filename)
        return response_data
    except Exception as e:
        logger.error("TTS processing failed for file: %s. Error: %s", pdf_file.filename, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during TTS processing.")
