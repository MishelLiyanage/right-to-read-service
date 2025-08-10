from fastapi import FastAPI
from src.main.controllers.tts_controller import router as tts_router
from src.main.controllers.health_controller import router as health_router
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import vertexai
import os

load_dotenv()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

VERTEX_AI_PROJECT = os.getenv('VERTEX_AI_PROJECT')
VERTEX_AI_LOCATION = os.getenv('VERTEX_AI_LOCATION')

try:
    if VERTEX_AI_PROJECT and VERTEX_AI_LOCATION:
        vertexai.init(project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)
        logger.info("Vertex AI initialized successfully.")
    else:
        logger.warning("Vertex AI credentials not configured. Some features may not work.")
except Exception as e:
    logger.error("Failed to initialize Vertex AI: %s", str(e))
    logger.warning("Continuing without Vertex AI. Some features may not work.")

app.include_router(tts_router, prefix="/api")
app.include_router(health_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")