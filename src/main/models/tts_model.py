from pydantic import BaseModel
from fastapi import UploadFile
from typing import Dict, Optional

class TTSRequest(BaseModel):
    page_number: int

class TTSResponse(BaseModel):
    status: str
    message: str
    annotated_image_path: Optional[str]
    json_path: Optional[str]
    vertex_trimmed_path: Optional[str]
    metadata_path: Optional[str]
    errors: Optional[Dict[str, str]] = None
