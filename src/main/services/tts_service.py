import os
import json
import tempfile
import logging
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF

from src.main.utils.image_processing_utils import (
    annotate_image_with_words, extract_page_as_base64, generate_color_palette
)
from src.main.utils.saving_utils import (
    save_annotated_image, save_audio_and_speech_marks, save_block_details_as_json
)
from src.main.utils.polly_session_utils import initialize_polly
from src.main.utils.generate_block_json_utils import generate_block_json
from src.main.utils.llm_response_processing_utils import clean_llm_response

logger = logging.getLogger(__name__)
polly_client = initialize_polly()

if polly_client is None:
    logger.warning("Polly client is not available. Audio generation will be skipped.")

async def process_tts_request(pdf_file):
    temp_pdf = None

    try:
        # Setup output directory
        book_name = os.path.splitext(pdf_file.filename)[0]
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        output_dir = os.path.join("output", f"{book_name}-{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Created output directory: %s", output_dir)

        # Save uploaded PDF to temp location
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(await pdf_file.read())
        temp_pdf.close()
        pdf_path = temp_pdf.name
        logger.info("Saved temporary PDF: %s", pdf_path)

        results = []

        # Process each page
        with fitz.open(pdf_path) as pdf:
            for page_number in range(len(pdf)):
                page = pdf.load_page(page_number)
                words = page.get_text("words")
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save image and base64
                base64_img, image_path, _ = extract_page_as_base64(pdf_path, page_number, output_dir, pdf_file.filename)
                logger.info("Saved base64 and image for page %d: %s", page_number, image_path)

                # Generate block colors and annotate image
                block_ids = set(w[5] for w in words)
                color_palette = generate_color_palette(block_ids)
                block_details = {}
                annotate_image_with_words(image, words, color_palette, block_details)

                # Save annotated image and JSON
                annotated_image_path = save_annotated_image(image, output_dir, pdf_file.filename, page_number)
                json_path = save_block_details_as_json(block_details, output_dir, pdf_file.filename, page_number)
                logger.info("Saved annotated image and block details for page %d", page_number)


                # Generate and clean LLM output
                block_json = generate_block_json(base64_img, block_details)
                

                # cleaned_output = clean_llm_response(block_json)

                


        #         # Save cleaned output
                

        #         # Generate audio and speech marks
                audio_metadata = {}
                entries = block_json.items() if isinstance(block_json, dict) else enumerate(block_json)

                for block_id, data in entries:
                    ssml = data.get("ssml")
                    if not ssml:
                        logger.warning("No SSML found for block %s on page %d", block_id, page_number)
                        continue
                    audio_path, marks_path = save_audio_and_speech_marks(
                        polly_client, f"{page_number}_{block_id}", ssml, output_dir , data.get("person_type")  , block_json , block_id
                    )
                    
                    vertex_path = os.path.join(output_dir, f"{pdf_file.filename}_page_{page_number}_trimmed_blocks.json")
                    with open(vertex_path, "w") as f:
                        json.dump(block_json, f, indent=4)
                    logger.info("Saved trimmed block JSON for page %d", page_number)

                    audio_metadata[block_id] = {
                        "audio_path": audio_path,
                        "speech_marks_path": marks_path
                    }
                    logger.info("Saved audio and speech marks for block %s on page %d", block_id, page_number)

                # Save audio metadata
                metadata_path = os.path.join(output_dir, f"page_{page_number}_audio_speech_marks_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(audio_metadata, f, indent=4)
                logger.info("Saved metadata for page %d", page_number)

                results.append({
                    "page_number": page_number,
                    "annotated_image_path": annotated_image_path,
                    "json_path": json_path,
                    "vertex_trimmed_path": vertex_path,
                    "metadata_path": metadata_path
                })

        return {
            "status": "success",
            "message": f"Processed {len(results)} page(s).",
            "results": results
        }

    except Exception as e:
        logger.error("Exception in process_tts_request: %s", str(e), exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "errors": {"process_tts_request": str(e)}
        }

    finally:
        if temp_pdf:
            try:
                os.remove(temp_pdf.name)
                logger.info("Deleted temporary PDF file: %s", temp_pdf.name)
            except PermissionError as e:
                logger.warning("Failed to delete temp file %s: %s", temp_pdf.name, e)
