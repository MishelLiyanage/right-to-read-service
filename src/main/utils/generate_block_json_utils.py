import base64
import json
import logging
import time  # For retry delay
from typing import Dict, Any, Optional
import os  # For path operations
from datetime import datetime  # For timestamped filenames

from vertexai.generative_models import GenerativeModel, Part, SafetySetting, HarmCategory, HarmBlockThreshold

# --- Configuration ---
MODEL_NAME = "gemini-2.5-pro-preview-05-06"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
OUTPUT_CHUNK_SIZE = 1  # Number of blocks to process per chunk

GENERATION_CONFIG = {
    "max_output_tokens": 65535,
    "temperature": 0.1,
    "top_p": 0.95,
}

SAFETY_SETTINGS = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def construct_gemini_prompt(blocks_json_str: str) -> str:
    """Constructs the detailed prompt for the Gemini model."""
    return f"""
You will be provided with a JSON string representing blocks of a PDF page, and an image of that PDF page.
The input JSON contains the block number, text, words, and bounding boxes of words.
Your goal is to process this JSON:
1. Analyze the image and text to understand the context and flow, focusing on elements suitable for helping children learn English.
2. Remove any blocks containing unnecessary details not part of the main story/learning content (e.g., publication names, page numbers, headers/footers not relevant to the story).
3. For the remaining essential blocks, you MUST append three new keys: "ssml", "dialog", and "person_type".
5. Make sure the output and the input JSONs have only one difference it is that they the out put have the new three fields dialog , person_type and ssml
4. The final output MUST be a single, complete, and valid JSON object string. Do NOT truncate the output.

Detailed Steps and Guidelines:

STEP 1: Understand the overall context/flow of the textbook page from the image and input JSON.
STEP 2: Identify and REMOVE any blocks from the input JSON that are not part of the core story or learning content (e.g., publication names, page numbers, irrelevant metadata). Only include blocks that are essential for the narrative or educational purpose.
STEP 3: Ensure the SSML and dialog information follows the logical flow of the story as seen in the page.
STEP 4: For each RETAINED block, ensure it has these fields: "text", "words", "bounding_boxes", and then ADD "ssml", "dialog", "person_type".
STEP 5: Add "dialog":
    - Set to "true" (as a string) if the block's text is part of a conversation or direct speech.
    - Set to "false" (as a string) otherwise.
STEP 6: Add "person_type":
    - If "dialog" is "true", analyze the image near the text block to determine the speaker. Assign one of: "young boy", "old man", "young girl", "old woman", "middle aged man", "middle aged woman".
    - If "dialog" is "false", set "person_type" to "null" (as a string).
STEP 7: Do not alter the original "text" of blocks that are part of the story.
STEP 8: For the "ssml" field, generate a simple SSML string. Wrap the original text with `<speak><prosody rate='slow'>...</prosody></speak>`.

JSON Output Format Guidelines:

GUIDELINE 0: The output MUST be a single, valid JSON object string. Double-check your output structure.
Example of the desired structure for each block in the output JSON:
{{
    "0": {{
        "text": "example text",
        "words": ["example", "word", "1"],
        "bounding_boxes": [ [[...],[...]], [[...],[...]], [[...],[...]] ],
        "ssml": "<speak><prosody rate='slow'>example text</prosody></speak>",
        "dialog": "false",
        "person_type": "null"
    }},
    "1": {{
        "text": "Hello there!",
        "words": ["Hello", "there!"],
        "bounding_boxes": [ [[...],[...]], [[...],[...]] ],
        "ssml": "<speak><prosody rate='slow'>Hello there!</prosody></speak>",
        "dialog": "true",
        "person_type": "young boy"
    }}
    // ... more blocks
}}

RULE YOU MUST ADHERE ALWAYS -Make sure the output and the input JSONs have only one difference it is that they the out put have the new three fields dialog , person_type and ssml


GUIDELINE 1: Ensure the SSML is compatible with AWS Polly (simple prosody as shown is fine).
GUIDELINE 2: Use only double quotes (") for keys and string values in the JSON. Do NOT use single quotes (').
GUIDELINE 3: The entire response must be a valid JSON string, parsable with standard JSON libraries (e.g., Python's `json.loads`).
GUIDELINE 4: Do not interpret or escape special characters within the text meant for SSML differently than they appear, unless necessary for valid SSML/JSON.
GUIDELINE 5: Do not include `<audio>` tags in the SSML.
GUIDELINE 6: CRITICAL: Ensure the generated JSON is COMPLETE and NOT TRUNCATED. The final character of your response must be the closing brace `}}` of the main JSON object.

Here is the input JSON data containing the blocks:
{blocks_json_str}

Respond ONLY with the processed JSON object string. Do not include any other text, explanations, or markdown formatting like ```json ... ``` around the JSON. Your response should start with `{{` and end with `}}`.
"""

def clean_llm_response_to_json_string(llm_response_text: str) -> Optional[str]:
    """
    Attempts to extract a valid JSON string from the LLM's raw output.
    Handles cases where the LLM might wrap the JSON in markdown or add extraneous text.
    """
    text = llm_response_text.strip()

    # Common markdown code block for JSON
    if text.startswith("```json"):
        text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Fallback: try to find the first '{' and last '}'
    # This is greedy and might fail for malformed JSONs with nested structures and premature endings.
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]
    else:
        logging.warning(f"Could not find clear JSON structure {{...}} in response: {llm_response_text[:200]}...") # Log snippet
        return text # Return as is, parsing will likely fail but gives context

def generate_block_json(base64_image_string: str, blocks_input_json_str: str) -> Optional[Dict[str, Any]]:
    """
    Generates SSML and dialog information for a chunk of text blocks.

    Args:
        base64_image_string: Base64 encoded string of the PDF page image (JPEG).
        blocks_input_json_str: JSON string containing a subset of the initial block information.

    Returns:
        A dictionary representing the processed JSON chunk, or None if processing fails after retries.
    """
    try:
        # Check if we can import vertexai (it may not be properly configured)
        from vertexai.generative_models import GenerativeModel
        
        image_part = Part.from_data(
            mime_type="image/jpeg",
            data=base64.b64decode(base64_image_string)
        )
    except ImportError as e:
        logging.error(f"Vertex AI not available: {e}")
        return create_fallback_block_json(blocks_input_json_str)
    except Exception as e:
        logging.error(f"Error setting up Vertex AI or decoding base64 image: {e}")
        return create_fallback_block_json(blocks_input_json_str)

    prompt_text = construct_gemini_prompt(blocks_input_json_str)

    model = GenerativeModel(
        MODEL_NAME,
        generation_config=GENERATION_CONFIG,
        safety_settings=SAFETY_SETTINGS
    )

    for attempt in range(MAX_RETRIES):
        logging.info(f"Attempting to generate content for chunk (Attempt {attempt + 1}/{MAX_RETRIES})...")
        try:
            response = model.generate_content(
                [image_part, prompt_text],
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS,
                stream=False
            )

            if not response.candidates or not response.candidates[0].content.parts:
                logging.warning("Received empty or unexpected response from Gemini for chunk.")
                if response.prompt_feedback:
                    logging.warning(f"Prompt feedback: {response.prompt_feedback}")
                if response.candidates and response.candidates[0].finish_reason not in (1, "STOP"):
                    logging.warning(f"Generation stopped for chunk due to: {response.candidates[0].finish_reason}")

                time.sleep(RETRY_DELAY_SECONDS)
                continue

            raw_response_text = response.text
            logging.info("Raw response received from Gemini for chunk.")
            # logging.debug(f"Raw response text for chunk: {raw_response_text}") # Uncomment for debugging

            cleaned_json_string = clean_llm_response_to_json_string(raw_response_text)
            if not cleaned_json_string:
                logging.error("Failed to extract a potential JSON string from the LLM response for chunk.")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            try:
                parsed_json = json.loads(cleaned_json_string)
                logging.info("Successfully parsed JSON from Gemini response for chunk.")
                return parsed_json # Success for this chunk!
            except json.JSONDecodeError as e:
                logging.error(f"JSONDecodeError on chunk attempt {attempt + 1}: {e}")
                logging.error(f"Problematic JSON string snippet (chunk): {cleaned_json_string}...")
                if attempt < MAX_RETRIES - 1:
                    logging.info(f"Retrying chunk in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logging.error("Max retries reached for chunk. Failed to get valid JSON.")
                    return None
            except Exception as e:
                logging.error(f"An unexpected error occurred on chunk attempt {attempt + 1}: {e}")
                try:
                    if 'raw_response_text' in locals() and raw_response_text:
                        logging.error(f"Raw response causing error (chunk): {raw_response_text[:500]}...")
                except Exception as log_e:
                    logging.error(f"Error logging raw response (chunk): {log_e}")
                if attempt < MAX_RETRIES - 1:
                    logging.info(f"Retrying chunk in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logging.error("Max retries reached for chunk. Failed due to an unexpected error.")
                    return None
        except Exception as e:
            logging.error(f"An outer error occurred during chunk processing (Attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                logging.info(f"Retrying chunk in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error("Max retries reached for chunk due to an outer error.")
                return None
    return None

def chunk_and_process_json(base64_image_string: str, blocks_input_json_str: str, chunk_size: int = OUTPUT_CHUNK_SIZE) -> Optional[Dict[str, Any]]:
    """Chunks the input JSON, processes each chunk, and merges the results."""
    try:
        all_blocks = json.loads(blocks_input_json_str)
        block_keys = list(all_blocks.keys())
        final_output = {}

        for i in range(0, len(block_keys), chunk_size):
            chunk_keys = block_keys[i : i + chunk_size]
            current_chunk = {key: all_blocks[key] for key in chunk_keys}
            chunk_json_str = json.dumps(current_chunk)

            processed_chunk = generate_block_json(base64_image_string, chunk_json_str)
            if processed_chunk:
                final_output.update(processed_chunk)
            else:
                logging.error(f"Failed to process chunk starting with block key: {chunk_keys[0] if chunk_keys else 'None'}. Results might be incomplete.")
                return None # Or decide to return partial results

        return final_output

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding the main input JSON for chunking: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during chunking and processing: {e}")
        return None

def create_fallback_block_json(blocks_input_json_str: str) -> Optional[Dict[str, Any]]:
    """
    Creates a fallback block JSON when Vertex AI is not available.
    Adds basic SSML, dialog=false, and person_type=null to all blocks.
    """
    try:
        blocks = json.loads(blocks_input_json_str)
        fallback_output = {}
        
        for block_id, block_data in blocks.items():
            fallback_output[block_id] = {
                "text": block_data.get("text", ""),
                "words": block_data.get("words", []),
                "bounding_boxes": block_data.get("bounding_boxes", []),
                "ssml": f"<speak><prosody rate='slow'>{block_data.get('text', '')}</prosody></speak>",
                "dialog": "false",
                "person_type": "null"
            }
        
        logging.info("Created fallback block JSON without AI processing")
        return fallback_output
        
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing input JSON for fallback: {e}")
        return None
    except Exception as e:
        logging.error(f"Error creating fallback block JSON: {e}")
        return None