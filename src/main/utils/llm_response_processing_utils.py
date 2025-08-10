import json

def clean_llm_response(response):
    # response = response.replace('```json', '').replace('```', '').strip()
    processed_response = json.loads(response)
    return processed_response