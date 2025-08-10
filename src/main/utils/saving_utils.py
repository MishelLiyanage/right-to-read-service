import os
import json
import io

# Mapping person types to Amazon Polly voice IDs
PERSON_TYPE_TO_VOICE = {
    "young boy": "Justin",
    "old man": "Matthew",
    "young girl": "Ivy",
    "old woman": "Kimberly",
    "middle aged man": "Joey",
    "middle aged woman": "Joanna"
}


def save_annotated_image(image, output_dir, output_name, page_number):
    """
    Returns the path for the annotated image. (Saving is currently disabled.)
    """
    annotated_image_path = os.path.join(output_dir, f"{output_name}_annotated_page_{page_number}_blocks.png")
    # image.save(annotated_image_path)  # Uncomment to actually save the image
    return annotated_image_path


def save_block_details_as_json(block_details, output_dir, output_name, page_number):
    """
    Saves block details to a JSON file and returns the path.
    """
    json_path = os.path.join(output_dir, f"{output_name}_page_{page_number}_blocks.json")
    with open(json_path, "w") as json_file:
        json.dump(block_details, json_file, indent=4)
    return json_path


def save_audio_and_speech_marks(polly_client, block_id, ssml_output, output_dir, person_type, block_json, block_key):
    """
    Generates audio and speech marks for a block and updates block_json with timing data.
    Returns paths to the audio and speech marks files.
    """
    if polly_client is None:
        # Create dummy files when Polly is not available
        audio_path = os.path.join(output_dir, f"block_{block_id}_audio.mp3")
        speech_marks_path = os.path.join(output_dir, f"block_{block_id}_speech_marks.json")
        
        # Create empty audio file (placeholder)
        with open(audio_path, "wb") as audio_file:
            audio_file.write(b"")  # Empty audio file
            
        # Create empty speech marks
        with open(speech_marks_path, "w") as marks_file:
            json.dump([], marks_file, indent=4)
            
        # Add empty timing info to the block JSON
        if str(block_key) in block_json:
            block_json[str(block_key)]["timing"] = []
            
        return audio_path, speech_marks_path
    
    # Normalize person type and get voice ID
    voice_id = PERSON_TYPE_TO_VOICE.get(person_type.lower() if person_type else None, "Joanna")

    # Generate audio
    audio_response = polly_client.synthesize_speech(
        Engine='standard',
        OutputFormat='mp3',
        Text=ssml_output,
        TextType='ssml',
        VoiceId=voice_id
    )
    audio_path = os.path.join(output_dir, f"block_{block_id}_audio.mp3")
    with open(audio_path, "wb") as audio_file:
        audio_file.write(audio_response['AudioStream'].read())

    # Generate speech marks
    speech_marks_response = polly_client.synthesize_speech(
        Engine='standard',
        OutputFormat='json',
        Text=ssml_output,
        TextType='ssml',
        VoiceId=voice_id,
        SpeechMarkTypes=['word']
    )

    # Parse speech marks stream
    speech_marks = []
    stream = io.TextIOWrapper(speech_marks_response['AudioStream'], encoding='utf-8')
    for line in stream:
        try:
            speech_marks.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue

    # Save speech marks to file
    speech_marks_path = os.path.join(output_dir, f"block_{block_id}_speech_marks.json")
    with open(speech_marks_path, "w") as marks_file:
        json.dump(speech_marks, marks_file, indent=4)

    # Add timing info to the block JSON
    if str(block_key) in block_json:
        block_json[str(block_key)]["timing"] = speech_marks

    return audio_path, speech_marks_path
