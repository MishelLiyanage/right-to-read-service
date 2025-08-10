from PIL import Image, ImageDraw, ImageFont
import fitz
import base64
import os
import random

def extract_page_as_base64(pdf_path, page_number, output_dir, output_name):
    os.makedirs(output_dir, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    image_bytes = pix.tobytes("png")
    base64_image_string = base64.b64encode(image_bytes).decode('utf-8')
    
    image_path = os.path.join(output_dir, f"{output_name}_page_{page_number}.png")
    with open(image_path, "wb") as img_file:
        img_file.write(image_bytes)
        
    return base64_image_string, image_path, page

def annotate_image_with_words(image, words, color_palette, block_details):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for word_info in words:
        x0, y0, x1, y1, word, block_no, line_no, word_no = word_info
        rect = [(x0, y0), (x1, y1)]
        color = color_palette[block_no]
        draw.rectangle(rect, outline=color, width=2)

        if block_no not in block_details:
            block_details[block_no] = {
                "text": "",
                "words": [],
                "bounding_boxes": []
            }

        block_details[block_no]["text"] += word + " "
        block_details[block_no]["words"].append(word)
        block_details[block_no]["bounding_boxes"].append(rect)

        if word_no == 0 and line_no == 0:
            label_position = (x0, y0 - 15)
            draw.text(label_position, f"Block {block_no}", fill=color, font=font)

    for block in block_details.values():
        block["text"] = block["text"].strip()

def generate_color_palette(block_numbers):
    return {block_no: tuple(random.randint(0, 255) for _ in range(3)) for block_no in block_numbers}