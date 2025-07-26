import re
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdf(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    text = ''
    for i, image in enumerate(images):
        if i < 2 or (31 <= i <= 40):
            continue
        text += pytesseract.image_to_string(image, lang='ben+eng')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\r]+', '\n', text)
    return text.strip()