from deep_translator import GoogleTranslator

def translate_text(text: str, max_length: int = 500) -> str:
    translator = GoogleTranslator(source='auto', target='en')
    segments = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    translated_segments = [translator.translate(segment) for segment in segments if segment.strip()]
    return " ".join(translated_segments)