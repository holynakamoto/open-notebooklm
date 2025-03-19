# app.py
import base64
import glob
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Optional

from modal import App, Image, Secret, fastapi_endpoint
import gradio as gr
from loguru import logger
from pypdf import PdfReader
from pydub import AudioSegment
import random

from constants import (
    APP_TITLE, CHARACTER_LIMIT, ERROR_MESSAGE_NOT_PDF, ERROR_MESSAGE_NO_INPUT,
    ERROR_MESSAGE_NOT_SUPPORTED_IN_MELO_TTS, ERROR_MESSAGE_READING_PDF,
    ERROR_MESSAGE_TOO_LONG, GRADIO_CACHE_DIR, GRADIO_CLEAR_CACHE_OLDER_THAN,
    MELO_TTS_LANGUAGE_MAPPING, NOT_SUPPORTED_IN_MELO_TTS, SUNO_LANGUAGE_MAPPING,
    UI_ALLOW_FLAGGING, UI_API_NAME, UI_CACHE_EXAMPLES, UI_CONCURRENCY_LIMIT,
    UI_DESCRIPTION, UI_EXAMPLES, UI_INPUTS, UI_OUTPUTS, UI_SHOW_API
)
from prompts import (
    LANGUAGE_MODIFIER, LENGTH_MODIFIERS, QUESTION_MODIFIER, SYSTEM_PROMPT, TONE_MODIFIER
)
from schema import ShortDialogue, MediumDialogue
from utils import generate_podcast_audio, generate_script, parse_url

app = App("open-notebooklm")
image = Image.debian_slim(python_version="3.11") \
    .pip_install_from_requirements("requirements.txt") \
    .apt_install("ffmpeg", "libsndfile1") \
    .pip_install("modal") \
    .add_local_python_source("constants", "prompts", "schema", "utils")

def generate_podcast(
    files: List[str],
    url: Optional[str],
    question: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    language: str,
    use_advanced_audio: bool,
) -> Tuple[str, str]:
    logger.info("Starting podcast generation: files=%s, url=%s, language=%s", files, url, language)
    text = ""
    random_voice_number = random.randint(0, 8)

    if not use_advanced_audio and language in NOT_SUPPORTED_IN_MELO_TTS:
        logger.error("Unsupported language for Melo TTS: %s", language)
        raise gr.Error(ERROR_MESSAGE_NOT_SUPPORTED_IN_MELO_TTS)

    if not files and not url:
        logger.error("No input provided")
        raise gr.Error(ERROR_MESSAGE_NO_INPUT)

    if files:
        logger.info("Processing %d files", len(files))
        for file in files:
            if not file.lower().endswith(".pdf"):
                logger.error("Invalid file type: %s", file)
                raise gr.Error(ERROR_MESSAGE_NOT_PDF)
            try:
                with Path(file).open("rb") as f:
                    reader = PdfReader(f)
                    extracted_text = "\n\n".join([page.extract_text() for page in reader.pages])
                    text += extracted_text
                    logger.debug("Extracted text from %s: %s chars", file, len(extracted_text))
            except Exception as e:
                logger.error("Failed to read PDF %s: %s", file, str(e))
                raise gr.Error(f"{ERROR_MESSAGE_READING_PDF}: {str(e)}")

    if url:
        logger.info("Processing URL: %s", url)
        try:
            url_text = parse_url(url)
            text += "\n\n" + url_text
            logger.debug("Extracted text from URL: %s chars", len(url_text))
        except ValueError as e:
            logger.error("Failed to parse URL %s: %s", url, str(e))
            raise gr.Error(str(e))

    if len(text) > CHARACTER_LIMIT:
        logger.error("Text exceeds character limit: %d > %d", len(text), CHARACTER_LIMIT)
        raise gr.Error(ERROR_MESSAGE_TOO_LONG)

    modified_system_prompt = SYSTEM_PROMPT
    if question:
        modified_system_prompt += f"\n\n{QUESTION_MODIFIER} {question}"
    if tone:
        modified_system_prompt += f"\n\n{TONE_MODIFIER} {tone}."
    if length:
        modified_system_prompt += f"\n\n{LENGTH_MODIFIERS[length]}"
    if language:
        modified_system_prompt += f"\n\n{LANGUAGE_MODIFIER} {language}."

    logger.debug("Modified system prompt: %s", modified_system_prompt[:100] + "..." if len(modified_system_prompt) > 100 else modified_system_prompt)

    if length == "Short (1-2 min)":
        llm_output = generate_script(modified_system_prompt, text, ShortDialogue)
    else:
        llm_output = generate_script(modified_system_prompt, text, MediumDialogue)

    logger.info("Generated dialogue: %s", llm_output)

    audio_segments = []
    transcript = ""
    total_characters = 0

    for line in llm_output.dialogue:
        logger.info("Generating audio for %s: %s", line.speaker, line.text[:50] + "..." if len(line.text) > 50 else line.text)
        if line.speaker == "Host (Jane)":
            speaker = f"**Host**: {line.text}"
        else:
            speaker = f"**{llm_output.name_of_guest}**: {line.text}"
        transcript += speaker + "\n\n"
        total_characters += len(line.text)

        language_for_tts = SUNO_LANGUAGE_MAPPING[language]
        if not use_advanced_audio:
            language_for_tts = MELO_TTS_LANGUAGE_MAPPING[language_for_tts]

        audio_file_path = generate_podcast_audio(
            line.text, line.speaker, language_for_tts, use_advanced_audio, random_voice_number
        )
        logger.debug("Generated audio file: %s", audio_file_path)
        audio_segment = AudioSegment.from_file(audio_file_path)
        audio_segments.append(audio_segment)

    logger.info("Combining %d audio segments", len(audio_segments))
    combined_audio = sum(audio_segments)
    temporary_directory = GRADIO_CACHE_DIR
    os.makedirs(temporary_directory, exist_ok=True)

    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    combined_audio.export(temporary_file.name, format="mp3")
    logger.debug("Exported combined audio to: %s", temporary_file.name)

    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if (
            os.path.isfile(file)
            and time.time() - os.path.getmtime(file) > GRADIO_CLEAR_CACHE_OLDER_THAN
        ):
            logger.debug("Removing old audio file: %s", file)
            os.remove(file)

    logger.info("Podcast generation completed: %d characters of audio", total_characters)
    return temporary_file.name, transcript

@app.function(image=image, gpu="A100", secrets=[Secret.from_name("FIREWORKS_API_KEY")])
@fastapi_endpoint(method="POST")
async def generate(request: dict):
    logger.info("Received request: %s", request)
    try:
        audio_path, transcript = generate_podcast(
            request.get("files", []),
            request.get("url"),
            request.get("question"),
            request.get("tone"),
            request.get("length"),
            request.get("language", "EN"),
            request.get("use_advanced_audio", False)
        )
        logger.debug("Reading audio file: %s", audio_path)
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        logger.debug("Encoded audio to base64: %d bytes", len(audio_base64))
        if os.path.exists(audio_path):
            logger.debug("Removing temporary audio file: %s", audio_path)
            os.remove(audio_path)
        logger.info("Request completed successfully")
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "transcript": transcript
        }
    except Exception as e:
        logger.error("Request failed: %s", str(e))
        raise Exception(str(e))

# Gradio interface for local testing
demo = gr.Interface(
    title=APP_TITLE,
    description=UI_DESCRIPTION,
    fn=generate_podcast,
    inputs=[
        gr.File(
            label=UI_INPUTS["file_upload"]["label"],
            file_types=UI_INPUTS["file_upload"]["file_types"],
            file_count=UI_INPUTS["file_upload"]["file_count"],
        ),
        gr.Textbox(
            label=UI_INPUTS["url"]["label"],
            placeholder=UI_INPUTS["url"]["placeholder"],
        ),
        gr.Textbox(label=UI_INPUTS["question"]["label"]),
        gr.Dropdown(
            label=UI_INPUTS["tone"]["label"],
            choices=UI_INPUTS["tone"]["choices"],
            value=UI_INPUTS["tone"]["value"],
        ),
        gr.Dropdown(
            label=UI_INPUTS["length"]["label"],
            choices=UI_INPUTS["length"]["choices"],
            value=UI_INPUTS["length"]["value"],
        ),
        gr.Dropdown(
            choices=UI_INPUTS["language"]["choices"],
            value=UI_INPUTS["language"]["value"],
            label=UI_INPUTS["language"]["label"],
        ),
        gr.Checkbox(
            label=UI_INPUTS["advanced_audio"]["label"],
            value=False,
        ),
    ],
    outputs=[
        gr.Audio(
            label=UI_OUTPUTS["audio"]["label"],
            format=UI_OUTPUTS["audio"]["format"]
        ),
        gr.Markdown(label=UI_OUTPUTS["transcript"]["label"]),
    ],
    allow_flagging="never",  # Updated from flagging_mode
    api_name=UI_API_NAME,
    theme=gr.themes.Default(),
    concurrency_limit=UI_CONCURRENCY_LIMIT,
    examples=UI_EXAMPLES,
    cache_examples=UI_CACHE_EXAMPLES,
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info("Starting Gradio interface on port %d", port)
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=UI_SHOW_API)
