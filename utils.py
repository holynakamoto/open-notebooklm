"""
utils.py

Functions:
- generate_script: Get the dialogue from the LLM.
- call_llm: Call the LLM with the given prompt and dialogue format.
- parse_url: Parse the given URL and return the text content.
- generate_podcast_audio: Generate audio for podcast using TTS or advanced audio models.
- _use_suno_model: Generate advanced audio using Bark.
- _use_melotts_api: Generate audio using TTS model.
- _get_melo_tts_params: Get TTS parameters based on speaker and language.
"""

# Standard library imports
import time
from typing import Any, Union

# Third-party imports
import torch
import numpy
import instructor
import requests
import os
from bark import SAMPLE_RATE, generate_audio, preload_models
from fireworks.client import Fireworks
from gradio_client import Client
from scipy.io.wavfile import write as write_wav
from loguru import logger

# Local imports
from constants import (
    FIREWORKS_API_KEY,
    FIREWORKS_MODEL_ID,
    FIREWORKS_MAX_TOKENS,
    FIREWORKS_TEMPERATURE,
    MELO_API_NAME,
    MELO_TTS_SPACES_ID,
    MELO_RETRY_ATTEMPTS,
    MELO_RETRY_DELAY,
    JINA_READER_URL,
    JINA_RETRY_ATTEMPTS,
    JINA_RETRY_DELAY,
)
from schema import ShortDialogue, MediumDialogue

# Allowlist NumPy scalar callable for PyTorch 2.6+ weights_only=True
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# Initialize Fireworks client with environment variable fallback
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", FIREWORKS_API_KEY)
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY must be set in environment variables.")
fw_client = Fireworks(api_key=FIREWORKS_API_KEY)
fw_client = instructor.from_fireworks(fw_client)

# Initialize Hugging Face client
hf_client = Client(MELO_TTS_SPACES_ID)

# Preload Bark models with explicit device handling
def preload_bark_models():
    """Preload Bark models with GPU support if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Preloading Bark models on device: {device}")
    try:
        if device.type == "cpu":
            logger.warning("Using CPU locally for compatibility")
        preload_models()  # No weights_only parameter, as it’s not supported in bark==0.1.5
        logger.info("Bark models preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload Bark models: {str(e)}")
        raise

# Uncomment to preload models at startup if desired
# preload_bark_models()


def generate_script(
    system_prompt: str,
    input_text: str,
    output_model: Union[ShortDialogue, MediumDialogue],
) -> Union[ShortDialogue, MediumDialogue]:
    """Get the dialogue from the LLM."""
    first_draft_dialogue = call_llm(system_prompt, input_text, output_model)
    system_prompt_with_dialogue = f"{system_prompt}\n\nHere is the first draft of the dialogue you provided:\n\n{first_draft_dialogue.model_dump_json()}."
    final_dialogue = call_llm(system_prompt_with_dialogue, "Please improve the dialogue. Make it more natural and engaging.", output_model)
    return final_dialogue


def call_llm(system_prompt: str, text: str, dialogue_format: Any) -> Any:
    """Call the LLM with the given prompt and dialogue format."""
    response = fw_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=FIREWORKS_MODEL_ID,
        max_tokens=FIREWORKS_MAX_TOKENS,
        temperature=FIREWORKS_TEMPERATURE,
        response_model=dialogue_format,
    )
    return response


def parse_url(url: str) -> str:
    """Parse the given URL and return the text content."""
    for attempt in range(JINA_RETRY_ATTEMPTS):
        try:
            full_url = f"{JINA_READER_URL}{url}"
            response = requests.get(full_url, timeout=60)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == JINA_RETRY_ATTEMPTS - 1:
                raise ValueError(
                    f"Failed to fetch URL after {JINA_RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            time.sleep(JINA_RETRY_DELAY)
    return response.text


def generate_podcast_audio(
    text: str, speaker: str, language: str, use_advanced_audio: bool, random_voice_number: int
) -> str:
    """Generate audio for podcast using TTS or advanced audio models."""
    if use_advanced_audio:
        return _use_suno_model(text, speaker, language, random_voice_number)
    else:
        return _use_melotts_api(text, speaker, language)


def _use_suno_model(text: str, speaker: str, language: str, random_voice_number: int) -> str:
    """Generate advanced audio using Bark."""
    # Log GPU availability
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        logger.warning("No GPU available, using CPU. Inference might be slow.")
        device = torch.device("cpu")

    # Ensure models are loaded on the correct device
    preload_bark_models()

    host_voice_num = str(random_voice_number)
    guest_voice_num = str(random_voice_number + 1)
    audio_array = generate_audio(
        text,
        history_prompt=f"v2/{language}_speaker_{host_voice_num if speaker == 'Host (Jane)' else guest_voice_num}",
    )
    file_path = f"audio_{language}_{speaker}.mp3"
    write_wav(file_path, SAMPLE_RATE, audio_array)
    return file_path


def _use_melotts_api(text: str, speaker: str, language: str) -> str:
    """Generate audio using TTS model."""
    accent, speed = _get_melo_tts_params(speaker, language)

    for attempt in range(MELO_RETRY_ATTEMPTS):
        try:
            return hf_client.predict(
                text=text,
                language=language,
                speaker=accent,
                speed=speed,
                api_name=MELO_API_NAME,
            )
        except Exception as e:
            if attempt == MELO_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(MELO_RETRY_DELAY)


def _get_melo_tts_params(speaker: str, language: str) -> tuple[str, float]:
    """Get TTS parameters based on speaker and language."""
    if speaker == "Guest":
        accent = "EN-US" if language == "EN" else language
        speed = 0.9
    else:  # host
        accent = "EN-Default" if language == "EN" else language
        speed = 1.1 if language != "EN" else 1
    return accent, speed
