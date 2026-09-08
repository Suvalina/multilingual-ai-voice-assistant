import os
import re
import time
import uuid
import tempfile
import subprocess
import shutil
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag

import streamlit as st
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

import fitz  # PyMuPDF
from docx import Document
import requests
from bs4 import BeautifulSoup

from database import (
    create_session as db_create_session,
    list_sessions as db_list_sessions,
    get_session as db_get_session,
    update_session_title as db_update_session_title,
    save_message as db_save_message,
    load_messages as db_load_messages,
    save_source as db_save_source,
    delete_session as db_delete_session,
    clear_all_history as db_clear_all_history,
)


# ============================================================
# OPTIONAL PLAYWRIGHT
# ============================================================

try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True

except Exception:
    PLAYWRIGHT_AVAILABLE = False


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL",
    "medium"
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
).strip()

FALLBACK_MODELS = [
    "gemini-2.5-flash"
]

MAX_WEBSITE_PAGES = 50
WEBSITE_TIMEOUT = 30000
MAX_SOURCE_TEXT = 50000


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
)


# ============================================================
# API KEY
# ============================================================

if not GEMINI_API_KEY:

    st.error(
        "❌ GEMINI_API_KEY is missing.\n\n"
        "Please add GEMINI_API_KEY to your .env file."
    )

    st.stop()


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# LANGUAGE NAMES
# ============================================================

LANGUAGE_NAMES = {

    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "ne": "Nepali",
    "gu": "Gujarati",
    "as": "Assamese",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",

    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "tr": "Turkish",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "el": "Greek",
    "he": "Hebrew",
    "hu": "Hungarian",
    "no": "Norwegian",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "sw": "Swahili",
    "af": "Afrikaans",
}


# ============================================================
# EDGE TTS VOICES
# ============================================================

TTS_VOICES = {

    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "bn": "bn-IN-TanishaaNeural",
    "or": "or-IN-SubhasiniNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "ne": "ne-NP-HemkalaNeural",
    "gu": "gu-IN-DhwaniNeural",
    "as": "en-IN-NeerjaNeural",
    "mr": "mr-IN-AarohiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-OjasNeural",
    "ur": "ur-PK-UzmaNeural",

    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "tr": "tr-TR-EmelNeural",
    "id": "id-ID-GadisNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural",
    "pl": "pl-PL-ZofiaNeural",
    "nl": "nl-NL-ColetteNeural",
    "sv": "sv-SE-SofieNeural",
    "da": "da-DK-ChristelNeural",
    "fi": "fi-FI-NooraNeural",
    "uk": "uk-UA-PolinaNeural",
    "cs": "cs-CZ-VlastaNeural",
    "ro": "ro-RO-AlinaNeural",
    "el": "el-GR-AthinaNeural",
    "he": "he-IL-HilaNeural",
    "hu": "hu-HU-NoemiNeural",
    "no": "nb-NO-IselinNeural",
    "sk": "sk-SK-ViktoriaNeural",
    "bg": "bg-BG-KalinaNeural",
    "hr": "hr-HR-GabrijelaNeural",
    "sr": "sr-RS-SophieNeural",
    "sl": "sl-SI-PetraNeural",
    "sw": "sw-KE-ZuriNeural",
    "af": "af-ZA-AdriNeural",
}


# ============================================================
# ROMAN LANGUAGE WORDS
# ============================================================

ROMAN_BENGALI = {
    "ami", "amar", "amake", "amra", "amader",
    "tumi", "tomar", "tomake", "tomra",
    "apni", "apnar", "apnake",
    "ki", "kothay", "kothai", "kivabe", "kirokom",
    "keno", "kokhon", "ke", "kar",
    "ache", "achi", "acho", "achen",
    "hobe", "hoy", "hoye", "hoyeche",
    "kor", "koro", "korbo", "korchi", "korte",
    "jabo", "jabe", "jacchi", "gechi",
    "bolo", "bolun", "bolbe", "bollen",
    "dao", "den", "dewa",
    "bhalo", "valo", "khub",
    "ekhane", "okhane", "sekhane",
    "aj", "kal", "ekhon", "pore",
    "jani", "janina", "bujhi", "bujhina",
    "chai", "chaiye", "dorkar", "proyojon",
    "amar", "tomar",
    "dhonnobad", "please",
    "porashona", "project", "office",
    "kotha", "karon", "jonno",
    "theke", "sathe", "ache",
    "hoyechhe", "hochhe",
    "dekh", "dekhao", "dekhi",
    "likho", "lekho", "likhe",
    "shob", "sob", "kichu",
    "onek", "kom", "beshi",
    "age", "por",
    "matha", "bari", "kaj",
    "somossa", "solution",
    "bujhiye", "bojhao", "samjhao",
}


ROMAN_HINDI = {
    "main", "mein", "mera", "meri", "mere",
    "mujhe", "mujhko", "hum", "hamara",
    "aap", "aapka", "aapki", "aapko",
    "tum", "tumhara", "tumhe",
    "kya", "kaise", "kyu", "kyon",
    "kahan", "kab", "kaun",
    "hai", "hain", "hoon", "ho",
    "tha", "thi", "the",
    "hoga", "hogi", "hon",
    "kar", "karo", "karna", "karta",
    "karti", "karte", "karunga",
    "ja", "jana", "jaunga", "jaungi",
    "acha", "accha", "achha",
    "bahut", "nahi", "nahin",
    "chahiye", "zarurat", "pata",
    "samajh", "samjhao", "batao",
    "bolo", "dekho", "dikhao",
    "abhi", "aaj", "kal",
    "phir", "liye", "se",
    "mera", "tera",
}


ROMAN_LANGUAGE_WORDS = {

    "fr": {
        "bonjour", "merci", "avec", "pour",
        "comment", "pourquoi", "quoi", "vous",
        "je", "suis", "dans", "une", "des",
    },

    "de": {
        "hallo", "danke", "bitte", "ich",
        "mein", "meine", "wie", "warum",
        "was", "ist", "sind", "nicht",
    },

    "es": {
        "hola", "gracias", "como", "cómo",
        "porque", "por", "para", "que",
        "quiero", "necesito", "donde",
    },

    "it": {
        "ciao", "grazie", "come", "perche",
        "perché", "cosa", "voglio", "sono",
        "dove", "non",
    },

    "pt": {
        "olá", "ola", "obrigado", "obrigada",
        "como", "porque", "quero", "preciso",
        "onde", "você", "voce",
    },

    "tr": {
        "merhaba", "tesekkur", "teşekkür",
        "nasıl", "nasil", "neden", "ne",
        "ben", "sen", "istiyorum",
    },

    "id": {
        "halo", "terima", "kasih", "bagaimana",
        "mengapa", "apa", "saya", "anda",
        "ingin", "butuh",
    },

    "vi": {
        "xin", "chào", "cam", "ơn", "cảm",
        "như", "thế", "nào", "tôi", "bạn",
        "muốn", "cần",
    },

    "nl": {
        "hallo", "dank", "dankje", "hoe",
        "waarom", "wat", "ik", "jij",
        "niet", "wil",
    },

    "sv": {
        "hej", "tack", "hur", "varför",
        "vad", "jag", "du", "inte",
    },

    "da": {
        "hej", "tak", "hvordan", "hvorfor",
        "hvad", "jeg", "du", "ikke",
    },

    "fi": {
        "hei", "kiitos", "miten", "miksi",
        "mitä", "minä", "sinä", "en",
    },

    "pl": {
        "czesc", "cześć", "dziekuje",
        "dziękuję", "jak", "dlaczego",
        "co", "jest", "nie", "chce",
    },

    "ro": {
        "salut", "multumesc", "mulțumesc",
        "cum", "de", "ce", "ce", "vreau",
        "sunt", "nu",
    },

    "hu": {
        "szia", "köszönöm", "koszonom",
        "hogyan", "miért", "miert",
        "mit", "én", "en", "nem",
    },

    "cs": {
        "ahoj", "děkuji", "dekuji",
        "jak", "proč", "proc", "co",
        "jsem", "není", "neni",
    },

    "sk": {
        "ahoj", "ďakujem", "dakujem",
        "ako", "prečo", "preco",
        "čo", "co", "som", "nie",
    },

    "sl": {
        "zdravo", "hvala", "kako",
        "zakaj", "kaj", "jaz", "ti",
        "ni",
    },

    "hr": {
        "bok", "hvala", "kako", "zašto",
        "zasto", "što", "sto", "ja", "ti",
        "nije",
    },

    "sr": {
        "zdravo", "hvala", "kako", "zašto",
        "zasto", "šta", "sta", "ja", "ti",
        "nije",
    },

    "sw": {
        "habari", "asante", "vipi", "nini",
        "kwa", "mimi", "wewe", "sawa",
    },

    "af": {
        "hallo", "dankie", "hoe", "hoekom",
        "wat", "ek", "jy", "nie",
    },
}


# ============================================================
# SCRIPT DETECTION
# ============================================================

def detect_script_language(text):

    if not text:
        return None

    counts = {}

    ranges = {

        "bn": (0x0980, 0x09FF),
        "hi": (0x0900, 0x097F),
        "pa": (0x0A00, 0x0A7F),
        "gu": (0x0A80, 0x0AFF),
        "or": (0x0B00, 0x0B7F),
        "ta": (0x0B80, 0x0BFF),
        "te": (0x0C00, 0x0C7F),
        "kn": (0x0C80, 0x0CFF),
        "ml": (0x0D00, 0x0D7F),
        "th": (0x0E00, 0x0E7F),
        "he": (0x0590, 0x05FF),
        "ar": (0x0600, 0x06FF),
        "ja": (0x3040, 0x30FF),
        "ko": (0xAC00, 0xD7AF),
        "zh": (0x4E00, 0x9FFF),
        "ru": (0x0400, 0x04FF),
        "el": (0x0370, 0x03FF),
    }

    for char in text:

        code = ord(char)

        for lang, (start, end) in ranges.items():

            if start <= code <= end:

                counts[lang] = counts.get(lang, 0) + 1
                break

    if not counts:
        return None

    language, count = max(
        counts.items(),
        key=lambda x: x[1]
    )

    if count >= 2:
        return language

    return None


# ============================================================
# URDU DETECTION
# ============================================================

def contains_urdu_specific_characters(text):

    return any(
        char in set("ٹڈڑںھہےےژچگپ")
        for char in text
    )


# ============================================================
# WORD NORMALIZATION
# ============================================================

def normalize_words(text):

    if not text:
        return []

    text = text.lower().strip()

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    return re.findall(
        r"[^\W_]+",
        text,
        flags=re.UNICODE
    )


# ============================================================
# ROMAN LANGUAGE DETECTION
# ============================================================

def detect_roman_language(text):

    words = set(
        normalize_words(text)
    )

    if not words:
        return "en"

    # --------------------------------------------------------
    # Bengali
    # --------------------------------------------------------

    bn_score = len(
        words.intersection(
            ROMAN_BENGALI
        )
    )

    # --------------------------------------------------------
    # Hindi
    # --------------------------------------------------------

    hi_score = len(
        words.intersection(
            ROMAN_HINDI
        )
    )

    # Strong Bengali phrases
    bn_phrases = [
        "ami chai",
        "amar jonno",
        "amake bolo",
        "kivabe korbo",
        "ki korbo",
        "ki kore",
        "bujhiye dao",
        "bujhiye bolo",
        "ekhane bolo",
        "amar project",
        "office e",
        "kothay change",
        "step by step bolo",
    ]

    # Strong Hindi phrases
    hi_phrases = [
        "mujhe batao",
        "mujhe samjhao",
        "kaise karu",
        "kya karu",
        "mere liye",
        "mujhe chahiye",
        "step by step batao",
        "kahan change",
    ]

    normalized = " ".join(
        normalize_words(text)
    )

    for phrase in bn_phrases:

        if phrase in normalized:
            bn_score += 5

    for phrase in hi_phrases:

        if phrase in normalized:
            hi_score += 5

    # Bengali wins ties
    if bn_score >= 2 and bn_score >= hi_score:
        return "bn"

    if hi_score >= 2:
        return "hi"

    # --------------------------------------------------------
    # Other Roman languages
    # --------------------------------------------------------

    scores = {}

    for language, vocabulary in ROMAN_LANGUAGE_WORDS.items():

        score = len(
            words.intersection(
                vocabulary
            )
        )

        if score:
            scores[language] = score

    distinctive = {

        "bonjour": "fr",
        "merci": "fr",

        "hola": "es",
        "gracias": "es",

        "ciao": "it",

        "obrigado": "pt",
        "obrigada": "pt",

        "merhaba": "tr",

        "habari": "sw",

        "szia": "hu",

        "ahoj": "cs",

        "hei": "fi",

        "takk": "no",
    }

    for word, language in distinctive.items():

        if word in words:
            return language

    if scores:

        best_language, best_score = max(
            scores.items(),
            key=lambda x: x[1]
        )

        if best_score >= 2:
            return best_language

    return "en"


# ============================================================
# TEXT LANGUAGE
# ============================================================

def detect_explicit_language_request(text):

    """Detect direct requests such as 'banglay bolo' or 'answer in Hindi'."""

    if not text:
        return None

    normalized = " ".join(normalize_words(text))

    patterns = {
        "bn": [
            r"bangla(?:y|te)?\s+(?:bolo|bolun|dao|den|likho|lekho|answer|reply)",
            r"banglish\s+(?:e|te)?\s*(?:bolo|answer|reply|dao)",
            r"bengali\s+(?:te|in)?\s*(?:bolo|answer|reply)",
            r"বাংলা(?:য়|য়|তে)?",
        ],
        "hi": [
            r"hindi\s+(?:me|mein|mai|in)?\s*(?:bolo|batao|answer|reply|do)",
            r"hinglish\s+(?:me|mein|e)?\s*(?:bolo|answer|reply)",
            r"हिंदी\s*(?:में|मे)?",
        ],
        "en": [
            r"(?:answer|reply|respond)\s+in\s+english",
            r"english\s+(?:e|te|me)?\s*(?:bolo|answer|reply)",
        ],
        "ta": [r"tamil\s+(?:la|il|in)?\s*(?:answer|reply|bolo)", r"தமிழில்"],
        "te": [r"telugu\s+(?:lo|in)?\s*(?:answer|reply|cheppu)", r"తెలుగులో"],
        "or": [r"odia\s+(?:re|te)?\s*(?:answer|reply|bolo)", r"ଓଡ଼ିଆରେ"],
        "mr": [r"marathi\s+(?:madhe|in)?\s*(?:answer|reply|sanga)", r"मराठीत"],
        "gu": [r"gujarati\s+(?:ma|in)?\s*(?:answer|reply)", r"ગુજરાતીમાં"],
    }

    for language, expressions in patterns.items():
        for expression in expressions:
            if re.search(expression, text, flags=re.IGNORECASE):
                return language

    return None


def detect_text_language(text):

    if not text or not text.strip():
        return "en"

    explicit_language = detect_explicit_language_request(text)
    if explicit_language in LANGUAGE_NAMES:
        return explicit_language

    script_language = detect_script_language(text)

    if script_language:

        if script_language == "ar":

            if contains_urdu_specific_characters(text):
                return "ur"

            return "ar"

        return script_language

    return detect_roman_language(text)


# ============================================================
# LANGUAGE INSTRUCTION
# ============================================================

def get_language_instruction(language_code):

    if language_code == "bn":

        return """
Answer in Bengali.

If the current user question is written in Bengali script,
answer in natural Bengali script.

If the current user question is written in Romanized Bengali
/Banglish, answer in natural Banglish using Roman letters.

Do NOT automatically convert Banglish into Bengali script.
Do NOT convert Bengali into English.
"""

    if language_code == "hi":

        return """
Answer in Hindi.

If the current user question is written in Devanagari,
answer in natural Hindi Devanagari.

If the current user question is written in Romanized Hindi
/Hinglish, answer in natural Hinglish using Roman letters.

Do NOT automatically convert Hinglish into Devanagari.
Do NOT convert Hindi into English.
"""

    if language_code == "en":

        return """
Answer in natural English.
"""

    language_name = LANGUAGE_NAMES.get(
        language_code,
        "English"
    )

    return f"""
Answer in {language_name}.

Use the same natural language style as the CURRENT USER
QUESTION.

The uploaded source language must NOT override this.
"""


# ============================================================
# WHISPER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_whisper():

    return WhisperModel(
        WHISPER_MODEL_NAME,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )


# ============================================================
# FFMPEG
# ============================================================

def check_ffmpeg():

    return shutil.which("ffmpeg") is not None


def convert_audio_to_wav(input_path):

    if not check_ffmpeg():

        raise RuntimeError(
            "FFmpeg is required. "
            "Install FFmpeg and add it to PATH."
        )

    output_path = os.path.join(
        tempfile.gettempdir(),
        f"audio_{uuid.uuid4().hex}.wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    try:

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=True,
        )

        return output_path

    except subprocess.CalledProcessError as e:

        error_text = e.stderr.decode(
            "utf-8",
            errors="ignore"
        )

        raise RuntimeError(
            f"FFmpeg audio conversion failed:\n{error_text[-2000:]}"
        )

    except subprocess.TimeoutExpired:

        raise RuntimeError(
            "FFmpeg audio conversion timed out."
        )


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(
    model,
    audio_file,
    return_segments=False
):

    segments, info = model.transcribe(

        audio_file,

        language=None,

        task="transcribe",

        beam_size=5,

        best_of=5,

        temperature=0,

        condition_on_previous_text=False,

        vad_filter=True,

        vad_parameters={
            "min_silence_duration_ms": 500
        },
    )

    segment_data = []

    transcript_parts = []

    for segment in segments:

        text = segment.text.strip()

        if not text:
            continue

        transcript_parts.append(text)

        segment_data.append({

            "start": float(
                segment.start
            ),

            "end": float(
                segment.end
            ),

            "text": text,
        })

    transcript = " ".join(
        transcript_parts
    ).strip()

    whisper_language = getattr(
        info,
        "language",
        "en"
    )

    whisper_probability = getattr(
        info,
        "language_probability",
        0
    )

    if whisper_language == "bh":
        whisper_language = "hi"

    text_language = detect_text_language(
        transcript
    )

    script_language = detect_script_language(
        transcript
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Roman Bengali/Hindi detection gets priority
    # over Whisper's generic language prediction.
    # --------------------------------------------------------

    if script_language:

        final_language = script_language

    elif text_language in {
        "bn",
        "hi",
    }:

        final_language = text_language

    elif (
        whisper_language in LANGUAGE_NAMES
        and
        whisper_probability >= 0.60
    ):

        final_language = whisper_language

    elif text_language in LANGUAGE_NAMES:

        final_language = text_language

    else:

        final_language = "en"

    print("=" * 70)
    print("WHISPER LANGUAGE:", whisper_language)
    print("WHISPER PROBABILITY:", whisper_probability)
    print("TEXT LANGUAGE:", text_language)
    print("SCRIPT LANGUAGE:", script_language)
    print("FINAL LANGUAGE:", final_language)
    print("TRANSCRIPT:", transcript)
    print("=" * 70)

    if return_segments:

        return (
            final_language,
            transcript,
            segment_data,
        )

    return (
        final_language,
        transcript,
    )


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_data(file_bytes):

    pages = []

    try:

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        for index, page in enumerate(document):

            text = page.get_text(
                "text"
            ).strip()

            pages.append({

                "page_number":
                index + 1,

                "text":
                text,
            })

        document.close()

    except Exception as e:

        print(
            "PDF extraction error:",
            repr(e)
        )

    return pages


# ============================================================
# PDF PAGE RENDER
# ============================================================

def render_pdf_page(
    file_bytes,
    page_number
):

    try:

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        index = page_number - 1

        if index < 0 or index >= len(document):

            document.close()
            return None

        page = document[index]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(1.6, 1.6),
            alpha=False
        )

        image_bytes = pix.tobytes(
            "png"
        )

        document.close()

        return image_bytes

    except Exception as e:

        print(
            "PDF render error:",
            repr(e)
        )

        return None


# ============================================================
# PDF RELEVANT CROP
# ============================================================

def render_pdf_relevant_crop(
    file_bytes,
    page_number,
    question,
    padding=30
):

    try:

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        page = document[
            page_number - 1
        ]

        words = page.get_text(
            "words"
        )

        question_tokens = set(
            relevance_tokens(question)
        )

        matched_rects = []

        for word in words:

            word_text = str(
                word[4]
            ).lower()

            normalized = set(
                normalize_words(
                    word_text
                )
            )

            if normalized.intersection(
                question_tokens
            ):

                matched_rects.append(
                    fitz.Rect(
                        word[0],
                        word[1],
                        word[2],
                        word[3],
                    )
                )

        if matched_rects:

            rect = matched_rects[0]

            for current in matched_rects[1:]:
                rect |= current

        else:

            blocks = page.get_text(
                "blocks"
            )

            best_score = -1
            best_rect = None

            for block in blocks:

                block_text = block[4]

                score = calculate_relevance(
                    question,
                    block_text
                )

                if score > best_score:

                    best_score = score

                    best_rect = fitz.Rect(
                        block[:4]
                    )

            rect = (
                best_rect
                if best_rect
                else page.rect
            )

        rect.x0 = max(
            page.rect.x0,
            rect.x0 - padding
        )

        rect.y0 = max(
            page.rect.y0,
            rect.y0 - padding
        )

        rect.x1 = min(
            page.rect.x1,
            rect.x1 + padding
        )

        rect.y1 = min(
            page.rect.y1,
            rect.y1 + padding
        )

        # Avoid extremely tiny crop
        if rect.width < 100:
            rect.x1 = min(
                page.rect.x1,
                rect.x0 + 100
            )

        if rect.height < 60:
            rect.y1 = min(
                page.rect.y1,
                rect.y0 + 60
            )

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            clip=rect,
            alpha=False
        )

        image_bytes = pix.tobytes(
            "png"
        )

        document.close()

        return image_bytes

    except Exception as e:

        print(
            "PDF crop error:",
            repr(e)
        )

        return None


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_text_from_docx(
    file_bytes
):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".docx"
        ) as tmp:

            tmp.write(file_bytes)

            temp_path = tmp.name

        document = Document(
            temp_path
        )

        parts = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                parts.append(text)

        for table in document.tables:

            for row in table.rows:

                row_text = []

                for cell in row.cells:

                    cell_text = cell.text.strip()

                    if cell_text:
                        row_text.append(
                            cell_text
                        )

                if row_text:

                    parts.append(
                        " | ".join(row_text)
                    )

        return "\n".join(parts).strip()

    except Exception as e:

        print(
            "DOCX extraction error:",
            repr(e)
        )

        return ""

    finally:

        if (
            temp_path
            and
            os.path.exists(temp_path)
        ):

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# OFFICE -> PDF
# ============================================================

def convert_office_to_pdf(
    file_bytes,
    extension
):

    office_binary = (
        shutil.which("soffice")
        or
        shutil.which("libreoffice")
    )

    if not office_binary:
        return None

    temp_dir = tempfile.mkdtemp(
        prefix="office_convert_"
    )

    source_path = os.path.join(
        temp_dir,
        f"source{extension}"
    )

    try:

        with open(
            source_path,
            "wb"
        ) as file:

            file.write(
                file_bytes
            )

        subprocess.run(

            [
                office_binary,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temp_dir,
                source_path,
            ],

            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,

            timeout=120,

            check=True,
        )

        output_path = os.path.join(
            temp_dir,
            "source.pdf"
        )

        if not os.path.exists(
            output_path
        ):

            pdf_files = list(
                Path(temp_dir).glob(
                    "*.pdf"
                )
            )

            if not pdf_files:
                return None

            output_path = str(
                pdf_files[0]
            )

        with open(
            output_path,
            "rb"
        ) as file:

            return file.read()

    except Exception as e:

        print(
            "Office conversion error:",
            repr(e)
        )

        return None

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# GENERIC PDF PAGES
# ============================================================

def extract_pages_from_pdf_bytes(
    file_bytes
):

    pages = []

    try:

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        for index, page in enumerate(document):

            pages.append({

                "page_number":
                index + 1,

                "text":
                page.get_text(
                    "text"
                ).strip(),
            })

        document.close()

    except Exception as e:

        print(
            "Converted PDF extraction error:",
            repr(e)
        )

    return pages


# ============================================================
# VIDEO
# ============================================================

def save_video_to_temp(
    file_bytes,
    extension
):

    path = os.path.join(
        tempfile.gettempdir(),
        f"video_{uuid.uuid4().hex}{extension}"
    )

    with open(
        path,
        "wb"
    ) as file:

        file.write(
            file_bytes
        )

    return path


def extract_video_frame(
    video_path,
    timestamp
):

    output_path = os.path.join(
        tempfile.gettempdir(),
        f"frame_{uuid.uuid4().hex}.jpg"
    )

    command = [

        "ffmpeg",
        "-y",

        "-ss",
        str(max(0, timestamp)),

        "-i",
        video_path,

        "-frames:v",
        "1",

        "-q:v",
        "2",

        output_path,
    ]

    try:

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=True,
        )

        if not os.path.exists(
            output_path
        ):

            return None

        with open(
            output_path,
            "rb"
        ) as file:

            data = file.read()

        return data

    except Exception as e:

        print(
            "Video frame extraction error:",
            repr(e)
        )

        return None

    finally:

        if os.path.exists(
            output_path
        ):

            try:
                os.remove(output_path)
            except Exception:
                pass


def format_timestamp(seconds):

    seconds = max(
        0,
        float(seconds)
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# RELEVANCE
# ============================================================

STOP_WORDS = {

    "the", "a", "an", "is", "are",
    "was", "were", "am", "be",
    "to", "of", "in", "on", "for",
    "and", "or", "but", "with",
    "this", "that", "it", "as",
    "at", "by", "from",

    "ami", "amar", "amake",
    "tumi", "tomar", "tomake",
    "apni", "apnar",
    "ki", "kothay", "kivabe",
    "ache", "achi", "acho",
    "kor", "koro", "korbo",
    "bolo", "dao",

    "main", "mera", "mujhe",
    "aap", "aapko", "tum",
    "kya", "kaise", "kahan",
    "hai", "hain", "hoon",
    "kar", "karo", "batao",
}


def relevance_tokens(text):

    words = normalize_words(
        text
    )

    return [

        word

        for word in words

        if len(word) >= 2
        and word not in STOP_WORDS
    ]


def calculate_relevance(
    question,
    content
):

    if not question or not content:
        return 0

    question_tokens = set(
        relevance_tokens(question)
    )

    content_tokens = set(
        relevance_tokens(content)
    )

    overlap = question_tokens.intersection(
        content_tokens
    )

    score = len(overlap) * 2

    normalized_question = " ".join(
        normalize_words(question)
    )

    normalized_content = " ".join(
        normalize_words(content)
    )

    if (
        normalized_question
        and
        len(normalized_question) > 5
        and
        normalized_question
        in normalized_content
    ):

        score += 5

    return score


def find_relevant_pdf_pages(
    question,
    pages,
    max_pages=2
):

    if not pages:
        return []

    scored = []

    for page in pages:

        score = calculate_relevance(
            question,
            page.get("text", "")
        )

        scored.append(
            (
                score,
                page
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    positive = [
        page
        for score, page in scored
        if score > 0
    ]

    if positive:
        return positive[:max_pages]

    return [
        page
        for _, page
        in scored[:max_pages]
    ]


def find_relevant_video_segments(
    question,
    segments,
    max_segments=2
):

    if not segments:
        return []

    scored = []

    for segment in segments:

        score = calculate_relevance(
            question,
            segment.get(
                "text",
                ""
            )
        )

        scored.append(
            (
                score,
                segment
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    positive = [
        segment
        for score, segment in scored
        if score > 0
    ]

    if positive:
        return positive[:max_segments]

    return [
        segment
        for _, segment
        in scored[:max_segments]
    ]


# ============================================================
# WEBSITE HELPERS
# ============================================================

AUTH_KEYWORDS = {

    "login",
    "log in",
    "signin",
    "sign in",
    "authenticate",
    "authentication",
    "account",
    "username",
    "email",
    "password",
}


SECURITY_CHALLENGE_KEYWORDS = {

    "captcha",
    "recaptcha",
    "verification code",
    "verify code",
    "one-time password",
    "otp",
    "two-factor",
    "2fa",
    "authenticator",
    "security code",
    "passkey",
}


def normalize_url(
    url,
    base_url=None
):

    if not url:
        return None

    url = str(url).strip()

    if base_url:
        url = urljoin(
            base_url,
            url
        )

    url, _ = urldefrag(
        url
    )

    if not re.match(
        r"^https?://",
        url,
        re.IGNORECASE
    ):

        url = "https://" + url

    parsed = urlparse(
        url
    )

    if not parsed.netloc:
        return None

    parsed = parsed._replace(
        path=parsed.path or "/",
        fragment=""
    )

    return parsed.geturl()


def same_domain(
    url1,
    url2
):

    try:

        host1 = (
            urlparse(url1)
            .netloc
            .lower()
            .split(":")[0]
            .removeprefix("www.")
        )

        host2 = (
            urlparse(url2)
            .netloc
            .lower()
            .split(":")[0]
            .removeprefix("www.")
        )

        return host1 == host2

    except Exception:
        return False


def is_crawlable_url(url):

    if not url:
        return False

    lowered = url.lower()

    blocked_prefixes = (
        "mailto:",
        "tel:",
        "javascript:",
        "data:",
        "whatsapp:",
    )

    if lowered.startswith(
        blocked_prefixes
    ):

        return False

    blocked_extensions = (

        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",

        ".mp3",
        ".wav",
        ".mp4",
        ".mov",
        ".avi",

        ".zip",
        ".rar",
        ".7z",

        ".exe",
        ".dmg",

        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    )

    path = urlparse(
        lowered
    ).path

    return not path.endswith(
        blocked_extensions
    )


def clean_website_text(text):

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def _visible_locator(
    page,
    selectors
):

    for selector in selectors:

        try:

            locator = page.locator(
                selector
            )

            count = min(
                locator.count(),
                10
            )

            for index in range(count):

                element = locator.nth(
                    index
                )

                try:

                    if (
                        element.is_visible()
                        and
                        element.is_enabled()
                    ):

                        return element

                except Exception:
                    continue

        except Exception:
            continue

    return None


def detect_security_challenge(
    page
):

    try:

        body = page.locator(
            "body"
        ).inner_text(
            timeout=3000
        ).lower()

    except Exception:

        return None

    for keyword in SECURITY_CHALLENGE_KEYWORDS:

        if keyword in body:
            return keyword

    return None


# ============================================================
# INSPECT WEBSITE ACCESS
# ============================================================

def inspect_website_access(url):

    normalized = normalize_url(
        url
    )

    result = {

        "requires_login": False,

        "login_url":
        normalized,

        "final_url":
        normalized,

        "status_code":
        None,

        "error":
        None,
    }

    if not normalized:

        result["error"] = (
            "Invalid website URL."
        )

        return result

    # --------------------------------------------------------
    # Without Playwright
    # --------------------------------------------------------

    if not PLAYWRIGHT_AVAILABLE:

        try:

            response = requests.get(
                normalized,
                timeout=20,
                headers={
                    "User-Agent":
                    "Mozilla/5.0"
                },
                allow_redirects=True,
            )

            result["status_code"] = (
                response.status_code
            )

            result["final_url"] = (
                response.url
            )

            if response.status_code in {
                401,
                403,
            }:

                result[
                    "requires_login"
                ] = True

            return result

        except Exception as e:

            result["error"] = repr(e)

            return result

    # --------------------------------------------------------
    # Playwright
    # --------------------------------------------------------

    try:

        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )

            context = browser.new_context(
                viewport={
                    "width": 1440,
                    "height": 1000,
                }
            )

            page = context.new_page()

            response = page.goto(

                normalized,

                wait_until="domcontentloaded",

                timeout=WEBSITE_TIMEOUT
            )

            try:

                page.wait_for_load_state(
                    "networkidle",
                    timeout=10000
                )

            except Exception:
                pass

            result["final_url"] = page.url

            if response:

                result["status_code"] = (
                    response.status
                )

            current = page.url.lower()

            body = ""

            try:

                body = page.locator(
                    "body"
                ).inner_text(
                    timeout=3000
                ).lower()

            except Exception:
                pass

            login_path = any(
                path in current
                for path in (
                    "/login",
                    "/signin",
                    "/sign-in",
                    "/authenticate",
                )
            )

            has_password = (
                page.locator(
                    "input[type='password']"
                ).count() > 0
            )

            has_login_keyword = any(
                keyword in body
                for keyword in (
                    "sign in",
                    "log in",
                    "login",
                    "password",
                )
            )

            if (
                login_path
                or
                (
                    has_password
                    and
                    has_login_keyword
                )
                or
                result["status_code"]
                in {401, 403}
            ):

                result[
                    "requires_login"
                ] = True

                result[
                    "login_url"
                ] = page.url

            context.close()
            browser.close()

    except Exception as e:

        result["error"] = repr(e)

    return result


# ============================================================
# WEBSITE LOGIN
# ============================================================

def login_to_website(
    login_url,
    login_id,
    login_password
):

    if not PLAYWRIGHT_AVAILABLE:

        return {

            "success": False,

            "error":
            "Playwright is not installed.",
        }

    if not login_id or not login_password:

        return {

            "success": False,

            "error":
            "Login ID and password are required.",
        }

    browser = None
    context = None

    try:

        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )

            context = browser.new_context(

                viewport={
                    "width": 1440,
                    "height": 1000,
                },

                user_agent=(
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/138.0 Safari/537.36"
                ),
            )

            page = context.new_page()

            page.goto(
                login_url,
                wait_until="domcontentloaded",
                timeout=WEBSITE_TIMEOUT
            )

            try:

                page.wait_for_load_state(
                    "networkidle",
                    timeout=10000
                )

            except Exception:
                pass

            username_selectors = [

                "input[type='email']",
                "input[name*='email' i]",
                "input[id*='email' i]",
                "input[name*='username' i]",
                "input[id*='username' i]",
                "input[name*='user' i]",
                "input[id*='user' i]",
                "input[type='text']",
            ]

            password_selectors = [

                "input[type='password']",
                "input[name*='password' i]",
                "input[id*='password' i]",
            ]

            username = _visible_locator(
                page,
                username_selectors
            )

            password = _visible_locator(
                page,
                password_selectors
            )

            if not username:

                return {

                    "success": False,

                    "error":
                    "Could not locate the login ID field.",
                }

            if not password:

                return {

                    "success": False,

                    "error":
                    "Could not locate the password field.",
                }

            username.fill(
                login_id
            )

            password.fill(
                login_password
            )

            submit_selectors = [

                "button[type='submit']",
                "input[type='submit']",
                "button:has-text('Login')",
                "button:has-text('Log in')",
                "button:has-text('Sign in')",
                "button:has-text('Signin')",
            ]

            submit = _visible_locator(
                page,
                submit_selectors
            )

            if submit:

                submit.click()

            else:

                password.press(
                    "Enter"
                )

            try:

                page.wait_for_load_state(
                    "networkidle",
                    timeout=15000
                )

            except Exception:
                pass

            page.wait_for_timeout(
                1500
            )

            challenge = detect_security_challenge(
                page
            )

            if challenge:

                return {

                    "success": False,

                    "security_challenge":
                    challenge,

                    "error":
                    (
                        "Website requires "
                        "security verification."
                    ),
                }

            current_url = page.url

            current_lower = (
                current_url.lower()
            )

            login_still_visible = (

                page.locator(
                    "input[type='password']"
                ).count() > 0
                and
                any(
                    path in current_lower
                    for path in (
                        "/login",
                        "/signin",
                        "/sign-in",
                    )
                )
            )

            if login_still_visible:

                return {

                    "success": False,

                    "error":
                    "Login failed or the login page is still active.",
                }

            storage_state = (
                context.storage_state()
            )

            final_url = page.url

            return {

                "success": True,

                "storage_state":
                storage_state,

                "final_url":
                final_url,
            }

    except Exception as e:

        return {

            "success": False,

            "error":
            repr(e),
        }

    finally:

        try:

            if context:
                context.close()

        except Exception:
            pass

        try:

            if browser:
                browser.close()

        except Exception:
            pass


# ============================================================
# WEBSITE HTML PAGE
# ============================================================

def _website_page_from_html(
    url,
    response
):

    try:

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe",
                "template",
            ]
        ):

            tag.decompose()

        title = ""

        if soup.title:

            title = clean_website_text(
                soup.title.get_text(
                    " ",
                    strip=True
                )
            )

        elements = []

        for element in soup.select(
            "h1,h2,h3,h4,h5,h6,"
            "p,li,article,section,"
            "td,th,button,a"
        ):

            text = clean_website_text(
                element.get_text(
                    " ",
                    strip=True
                )
            )

            if text:

                elements.append({

                    "tag":
                    element.name,

                    "text":
                    text[:1500],
                })

        links = []

        for anchor in soup.select(
            "a[href]"
        ):

            href = anchor.get(
                "href"
            )

            candidate = normalize_url(
                href,
                url
            )

            if (
                candidate
                and
                same_domain(
                    candidate,
                    url
                )
                and
                is_crawlable_url(
                    candidate
                )
            ):

                links.append(
                    candidate
                )

        text = clean_website_text(
            soup.get_text(
                " ",
                strip=True
            )
        )

        return {

            "url":
            url,

            "title":
            title,

            "text":
            text[:50000],

            "elements":
            elements[:500],

            "links":
            list(
                dict.fromkeys(
                    links
                )
            )[:100],

            "requires_login":
            False,
        }

    except Exception as e:

        print(
            "Website HTML parsing error:",
            repr(e)
        )

        return None


# ============================================================
# WEBSITE FETCH
# ============================================================

def fetch_website(url):

    start_url = normalize_url(
        url
    )

    if not start_url:

        raise RuntimeError(
            "Invalid website URL."
        )

    session = requests.Session()

    session.headers.update({

        "User-Agent":
        (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/138.0 Safari/537.36"
        )
    })

    queue = [
        start_url
    ]

    visited = set()
    pages = []

    while (
        queue
        and
        len(pages) < MAX_WEBSITE_PAGES
    ):

        current_url = queue.pop(0)

        if current_url in visited:
            continue

        visited.add(
            current_url
        )

        try:

            response = session.get(

                current_url,

                timeout=20,

                allow_redirects=True
            )

            final_url = normalize_url(
                response.url
            )

            if (
                final_url
                and
                same_domain(
                    final_url,
                    start_url
                )
            ):

                current_url = final_url

            if response.status_code in {
                401,
                403,
            }:

                pages.append({

                    "url":
                    current_url,

                    "title":
                    "Login Required",

                    "text":
                    (
                        "This page requires login "
                        "or authorization."
                    ),

                    "elements":
                    [],

                    "links":
                    [],

                    "requires_login":
                    True,
                })

                continue

            response.raise_for_status()

            content_type = (
                response.headers.get(
                    "content-type",
                    ""
                ).lower()
            )

            if (
                "text/html"
                not in content_type
                and
                "application/xhtml"
                not in content_type
            ):

                continue

            page_data = _website_page_from_html(
                current_url,
                response
            )

            if not page_data:
                continue

            pages.append(
                page_data
            )

            for link in page_data.get(
                "links",
                []
            ):

                if (
                    link not in visited
                    and
                    link not in queue
                    and
                    same_domain(
                        link,
                        start_url
                    )
                ):

                    queue.append(
                        link
                    )

        except Exception as e:

            print(
                "Website fetch error:",
                current_url,
                repr(e)
            )

            continue

    combined_text = "\n\n".join(

        f"PAGE: {page['title']}\n"
        f"URL: {page['url']}\n"
        f"{page['text']}"

        for page in pages
    )

    return {

        "url":
        start_url,

        "title":
        (
            pages[0]["title"]
            if pages
            else ""
        ),

        "text":
        combined_text[:120000],

        "pages":
        pages,

        "authenticated":
        False,
    }


# ============================================================
# AUTHENTICATED WEBSITE CRAWL
# ============================================================

def crawl_authenticated_website(
    start_url,
    storage_state,
    max_pages=MAX_WEBSITE_PAGES
):

    if not PLAYWRIGHT_AVAILABLE:

        raise RuntimeError(
            "Playwright is required for authenticated websites."
        )

    start_url = normalize_url(
        start_url
    )

    if not start_url:

        raise RuntimeError(
            "Invalid website URL."
        )

    queue = [
        start_url
    ]

    visited = set()
    pages = []

    final_storage_state = storage_state

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        context = browser.new_context(

            storage_state=storage_state,

            viewport={
                "width": 1440,
                "height": 1000,
            },

            user_agent=(
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/138.0 Safari/537.36"
            ),
        )

        page = context.new_page()

        while (
            queue
            and
            len(pages) < max_pages
        ):

            current_url = queue.pop(0)

            if current_url in visited:
                continue

            visited.add(
                current_url
            )

            try:

                page.goto(

                    current_url,

                    wait_until="domcontentloaded",

                    timeout=WEBSITE_TIMEOUT
                )

                try:

                    page.wait_for_load_state(
                        "networkidle",
                        timeout=8000
                    )

                except Exception:
                    pass

                page.wait_for_timeout(
                    700
                )

                current_url = normalize_url(
                    page.url
                ) or current_url

                if not same_domain(
                    current_url,
                    start_url
                ):

                    continue

                title = clean_website_text(
                    page.title()
                )

                try:

                    body_text = clean_website_text(
                        page.locator(
                            "body"
                        ).inner_text(
                            timeout=3000
                        )
                    )

                except Exception:

                    body_text = ""

                elements = []

                locator = page.locator(

                    "h1,h2,h3,h4,h5,h6,"
                    "p,li,article,section,"
                    "td,th,button,a"
                )

                count = min(
                    locator.count(),
                    1500
                )

                for i in range(count):

                    try:

                        element = locator.nth(
                            i
                        )

                        if not element.is_visible():
                            continue

                        text = clean_website_text(

                            element.inner_text(
                                timeout=500
                            )
                        )

                        if not text:
                            continue

                        tag = element.evaluate(
                            "el => el.tagName.toLowerCase()"
                        )

                        elements.append({

                            "tag":
                            tag,

                            "text":
                            text[:1500],
                        })

                    except Exception:
                        continue

                links = []

                anchors = page.locator(
                    "a[href]"
                )

                for i in range(
                    min(
                        anchors.count(),
                        1000
                    )
                ):

                    try:

                        href = anchors.nth(
                            i
                        ).get_attribute(
                            "href"
                        )

                        candidate = normalize_url(
                            href,
                            current_url
                        )

                        if (
                            candidate
                            and
                            same_domain(
                                candidate,
                                start_url
                            )
                            and
                            is_crawlable_url(
                                candidate
                            )
                        ):

                            links.append(
                                candidate
                            )

                    except Exception:
                        continue

                links = list(
                    dict.fromkeys(
                        links
                    )
                )

                pages.append({

                    "url":
                    current_url,

                    "title":
                    title,

                    "text":
                    body_text[:50000],

                    "elements":
                    elements[:500],

                    "links":
                    links[:100],

                    "requires_login":
                    False,
                })

                for link in links:

                    if (
                        link not in visited
                        and
                        link not in queue
                    ):

                        queue.append(
                            link
                        )

            except Exception as e:

                print(
                    "Authenticated crawl error:",
                    repr(e)
                )

                continue

        try:

            final_storage_state = (
                context.storage_state()
            )

        except Exception:
            pass

        context.close()
        browser.close()

    combined_text = "\n\n".join(

        f"PAGE: {page['title']}\n"
        f"URL: {page['url']}\n"
        f"{page['text']}"

        for page in pages
    )

    return {

        "url":
        start_url,

        "title":
        (
            pages[0]["title"]
            if pages
            else ""
        ),

        "text":
        combined_text[:120000],

        "pages":
        pages,

        "authenticated":
        True,

        "storage_state":
        final_storage_state,
    }


# ============================================================
# FIND RELEVANT WEBSITE PAGE
# ============================================================

def find_relevant_website_page(
    question,
    website_data
):

    pages = (

        website_data.get(
            "pages",
            []
        )

        if website_data

        else []
    )

    if not pages:
        return None

    scored = []

    question_tokens = relevance_tokens(
        question
    )

    question_normalized = " ".join(
        normalize_words(
            question
        )
    )

    for page in pages:

        title = page.get(
            "title",
            ""
        )

        url = page.get(
            "url",
            ""
        )

        text = page.get(
            "text",
            ""
        )

        elements_text = " ".join(

            item.get(
                "text",
                ""
            )

            for item in page.get(
                "elements",
                []
            )
        )

        content = (
            f"{title} "
            f"{url} "
            f"{text} "
            f"{elements_text}"
        )

        score = calculate_relevance(
            question,
            content
        )

        title_score = calculate_relevance(
            question,
            title
        )

        score += (
            title_score * 4
        )

        url_lower = url.lower()

        for token in question_tokens:

            if len(token) >= 4:

                if token in url_lower:
                    score += 5

        normalized_content = " ".join(
            normalize_words(
                content
            )
        )

        if (
            len(question_normalized) > 8
            and
            question_normalized
            in normalized_content
        ):

            score += 20

        scored.append(
            (
                score,
                page
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored[0][1]


# ============================================================
# EXACT WEBSITE SCREENSHOT
# ============================================================

def capture_website_sections(
    url,
    storage_state=None,
    question=None
):

    if not PLAYWRIGHT_AVAILABLE:
        return []

    if not question:
        return []

    results = []

    try:

        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )

            browser_args = {

                "viewport": {
                    "width": 1440,
                    "height": 1000,
                },

                "device_scale_factor": 1,

                "user_agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/138.0 Safari/537.36"
                ),
            }

            if storage_state:
                browser_args[
                    "storage_state"
                ] = storage_state

            context = browser.new_context(
                **browser_args
            )

            page = context.new_page()

            page.goto(

                url,

                wait_until="domcontentloaded",

                timeout=WEBSITE_TIMEOUT
            )

            try:

                page.wait_for_load_state(
                    "networkidle",
                    timeout=10000
                )

            except Exception:
                pass

            page.wait_for_timeout(
                1200
            )

            current = page.url.lower()

            if any(

                x in current

                for x in (
                    "/login",
                    "/signin",
                    "/sign-in",
                    "/authenticate",
                )
            ):

                context.close()
                browser.close()

                return []

            # ------------------------------------------------
            # Trigger lazy content
            # ------------------------------------------------

            for _ in range(10):

                try:

                    page.mouse.wheel(
                        0,
                        1200
                    )

                    page.wait_for_timeout(
                        250
                    )

                except Exception:

                    break

            page.evaluate(
                "window.scrollTo(0, 0)"
            )

            page.wait_for_timeout(
                500
            )

            selector = (

                "h1,h2,h3,h4,h5,h6,"
                "p,li,td,th,"
                "article,section,"
                "button,a,label,"
                "[role='heading'],"
                "[role='button']"
            )

            locator = page.locator(
                selector
            )

            count = min(
                locator.count(),
                3000
            )

            candidates = []

            q_normalized = " ".join(
                normalize_words(
                    question
                )
            )

            for i in range(count):

                try:

                    element = locator.nth(
                        i
                    )

                    if not element.is_visible():
                        continue

                    text = clean_website_text(

                        element.inner_text(
                            timeout=1000
                        )
                    )

                    if not text:
                        continue

                    if len(text) > 6000:
                        continue

                    box = element.bounding_box()

                    if not box:
                        continue

                    if box["width"] < 20:
                        continue

                    if box["height"] < 10:
                        continue

                    score = calculate_relevance(
                        question,
                        text
                    )

                    if score <= 0:
                        continue

                    tag = element.evaluate(
                        "el => el.tagName.toLowerCase()"
                    )

                    if tag.startswith("h"):
                        score += 8

                    t_normalized = " ".join(
                        normalize_words(
                            text
                        )
                    )

                    if (
                        q_normalized
                        and
                        len(q_normalized) > 6
                        and
                        q_normalized
                        in t_normalized
                    ):

                        score += 15

                    area = (
                        box["width"]
                        *
                        box["height"]
                    )

                    if area > (
                        1440 * 1000 * 0.8
                    ):

                        score -= 20

                    elif area > (
                        1440 * 1000 * 0.5
                    ):

                        score -= 10

                    try:

                        metadata = element.evaluate(

                            """
                            el => ({
                                id: el.id || "",
                                className:
                                    typeof el.className === "string"
                                    ? el.className
                                    : "",
                                aria:
                                    el.getAttribute("aria-label") || "",
                                title:
                                    el.getAttribute("title") || "",
                                href:
                                    el.getAttribute("href") || ""
                            })
                            """
                        )

                    except Exception:

                        metadata = {}

                    attribute_text = " ".join([

                        metadata.get(
                            "id",
                            ""
                        ),

                        metadata.get(
                            "className",
                            ""
                        ),

                        metadata.get(
                            "aria",
                            ""
                        ),

                        metadata.get(
                            "title",
                            ""
                        ),

                        metadata.get(
                            "href",
                            ""
                        ),
                    ])

                    score += calculate_relevance(
                        question,
                        attribute_text
                    )

                    candidates.append({

                        "score":
                        score,

                        "element":
                        element,

                        "text":
                        text,

                        "tag":
                        tag,

                        "box":
                        box,
                    })

                except Exception:
                    continue

            candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            # ------------------------------------------------
            # Remove nested duplicates
            # ------------------------------------------------

            selected = []

            for candidate in candidates:

                box = candidate[
                    "box"
                ]

                duplicate = False

                for old in selected:

                    old_box = old[
                        "box"
                    ]

                    if (

                        box["x"] >= old_box["x"]

                        and

                        box["y"] >= old_box["y"]

                        and

                        box["x"]
                        +
                        box["width"]
                        <=
                        old_box["x"]
                        +
                        old_box["width"]

                        and

                        box["y"]
                        +
                        box["height"]
                        <=
                        old_box["y"]
                        +
                        old_box["height"]
                    ):

                        duplicate = True

                        break

                if not duplicate:

                    selected.append(
                        candidate
                    )

                if len(selected) >= 3:
                    break

            # ------------------------------------------------
            # Capture actual screenshots
            # ------------------------------------------------

            for candidate in selected:

                try:

                    element = candidate[
                        "element"
                    ]

                    element.scroll_into_view_if_needed(
                        timeout=5000
                    )

                    page.wait_for_timeout(
                        300
                    )

                    box = element.bounding_box()

                    if not box:
                        continue

                    padding = 25

                    viewport_width = 1440
                    viewport_height = 1000

                    x = max(
                        0,
                        box["x"] - padding
                    )

                    y = max(
                        0,
                        box["y"] - padding
                    )

                    width = min(

                        viewport_width - x,

                        box["width"]
                        +
                        padding * 2
                    )

                    height = min(

                        viewport_height - y,

                        box["height"]
                        +
                        padding * 2
                    )

                    clip = {

                        "x":
                        x,

                        "y":
                        y,

                        "width":
                        max(
                            20,
                            width
                        ),

                        "height":
                        max(
                            20,
                            height
                        ),
                    }

                    # Temporary outline
                    try:

                        element.evaluate(

                            """
                            el => {
                                el.dataset.aiOriginalOutline =
                                    el.style.outline;

                                el.style.outline =
                                    '3px solid red';

                                el.style.outlineOffset =
                                    '2px';
                            }
                            """
                        )

                    except Exception:
                        pass

                    image = page.screenshot(

                        type="png",

                        clip=clip,

                        animations="disabled",
                    )

                    try:

                        element.evaluate(

                            """
                            el => {
                                el.style.outline =
                                    el.dataset.aiOriginalOutline || '';

                                delete el.dataset.aiOriginalOutline;
                            }
                            """
                        )

                    except Exception:
                        pass

                    results.append({

                        "text":
                        candidate[
                            "text"
                        ],

                        "image":
                        image,

                        "mime_type":
                        "image/png",

                        "url":
                        page.url,

                        "tag":
                        candidate[
                            "tag"
                        ],

                        "score":
                        candidate[
                            "score"
                        ],

                        "exact_position":
                        candidate[
                            "box"
                        ],
                    })

                except Exception as e:

                    print(
                        "Website target capture error:",
                        repr(e)
                    )

            context.close()
            browser.close()

    except Exception as e:

        print(
            "Website screenshot error:",
            repr(e)
        )

    return results


# ============================================================
# EXACT WEBSITE SECTION
# ============================================================

def capture_exact_website_section(
    page_url,
    question,
    storage_state=None
):

    sections = capture_website_sections(

        page_url,

        storage_state=storage_state,

        question=question
    )

    if not sections:

        inspection = inspect_website_access(
            page_url
        )

        if inspection.get(
            "requires_login"
        ):

            return {

                "success":
                False,

                "requires_login":
                True,

                "login_url":
                inspection.get(
                    "login_url",
                    page_url
                ),
            }

        return {

            "success":
            False,

            "requires_login":
            False,

            "error":
            (
                "Could not locate the exact "
                "website position."
            ),
        }

    section = sections[0]

    return {

        "success":
        True,

        "requires_login":
        False,

        "image":
        section["image"],

        "url":
        section.get(
            "url",
            page_url
        ),

        "section_text":
        section.get(
            "text",
            ""
        ),

        "tag":
        section.get(
            "tag",
            ""
        ),

        "score":
        section.get(
            "score",
            0
        ),

        "exact_position":
        section.get(
            "exact_position"
        ),
    }


# ============================================================
# WEBSITE QUESTION SOURCE
# ============================================================

def get_website_question_source(
    question,
    source
):

    website_data = (

        source.get(
            "website_data"
        )

        if source

        else None
    )

    page = find_relevant_website_page(
        question,
        website_data
    )

    if not page:
        return [], ""

    state = (

        source.get(
            "storage_state"
        )

        or
        website_data.get(
            "storage_state"
        )
    )

    shot = capture_exact_website_section(

        page.get(
            "url"
        ),

        question,

        state
    )

    if shot.get(
        "requires_login"
    ):

        return [

            {

                "type":
                "website_login_required",

                "login_url":
                shot.get(
                    "login_url",
                    page.get(
                        "url"
                    )
                ),
            }

        ], ""

    if not shot.get(
        "success"
    ):

        return [], (

            f"Relevant website page:\n"
            f"{page.get('url')}\n\n"

            f"Page title:\n"
            f"{page.get('title')}\n\n"

            f"Content:\n"
            f"{page.get('text', '')[:15000]}"
        )

    media = {

        "type":
        "image",

        "data":
        shot["image"],

        "caption":
        "🌐 Exact website position",

        "mime_type":
        "image/png",

        "url":
        shot["url"],

        "section_text":
        shot["section_text"],

        "tag":
        shot.get(
            "tag",
            ""
        ),

        "exact_position":
        shot.get(
            "exact_position"
        ),

        "score":
        shot.get(
            "score",
            0
        ),
    }

    context = (

        f"Relevant website page:\n"
        f"{shot['url']}\n\n"

        f"Exact selected website element:\n"
        f"{shot['section_text']}\n\n"

        f"DOM element type:\n"
        f"{shot.get('tag', '')}"
    )

    return [
        media
    ], context


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

def process_uploaded_file(
    uploaded_file
):

    filename = uploaded_file.name

    extension = Path(
        filename
    ).suffix.lower()

    file_bytes = uploaded_file.getvalue()

    file_hash = get_file_hash(
        file_bytes
    )

    result = {

        "source_type":
        "unknown",

        "filename":
        filename,

        "extension":
        extension,

        "file_hash":
        file_hash,

        "file_bytes":
        file_bytes,

        "uploaded_context":
        "",

        "image_part":
        None,

        "file_part":
        None,

        "pdf_pages":
        [],

        "render_pdf_bytes":
        None,

        "video_path":
        None,

        "video_segments":
        [],

        "video_language":
        "en",
    }

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == ".pdf":

        result["source_type"] = "pdf"

        pdf_pages = extract_pdf_data(
            file_bytes
        )

        result["pdf_pages"] = pdf_pages

        result["uploaded_context"] = (
            "\n\n".join(
                page["text"]
                for page in pdf_pages
                if page["text"]
            )
        )

        result["file_part"] = (
            types.Part.from_bytes(
                data=file_bytes,
                mime_type="application/pdf"
            )
        )

        return result

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    if extension == ".docx":

        result["source_type"] = "docx"

        result["uploaded_context"] = (
            extract_text_from_docx(
                file_bytes
            )
        )

        result["file_part"] = (
            types.Part.from_bytes(

                data=file_bytes,

                mime_type=(
                    "application/"
                    "vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                )
            )
        )

        converted_pdf = (
            convert_office_to_pdf(
                file_bytes,
                ".docx"
            )
        )

        if converted_pdf:

            result["pdf_pages"] = (
                extract_pages_from_pdf_bytes(
                    converted_pdf
                )
            )

            result["render_pdf_bytes"] = (
                converted_pdf
            )

        return result

    # --------------------------------------------------------
    # PPTX
    # --------------------------------------------------------

    if extension == ".pptx":

        result["source_type"] = "pptx"

        try:

            from pptx import Presentation

            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".pptx"

            ) as tmp:

                tmp.write(
                    file_bytes
                )

                pptx_path = tmp.name

            try:

                presentation = Presentation(
                    pptx_path
                )

                slide_texts = []

                for index, slide in enumerate(
                    presentation.slides
                ):

                    parts = []

                    for shape in slide.shapes:

                        if (
                            hasattr(
                                shape,
                                "text"
                            )
                            and
                            shape.text.strip()
                        ):

                            parts.append(
                                shape.text.strip()
                            )

                    slide_texts.append({

                        "page_number":
                        index + 1,

                        "text":
                        "\n".join(
                            parts
                        ),
                    })

                result["pdf_pages"] = (
                    slide_texts
                )

                result["uploaded_context"] = (
                    "\n\n".join(
                        page["text"]
                        for page in slide_texts
                    )
                )

            finally:

                try:
                    os.remove(
                        pptx_path
                    )
                except Exception:
                    pass

        except Exception as e:

            print(
                "PPTX extraction error:",
                repr(e)
            )

        result["file_part"] = (
            types.Part.from_bytes(

                data=file_bytes,

                mime_type=(
                    "application/"
                    "vnd.openxmlformats-officedocument."
                    "presentationml.presentation"
                )
            )
        )

        converted_pdf = (
            convert_office_to_pdf(
                file_bytes,
                ".pptx"
            )
        )

        if converted_pdf:

            result["render_pdf_bytes"] = (
                converted_pdf
            )

            result["pdf_pages"] = (
                extract_pages_from_pdf_bytes(
                    converted_pdf
                )
            )

        return result

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if extension in {
        ".txt",
        ".md",
        ".csv",
    }:

        result["source_type"] = "text"

        result["uploaded_context"] = (
            file_bytes.decode(
                "utf-8",
                errors="ignore"
            ).strip()
        )

        return result

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_extensions = {

        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    if extension in image_extensions:

        result["source_type"] = "image"

        result["image_part"] = (
            types.Part.from_bytes(

                data=file_bytes,

                mime_type=image_extensions[
                    extension
                ]
            )
        )

        return result

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if extension in {

        ".mp4",
        ".mov",
        ".m4v",
        ".avi",
        ".mkv",
    }:

        result["source_type"] = "video"

        if not check_ffmpeg():

            raise RuntimeError(

                "FFmpeg is required for video upload. "
                "Please install FFmpeg and add it to PATH."
            )

        video_path = save_video_to_temp(

            file_bytes,
            extension
        )

        result["video_path"] = (
            video_path
        )

        wav_path = None

        try:

            wav_path = (
                convert_audio_to_wav(
                    video_path
                )
            )

            whisper_model = load_whisper()

            (
                language,
                transcript,
                segments,
            ) = transcribe_audio(

                whisper_model,

                wav_path,

                return_segments=True
            )

            result["video_language"] = (
                language
            )

            result["video_segments"] = (
                segments
            )

            result["uploaded_context"] = (
                "\n".join(

                    f"[{format_timestamp(item['start'])}] "
                    f"{item['text']}"

                    for item in segments
                )
            )

        finally:

            if (
                wav_path
                and
                os.path.exists(wav_path)
            ):

                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        return result

    return result


# ============================================================
# QUESTION SOURCE MEDIA
# ============================================================

def get_question_source_media(
    question
):

    source = (
        st.session_state.source_data
    )

    if not source:
        return [], ""

    source_type = source.get(
        "source_type"
    )

    media = []
    extra_context = ""

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if source_type == "pdf":

        pages = find_relevant_pdf_pages(

            question,

            source.get(
                "pdf_pages",
                []
            ),

            max_pages=2
        )

        for page in pages:

            image_bytes = (
                render_pdf_relevant_crop(

                    source["file_bytes"],

                    page["page_number"],

                    question
                )
            )

            if image_bytes:

                media.append({

                    "type":
                    "image",

                    "data":
                    image_bytes,

                    "caption":
                    (
                        "📄 Exact relevant "
                        "PDF portion — "
                        f"Page "
                        f"{page['page_number']}"
                    ),

                    "mime_type":
                    "image/png",

                    "page_number":
                    page["page_number"],
                })

        extra_context = "\n\n".join(

            page.get(
                "text",
                ""
            )

            for page in pages
        )

    # --------------------------------------------------------
    # DOCX / PPTX
    # --------------------------------------------------------

    elif source_type in {
        "docx",
        "pptx",
    }:

        pages = find_relevant_pdf_pages(

            question,

            source.get(
                "pdf_pages",
                []
            ),

            max_pages=2
        )

        render_pdf_bytes = source.get(
            "render_pdf_bytes"
        )

        if render_pdf_bytes:

            for page in pages:

                image_bytes = (
                    render_pdf_relevant_crop(

                        render_pdf_bytes,

                        page["page_number"],

                        question
                    )
                )

                if image_bytes:

                    label = (
                        "DOCX"
                        if source_type == "docx"
                        else "PPTX"
                    )

                    media.append({

                        "type":
                        "image",

                        "data":
                        image_bytes,

                        "caption":
                        (
                            "📄 Exact relevant "
                            f"{label} portion — "
                            f"Page/Slide "
                            f"{page['page_number']}"
                        ),

                        "mime_type":
                        "image/png",

                        "page_number":
                        page["page_number"],
                    })

        extra_context = "\n\n".join(

            page.get(
                "text",
                ""
            )

            for page in pages
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    elif source_type == "video":

        segments = (
            find_relevant_video_segments(

                question,

                source.get(
                    "video_segments",
                    []
                ),

                max_segments=2
            )
        )

        video_path = source.get(
            "video_path"
        )

        if (
            video_path
            and
            os.path.exists(video_path)
        ):

            for segment in segments:

                duration = (
                    segment["end"]
                    -
                    segment["start"]
                )

                timestamp = (
                    segment["start"]
                    +
                    max(
                        0.1,
                        duration * 0.45
                    )
                )

                frame = (
                    extract_video_frame(

                        video_path,

                        timestamp
                    )
                )

                if frame:

                    media.append({

                        "type":
                        "image",

                        "data":
                        frame,

                        "caption":
                        (
                            "🎬 Exact relevant "
                            "video frame at "
                            f"{format_timestamp(timestamp)}"
                        ),

                        "timestamp":
                        timestamp,

                        "video_text":
                        segment.get(
                            "text",
                            ""
                        ),

                        "mime_type":
                        "image/jpeg",
                    })

        extra_context = "\n".join(

            f"[{format_timestamp(s['start'])}] "
            f"{s.get('text', '')}"

            for s in segments
        )

    # --------------------------------------------------------
    # WEBSITE
    # --------------------------------------------------------

    elif source_type == "website":

        website_media, website_context = (
            get_website_question_source(

                question,

                source
            )
        )

        media.extend(
            website_media
        )

        extra_context = (
            website_context
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    elif source_type == "image":

        image_data = source.get(
            "file_bytes",
            b""
        )

        if image_data:

            image_extensions = {

                ".jpg":
                "image/jpeg",

                ".jpeg":
                "image/jpeg",

                ".png":
                "image/png",

                ".webp":
                "image/webp",

                ".gif":
                "image/gif",
            }

            mime = image_extensions.get(

                source.get(
                    "extension",
                    ".png"
                ).lower(),

                "image/png"
            )

            media.append({

                "type":
                "image",

                "data":
                image_data,

                "caption":
                "🖼️ Uploaded image",

                "mime_type":
                mime,

                "already_sent":
                True,
            })

    return media, extra_context


# ============================================================
# HISTORY
# ============================================================

def build_history():

    recent = (
        st.session_state.messages[-12:]
    )

    history = []

    for message in recent:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        if not content:
            continue

        if role == "user":

            history.append(
                f"User: {content}"
            )

        elif role == "assistant":

            history.append(
                f"Assistant: {content}"
            )

    return "\n".join(
        history
    )


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_model(
    model_name,
    prompt,
    image_part=None,
    file_part=None,
    extra_parts=None
):

    contents = []

    if image_part is not None:
        contents.append(
            image_part
        )

    if file_part is not None:
        contents.append(
            file_part
        )

    if extra_parts:
        contents.extend(
            extra_parts
        )

    contents.append(
        prompt
    )

    response = client.models.generate_content(

        model=model_name,

        contents=contents
    )

    if response is None:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:

        raise RuntimeError(
            "Gemini returned no text response."
        )

    return text.strip()


# ============================================================
# RETRYABLE GEMINI ERROR
# ============================================================

def is_retryable_gemini_error(
    error
):

    text = str(
        error
    ).lower()

    retry_words = [

        "429",
        "500",
        "502",
        "503",
        "504",

        "resource exhausted",
        "rate limit",
        "temporarily unavailable",
        "internal server error",
        "deadline exceeded",
        "timeout",
    ]

    return any(
        word in text
        for word in retry_words
    )


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(
    prompt,
    image_part=None,
    file_part=None,
    extra_parts=None
):

    models = []

    if GEMINI_MODEL:

        models.append(
            GEMINI_MODEL
        )

    for model in FALLBACK_MODELS:

        if model not in models:

            models.append(
                model
            )

    last_error = None

    for model_index, model_name in enumerate(
        models
    ):

        attempts = (
            3
            if model_index == 0
            else 2
        )

        for attempt in range(
            attempts
        ):

            try:

                return generate_with_model(

                    model_name=model_name,

                    prompt=prompt,

                    image_part=image_part,

                    file_part=file_part,

                    extra_parts=extra_parts,
                )

            except Exception as e:

                last_error = e

                print(
                    "Gemini error:",
                    model_name,
                    repr(e)
                )

                if not is_retryable_gemini_error(
                    e
                ):

                    break

                if attempt < attempts - 1:

                    time.sleep(
                        2 ** attempt
                    )

    raise RuntimeError(

        str(last_error)
        if last_error
        else
        "Gemini request failed."
    )


# ============================================================
# ASK AI
# ============================================================

def ask_ai(
    user_message,
    language_code,
    relevant_media=None,
    source_context=""
):

    if language_code not in LANGUAGE_NAMES:

        language_code = "en"

    # --------------------------------------------------------
    # Detect CURRENT question again
    # --------------------------------------------------------

    detected_question_language = (
        detect_text_language(
            user_message
        )
    )

    if (
        detected_question_language
        in LANGUAGE_NAMES
    ):

        language_code = (
            detected_question_language
        )

    language_name = LANGUAGE_NAMES.get(
        language_code,
        "English"
    )

    language_instruction = (
        get_language_instruction(
            language_code
        )
    )

    history = build_history()

    source = (
        st.session_state.source_data
    )

    uploaded_context = ""

    image_part = None
    file_part = None
    extra_parts = []

    source_instruction = ""

    # --------------------------------------------------------
    # Relevant visual media
    # --------------------------------------------------------

    if relevant_media:

        for media in relevant_media:

            if media.get(
                "already_sent"
            ):

                # Uploaded image is sent
                # through image_part below.
                continue

            image_data = media.get(
                "data"
            )

            if image_data:

                try:

                    extra_parts.append(

                        types.Part.from_bytes(

                            data=image_data,

                            mime_type=media.get(

                                "mime_type",

                                "image/png"
                            )
                        )
                    )

                except Exception as e:

                    print(
                        "Visual context error:",
                        repr(e)
                    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    if source:

        uploaded_context = source.get(
            "uploaded_context",
            ""
        )

        image_part = source.get(
            "image_part"
        )

        file_part = source.get(
            "file_part"
        )

        source_type = source.get(
            "source_type"
        )

        filename = source.get(
            "filename",
            ""
        )

        source_instruction = f"""

============================================================
CURRENT SOURCE
============================================================

Source type:
{source_type}

Source name:
{filename}

Use the uploaded source only when relevant
to the CURRENT USER QUESTION.

Do not invent information.

The source language MUST NOT determine
the answer language.

The CURRENT USER QUESTION determines
the answer language.
"""

        if uploaded_context:

            source_instruction += f"""

SOURCE TEXT / TRANSCRIPT:

{uploaded_context[:MAX_SOURCE_TEXT]}

============================================================
END SOURCE TEXT
============================================================
"""

    # --------------------------------------------------------
    # Exact source context
    # --------------------------------------------------------

    if source_context:

        source_instruction += f"""

============================================================
EXACTLY SELECTED SOURCE CONTEXT
============================================================

The following content was selected specifically
for the CURRENT USER QUESTION.

Use it as the highest-relevance source context.

{source_context[:20000]}

============================================================
END EXACTLY SELECTED SOURCE CONTEXT
============================================================
"""

    # --------------------------------------------------------
    # Source-specific rules
    # --------------------------------------------------------

    if source:

        source_type = source.get(
            "source_type"
        )

        if source_type == "video":

            source_instruction += """

VIDEO RULES:

The uploaded video has been transcribed
with timestamps.

Relevant video frames may be attached.

Use the actual transcript and actual
frames when answering.

Do not invent timestamps.

If useful, mention the relevant timestamp.
"""

        elif source_type == "website":

            source_instruction += """

WEBSITE RULES:

The user supplied a website URL.

Use the analyzed website content.

If an exact website screenshot is attached,
inspect the actual screenshot carefully.

The screenshot comes from the actual
rendered website DOM.

Do not generate or imagine a replacement
website screenshot.

Do not claim that a different website area
was inspected.

If the exact screenshot contains the answer,
prioritize it.

If necessary, use the analyzed website text
as supporting context.
"""

        elif source_type in {
            "pdf",
            "docx",
            "pptx",
        }:

            source_instruction += """

DOCUMENT RULES:

Use the uploaded document content.

If an exact relevant page/slide crop
is attached, inspect that visual carefully.

The screenshot is an actual rendering
of the original document.

Do not invent visual information.

If the exact crop contains the answer,
prioritize it.
"""

        elif source_type == "image":

            source_instruction += """

IMAGE RULES:

Inspect the uploaded image carefully.

Answer only using information supported
by the image and relevant user context.

Do not invent unsupported visual details.
"""

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a professional multilingual AI voice assistant.

{language_instruction}

============================================================
CURRENT USER QUESTION
============================================================

{user_message}

============================================================
LANGUAGE PRIORITY
============================================================

The CURRENT USER QUESTION determines the answer language.

Priority:

1. Current user question
2. Explicit language request
3. Conversation context
4. Uploaded source

The uploaded source language must NEVER override
the current user question language.

============================================================
DETECTED LANGUAGE
============================================================

Code:
{language_code}

Language:
{language_name}

============================================================
ABSOLUTE CURRENT-QUESTION LANGUAGE RULE
============================================================

Answer the CURRENT USER QUESTION in the SAME
language/style used by the user.

If the user writes Banglish, answer in natural Banglish.

If the user writes Bengali script, answer in Bengali script.

If the user writes Hinglish, answer in natural Hinglish.

If the user writes Hindi script, answer in Hindi script.

If the user writes English, answer in English.

Never automatically translate the answer into
the source language.

Do not change Banglish into Bengali script unless
the user explicitly asks for Bengali script.

Do not change Hinglish into Hindi script unless
the user explicitly asks for Hindi script.

============================================================
EXACT SOURCE VISUAL RULE
============================================================

If an exact source screenshot is attached:

1. Treat it as the actual source visual.
2. Inspect it carefully.
3. Answer using the exact visible content.
4. Do not claim that another area was inspected.
5. Do not invent details outside the visible source.
6. If the screenshot contains the answer, prioritize it.
7. If the screenshot does not contain enough information,
   use the source text/context.
8. Never generate or imagine a replacement screenshot.

For websites, the screenshot is captured from the
actual rendered website DOM position.

For PDFs, the screenshot is an actual crop from
the original PDF page.

For DOCX/PPTX, the screenshot is an actual crop
from the rendered document/slide.

For videos, the screenshot is an actual frame
extracted from the uploaded video.

============================================================
GENERAL BEHAVIOUR
============================================================

- Be accurate.
- Be helpful.
- Do not invent facts.
- Prefer source evidence when a source is available.
- If the source does not contain the answer,
  clearly say so.
- Do not mention internal implementation details
  unless the user asks.
- Answer directly.
- Keep the answer reasonably concise unless
  the user asks for detail.

============================================================
RECENT CONVERSATION
============================================================

{history}

============================================================
SOURCE INFORMATION
============================================================

{source_instruction}

============================================================
ABSOLUTE LANGUAGE LOCK
============================================================

Answer ONLY in the language/style of the CURRENT USER QUESTION.

Current language:
{language_name}

Never allow the uploaded source language
to override the answer language.

Return ONLY the actual answer.
"""

    return ask_gemini(

        prompt=prompt,

        image_part=image_part,

        file_part=file_part,

        extra_parts=extra_parts
    )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(
    text,
    language_code
):

    if not text:
        return None

    try:

        import edge_tts

    except Exception as e:

        print(
            "TTS import error:",
            repr(e)
        )

        return None

    voice = TTS_VOICES.get(

        language_code,

        TTS_VOICES["en"]
    )

    output_path = os.path.join(

        tempfile.gettempdir(),

        f"tts_{uuid.uuid4().hex}.mp3"
    )

    async def generate():

        communicator = edge_tts.Communicate(

            text,

            voice
        )

        await communicator.save(
            output_path
        )

    try:

        import asyncio

        asyncio.run(
            generate()
        )

        if (
            not os.path.exists(
                output_path
            )
            or
            os.path.getsize(
                output_path
            ) < 100
        ):

            return None

        return output_path

    except Exception as e:

        print(
            "TTS ERROR:",
            repr(e)
        )

        return None


# ============================================================
# SESSION DEFAULTS
# ============================================================

DEFAULTS = {

    "active_session_id": None,

    "messages_loaded": False,

    "messages": [],

    "pending_transcript": "",

    "pending_language": "en",

    "recorder_key": 0,

    "voice_ready": False,

    "source_data": None,

    "source_filename": "",

    "website_url_input": "",

    "website_login_required": False,

    "website_login_url": "",

    "website_original_url": "",

    "website_authenticated": False,

    "pending_website_question": "",
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# DATABASE SESSION INITIALIZATION
# ============================================================

def ensure_active_session():

    active_id = st.session_state.get("active_session_id")

    if active_id and db_get_session(active_id):
        if not st.session_state.get("messages_loaded", False):
            st.session_state.messages = db_load_messages(active_id)
            st.session_state.messages_loaded = True
        return active_id

    sessions = db_list_sessions(limit=1)

    if sessions:
        active_id = sessions[0]["id"]
    else:
        active_id = db_create_session("New Chat")

    st.session_state.active_session_id = active_id
    st.session_state.messages = db_load_messages(active_id)
    st.session_state.messages_loaded = True
    return active_id


ensure_active_session()


# ============================================================
# CLEAR SOURCE
# ============================================================

def clear_source():

    source = (
        st.session_state.source_data
    )

    if source:

        video_path = source.get(
            "video_path"
        )

        if (
            video_path
            and
            os.path.exists(video_path)
        ):

            try:

                os.remove(
                    video_path
                )

            except Exception:
                pass

    st.session_state.source_data = None

    st.session_state.source_filename = ""

    st.session_state.website_login_required = False

    st.session_state.website_login_url = ""

    st.session_state.website_original_url = ""

    st.session_state.website_authenticated = False

    st.session_state.pending_website_question = ""

    # Clear login fields if they exist
    for key in (
        "website_login_id",
        "website_login_password",
    ):

        if key in st.session_state:

            try:
                del st.session_state[key]
            except Exception:
                pass


# ============================================================
# NEW CHAT
# ============================================================

def start_new_chat():

    active_id = db_create_session("New Chat")

    st.session_state.active_session_id = active_id
    st.session_state.messages = []
    st.session_state.messages_loaded = True
    st.session_state.pending_transcript = ""
    st.session_state.pending_language = "en"
    st.session_state.recorder_key += 1
    st.session_state.voice_ready = False
    clear_source()


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

def process_user_message(
    user_message,
    language_code
):

    if not user_message:
        return

    user_message = (
        user_message.strip()
    )

    if not user_message:
        return

    detected = detect_text_language(
        user_message
    )

    if detected in LANGUAGE_NAMES:

        language_code = detected

    if language_code not in LANGUAGE_NAMES:

        language_code = "en"

    # --------------------------------------------------------
    # Find exact source media
    # --------------------------------------------------------

    relevant_media = []

    source_context = ""

    try:

        (
            relevant_media,
            source_context
        ) = get_question_source_media(

            user_message
        )

    except Exception as e:

        print(
            "SOURCE MEDIA ERROR:",
            repr(e)
        )

    # --------------------------------------------------------
    # Website login
    # --------------------------------------------------------

    login_items = [

        m

        for m in relevant_media

        if (
            isinstance(
                m,
                dict
            )
            and
            m.get(
                "type"
            )
            ==
            "website_login_required"
        )
    ]

    if login_items:

        st.session_state[
            "pending_website_question"
        ] = user_message

        st.session_state[
            "website_login_required"
        ] = True

        st.session_state[
            "website_login_url"
        ] = (

            login_items[0].get(
                "login_url"
            )

            or

            st.session_state.get(
                "website_url_input",
                ""
            )
        )

        st.warning(

            "🔐 This page requires login. "
            "Please enter your Login ID/Email/Username "
            "and Password above."
        )

        return

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role":
        "user",

        "content":
        user_message,

        "language":
        language_code,
    })

    # Persist the user message immediately.
    session_id = ensure_active_session()
    db_save_message(
        session_id,
        "user",
        user_message,
        language_code,
    )

    current_session = db_get_session(session_id)
    if current_session and current_session.get("title") == "New Chat":
        db_update_session_title(
            session_id,
            user_message[:80].strip() or "New Chat"
        )

    try:

        with st.spinner(
            "🤖 Thinking..."
        ):

            response = ask_ai(

                user_message=user_message,

                language_code=language_code,

                relevant_media=relevant_media,

                source_context=source_context,
            )

    except Exception as e:

        st.error(
            "❌ Gemini request failed."
        )

        print(
            "GEMINI ERROR:",
            repr(e)
        )

        if (

            st.session_state.messages

            and

            st.session_state.messages[-1].get(
                "role"
            )
            ==
            "user"
        ):

            st.session_state.messages.pop()

        st.exception(
            RuntimeError(
                str(e)
            )
        )

        return

    assistant_message = {

        "role":
        "assistant",

        "content":
        response,

        "tts_language":
        language_code,

        "source_media":
        relevant_media,
    }

    for media in relevant_media:

        if media.get(
            "timestamp"
        ) is not None:

            assistant_message.setdefault(
                "timestamps",
                []
            ).append(
                media[
                    "timestamp"
                ]
            )

    st.session_state.messages.append(
        assistant_message
    )

    # Persist answer + exact source visuals in SQL.
    db_save_message(
        session_id,
        "assistant",
        response,
        language_code,
        relevant_media,
    )

    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    with st.spinner(
        "🔊 Generating voice..."
    ):

        audio_file = text_to_speech(

            text=response,

            language_code=language_code
        )

    if audio_file:

        st.session_state.messages[
            -1
        ]["audio_file"] = (
            audio_file
        )

    else:

        st.warning(

            "⚠️ Text response generated, "
            "but voice generation failed. "
            "Check the terminal for TTS ERROR."
        )


# ============================================================
# RENDER MESSAGES
# ============================================================

def render_messages():

    if not st.session_state.messages:

        st.info(
            "👋 Ask me something or upload a source."
        )

        return

    for message in st.session_state.messages:

        role = message.get(
            "role"
        )

        content = message.get(
            "content",
            ""
        )

        if role == "user":

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    content
                )

        else:

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    content
                )

                source_media = message.get(
                    "source_media",
                    []
                )

                for media in source_media:

                    if media.get(
                        "type"
                    ) == "image":

                        image_data = media.get(
                            "data"
                        )

                        if image_data:

                            st.image(

                                image_data,

                                caption=media.get(
                                    "caption",
                                    "Source visual"
                                ),

                                use_container_width=True
                            )

                        if media.get(
                            "timestamp"
                        ) is not None:

                            st.caption(

                                "⏱️ Timestamp: "
                                +
                                format_timestamp(
                                    media[
                                        "timestamp"
                                    ]
                                )
                            )

                        if media.get(
                            "video_text"
                        ):

                            st.caption(

                                "🎬 Transcript: "
                                +
                                media[
                                    "video_text"
                                ]
                            )

                    elif media.get(
                        "type"
                    ) == "website_login_required":

                        st.warning(
                            "🔐 Website login is required."
                        )

                audio_file = message.get(
                    "audio_file"
                )

                if audio_file:

                    if os.path.exists(
                        audio_file
                    ):

                        try:

                            with open(
                                audio_file,
                                "rb"
                            ) as audio:

                                st.audio(
                                    audio.read(),
                                    format="audio/mp3"
                                )

                        except Exception as e:

                            print(
                                "Audio render error:",
                                repr(e)
                            )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    ensure_active_session()

    with st.sidebar:

        st.title("🎙️ AI Voice Assistant")

        st.caption(
            "Multilingual AI assistant with document, video, "
            "website understanding and SQL chat history."
        )

        st.divider()

        if st.button("🆕 New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.divider()
        st.subheader("📚 Chat History")

        sessions = db_list_sessions(limit=50)

        if not sessions:
            st.caption("No saved chats yet.")
        else:
            for item in sessions:
                session_id = item["id"]
                title = item.get("title") or "New Chat"
                label = title[:42] + ("…" if len(title) > 42 else "")

                c1, c2 = st.columns([5, 1])
                with c1:
                    if st.button(
                        ("🟢 " if session_id == st.session_state.active_session_id else "💬 ") + label,
                        key=f"open_chat_{session_id}",
                        use_container_width=True,
                    ):
                        st.session_state.active_session_id = session_id
                        st.session_state.messages = db_load_messages(session_id)
                        st.session_state.messages_loaded = True
                        clear_source()
                        st.rerun()
                with c2:
                    if st.button("🗑️", key=f"delete_chat_{session_id}"):
                        db_delete_session(session_id)
                        if session_id == st.session_state.active_session_id:
                            remaining = db_list_sessions(limit=1)
                            if remaining:
                                st.session_state.active_session_id = remaining[0]["id"]
                                st.session_state.messages = db_load_messages(remaining[0]["id"])
                            else:
                                new_id = db_create_session("New Chat")
                                st.session_state.active_session_id = new_id
                                st.session_state.messages = []
                            st.session_state.messages_loaded = True
                        st.rerun()

        st.divider()

        if st.button("🧹 Delete All Chat History", use_container_width=True):
            db_clear_all_history()
            new_id = db_create_session("New Chat")
            st.session_state.active_session_id = new_id
            st.session_state.messages = []
            st.session_state.messages_loaded = True
            st.rerun()

        if st.button("🗑️ Clear Uploaded Source", use_container_width=True):
            clear_source()
            st.rerun()

        st.divider()
        st.subheader("📌 Current Source")

        source = st.session_state.source_data
        if source:
            source_type = source.get("source_type", "unknown")
            filename = source.get("filename", "")
            st.success(f"📎 {source_type.upper()}")
            st.caption(filename)
        else:
            st.caption("No source uploaded.")

        st.divider()
        st.subheader("🌍 Supported Languages")
        st.write(", ".join(LANGUAGE_NAMES.values()))

        st.divider()
        st.subheader("⚙️ System Status")
        st.write("Whisper:", "✅" if WHISPER_MODEL_NAME else "❌")
        st.write("Gemini:", "✅" if GEMINI_API_KEY else "❌")
        st.write("Playwright:", "✅" if PLAYWRIGHT_AVAILABLE else "❌")
        st.write("FFmpeg:", "✅" if check_ffmpeg() else "❌")
        st.write("SQL Database:", "✅")


# ============================================================
# MAIN
# ============================================================

def main():

    render_sidebar()

    st.title(
        "🎙️ Multilingual AI Voice Assistant"
    )

    st.caption(

        "Ask questions by voice or text. "
        "Upload documents/videos/images or analyze "
        "a website and get source-grounded answers."
    )

    # ========================================================
    # WEBSITE ANALYSIS
    # ========================================================

    st.header(
        "🌐 Website Analysis"
    )

    website_url = st.text_input(

        "Enter website URL",

        value=st.session_state.get(
            "website_url_input",
            ""
        ),

        placeholder=(
            "https://example.com"
        )
    )

    st.session_state[
        "website_url_input"
    ] = website_url

    if st.button(
        "🔍 Analyze Website",
        use_container_width=False
    ):

        if not website_url.strip():

            st.warning(
                "Please enter a website URL."
            )

        else:

            with st.spinner(
                "🌐 Inspecting website..."
            ):

                inspection = (
                    inspect_website_access(
                        website_url
                    )
                )

            if inspection.get(
                "requires_login"
            ):

                st.session_state[
                    "website_login_required"
                ] = True

                st.session_state[
                    "website_login_url"
                ] = (

                    inspection.get(
                        "login_url"
                    )

                    or

                    website_url
                )

                st.session_state[
                    "website_original_url"
                ] = website_url

                st.warning(
                    "🔐 This website requires login."
                )

            else:

                with st.spinner(
                    "🌐 Crawling website..."
                ):

                    try:

                        website_data = (
                            fetch_website(
                                website_url
                            )
                        )

                        st.session_state[
                            "source_data"
                        ] = {

                            "source_type":
                            "website",

                            "filename":
                            normalize_url(
                                website_url
                            ),

                            "extension":
                            "",

                            "file_hash":
                            hashlib.sha256(
                                website_url.encode()
                            ).hexdigest(),

                            "file_bytes":
                            b"",

                            "uploaded_context":
                            website_data.get(
                                "text",
                                ""
                            ),

                            "image_part":
                            None,

                            "file_part":
                            None,

                            "pdf_pages":
                            [],

                            "render_pdf_bytes":
                            None,

                            "video_path":
                            None,

                            "video_segments":
                            [],

                            "video_language":
                            "en",

                            "website_data":
                            website_data,

                            "storage_state":
                            None,

                            "authenticated":
                            False,
                        }

                        st.session_state[
                            "source_filename"
                        ] = website_url

                        db_save_source(
                            ensure_active_session(),
                            st.session_state["source_data"]
                        )

                        st.session_state[
                            "website_authenticated"
                        ] = False

                        st.success(
                            "✅ Website analyzed successfully."
                        )

                        st.info(

                            f"Found "
                            f"{len(website_data.get('pages', []))} "
                            f"accessible pages."
                        )

                    except Exception as e:

                        st.error(
                            "❌ Website analysis failed."
                        )

                        st.exception(e)

    # ========================================================
    # WEBSITE LOGIN
    # ========================================================

    if st.session_state.get(
        "website_login_required"
    ):

        st.subheader(
            "🔐 Website Login"
        )

        login_url = st.session_state.get(
            "website_login_url",
            website_url
        )

        st.caption(
            f"Login page: {login_url}"
        )

        login_id = st.text_input(

            "Login ID / Email / Username",

            key="website_login_id"
        )

        login_password = st.text_input(

            "Password",

            type="password",

            key="website_login_password"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔓 Login & Analyze",
                use_container_width=True
            ):

                with st.spinner(
                    "🔐 Logging in..."
                ):

                    login_result = (
                        login_to_website(

                            login_url,

                            login_id,

                            login_password
                        )
                    )

                if login_result.get(
                    "success"
                ):

                    st.success(
                        "✅ Login successful."
                    )

                    original_url = (
                        st.session_state.get(
                            "website_original_url",
                            website_url
                        )
                    )

                    with st.spinner(
                        "🌐 Crawling authenticated website..."
                    ):

                        try:

                            website_data = (
                                crawl_authenticated_website(

                                    original_url,

                                    login_result[
                                        "storage_state"
                                    ],

                                    max_pages=MAX_WEBSITE_PAGES
                                )
                            )

                            storage_state = (
                                website_data.get(
                                    "storage_state"
                                )
                            )

                            st.session_state[
                                "source_data"
                            ] = {

                                "source_type":
                                "website",

                                "filename":
                                original_url,

                                "extension":
                                "",

                                "file_hash":
                                hashlib.sha256(
                                    original_url.encode()
                                ).hexdigest(),

                                "file_bytes":
                                b"",

                                "uploaded_context":
                                website_data.get(
                                    "text",
                                    ""
                                ),

                                "image_part":
                                None,

                                "file_part":
                                None,

                                "pdf_pages":
                                [],

                                "render_pdf_bytes":
                                None,

                                "video_path":
                                None,

                                "video_segments":
                                [],

                                "video_language":
                                "en",

                                "website_data":
                                website_data,

                                "storage_state":
                                storage_state,

                                "authenticated":
                                True,
                            }

                            st.session_state[
                                "source_filename"
                            ] = original_url

                            db_save_source(
                                ensure_active_session(),
                                st.session_state["source_data"]
                            )

                            st.session_state[
                                "website_authenticated"
                            ] = True

                            st.session_state[
                                "website_login_required"
                            ] = False

                            st.success(
                                "✅ Authenticated website analyzed."
                            )

                            # Do not automatically submit
                            # credentials anywhere else.

                        except Exception as e:

                            st.error(
                                "❌ Authenticated crawl failed."
                            )

                            st.exception(e)

                else:

                    if login_result.get(
                        "security_challenge"
                    ):

                        st.warning(

                            "⚠️ This website requires "
                            "CAPTCHA/OTP/2FA/passkey verification. "
                            "Automatic login cannot bypass "
                            "security verification."
                        )

                    else:

                        st.error(

                            login_result.get(
                                "error",
                                "Login failed."
                            )
                        )

        with col2:

            if st.button(
                "❌ Cancel Login",
                use_container_width=True
            ):

                st.session_state[
                    "website_login_required"
                ] = False

                st.session_state[
                    "pending_website_question"
                ] = ""

                st.rerun()

    # ========================================================
    # AUTHENTICATED STATUS
    # ========================================================

    if st.session_state.get(
        "website_authenticated"
    ):

        st.success(
            "🔓 Authenticated website source is active."
        )

        if st.button(
            "🚪 Logout Website"
        ):

            clear_source()

            st.rerun()

    # ========================================================
    # WEBSITE CONTENT
    # ========================================================

    source = (
        st.session_state.source_data
    )

    if (
        source
        and
        source.get(
            "source_type"
        )
        ==
        "website"
    ):

        website_data = source.get(
            "website_data",
            {}
        )

        with st.expander(
            "🌐 Analyzed Website Content",
            expanded=False
        ):

            st.write(
                "Pages:",
                len(
                    website_data.get(
                        "pages",
                        []
                    )
                )
            )

            for page in website_data.get(
                "pages",
                []
            )[:20]:

                st.markdown(
                    f"**{page.get('title', '')}**"
                )

                st.caption(
                    page.get(
                        "url",
                        ""
                    )
                )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.header(
        "📎 Upload Source"
    )

    uploaded_file = st.file_uploader(

        "Upload PDF, DOCX, PPTX, TXT, CSV, image or video",

        type=[

            "pdf",
            "docx",
            "pptx",

            "txt",
            "md",
            "csv",

            "jpg",
            "jpeg",
            "png",
            "webp",
            "gif",

            "mp4",
            "mov",
            "m4v",
            "avi",
            "mkv",
        ]
    )

    if uploaded_file:

        uploaded_hash = get_file_hash(
            uploaded_file.getvalue()
        )

        existing_hash = None

        if st.session_state.source_data:

            existing_hash = (
                st.session_state.source_data.get(
                    "file_hash"
                )
            )

        if uploaded_hash != existing_hash:

            try:

                with st.spinner(
                    "📚 Processing uploaded source..."
                ):

                    processed = (
                        process_uploaded_file(
                            uploaded_file
                        )
                    )

                st.session_state[
                    "source_data"
                ] = processed

                st.session_state[
                    "source_filename"
                ] = uploaded_file.name

                db_save_source(
                    ensure_active_session(),
                    processed
                )

                st.success(

                    f"✅ {uploaded_file.name} "
                    f"processed successfully."
                )

            except Exception as e:

                st.error(
                    "❌ File processing failed."
                )

                st.exception(e)

    # ========================================================
    # DISPLAY CURRENT SOURCE
    # ========================================================

    source = (
        st.session_state.source_data
    )

    if source:

        source_type = source.get(
            "source_type"
        )

        st.subheader(
            "📌 Current Source"
        )

        st.write(
            f"**Type:** {source_type}"
        )

        st.write(
            f"**Name:** "
            f"{source.get('filename', '')}"
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if source_type == "image":

            image_data = source.get(
                "file_bytes"
            )

            if image_data:

                st.image(

                    image_data,

                    caption="🖼️ Uploaded image",

                    use_container_width=True
                )

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        if source_type == "video":

            video_path = source.get(
                "video_path"
            )

            if (
                video_path
                and
                os.path.exists(video_path)
            ):

                try:

                    with open(
                        video_path,
                        "rb"
                    ) as video:

                        st.video(
                            video.read()
                        )

                except Exception as e:

                    print(
                        "Video preview error:",
                        repr(e)
                    )

            st.caption(

                "Detected language: "
                +
                LANGUAGE_NAMES.get(

                    source.get(
                        "video_language",
                        "en"
                    ),

                    "English"
                )
            )

            with st.expander(
                "🎬 Video Transcript"
            ):

                st.text(
                    source.get(
                        "uploaded_context",
                        ""
                    )
                )

        # ----------------------------------------------------
        # EXTRACTED TEXT
        # ----------------------------------------------------

        uploaded_context = source.get(
            "uploaded_context",
            ""
        )

        if uploaded_context:

            with st.expander(
                "📄 Extracted Source Text"
            ):

                st.text(
                    uploaded_context[
                        :20000
                    ]
                )

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    render_messages()

    # ========================================================
    # VOICE INPUT
    # ========================================================

    st.header(
        "🎙️ Voice Input"
    )

    audio_value = st.audio_input(

        "Record your question",

        key=(
            f"voice_recorder_"
            f"{st.session_state.recorder_key}"
        )
    )

    if audio_value:

        if st.button(
            "📝 Transcribe Voice",
            use_container_width=False
        ):

            wav_path = None
            input_path = None

            try:

                with st.spinner(
                    "🎧 Transcribing..."
                ):

                    audio_bytes = (
                        audio_value.getvalue()
                    )

                    input_path = os.path.join(

                        tempfile.gettempdir(),

                        f"voice_"
                        f"{uuid.uuid4().hex}.webm"
                    )

                    with open(
                        input_path,
                        "wb"
                    ) as audio_file:

                        audio_file.write(
                            audio_bytes
                        )

                    wav_path = (
                        convert_audio_to_wav(
                            input_path
                        )
                    )

                    model = load_whisper()

                    (
                        language,
                        transcript,
                        segments,
                    ) = transcribe_audio(

                        model,

                        wav_path,

                        return_segments=True
                    )

                    st.session_state[
                        "pending_transcript"
                    ] = transcript

                    st.session_state[
                        "pending_language"
                    ] = language

                    st.session_state[
                        "voice_ready"
                    ] = True

                    st.success(
                        "✅ Voice transcribed."
                    )

            except Exception as e:

                st.error(
                    "❌ Voice transcription failed."
                )

                st.exception(e)

            finally:

                if (
                    wav_path
                    and
                    os.path.exists(wav_path)
                ):

                    try:
                        os.remove(
                            wav_path
                        )
                    except Exception:
                        pass

                if (
                    input_path
                    and
                    os.path.exists(input_path)
                ):

                    try:
                        os.remove(
                            input_path
                        )
                    except Exception:
                        pass

    # ========================================================
    # EDITED VOICE QUESTION
    # ========================================================

    if st.session_state.get(
        "voice_ready"
    ):

        st.subheader(
            "📝 Voice Question"
        )

        final_question = st.text_area(

            "You can edit the transcription before sending.",

            value=st.session_state.get(
                "pending_transcript",
                ""
            ),

            height=100,

            key="editable_voice_question"
        )

        if st.button(
            "🚀 Send Voice Question",
            use_container_width=True
        ):

            final_question = (
                final_question.strip()
            )

            if not final_question:

                st.warning(
                    "Please enter a question."
                )

            else:

                # IMPORTANT:
                # Edited question decides language.

                final_language = (
                    detect_text_language(
                        final_question
                    )
                )

                if final_language not in LANGUAGE_NAMES:

                    final_language = "en"

                process_user_message(

                    user_message=final_question,

                    language_code=final_language
                )

                st.session_state[
                    "pending_transcript"
                ] = ""

                st.session_state[
                    "voice_ready"
                ] = False

                st.session_state[
                    "recorder_key"
                ] += 1

                st.rerun()

    # ========================================================
    # PENDING WEBSITE QUESTION
    # ========================================================

    pending_website_question = (
        st.session_state.get(
            "pending_website_question",
            ""
        )
    )

    if (
        pending_website_question
        and
        not st.session_state.get(
            "website_login_required"
        )
    ):

        st.info(
            "🔄 Processing your previous website question..."
        )

        question = (
            pending_website_question
        )

        st.session_state[
            "pending_website_question"
        ] = ""

        language = detect_text_language(
            question
        )

        process_user_message(

            question,

            language
        )

        st.rerun()

    # ========================================================
    # TYPED CHAT
    # ========================================================

    user_text = st.chat_input(
        "Type your message here..."
    )

    if user_text:

        language_code = (
            detect_text_language(
                user_text
            )
        )

        print("=" * 70)
        print("TYPED MESSAGE")
        print("=" * 70)
        print("Text:", user_text)
        print("Language:", language_code)
        print("=" * 70)

        process_user_message(

            user_message=user_text,

            language_code=language_code
        )

        st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()