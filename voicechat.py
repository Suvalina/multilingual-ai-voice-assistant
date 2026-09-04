import os
import time
import uuid
import tempfile
import asyncio
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel
import edge_tts

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pypdf import PdfReader
from docx import Document


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
    "gemini-3.5-flash-lite"
).strip()

FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
)


# ============================================================
# API KEY CHECK
# ============================================================

if not GEMINI_API_KEY:

    st.error(
        "❌ GEMINI_API_KEY is missing. "
        "Please add it to Streamlit Secrets or your .env file."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

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
    "as": "as-IN-PriyomNeural",
    "mr": "mr-IN-AarohiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "pa": "pa-IN-OjasNeural",
    "ur": "ur-PK-AsadNeural",

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
# SESSION STATE
# ============================================================

DEFAULTS = {
    "messages": [],
    "conversation_titles": [],
    "pending_transcript": "",
    "pending_language": "en",
    "recorder_key": 0,
    "uploaded_context": "",
    "uploaded_filename": "",
    "image_part": None,
    "file_part": None,
}

for key, value in DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# TEXT LANGUAGE DETECTION
# ============================================================

def detect_text_language(text):
    """
    Detect language from:

    1. Unicode script
    2. Romanized Bengali
    3. Common Romanized Indian words

    Examples:

    আমি কেমন আছি       -> bn
    ami kemon achi     -> bn
    ami bhalo achi     -> bn
    tumi ki korcho     -> bn
    amar naam Suval   -> bn
    How are you       -> en
    आप कैसे हैं       -> hi
    """

    if not text:
        return "en"

    text = text.strip()

    if not text:
        return "en"

    # ========================================================
    # 1. UNICODE SCRIPT DETECTION
    # ========================================================

    chars = [
        ch
        for ch in text
        if ch.isalpha()
    ]

    if chars:

        script_counts = {}

        for ch in chars:

            code = ord(ch)
            detected = None

            # Bengali
            if 0x0980 <= code <= 0x09FF:
                detected = "bn"

            # Devanagari
            elif 0x0900 <= code <= 0x097F:
                detected = "hi"

            # Gujarati
            elif 0x0A80 <= code <= 0x0AFF:
                detected = "gu"

            # Punjabi / Gurmukhi
            elif 0x0A00 <= code <= 0x0A7F:
                detected = "pa"

            # Odia
            elif 0x0B00 <= code <= 0x0B7F:
                detected = "or"

            # Tamil
            elif 0x0B80 <= code <= 0x0BFF:
                detected = "ta"

            # Telugu
            elif 0x0C00 <= code <= 0x0C7F:
                detected = "te"

            # Kannada
            elif 0x0C80 <= code <= 0x0CFF:
                detected = "kn"

            # Malayalam
            elif 0x0D00 <= code <= 0x0D7F:
                detected = "ml"

            # Thai
            elif 0x0E00 <= code <= 0x0E7F:
                detected = "th"

            # Hebrew
            elif 0x0590 <= code <= 0x05FF:
                detected = "he"

            # Arabic / Urdu
            elif 0x0600 <= code <= 0x06FF:
                detected = "ur"

            # Japanese
            elif (
                0x3040 <= code <= 0x309F
                or 0x30A0 <= code <= 0x30FF
            ):
                detected = "ja"

            # Korean
            elif 0xAC00 <= code <= 0xD7AF:
                detected = "ko"

            # Chinese
            elif 0x4E00 <= code <= 0x9FFF:
                detected = "zh"

            if detected:

                script_counts[detected] = (
                    script_counts.get(
                        detected,
                        0
                    ) + 1
                )

        if script_counts:

            return max(
                script_counts,
                key=script_counts.get
            )

    # ========================================================
    # 2. ROMANIZED BENGALI
    # ========================================================

    lower_text = text.lower()

    normalized_text = ""

    for ch in lower_text:

        if ch.isalnum() or ch.isspace():

            normalized_text += ch

        else:

            normalized_text += " "

    words = set(
        normalized_text.split()
    )

    roman_bengali_words = {
        # Pronouns
        "ami",
        "amar",
        "amake",
        "amra",
        "amader",

        "tumi",
        "tomar",
        "tomake",
        "tomra",
        "tomader",

        "apni",
        "apnar",
        "apnake",

        "se",
        "she",
        "tar",
        "take",

        # Questions
        "ki",
        "ke",
        "kake",
        "kemon",
        "keno",
        "kothay",
        "kotha",
        "kokhon",
        "kivabe",
        "kirokom",

        # Being
        "achi",
        "achho",
        "acho",
        "ache",
        "achen",

        "chilam",
        "chilo",
        "chhilo",

        # Doing
        "korchi",
        "korcho",
        "korchen",
        "korbo",
        "korbe",
        "kore",
        "koro",
        "korben",

        # Going
        "jacchi",
        "jachhi",
        "jachcho",
        "jaccho",
        "jabo",
        "jabe",

        # Coming
        "asche",
        "aschhe",
        "aschi",
        "ashchi",

        # Eating
        "khacchi",
        "khaccho",
        "khabo",
        "kheye",

        # Good / quantity
        "bhalo",
        "valo",
        "valobasha",
        "bhalobasha",
        "onek",
        "khub",
        "ektu",
        "sob",
        "shob",
        "kichu",
        "kono",

        # Location / time
        "ekhane",
        "okhane",
        "sekhane",
        "sekhaney",
        "aj",
        "aaj",
        "kal",
        "ekhon",

        # Name
        "naam",
        "nam",

        # Want / need
        "hobe",
        "hoy",
        "hoye",
        "hoyeche",
        "dorkar",
        "proyojon",
        "chai",
        "chao",
        "chaichi",

        # Commands
        "dao",
        "den",
        "dekh",
        "dekho",
        "bolo",
        "bol",
        "bolchi",
        "bolcho",

        # Knowledge
        "jante",
        "jani",
        "janina",

        # Ability
        "parbo",
        "pari",
        "parena",

        # Negative
        "na",
        "nei",
        "noy",

        # Yes
        "hya",
        "ha",
        "haan",

        # Common words
        "dhonnobad",
        "bari",
        "ghor",
        "bondhu",
        "ma",
        "baba",
        "dada",
        "didi",
        "bhai",
        "bon",

        # Study / work
        "porashona",
        "porchi",
        "porte",
        "kaj",
        "chakri",

        # Feelings
        "bhalo",
        "kharap",
        "sundor",
        "valo",
    }

    bengali_matches = len(
        words.intersection(
            roman_bengali_words
        )
    )

    # Strong Bengali words
    strong_bengali_words = {
        "ami",
        "amar",
        "amake",
        "tumi",
        "tomar",
        "tomake",
        "kemon",
        "achi",
        "achho",
        "acho",
        "korchi",
        "korcho",
        "bhalo",
        "valo",
        "kothay",
        "kivabe",
        "keno",
        "ki",
        "ekhane",
        "okhane",
        "jabo",
        "jacchi",
        "asche",
        "bolchi",
        "jani",
        "janina",
        "naam",
        "bari",
        "bondhu",
    }

    strong_matches = len(
        words.intersection(
            strong_bengali_words
        )
    )

    # Short Roman Bengali sentences
    if strong_matches >= 1:

        return "bn"

    if bengali_matches >= 2:

        return "bn"

    # ========================================================
    # 3. DEFAULT ENGLISH
    # ========================================================

    return "en"


# ============================================================
# WHISPER MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_whisper():

    model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )

    return model


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================

def transcribe_audio(
    model,
    audio_file
):

    try:

        segments, info = model.transcribe(
            audio_file,
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500
            },
        )

        segments = list(
            segments
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        whisper_language = (
            info.language
            if info.language
            else "en"
        )

        whisper_language = (
            whisper_language
            .strip()
            .lower()
        )

        if whisper_language == "bh":

            whisper_language = "hi"

        if whisper_language not in LANGUAGE_NAMES:

            whisper_language = "en"

        # ====================================================
        # SCRIPT DETECTION
        # ====================================================

        script_language = detect_text_language(
            text
        )

        # If transcript contains Indian/non-English
        # script, trust script detection.
        if script_language != "en":

            language_code = script_language

        else:

            language_code = whisper_language

        if language_code not in LANGUAGE_NAMES:

            language_code = "en"

        return language_code, text

    except Exception as e:

        raise RuntimeError(
            f"Whisper transcription failed: {repr(e)}"
        )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(
    file_bytes
):

    text_parts = []

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(file_bytes)

        tmp_path = tmp.name

    try:

        reader = PdfReader(
            tmp_path
        )

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text_parts.append(
                    page_text
                )

    finally:

        try:

            os.remove(
                tmp_path
            )

        except Exception:
            pass

    return "\n".join(
        text_parts
    ).strip()


# ============================================================
# DOCX TEXT EXTRACTION
# ============================================================

def extract_text_from_docx(
    file_bytes
):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".docx"
    ) as tmp:

        tmp.write(file_bytes)

        tmp_path = tmp.name

    try:

        document = Document(
            tmp_path
        )

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(
            paragraphs
        ).strip()

    finally:

        try:

            os.remove(
                tmp_path
            )

        except Exception:
            pass


# ============================================================
# FILE PROCESSING
# ============================================================

def process_uploaded_file(
    uploaded_file
):

    if uploaded_file is None:

        return "", None, None

    filename = uploaded_file.name

    extension = Path(
        filename
    ).suffix.lower()

    file_bytes = uploaded_file.getvalue()

    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        text = extract_text_from_pdf(
            file_bytes
        )

        file_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type="application/pdf"
        )

        return text, None, file_part

    # ========================================================
    # DOCX
    # ========================================================

    if extension == ".docx":

        text = extract_text_from_docx(
            file_bytes
        )

        file_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
        )

        return text, None, file_part

    # ========================================================
    # TXT / MD / CSV
    # ========================================================

    if extension in [
        ".txt",
        ".md",
        ".csv"
    ]:

        try:

            text = file_bytes.decode(
                "utf-8",
                errors="ignore"
            )

        except Exception:

            text = ""

        return text.strip(), None, None

    # ========================================================
    # IMAGES
    # ========================================================

    image_extensions = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    if extension in image_extensions:

        mime_type = image_extensions[
            extension
        ]

        image_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=mime_type
        )

        return "", image_part, None

    return "", None, None


# ============================================================
# LANGUAGE INSTRUCTION
# ============================================================

def get_language_instruction(
    language_code
):

    language_name = LANGUAGE_NAMES.get(
        language_code,
        "English"
    )

    instruction = f"""
IMPORTANT LANGUAGE RULE:

The user's language is {language_name}.

You MUST reply ONLY in {language_name}.

The response language MUST match the user's language.

If the user speaks or writes Bengali,
answer completely in Bengali.

If the user speaks or writes Romanized Bengali,
understand it as Bengali and answer completely in Bengali.

If the user speaks or writes Hindi,
answer completely in Hindi.

If the user speaks or writes English,
answer completely in English.

If the user speaks or writes Odia,
answer completely in Odia.

If the user speaks or writes Tamil,
answer completely in Tamil.

If the user speaks or writes Telugu,
answer completely in Telugu.

If the user speaks or writes Kannada,
answer completely in Kannada.

If the user speaks or writes Malayalam,
answer completely in Malayalam.

If the user speaks or writes Gujarati,
answer completely in Gujarati.

If the user speaks or writes Punjabi,
answer completely in Punjabi.

Do NOT translate the user's question into English
unless explicitly requested.

Do NOT switch to English unnecessarily.

Do NOT mix languages unnecessarily.

The final answer must remain in {language_name}.

Use natural conversational language suitable for voice output.
"""

    return instruction


# ============================================================
# BUILD CHAT HISTORY
# ============================================================

def build_history():

    recent_messages = (
        st.session_state.messages[-12:]
    )

    history = []

    for message in recent_messages:

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
# GEMINI SINGLE REQUEST
# ============================================================

def generate_with_model(
    model_name,
    prompt,
    image_part=None,
    file_part=None,
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

    contents.append(
        prompt
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
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

    error_text = str(
        error
    ).upper()

    retry_codes = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "INTERNAL",
        "BAD_GATEWAY",
        "DEADLINE",
        "TIMEOUT",
    ]

    return any(
        code in error_text
        for code in retry_codes
    )


# ============================================================
# GEMINI REQUEST WITH RETRY + FALLBACK
# ============================================================

def ask_gemini(
    prompt,
    image_part=None,
    file_part=None,
):

    models_to_try = [
        GEMINI_MODEL
    ]

    for fallback in FALLBACK_MODELS:

        if fallback not in models_to_try:

            models_to_try.append(
                fallback
            )

    last_error = None

    for model_index, model_name in enumerate(
        models_to_try
    ):

        max_attempts = (
            3
            if model_index == 0
            else 2
        )

        for attempt in range(
            max_attempts
        ):

            try:

                if (
                    model_index > 0
                    and attempt == 0
                ):

                    st.info(
                        f"🔄 {GEMINI_MODEL} is temporarily busy. "
                        f"Trying fallback model: {model_name}"
                    )

                result = generate_with_model(
                    model_name=model_name,
                    prompt=prompt,
                    image_part=image_part,
                    file_part=file_part,
                )

                if result:

                    return result

            except Exception as e:

                last_error = e

                print(
                    f"Gemini error | "
                    f"model={model_name} | "
                    f"attempt={attempt + 1} | "
                    f"error={repr(e)}"
                )

                if not is_retryable_gemini_error(
                    e
                ):

                    raise RuntimeError(
                        f"Gemini API error: {e}"
                    )

                if attempt < max_attempts - 1:

                    delay = 5 * (
                        2 ** attempt
                    )

                    delay = min(
                        delay,
                        20
                    )

                    time.sleep(
                        delay
                    )

    raise RuntimeError(
        "Gemini API is temporarily unavailable. "
        "The primary model and fallback models "
        "all failed.\n\n"
        f"Last error: {last_error}"
    )


# ============================================================
# ASK AI
# ============================================================

def ask_ai(
    user_message,
    language_code,
    image_part=None,
    file_part=None,
    uploaded_context="",
):

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

    attachment_instruction = ""

    # ========================================================
    # UPLOADED TEXT
    # ========================================================

    if uploaded_context:

        attachment_instruction = f"""
The user uploaded a file.

Here is extracted text from the file:

---------------- FILE CONTENT ----------------

{uploaded_context}

---------------- END FILE CONTENT ------------

Use this content when answering the user's question.

Do not ignore relevant information from the uploaded file.
"""

    # ========================================================
    # IMAGE
    # ========================================================

    if image_part is not None:

        attachment_instruction += """
An image has also been attached.

Analyze the image carefully and answer based on
the visible content.
"""

    # ========================================================
    # DOCUMENT
    # ========================================================

    if file_part is not None:

        attachment_instruction += """
A document has also been attached.

Use the document as supporting context when relevant.
"""

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are a helpful multilingual AI voice assistant.

{language_instruction}

GENERAL RULES:

1. Give accurate and useful answers.

2. Understand conversational questions.

3. Be polite and natural.

4. Do not mention internal API errors unless necessary.

5. Do not say that you are unable to understand the user
   unless the input is genuinely unclear.

6. If the user asks a technical question,
   explain clearly.

7. If the user asks for code,
   provide complete working code when appropriate.

8. Keep answers reasonably concise unless the user
   requests a detailed explanation.

9. Preserve the user's language exactly.

10. Do not unnecessarily change language.

11. The final response MUST be written in
    the detected user language.

12. Never switch to English just because the question
    contains some English words.

13. If the user uses Bengali script,
    respond in Bengali.

14. If the user uses Romanized Bengali,
    treat it as Bengali and respond in Bengali.

15. If the user uses Hindi / Devanagari script,
    respond in Hindi.

16. Use natural conversational language suitable
    for voice output.

17. Do not translate unless the user asks for translation.

18. Make the response easy to listen to using TTS.

RECENT CONVERSATION:

{history}

{attachment_instruction}

CURRENT USER MESSAGE:

{user_message}

FINAL INSTRUCTION:

Answer the user in {language_name}.

Do not change the response language.

If the user's input is Romanized Bengali,
the final answer must be Bengali.
"""

    return ask_gemini(
        prompt=prompt,
        image_part=image_part,
        file_part=file_part,
    )


# ============================================================
# TEXT TO SPEECH
# ============================================================

async def generate_tts(
    text,
    language_code,
    output_file,
):

    if language_code not in TTS_VOICES:

        language_code = "en"

    voice = TTS_VOICES.get(
        language_code,
        TTS_VOICES["en"]
    )

    print(
        f"TTS language: {language_code}"
    )

    print(
        f"TTS voice: {voice}"
    )

    communicate = edge_tts.Communicate(
        text,
        voice
    )

    await communicate.save(
        output_file
    )


def text_to_speech(
    text,
    language_code,
):

    if not text:

        return None

    if language_code not in TTS_VOICES:

        language_code = "en"

    output_file = os.path.join(
        tempfile.gettempdir(),
        f"tts_{uuid.uuid4().hex}.mp3"
    )

    try:

        asyncio.run(
            generate_tts(
                text,
                language_code,
                output_file,
            )
        )

        if os.path.exists(
            output_file
        ):

            return output_file

    except Exception as e:

        print(
            f"TTS error: {repr(e)}"
        )

    return None


# ============================================================
# NEW CHAT
# ============================================================

def start_new_chat():

    st.session_state.messages = []

    st.session_state.pending_transcript = ""

    st.session_state.pending_language = "en"

    st.session_state.uploaded_context = ""

    st.session_state.uploaded_filename = ""

    st.session_state.image_part = None

    st.session_state.file_part = None

    st.session_state.recorder_key += 1


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

def process_user_message(
    user_message,
    language_code,
):

    if not user_message:

        return

    user_message = user_message.strip()

    if not user_message:

        return

    if language_code not in LANGUAGE_NAMES:

        language_code = "en"

    language_name = LANGUAGE_NAMES.get(
        language_code,
        "English"
    )

    print(
        f"Processing message | "
        f"language={language_code} | "
        f"language_name={language_name}"
    )

    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # ========================================================
    # GEMINI
    # ========================================================

    try:

        with st.spinner(
            "🤖 Thinking..."
        ):

            response = ask_ai(
                user_message=user_message,
                language_code=language_code,
                image_part=(
                    st.session_state.image_part
                ),
                file_part=(
                    st.session_state.file_part
                ),
                uploaded_context=(
                    st.session_state.uploaded_context
                ),
            )

    except Exception as e:

        error_message = str(
            e
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Sorry, I could not process your request.\n\n"
                    f"Error: {error_message}"
                ),
            }
        )

        st.error(
            "❌ Gemini request failed. "
            "Please try again."
        )

        return

    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    # ========================================================
    # TTS
    # ========================================================

    with st.spinner(
        "🔊 Generating voice..."
    ):

        audio_file = text_to_speech(
            response,
            language_code,
        )

    if audio_file:

        st.session_state.messages[-1][
            "audio_file"
        ] = audio_file

        st.session_state.messages[-1][
            "tts_language"
        ] = language_code

    # ========================================================
    # CLEAR ATTACHMENT
    # ========================================================

    st.session_state.uploaded_context = ""

    st.session_state.uploaded_filename = ""

    st.session_state.image_part = None

    st.session_state.file_part = None


# ============================================================
# RENDER CHAT MESSAGES
# ============================================================

def render_messages():

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant"
        )

        content = message.get(
            "content",
            ""
        )

        with st.chat_message(
            role
        ):

            st.markdown(
                content
            )

            audio_file = message.get(
                "audio_file"
            )

            if (
                role == "assistant"
                and audio_file
                and os.path.exists(audio_file)
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

                except Exception:

                    pass


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.title(
            "🎙️ AI Voice Assistant"
        )

        st.caption(
            "Multilingual AI voice assistant"
        )

        st.divider()

        # ====================================================
        # NEW CHAT
        # ====================================================

        if st.button(
            "🆕 New Chat",
            use_container_width=True
        ):

            start_new_chat()

            st.rerun()

        st.divider()

        # ====================================================
        # SEARCH
        # ====================================================

        st.subheader(
            "🔎 Search Chat"
        )

        search_text = st.text_input(
            "Search messages",
            placeholder="Type to search..."
        )

        if search_text:

            search_text_lower = (
                search_text.lower()
            )

            found = False

            for message in st.session_state.messages:

                content = message.get(
                    "content",
                    ""
                )

                if (
                    search_text_lower
                    in content.lower()
                ):

                    found = True

                    st.write(
                        content[:200]
                    )

            if not found:

                st.caption(
                    "No matching message found."
                )

        st.divider()

        # ====================================================
        # LANGUAGES
        # ====================================================

        st.subheader(
            "🌍 Supported Languages"
        )

        language_count = len(
            LANGUAGE_NAMES
        )

        st.caption(
            f"{language_count} languages supported"
        )

        language_display = [
            f"{code.upper()} — {name}"
            for code, name
            in LANGUAGE_NAMES.items()
        ]

        with st.expander(
            "View languages"
        ):

            for language in language_display:

                st.write(
                    f"• {language}"
                )

        st.divider()

        # ====================================================
        # CLEAR HISTORY
        # ====================================================

        if st.button(
            "🗑️ Clear Chat History",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()

        st.divider()

        st.caption(
            f"Whisper model: {WHISPER_MODEL_NAME}"
        )

        st.caption(
            f"Gemini model: {GEMINI_MODEL}"
        )


# ============================================================
# MAIN APP
# ============================================================

def main():

    render_sidebar()

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "🎙️ Multilingual AI Voice Assistant"
    )

    st.markdown(
        "Speak naturally in your language and get an AI response."
    )

    st.divider()

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "📎 Upload a file or image",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "csv",
            "jpg",
            "jpeg",
            "png",
            "webp",
            "gif",
        ],
    )

    if uploaded_file is not None:

        if (
            st.session_state.uploaded_filename
            != uploaded_file.name
        ):

            try:

                with st.spinner(
                    "📄 Processing uploaded file..."
                ):

                    (
                        extracted_text,
                        image_part,
                        file_part,
                    ) = process_uploaded_file(
                        uploaded_file
                    )

                    st.session_state.uploaded_context = (
                        extracted_text
                    )

                    st.session_state.uploaded_filename = (
                        uploaded_file.name
                    )

                    st.session_state.image_part = (
                        image_part
                    )

                    st.session_state.file_part = (
                        file_part
                    )

                st.success(
                    f"✅ {uploaded_file.name} "
                    "uploaded successfully."
                )

            except Exception as e:

                st.error(
                    f"❌ File processing failed: {e}"
                )

        # ====================================================
        # IMAGE PREVIEW
        # ====================================================

        extension = Path(
            uploaded_file.name
        ).suffix.lower()

        if extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
        ]:

            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True,
            )

        # ====================================================
        # TEXT PREVIEW
        # ====================================================

        if st.session_state.uploaded_context:

            with st.expander(
                "📄 View extracted file text"
            ):

                preview_text = (
                    st.session_state.uploaded_context
                )

                if len(preview_text) > 5000:

                    preview_text = (
                        preview_text[:5000]
                        + "\n\n...[truncated]"
                    )

                st.text_area(
                    "Extracted content",
                    preview_text,
                    height=250,
                    disabled=True,
                )

    # ========================================================
    # EXISTING CHAT
    # ========================================================

    render_messages()

    # ========================================================
    # MICROPHONE
    # ========================================================

    st.subheader(
        "🎤 Voice Input"
    )

    recorder_key = (
        f"recorder_{st.session_state.recorder_key}"
    )

    audio_value = st.audio_input(
        "Click microphone and speak",
        sample_rate=16000,
        key=recorder_key,
    )

    if audio_value is not None:

        if st.button(
            "📝 Transcribe Voice",
            use_container_width=True
        ):

            try:

                # =================================================
                # LOAD WHISPER
                # =================================================

                whisper_spinner_text = (
                    f"⏳ Loading Whisper "
                    f"{WHISPER_MODEL_NAME} "
                    "(CPU INT8)..."
                )

                with st.spinner(
                    whisper_spinner_text
                ):

                    whisper_model = load_whisper()

                # =================================================
                # TEMP AUDIO
                # =================================================

                audio_path = os.path.join(
                    tempfile.gettempdir(),
                    f"voice_{uuid.uuid4().hex}.wav"
                )

                with open(
                    audio_path,
                    "wb"
                ) as audio_file:

                    audio_file.write(
                        audio_value.getvalue()
                    )

                # =================================================
                # TRANSCRIPTION
                # =================================================

                with st.spinner(
                    "🎧 Transcribing..."
                ):

                    (
                        language_code,
                        transcript,
                    ) = transcribe_audio(
                        whisper_model,
                        audio_path,
                    )

                # =================================================
                # DELETE TEMP AUDIO
                # =================================================

                try:

                    os.remove(
                        audio_path
                    )

                except Exception:

                    pass

                if not transcript:

                    st.warning(
                        "⚠️ No speech detected. "
                        "Please record again."
                    )

                else:

                    st.session_state.pending_transcript = (
                        transcript
                    )

                    st.session_state.pending_language = (
                        language_code
                    )

                    detected_name = (
                        LANGUAGE_NAMES.get(
                            language_code,
                            language_code
                        )
                    )

                    st.success(
                        f"Detected language: {detected_name}"
                    )

                    print(
                        f"Whisper/script detected language: "
                        f"{language_code}"
                    )

                    print(
                        f"Transcript: {transcript}"
                    )

            except Exception as e:

                st.error(
                    f"❌ Transcription failed: {e}"
                )

    # ========================================================
    # TRANSCRIPT
    # ========================================================

    if st.session_state.pending_transcript:

        st.subheader(
            "📝 Transcribed Text"
        )

        edited_transcript = st.text_area(
            "You can edit the transcript before sending:",
            value=st.session_state.pending_transcript,
            height=120,
        )

        detected_language = (
            st.session_state.pending_language
        )

        detected_name = (
            LANGUAGE_NAMES.get(
                detected_language,
                detected_language
            )
        )

        st.info(
            f"🌍 Response language: {detected_name}"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔄 Record Again",
                use_container_width=True
            ):

                st.session_state.pending_transcript = ""

                st.session_state.recorder_key += 1

                st.rerun()

        with col2:

            if st.button(
                "🚀 Send to AI",
                use_container_width=True
            ):

                final_text = (
                    edited_transcript.strip()
                )

                if final_text:

                    # =================================================
                    # RE-DETECT EDITED TEXT
                    # =================================================

                    edited_language = (
                        detect_text_language(
                            final_text
                        )
                    )

                    if edited_language != "en":

                        final_language = (
                            edited_language
                        )

                    else:

                        final_language = (
                            st.session_state.pending_language
                        )

                    final_language_name = (
                        LANGUAGE_NAMES.get(
                            final_language,
                            final_language
                        )
                    )

                    print(
                        f"Final voice language: "
                        f"{final_language}"
                    )

                    print(
                        f"Final voice language name: "
                        f"{final_language_name}"
                    )

                    process_user_message(
                        user_message=final_text,
                        language_code=final_language,
                    )

                    st.session_state.pending_transcript = ""

                    st.rerun()

    # ========================================================
    # NORMAL CHAT INPUT
    # ========================================================

    user_text = st.chat_input(
        "Type your message here..."
    )

    if user_text:

        # ====================================================
        # AUTOMATIC TYPED LANGUAGE DETECTION
        # ====================================================

        language_code = (
            detect_text_language(
                user_text
            )
        )

        typed_language_name = (
            LANGUAGE_NAMES.get(
                language_code,
                language_code
            )
        )

        print(
            f"Typed language detected: "
            f"{language_code}"
        )

        print(
            f"Typed language name: "
            f"{typed_language_name}"
        )

        process_user_message(
            user_message=user_text,
            language_code=language_code,
        )

        st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
