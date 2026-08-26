import os
import time
import uuid
import tempfile
import asyncio
from pathlib import Path

import streamlit as st
import whisper
import edge_tts

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pypdf import PdfReader
from docx import Document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL",
    "medium"
).strip()


# IMPORTANT:
# Your API key successfully tested with this model.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()


if not GEMINI_API_KEY:

    st.error(
        """
        ❌ GEMINI_API_KEY not found.

        Create a .env file:

        GEMINI_API_KEY=YOUR_API_KEY
        WHISPER_MODEL=medium
        GEMINI_MODEL=gemini-3.5-flash-lite
        """
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

try:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

except Exception as e:

    st.error(
        f"❌ Gemini initialization failed:\n\n{e}"
    )

    st.stop()


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
# PREFERRED LANGUAGES
# ============================================================

PREFERRED_LANGUAGES = {

    "en",
    "hi",
    "bn",
    "or",
    "ta",
    "te",
    "ne",
    "gu",
    "as",
}


# ============================================================
# TTS VOICES
# ============================================================

TTS_VOICES = {

    "en": "en-US-JennyNeural",

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

    "pa": "pa-IN-VaaniNeural",

    "ur": "ur-PK-AsadNeural",

    "fr": "fr-FR-DeniseNeural",

    "de": "de-DE-KatjaNeural",

    "es": "es-ES-ElviraNeural",

    "it": "it-IT-ElsaNeural",

    "pt": "pt-PT-RaquelNeural",

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

    "no": "nb-NO-PernilleNeural",

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

DEFAULT_STATE = {

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


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# WHISPER
# ============================================================

@st.cache_resource
def load_whisper():

    with st.spinner(
        f"⏳ Loading Whisper {WHISPER_MODEL_NAME}..."
    ):

        model = whisper.load_model(
            WHISPER_MODEL_NAME,
            download_root="asrmodel"
        )

    return model


# ============================================================
# PDF TEXT
# ============================================================

def extract_pdf_text(file_bytes):

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"pdf_{uuid.uuid4().hex}.pdf"
    )

    try:

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(file_bytes)


        reader = PdfReader(temp_path)

        pages = []

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:

                    pages.append(text)

            except Exception:

                pass


        return "\n\n".join(
            pages
        ).strip()


    except Exception:

        return ""


    finally:

        try:

            if os.path.exists(temp_path):

                os.remove(temp_path)

        except Exception:

            pass


# ============================================================
# DOCX TEXT
# ============================================================

def extract_docx_text(file_bytes):

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"docx_{uuid.uuid4().hex}.docx"
    )

    try:

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(file_bytes)


        document = Document(temp_path)

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                paragraphs.append(text)


        return "\n".join(
            paragraphs
        ).strip()


    except Exception:

        return ""


    finally:

        try:

            if os.path.exists(temp_path):

                os.remove(temp_path)

        except Exception:

            pass


# ============================================================
# ATTACHMENT PROCESSING
# ============================================================

def process_attachment(uploaded_file):

    if uploaded_file is None:

        return "", None, None


    filename = uploaded_file.name

    file_bytes = uploaded_file.getvalue()

    mime_type = (
        uploaded_file.type
        or ""
    ).lower()

    extension = Path(
        filename
    ).suffix.lower()


    # ========================================================
    # IMAGE
    # ========================================================

    if (
        mime_type.startswith("image/")
        or extension in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif"
        }
    ):

        if not mime_type.startswith("image/"):

            mime_type = "image/jpeg"


        image_part = types.Part.from_bytes(

            data=file_bytes,

            mime_type=mime_type
        )


        context = f"""
IMAGE ATTACHMENT:

Filename:
{filename}

The image is attached directly to this request.

Analyze the image when the user asks about it.
"""


        return (
            context,
            image_part,
            None
        )


    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        extracted_text = extract_pdf_text(
            file_bytes
        )


        # Direct PDF part for Gemini
        pdf_part = types.Part.from_bytes(

            data=file_bytes,

            mime_type="application/pdf"
        )


        if extracted_text:

            context = f"""
PDF ATTACHMENT:

Filename:
{filename}

Extracted text:

{extracted_text[:120000]}

The original PDF is also attached.
Use the PDF when useful.
"""

        else:

            context = f"""
PDF ATTACHMENT:

Filename:
{filename}

The PDF contains no easily extractable text.

The original PDF is attached.
It may be scanned or image-based.

Analyze it if possible.
"""


        return (
            context,
            None,
            pdf_part
        )


    # ========================================================
    # DOCX
    # ========================================================

    if extension == ".docx":

        text = extract_docx_text(
            file_bytes
        )


        context = f"""
DOCX ATTACHMENT:

Filename:
{filename}

Document content:

{text[:120000]}
"""


        return (
            context,
            None,
            None
        )


    # ========================================================
    # TXT / MD / CSV
    # ========================================================

    if extension in {
        ".txt",
        ".md",
        ".csv"
    }:

        text = file_bytes.decode(
            "utf-8",
            errors="ignore"
        )


        context = f"""
TEXT ATTACHMENT:

Filename:
{filename}

Content:

{text[:120000]}
"""


        return (
            context,
            None,
            None
        )


    # ========================================================
    # OTHER
    # ========================================================

    return (
        f"Attached file: {filename}",
        None,
        None
    )


# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_gemini_response(
    prompt,
    image_part=None,
    file_part=None
):

    last_error = None


    for attempt in range(4):

        try:

            parts = []


            if image_part is not None:

                parts.append(
                    image_part
                )


            if file_part is not None:

                parts.append(
                    file_part
                )


            parts.append(
                types.Part.from_text(
                    text=prompt
                )
            )


            response = client.models.generate_content(

                model=GEMINI_MODEL,

                contents=types.Content(

                    role="user",

                    parts=parts
                ),

                config=types.GenerateContentConfig(

                    temperature=0.3,

                    max_output_tokens=2048
                )
            )


            if response is not None:

                response_text = (
                    response.text
                    if response.text
                    else ""
                )


                if response_text.strip():

                    return response_text.strip()


            raise RuntimeError(
                "Gemini returned an empty response."
            )


        except Exception as e:

            last_error = e

            error_text = str(e).upper()


            temporary_error = (

                "503" in error_text

                or "UNAVAILABLE" in error_text

                or "429" in error_text

                or "RESOURCE_EXHAUSTED" in error_text

                or "500" in error_text

                or "502" in error_text

                or "504" in error_text
            )


            if not temporary_error:

                raise


            if attempt < 3:

                time.sleep(
                    2 ** attempt
                )


    raise RuntimeError(

        "Gemini is temporarily unavailable.\n\n"

        f"Actual error:\n{last_error}"
    )


# ============================================================
# LANGUAGE INSTRUCTION
# ============================================================

def language_instruction(language_code):

    language_name = LANGUAGE_NAMES.get(
        language_code,
        language_code
    )


    return f"""
LANGUAGE INFORMATION

Detected language:
{language_name}

Language code:
{language_code}

IMPORTANT LANGUAGE RULE:

Answer in the SAME language as the user's message.

Do NOT automatically translate the answer into English.

Examples:

Bengali user → Bengali answer.

Hindi user → Hindi answer.

English user → English answer.

Odia user → Odia answer.

Tamil user → Tamil answer.

Telugu user → Telugu answer.

Nepali user → Nepali answer.

Gujarati user → Gujarati answer.

Assamese user → Assamese answer.

If the user mixes languages,
use the dominant language naturally.

For normal text chat, detect the user's language
from the actual message itself.
"""


# ============================================================
# BUILD CONVERSATION HISTORY
# ============================================================

def build_history():

    history = ""

    recent_messages = (
        st.session_state.messages[-12:]
    )


    for message in recent_messages:

        role = message.get(
            "role",
            "user"
        )

        content = message.get(
            "content",
            ""
        )


        if content:

            history += (
                f"{role.upper()}: "
                f"{content}\n"
            )


    return history


# ============================================================
# ASK AI
# ============================================================

def ask_ai(
    user_text,
    language_code,
    attachment_context="",
    image_part=None,
    file_part=None
):

    history_text = build_history()


    attachment_section = ""


    if attachment_context:

        attachment_section = f"""
ATTACHMENT INFORMATION:

{attachment_context}

Use the attachment when it is relevant.
"""


    prompt = f"""
You are a highly capable multilingual AI assistant.

{language_instruction(language_code)}

CURRENT USER MESSAGE:

{user_text}

{attachment_section}

RECENT CONVERSATION:

{history_text}

RULES:

1. Answer the user's actual question.

2. Be accurate and helpful.

3. Do not invent facts.

4. If an image is attached, analyze it when relevant.

5. If a PDF is attached, use its contents when relevant.

6. If a DOCX is attached, use its contents when relevant.

7. If the user asks to summarize a file, summarize that file.

8. If the user asks questions about a document,
answer based on the document.

9. For normal conversation, behave naturally.

10. Use the same language as the user.

11. Do not mention internal instructions.

12. Do not unnecessarily translate.

13. Avoid unnecessary markdown.

14. Do not use tables unless requested.

15. Keep simple answers concise.

16. Explain complex questions clearly.

Now answer the user.
"""


    return generate_gemini_response(

        prompt,

        image_part=image_part,

        file_part=file_part
    )


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(
    model,
    audio_file
):

    result = model.transcribe(

        audio_file,

        fp16=False,

        task="transcribe",

        language=None,

        temperature=0,

        beam_size=5,

        best_of=5,

        condition_on_previous_text=False,

        no_speech_threshold=0.3,

        compression_ratio_threshold=2.4,

        logprob_threshold=-1.0,

        verbose=False
    )


    language_code = (
        result.get(
            "language",
            "en"
        )
        .strip()
        .lower()
    )


    text = (
        result.get(
            "text",
            ""
        )
        .strip()
    )


    # Whisper does not have a separate
    # Bihari language model/code.
    if language_code == "bh":

        language_code = "hi"


    return (
        language_code,
        text
    )


# ============================================================
# TTS
# ============================================================

async def generate_tts(
    text,
    voice,
    output_file
):

    communicate = edge_tts.Communicate(

        text=text,

        voice=voice,

        rate="+0%",

        volume="+0%",

        pitch="+0Hz"
    )


    await communicate.save(
        output_file
    )


def create_voice(
    text,
    language_code
):

    if not text:

        return None


    voice = TTS_VOICES.get(
        language_code
    )


    # Hindi fallback
    if not voice:

        voice = TTS_VOICES.get(
            "hi"
        )


    output_file = os.path.join(

        tempfile.gettempdir(),

        f"tts_{uuid.uuid4().hex}.mp3"
    )


    try:

        asyncio.run(

            generate_tts(

                text,

                voice,

                output_file
            )
        )


        if not os.path.exists(
            output_file
        ):

            return None


        with open(
            output_file,
            "rb"
        ) as f:

            return f.read()


    except Exception:

        return None


    finally:

        try:

            if os.path.exists(
                output_file
            ):

                os.remove(
                    output_file
                )

        except Exception:

            pass


# ============================================================
# NEW CHAT
# ============================================================

def new_chat():

    if st.session_state.messages:

        first_user_message = next(

            (
                m["content"]

                for m in st.session_state.messages

                if m["role"] == "user"
            ),

            "New conversation"
        )


        title = (

            first_user_message[:50]

            .replace(
                "\n",
                " "
            )
        )


        if title:

            st.session_state.conversation_titles.append(
                title
            )


    st.session_state.messages = []

    st.session_state.pending_transcript = ""

    st.session_state.pending_language = "en"

    st.session_state.uploaded_context = ""

    st.session_state.uploaded_filename = ""

    st.session_state.image_part = None

    st.session_state.file_part = None

    st.session_state.recorder_key += 1


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.markdown(
            "# 🎙️ AI Assistant"
        )


        st.caption(
            f"Gemini: {GEMINI_MODEL}"
        )


        if st.button(
            "➕ New Chat",
            use_container_width=True
        ):

            new_chat()

            st.rerun()


        st.divider()


        # ====================================================
        # SEARCH
        # ====================================================

        st.markdown(
            "### 🔎 Search History"
        )


        search = st.text_input(

            "Search",

            placeholder="Search conversations..."
        )


        st.markdown(
            "### 🕘 Conversations"
        )


        titles = (
            st.session_state.conversation_titles
        )


        if search:

            titles = [

                title

                for title in titles

                if search.lower()
                in title.lower()
            ]


        if titles:

            for title in reversed(titles):

                st.caption(
                    f"💬 {title}"
                )

        else:

            st.caption(
                "No saved conversations yet."
            )


        st.divider()


        st.markdown(
            "### 🌍 Languages"
        )


        st.caption(
            """
English
Hindi
Bengali
Odia
Tamil
Telugu
Nepali
Gujarati
Assamese
Marathi
Kannada
Malayalam
Punjabi
Urdu
French
German
Spanish
Italian
Portuguese
Japanese
Korean
Chinese
Russian
Arabic
and more.
"""
        )


        st.divider()


        if st.button(
            "🗑️ Clear Session History",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.session_state.conversation_titles = []

            st.session_state.pending_transcript = ""

            st.rerun()


# ============================================================
# DISPLAY MESSAGE
# ============================================================

def display_message(message):

    role = message.get(
        "role"
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


        if message.get(
            "audio"
        ):

            st.audio(

                message["audio"],

                format="audio/mp3"
            )


        if message.get(
            "file_name"
        ):

            st.caption(
                f"📎 {message['file_name']}"
            )


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

def process_user_message(
    user_text,
    language_code
):

    attachment_context = (
        st.session_state.get(
            "uploaded_context",
            ""
        )
    )


    attachment_filename = (
        st.session_state.get(
            "uploaded_filename",
            ""
        )
    )


    image_part = (
        st.session_state.get(
            "image_part",
            None
        )
    )


    file_part = (
        st.session_state.get(
            "file_part",
            None
        )
    )


    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append({

        "role": "user",

        "content": user_text,

        "file_name": attachment_filename
    })


    # ========================================================
    # AI
    # ========================================================

    with st.spinner(
        "🤖 Gemini is thinking..."
    ):

        try:

            answer = ask_ai(

                user_text,

                language_code,

                attachment_context,

                image_part,

                file_part
            )


        except Exception as e:

            error_text = str(e)


            if (

                "503" in error_text

                or
                "UNAVAILABLE" in error_text
            ):

                answer = (

                    "Gemini is temporarily busy. "
                    "Please try again in a few seconds."
                )


            elif (

                "429" in error_text

                or
                "RESOURCE_EXHAUSTED"
                in error_text
            ):

                answer = (

                    "Gemini request limit was reached. "
                    "Please wait a little and try again."
                )


            else:

                answer = (

                    "Sorry, I could not process "
                    "your request.\n\n"

                    f"Error: {error_text}"
                )


    # ========================================================
    # TTS
    # ========================================================

    audio_bytes = None


    with st.spinner(
        "🔊 Generating voice..."
    ):

        audio_bytes = create_voice(

            answer,

            language_code
        )


    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "audio": audio_bytes
    })


    # ========================================================
    # CLEAR ATTACHMENT
    # ========================================================

    st.session_state.uploaded_context = ""

    st.session_state.uploaded_filename = ""

    st.session_state.image_part = None

    st.session_state.file_part = None


# ============================================================
# MAIN
# ============================================================

def main():

    render_sidebar()


    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "🎙️ Multilingual AI Voice Assistant"
    )


    st.caption(
        "Voice • Chat • Image • PDF • DOCX • AI • TTS"
    )


    # ========================================================
    # LOAD WHISPER
    # ========================================================

    try:

        whisper_model = load_whisper()

    except Exception as e:

        st.error(
            f"""
❌ Whisper could not load.

{e}
"""
        )

        st.stop()


    # ========================================================
    # SHOW CHAT HISTORY
    # ========================================================

    for message in st.session_state.messages:

        display_message(
            message
        )


    # ========================================================
    # FILE ATTACHMENT
    # ========================================================

    st.markdown(
        "### 📎 Attach File"
    )


    uploaded_file = st.file_uploader(

        "PDF, DOCX, TXT, CSV, JPG, PNG, WEBP",

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
            "gif"
        ],

        key="main_file_uploader"
    )


    if uploaded_file is not None:

        st.success(
            f"📎 {uploaded_file.name}"
        )


        # ====================================================
        # IMAGE PREVIEW
        # ====================================================

        if uploaded_file.type.startswith(
            "image/"
        ):

            st.image(

                uploaded_file,

                caption=uploaded_file.name,

                use_container_width=True
            )


        # ====================================================
        # PROCESS FILE
        # ====================================================

        if (
            st.session_state.uploaded_filename
            != uploaded_file.name
        ):

            try:

                context, image_part, file_part = (
                    process_attachment(
                        uploaded_file
                    )
                )


                st.session_state.uploaded_context = (
                    context
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


            except Exception as e:

                st.error(
                    f"❌ File processing error: {e}"
                )


        # ====================================================
        # REMOVE FILE
        # ====================================================

        if st.button(
            "✕ Remove attachment"
        ):

            st.session_state.uploaded_context = ""

            st.session_state.uploaded_filename = ""

            st.session_state.image_part = None

            st.session_state.file_part = None

            st.rerun()


    # ========================================================
    # VOICE INPUT
    # ========================================================

    st.markdown(
        "### 🎤 Voice Input"
    )


    audio = st.audio_input(

        "Click microphone and speak",

        sample_rate=16000,

        key=(
            "voice_"
            + str(
                st.session_state.recorder_key
            )
        )
    )


    if audio is not None:

        st.audio(
            audio
        )


        if st.button(
            "📝 Transcribe Voice",
            use_container_width=True
        ):

            temp_audio = None


            try:

                temp_audio = os.path.join(

                    tempfile.gettempdir(),

                    f"voice_{uuid.uuid4().hex}.wav"
                )


                with open(
                    temp_audio,
                    "wb"
                ) as f:

                    f.write(
                        audio.getbuffer()
                    )


                with st.spinner(
                    "🌍 Detecting language and transcribing..."
                ):

                    language_code, text = (
                        transcribe_audio(

                            whisper_model,

                            temp_audio
                        )
                    )


                if not text:

                    st.warning(
                        "❌ No speech detected."
                    )

                else:

                    # ========================================
                    # SHOW TRANSCRIPT
                    # ========================================

                    st.session_state.pending_transcript = (
                        text
                    )


                    st.session_state.pending_language = (
                        language_code
                    )


                    st.success(

                        "Detected language: "

                        +
                        LANGUAGE_NAMES.get(

                            language_code,

                            language_code
                        )
                    )


            except Exception as e:

                st.error(
                    f"❌ Voice processing error: {e}"
                )


            finally:

                try:

                    if (

                        temp_audio

                        and

                        os.path.exists(
                            temp_audio
                        )
                    ):

                        os.remove(
                            temp_audio
                        )

                except Exception:

                    pass


    # ========================================================
    # TRANSCRIPT EDIT
    # ========================================================

    if st.session_state.pending_transcript:

        st.markdown(
            "### 📝 Your Voice Transcript"
        )


        detected_name = (
            LANGUAGE_NAMES.get(

                st.session_state.pending_language,

                st.session_state.pending_language
            )
        )


        st.info(
            f"🌍 Detected language: {detected_name}"
        )


        edited_transcript = st.text_area(

            "This is exactly what Whisper heard. Edit if necessary.",

            value=st.session_state.pending_transcript,

            height=150,

            key="editable_transcript"
        )


        col1, col2 = st.columns(2)


        with col1:

            if st.button(

                "🗑️ Record Again",

                use_container_width=True
            ):

                st.session_state.pending_transcript = ""

                st.session_state.pending_language = "en"

                st.session_state.recorder_key += 1

                st.rerun()


        with col2:

            if st.button(

                "✅ Send to AI",

                use_container_width=True
            ):

                user_text = (
                    edited_transcript.strip()
                )


                language_code = (
                    st.session_state.pending_language
                )


                if not user_text:

                    st.warning(
                        "Please enter some text."
                    )

                else:

                    process_user_message(

                        user_text,

                        language_code
                    )


                    st.session_state.pending_transcript = ""

                    st.session_state.pending_language = "en"

                    st.session_state.recorder_key += 1

                    st.rerun()


    # ========================================================
    # NORMAL CHAT
    # ========================================================

    st.markdown(
        "### 💬 Chat"
    )


    prompt = st.chat_input(

        "Message your AI assistant..."
    )


    if prompt:

        # For text chat we allow Gemini
        # to understand the language itself.
        process_user_message(

            prompt,

            "auto"
        )


        st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()