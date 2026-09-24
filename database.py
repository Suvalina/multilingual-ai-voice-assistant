import os
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

_LOCK = threading.RLock()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "voice_assistant.db"


# ============================================================
# DATABASE PATH
# ============================================================

def _resolve_db_path():
    value = (
        os.getenv("DATABASE_URL")
        or os.getenv("DB_URL")
        or os.getenv("DB_PATH")
        or ""
    ).strip()

    if not value:
        return DEFAULT_DB_PATH

    if value.startswith("sqlite:///"):
        return Path(value[10:])

    if value.startswith("sqlite://"):
        return Path(value[9:])

    # This project intentionally uses SQLite unless
    # a separate database adapter is added.
    # Ignore non-SQLite DATABASE_URL values.
    if "://" in value:
        return DEFAULT_DB_PATH

    return Path(value)


DB_PATH = _resolve_db_path()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def _connect():
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    return conn


# ============================================================
# TIME
# ============================================================

def _now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# SCHEMA HELPERS
# ============================================================

def _table_columns(conn, table_name):
    """
    Return all column names from a SQLite table.
    """

    try:
        return {
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
        }

    except Exception:
        return set()


def _ensure_column(
    conn,
    table_name,
    column_name,
    column_type,
):
    """
    Add a missing column to an existing table.

    This makes old databases compatible with
    newer versions of the application.
    """

    columns = _table_columns(
        conn,
        table_name,
    )

    if column_name not in columns:

        conn.execute(
            f"ALTER TABLE {table_name} "
            f"ADD COLUMN {column_name} {column_type}"
        )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():
    """
    Create the SQLite schema and safely migrate
    older database versions.

    IMPORTANT:
    Existing database data is preserved.
    """

    with _LOCK:

        conn = _connect()

        try:

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    language TEXT,
                    tts_language TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id)
                        REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS message_media (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    media_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(message_id)
                        REFERENCES messages(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS session_sources (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    filename TEXT,
                    source_type TEXT,
                    source_url TEXT,
                    file_hash TEXT,
                    context TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id)
                        REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_messages_session
                    ON messages(session_id, created_at);

                CREATE INDEX IF NOT EXISTS
                    idx_sources_session
                    ON session_sources(session_id, created_at);

                CREATE INDEX IF NOT EXISTS
                    idx_message_media_message
                    ON message_media(message_id, created_at);
                """
            )

            # ====================================================
            # MIGRATIONS
            # ====================================================
            #
            # These are important because older versions of
            # voice_assistant.db may already exist.
            #
            # CREATE TABLE IF NOT EXISTS does NOT update an
            # existing table, so missing columns must be added
            # manually.
            # ====================================================

            # messages table
            _ensure_column(
                conn,
                "messages",
                "language",
                "TEXT",
            )

            _ensure_column(
                conn,
                "messages",
                "tts_language",
                "TEXT",
            )

            # chat_sessions table
            _ensure_column(
                conn,
                "chat_sessions",
                "title",
                "TEXT NOT NULL DEFAULT 'New Chat'",
            )

            # message_media table
            _ensure_column(
                conn,
                "message_media",
                "media_json",
                "TEXT",
            )

            _ensure_column(
                conn,
                "message_media",
                "created_at",
                "TEXT",
            )

            # session_sources table
            _ensure_column(
                conn,
                "session_sources",
                "filename",
                "TEXT",
            )

            _ensure_column(
                conn,
                "session_sources",
                "source_type",
                "TEXT",
            )

            _ensure_column(
                conn,
                "session_sources",
                "source_url",
                "TEXT",
            )

            _ensure_column(
                conn,
                "session_sources",
                "file_hash",
                "TEXT",
            )

            _ensure_column(
                conn,
                "session_sources",
                "context",
                "TEXT",
            )

            _ensure_column(
                conn,
                "session_sources",
                "metadata_json",
                "TEXT",
            )

            _ensure_column(
                conn,
                "session_sources",
                "created_at",
                "TEXT",
            )

            conn.commit()

        finally:

            conn.close()


# ============================================================
# CREATE SESSION
# ============================================================

def create_session(
    title="New Chat",
    session_id=None,
):
    """
    Create a new chat session.
    """

    init_db()

    sid = session_id or str(uuid.uuid4())
    now = _now()

    with _LOCK:

        conn = _connect()

        try:

            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions
                (
                    id,
                    title,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    sid,
                    title or "New Chat",
                    now,
                    now,
                ),
            )

            conn.commit()

        finally:

            conn.close()

    return sid


# ============================================================
# UPDATE SESSION
# ============================================================

def touch_session(
    session_id,
    title=None,
):
    """
    Update the session timestamp and optionally
    update its title.
    """

    if not session_id:
        return

    now = _now()

    with _LOCK:

        conn = _connect()

        try:

            if title:

                conn.execute(
                    """
                    UPDATE chat_sessions
                    SET title = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        title,
                        now,
                        session_id,
                    ),
                )

            else:

                conn.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        session_id,
                    ),
                )

            conn.commit()

        finally:

            conn.close()


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    session_id,
    role,
    content,
    language=None,
    tts_language=None,
    media=None,
):
    """
    Save a user/assistant message.

    Optional media metadata is stored in message_media.
    Actual binary files are intentionally NOT stored.
    """

    if not session_id or not content:
        return None

    init_db()

    message_id = str(uuid.uuid4())
    now = _now()

    with _LOCK:

        conn = _connect()

        try:

            # ------------------------------------------------
            # Save message
            # ------------------------------------------------

            conn.execute(
                """
                INSERT INTO messages
                (
                    id,
                    session_id,
                    role,
                    content,
                    language,
                    tts_language,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    language,
                    tts_language,
                    now,
                ),
            )

            # ------------------------------------------------
            # Save media metadata
            # ------------------------------------------------

            if media:

                safe_media = []

                for item in media:

                    if not isinstance(item, dict):
                        continue

                    cleaned = {}

                    for key, value in item.items():

                        # Never store temporary binary/audio data
                        if key in {
                            "data",
                            "audio_file",
                        }:
                            continue

                        try:

                            json.dumps(
                                value,
                                ensure_ascii=False,
                            )

                            cleaned[key] = value

                        except Exception:

                            cleaned[key] = str(value)

                    safe_media.append(cleaned)

                if safe_media:

                    conn.execute(
                        """
                        INSERT INTO message_media
                        (
                            id,
                            message_id,
                            media_json,
                            created_at
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            message_id,
                            json.dumps(
                                safe_media,
                                ensure_ascii=False,
                            ),
                            now,
                        ),
                    )

            # ------------------------------------------------
            # Update session timestamp
            # ------------------------------------------------

            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    session_id,
                ),
            )

            conn.commit()

        finally:

            conn.close()

    return message_id


# ============================================================
# LOAD MESSAGES
# ============================================================

def load_messages(session_id):
    """
    Load all messages belonging to a chat session.
    """

    if not session_id:
        return []

    init_db()

    with _LOCK:

        conn = _connect()

        try:

            rows = conn.execute(
                """
                SELECT
                    id,
                    role,
                    content,
                    language,
                    tts_language,
                    created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()

            result = []

            for row in rows:

                item = {
                    "role": row["role"],
                    "content": row["content"],
                }

                if row["language"]:
                    item["language"] = row["language"]

                if row["tts_language"]:
                    item["tts_language"] = row["tts_language"]

                # --------------------------------------------
                # Load media attached to this message
                # --------------------------------------------

                media_row = conn.execute(
                    """
                    SELECT media_json
                    FROM message_media
                    WHERE message_id = ?
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (row["id"],),
                ).fetchone()

                if media_row:

                    try:

                        media_json = media_row["media_json"]

                        if media_json:

                            item["source_media"] = json.loads(
                                media_json
                            )

                    except Exception:

                        pass

                result.append(item)

            return result

        finally:

            conn.close()


# ============================================================
# CLEAR SESSION MESSAGES
# ============================================================

def clear_session_messages(session_id):
    """
    Delete all messages from a specific session.
    """

    if not session_id:
        return

    init_db()

    with _LOCK:

        conn = _connect()

        try:

            conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            )

            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    _now(),
                    session_id,
                ),
            )

            conn.commit()

        finally:

            conn.close()


# ============================================================
# SAVE SOURCE
# ============================================================

def save_source(
    session_id,
    source,
):
    """
    Save uploaded document, video, image or website
    source information for a session.
    """

    if not session_id or not source:
        return None

    init_db()

    source_id = str(uuid.uuid4())
    now = _now()

    metadata = {}

    for key, value in source.items():

        # Do not store large/temporary data
        if key in {
            "uploaded_context",
            "video_segments",
            "pages",
            "images",
            "frames",
            "video_path",
        }:
            continue

        try:

            json.dumps(
                value,
                ensure_ascii=False,
            )

            metadata[key] = value

        except Exception:

            metadata[key] = str(value)

    with _LOCK:

        conn = _connect()

        try:

            conn.execute(
                """
                INSERT INTO session_sources
                (
                    id,
                    session_id,
                    filename,
                    source_type,
                    source_url,
                    file_hash,
                    context,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    session_id,
                    source.get("filename") or "",
                    source.get("source_type") or "",
                    source.get("url")
                    or source.get("source_url")
                    or "",
                    source.get("file_hash") or "",
                    source.get("uploaded_context") or "",
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    ),
                    now,
                ),
            )

            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    session_id,
                ),
            )

            conn.commit()

        finally:

            conn.close()

    return source_id


# ============================================================
# LIST CHAT SESSIONS
# ============================================================

def list_sessions(limit=50):
    """
    Return recent chat sessions.
    """

    init_db()

    with _LOCK:

        conn = _connect()

        try:

            rows = conn.execute(
                """
                SELECT
                    id,
                    title,
                    created_at,
                    updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

        finally:

            conn.close()


# ============================================================
# DELETE SESSION
# ============================================================

def delete_session(session_id):
    """
    Delete an entire chat session.

    Due to ON DELETE CASCADE,
    messages/media/sources belonging to it
    are also removed.
    """

    if not session_id:
        return

    init_db()

    with _LOCK:

        conn = _connect()

        try:

            conn.execute(
                """
                DELETE FROM chat_sessions
                WHERE id = ?
                """,
                (session_id,),
            )

            conn.commit()

        finally:

            conn.close()


# ============================================================
# GET SINGLE SESSION
# ============================================================

def get_session(session_id):
    """
    Get information about one chat session.
    """

    if not session_id:
        return None

    init_db()

    with _LOCK:

        conn = _connect()

        try:

            row = conn.execute(
                """
                SELECT
                    id,
                    title,
                    created_at,
                    updated_at
                FROM chat_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()

            return dict(row) if row else None

        finally:

            conn.close()


# ============================================================
# INITIAL DATABASE SETUP
# ============================================================

init_db()