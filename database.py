import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

if DATABASE_URL:
    DB_URL = DATABASE_URL
else:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    DB_URL = "sqlite:///" + str((data_dir / "voice_assistant.db").resolve()).replace("\\", "/")

engine = create_engine(DB_URL, pool_pre_ping=True, future=True)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _params(**kwargs):
    return kwargs


def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id VARCHAR(64) PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(64) PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                role VARCHAR(32) NOT NULL,
                content TEXT NOT NULL,
                language VARCHAR(16),
                created_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS message_media (
                id VARCHAR(64) PRIMARY KEY,
                message_id VARCHAR(64) NOT NULL,
                media_type VARCHAR(64) NOT NULL,
                caption TEXT,
                timestamp_seconds DOUBLE PRECISION,
                video_text TEXT,
                mime_type VARCHAR(128),
                data BYTEA,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS session_sources (
                id VARCHAR(64) PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                source_type VARCHAR(64),
                filename TEXT,
                url TEXT,
                file_hash VARCHAR(128),
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        # SQLite does not enforce foreign keys by default; indexes still help history queries.
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_media_message ON message_media(message_id, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sources_session ON session_sources(session_id, updated_at)"))


def create_session(title="New Chat"):
    session_id = uuid.uuid4().hex
    now = _now()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO chat_sessions(id, title, created_at, updated_at)
            VALUES (:id, :title, :created_at, :updated_at)
        """), _params(id=session_id, title=title[:120] or "New Chat", created_at=now, updated_at=now))
    return session_id


def list_sessions(limit=50):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, title, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def get_session(session_id):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id, title, created_at, updated_at
            FROM chat_sessions WHERE id=:id
        """), {"id": session_id}).mappings().first()
    return dict(row) if row else None


def update_session_title(session_id, title):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE chat_sessions SET title=:title, updated_at=:updated_at WHERE id=:id
        """), {"title": title[:120] or "New Chat", "updated_at": _now(), "id": session_id})


def touch_session(session_id):
    with engine.begin() as conn:
        conn.execute(text("UPDATE chat_sessions SET updated_at=:updated_at WHERE id=:id"),
                     {"updated_at": _now(), "id": session_id})


def save_message(session_id, role, content, language=None, source_media=None):
    message_id = uuid.uuid4().hex
    now = _now()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages(id, session_id, role, content, language, created_at)
            VALUES (:id, :session_id, :role, :content, :language, :created_at)
        """), {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content or "",
            "language": language,
            "created_at": now,
        })

        for media in source_media or []:
            if not isinstance(media, dict):
                continue
            media_type = media.get("type")
            data = media.get("data")
            if media_type != "image" or not isinstance(data, (bytes, bytearray)):
                continue
            metadata = {}
            for key in ("source_type", "source_position", "page_number", "url", "section", "timestamp"):
                if key in media and media.get(key) is not None:
                    metadata[key] = media.get(key)
            conn.execute(text("""
                INSERT INTO message_media(
                    id, message_id, media_type, caption, timestamp_seconds,
                    video_text, mime_type, data, metadata_json, created_at
                ) VALUES (
                    :id, :message_id, :media_type, :caption, :timestamp_seconds,
                    :video_text, :mime_type, :data, :metadata_json, :created_at
                )
            """), {
                "id": uuid.uuid4().hex,
                "message_id": message_id,
                "media_type": media_type,
                "caption": media.get("caption", "Source visual"),
                "timestamp_seconds": media.get("timestamp"),
                "video_text": media.get("video_text"),
                "mime_type": media.get("mime_type", "image/png"),
                "data": bytes(data),
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
                "created_at": now,
            })

        conn.execute(text("UPDATE chat_sessions SET updated_at=:updated_at WHERE id=:id"),
                     {"updated_at": now, "id": session_id})

    return message_id


def load_messages(session_id):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, role, content, language, created_at
            FROM messages
            WHERE session_id=:session_id
            ORDER BY created_at ASC
        """), {"session_id": session_id}).mappings().all()

        media_rows = conn.execute(text("""
            SELECT mm.message_id, mm.media_type, mm.caption, mm.timestamp_seconds,
                   mm.video_text, mm.mime_type, mm.data, mm.metadata_json
            FROM message_media mm
            JOIN messages m ON m.id=mm.message_id
            WHERE m.session_id=:session_id
            ORDER BY mm.created_at ASC
        """), {"session_id": session_id}).mappings().all()

    media_by_message = {}
    for row in media_rows:
        item = {
            "type": row["media_type"],
            "caption": row["caption"] or "Source visual",
            "timestamp": row["timestamp_seconds"],
            "video_text": row["video_text"],
            "mime_type": row["mime_type"] or "image/png",
            "data": row["data"],
        }
        try:
            item.update(json.loads(row["metadata_json"] or "{}"))
        except Exception:
            pass
        media_by_message.setdefault(row["message_id"], []).append(item)

    messages = []
    for row in rows:
        item = {
            "role": row["role"],
            "content": row["content"],
            "language": row["language"] or "en",
            "created_at": row["created_at"],
        }
        if row["id"] in media_by_message:
            item["source_media"] = media_by_message[row["id"]]
        messages.append(item)
    return messages


def save_source(session_id, source):
    if not source:
        return
    source_type = source.get("source_type", "")
    filename = source.get("filename", "")
    url = filename if source_type == "website" else source.get("url", "")
    file_hash = source.get("file_hash") or source.get("hash", "")
    metadata = {
        "extension": source.get("extension", ""),
        "authenticated": bool(source.get("authenticated", False)),
        "page_count": len(source.get("pdf_pages", []) or []),
        "website_pages": len((source.get("website_data") or {}).get("pages", []) or []),
        "video_language": source.get("video_language", ""),
    }
    now = _now()
    with engine.begin() as conn:
        existing = conn.execute(text("""
            SELECT id FROM session_sources WHERE session_id=:session_id ORDER BY updated_at DESC LIMIT 1
        """), {"session_id": session_id}).scalar()
        if existing:
            conn.execute(text("""
                UPDATE session_sources
                SET source_type=:source_type, filename=:filename, url=:url,
                    file_hash=:file_hash, metadata_json=:metadata_json, updated_at=:updated_at
                WHERE id=:id
            """), {
                "id": existing, "source_type": source_type, "filename": filename, "url": url,
                "file_hash": file_hash, "metadata_json": json.dumps(metadata, ensure_ascii=False), "updated_at": now,
            })
        else:
            conn.execute(text("""
                INSERT INTO session_sources(
                    id, session_id, source_type, filename, url, file_hash,
                    metadata_json, created_at, updated_at
                ) VALUES (:id,:session_id,:source_type,:filename,:url,:file_hash,:metadata_json,:created_at,:updated_at)
            """), {
                "id": uuid.uuid4().hex, "session_id": session_id, "source_type": source_type,
                "filename": filename, "url": url, "file_hash": file_hash,
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
                "created_at": now, "updated_at": now,
            })


def delete_session(session_id):
    with engine.begin() as conn:
        message_ids = [r[0] for r in conn.execute(text("SELECT id FROM messages WHERE session_id=:id"), {"id": session_id}).all()]
        if message_ids:
            for mid in message_ids:
                conn.execute(text("DELETE FROM message_media WHERE message_id=:id"), {"id": mid})
        conn.execute(text("DELETE FROM messages WHERE session_id=:id"), {"id": session_id})
        conn.execute(text("DELETE FROM session_sources WHERE session_id=:id"), {"id": session_id})
        conn.execute(text("DELETE FROM chat_sessions WHERE id=:id"), {"id": session_id})


def clear_all_history():
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM message_media"))
        conn.execute(text("DELETE FROM messages"))
        conn.execute(text("DELETE FROM session_sources"))
        conn.execute(text("DELETE FROM chat_sessions"))


try:
    init_db()
except Exception as exc:
    print("DATABASE INIT ERROR:", repr(exc))
