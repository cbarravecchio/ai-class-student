"""SQLite storage for the local Gradio chat's profile and conversations."""

import json
import hashlib
import math
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np


DATABASE_PATH = Path(__file__).with_name("chat_memory.sqlite3")
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBEDDING_CONTENT_VERSION = 3
_EMBEDDING_INDEX_VERSION = 4
_VECTOR_DIMENSIONS = 384
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_MIN_SIMILARITY = 0.53
_RRF_K = 60
_EXACT_SEARCH_MAX_VECTORS = 2048
_MAX_CHUNK_CHARS = 1600
_CHUNK_OVERLAP_CHARS = 200
_ANN_CHECKPOINT_INTERVAL = 256
_embedding_model = None
_ann_indexes = {}


def _connect(database_path=DATABASE_PATH):
    return sqlite3.connect(database_path, timeout=10)


def _embed_texts(texts):
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding

        _embedding_model = TextEmbedding(model_name=_EMBEDDING_MODEL_NAME)
    return list(_embedding_model.embed(texts))


def _normalize_embedding(embedding):
    values = [float(value) for value in embedding]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not norm:
        return values
    return [value / norm for value in values]


def _serialize_embedding(embedding):
    return np.asarray(_normalize_embedding(embedding), dtype=np.float32).tobytes()


def _deserialize_embedding(embedding):
    if isinstance(embedding, str):
        return np.asarray(json.loads(embedding), dtype=np.float32)
    return np.frombuffer(embedding, dtype=np.float32)


def _split_text(text):
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + _MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + _MAX_CHUNK_CHARS // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - _CHUNK_OVERLAP_CHARS)
    return [chunk for chunk in chunks if chunk]


def _new_ann_index():
    from usearch.index import Index

    return Index(
        ndim=_VECTOR_DIMENSIONS,
        metric="cos",
        dtype="f32",
        connectivity=32,
        expansion_add=128,
        expansion_search=64,
    )


def _ann_snapshot_path(database_path, revision):
    database_path = Path(database_path)
    return database_path.with_name(
        f"{database_path.name}.ann.{revision}.usearch"
    )


def _checkpoint_ann_index(database_path, index, revision):
    snapshot_path = _ann_snapshot_path(database_path, revision)
    temporary_path = snapshot_path.with_name(
        f"{snapshot_path.name}.{uuid.uuid4().hex}.tmp"
    )
    index.save(temporary_path)
    os.replace(temporary_path, snapshot_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO chat_ann_checkpoint (id, revision, file_name)
            VALUES (1, ?, ?)
            """,
            (revision, snapshot_path.name),
        )
        connection.execute(
            "DELETE FROM chat_ann_changes WHERE revision <= ?", (revision,)
        )
    for candidate in Path(database_path).parent.glob(
        f"{Path(database_path).name}.ann.*.usearch"
    ):
        if candidate != snapshot_path:
            candidate.unlink(missing_ok=True)


def _rebuild_ann_index(database_path, revision):
    index = _new_ann_index()
    with _connect(database_path) as connection:
        rows = connection.execute(
            "SELECT id, embedding FROM chat_message_embeddings ORDER BY id"
        )
        keys = []
        vectors = []
        for vector_id, serialized_embedding in rows:
            keys.append(vector_id)
            vectors.append(_deserialize_embedding(serialized_embedding))
            if len(keys) >= 4096:
                index.add(
                    np.asarray(keys, dtype=np.uint64),
                    np.asarray(vectors, dtype=np.float32),
                )
                keys.clear()
                vectors.clear()
        if keys:
            index.add(
                np.asarray(keys, dtype=np.uint64),
                np.asarray(vectors, dtype=np.float32),
            )
    _checkpoint_ann_index(database_path, index, revision)
    _ann_indexes[str(Path(database_path).resolve())] = (index, revision)
    return index, revision


def _get_ann_index(database_path):
    cache_key = str(Path(database_path).resolve())
    with _connect(database_path) as connection:
        state = connection.execute(
            "SELECT revision FROM chat_embedding_state WHERE id = 1"
        ).fetchone()
        if state is None:
            raise RuntimeError("The chat embedding index has not been initialized.")
        data_revision = state[0]
        checkpoint = connection.execute(
            "SELECT revision, file_name FROM chat_ann_checkpoint WHERE id = 1"
        ).fetchone()

    cached = _ann_indexes.get(cache_key)
    if cached and cached[1] == data_revision:
        return cached

    if cached and checkpoint and cached[1] >= checkpoint[0]:
        index, loaded_revision = cached
    elif checkpoint:
        snapshot_path = Path(database_path).with_name(checkpoint[1])
        if checkpoint[0] <= data_revision and snapshot_path.is_file():
            from usearch.index import Index

            index = Index(
                ndim=_VECTOR_DIMENSIONS,
                metric="cos",
                dtype="f32",
                connectivity=32,
                expansion_add=128,
                expansion_search=64,
                path=snapshot_path,
            )
            loaded_revision = checkpoint[0]
        else:
            return _rebuild_ann_index(database_path, data_revision)
    else:
        return _rebuild_ann_index(database_path, data_revision)

    with _connect(database_path) as connection:
        changes = connection.execute(
            """
            SELECT revision, operation, vector_id, embedding
            FROM chat_ann_changes
            WHERE revision > ? AND revision <= ?
            ORDER BY id
            """,
            (loaded_revision, data_revision),
        ).fetchall()
        checkpoint_revision = checkpoint[0] if checkpoint else loaded_revision
        pending_change_count = connection.execute(
            """
            SELECT COUNT(*) FROM chat_ann_changes
            WHERE revision > ? AND revision <= ?
            """,
            (checkpoint_revision, data_revision),
        ).fetchone()[0]
    try:
        for _, operation, vector_id, serialized_embedding in changes:
            if operation == "remove":
                index.remove(np.asarray([vector_id], dtype=np.uint64))
            else:
                index.add(
                    np.asarray([vector_id], dtype=np.uint64),
                    np.asarray(
                        [_deserialize_embedding(serialized_embedding)],
                        dtype=np.float32,
                    ),
                )

        if pending_change_count >= _ANN_CHECKPOINT_INTERVAL:
            return _rebuild_ann_index(database_path, data_revision)
    except Exception:
        _ann_indexes.pop(cache_key, None)
        raise
    _ann_indexes[cache_key] = (index, data_revision)
    return index, data_revision


def initialize_database(database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        embedding_table_recreated = False
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_claim_embeddings (
                claim_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (claim_id) REFERENCES memory_claims(id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_message_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate'
                    CHECK (status IN ('candidate', 'confirmed', 'superseded', 'disputed')),
                confidence REAL NOT NULL DEFAULT 0.5
                    CHECK (confidence >= 0 AND confidence <= 1),
                source_conversation_id TEXT NOT NULL,
                source_message_index INTEGER NOT NULL,
                source_excerpt TEXT NOT NULL,
                valid_from TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                valid_until TEXT,
                supersedes_id INTEGER REFERENCES memory_claims(id),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_conversation_id) REFERENCES conversations(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                elapsed_ms REAL NOT NULL,
                index_size INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                lexical_weight INTEGER NOT NULL,
                result_count INTEGER NOT NULL,
                query_hash TEXT,
                feedback INTEGER CHECK (feedback IN (0, 1))
            )
            """
        )
        metric_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(retrieval_metrics)")
        }
        if "query_hash" not in metric_columns:
            connection.execute(
                "ALTER TABLE retrieval_metrics ADD COLUMN query_hash TEXT"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_metric_results (
                event_id INTEGER NOT NULL,
                record_key TEXT NOT NULL,
                rank INTEGER NOT NULL,
                feedback INTEGER CHECK (feedback IN (0, 1)),
                PRIMARY KEY (event_id, record_key),
                FOREIGN KEY (event_id) REFERENCES retrieval_metrics(id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS retrieval_metric_results_feedback
            ON retrieval_metric_results (record_key, feedback)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_relations (
                source_claim_id INTEGER NOT NULL,
                target_claim_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_claim_id, target_claim_id, relation_type),
                CHECK (source_claim_id != target_claim_id),
                FOREIGN KEY (source_claim_id) REFERENCES memory_claims(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (target_claim_id) REFERENCES memory_claims(id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_a_id INTEGER NOT NULL,
                claim_b_id INTEGER NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'resolved', 'dismissed')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (claim_a_id, claim_b_id),
                FOREIGN KEY (claim_a_id) REFERENCES memory_claims(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (claim_b_id) REFERENCES memory_claims(id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS memory_claims_status_category
            ON memory_claims (status, category)
            """
        )
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(chat_message_embeddings)"
            )
        }
        if not {"id", "message_index", "chunk_index"}.issubset(columns):
            embedding_table_recreated = True
            connection.execute("DROP TABLE chat_message_embeddings")
            connection.execute(
                """
                CREATE TABLE chat_message_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chat_memory_fts USING fts5(content)"
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chat_embeddings_fts_insert
            AFTER INSERT ON chat_message_embeddings BEGIN
                INSERT INTO chat_memory_fts (rowid, content)
                VALUES (new.id, new.content);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chat_embeddings_fts_delete
            AFTER DELETE ON chat_message_embeddings BEGIN
                DELETE FROM chat_memory_fts WHERE rowid = old.id;
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS memory_claims_fts_insert
            AFTER INSERT ON memory_claims BEGIN
                INSERT INTO chat_memory_fts (rowid, content)
                VALUES (-new.id, new.content);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS memory_claims_fts_update
            AFTER UPDATE OF content ON memory_claims BEGIN
                DELETE FROM chat_memory_fts WHERE rowid = -old.id;
                INSERT INTO chat_memory_fts (rowid, content)
                VALUES (-new.id, new.content);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS memory_claims_fts_delete
            AFTER DELETE ON memory_claims BEGIN
                DELETE FROM chat_memory_fts WHERE rowid = -old.id;
            END
            """
        )
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS conversation_summary_fts USING fts5(summary)"
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS conversation_summaries_fts_insert
            AFTER INSERT ON conversation_summaries BEGIN
                INSERT INTO conversation_summary_fts (rowid, summary)
                VALUES (new.id, new.summary);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS conversation_summaries_fts_update
            AFTER UPDATE OF summary ON conversation_summaries BEGIN
                DELETE FROM conversation_summary_fts WHERE rowid = old.id;
                INSERT INTO conversation_summary_fts (rowid, summary)
                VALUES (new.id, new.summary);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS conversation_summaries_fts_delete
            AFTER DELETE ON conversation_summaries BEGIN
                DELETE FROM conversation_summary_fts WHERE rowid = old.id;
            END
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_memory_fts_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initialized INTEGER NOT NULL
            )
            """
        )
        fts_initialized = connection.execute(
            "SELECT initialized FROM chat_memory_fts_state WHERE id = 1"
        ).fetchone()
        if fts_initialized is None or embedding_table_recreated:
            connection.execute("DELETE FROM chat_memory_fts")
            connection.execute(
                """
                INSERT INTO chat_memory_fts (rowid, content)
                SELECT id, content FROM chat_message_embeddings
                """
            )
            connection.execute(
                """
                INSERT INTO chat_memory_fts (rowid, content)
                SELECT -id, content FROM memory_claims
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO chat_memory_fts_state (id, initialized)
                VALUES (1, 1)
                """
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS chat_message_embeddings_conversation_id
            ON chat_message_embeddings (conversation_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_embedding_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                model TEXT NOT NULL,
                indexed INTEGER NOT NULL,
                index_version INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        state_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(chat_embedding_state)")
        }
        if "index_version" not in state_columns:
            connection.execute(
                "ALTER TABLE chat_embedding_state "
                "ADD COLUMN index_version INTEGER NOT NULL DEFAULT 0"
            )
        if "revision" not in state_columns:
            connection.execute(
                "ALTER TABLE chat_embedding_state "
                "ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_ann_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                revision INTEGER NOT NULL,
                operation TEXT NOT NULL,
                vector_id INTEGER NOT NULL,
                embedding BLOB
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS chat_ann_changes_revision
            ON chat_ann_changes (revision, id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_ann_checkpoint (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                revision INTEGER NOT NULL,
                file_name TEXT NOT NULL
            )
            """
        )
        indexed = connection.execute(
            "SELECT model, indexed, index_version, revision "
            "FROM chat_embedding_state WHERE id = 1"
        ).fetchone()
        needs_embedding_rebuild = (
            indexed is None
            or indexed[0] != _EMBEDDING_MODEL_NAME
            or indexed[2] < _EMBEDDING_CONTENT_VERSION
        )
        needs_index_upgrade = (
            indexed is None
            or indexed[1] != 1
            or indexed[2] != _EMBEDDING_INDEX_VERSION
        )
        if needs_index_upgrade:
            _ann_indexes.pop(str(Path(database_path).resolve()), None)
            connection.execute("DELETE FROM chat_ann_changes")
            connection.execute("DELETE FROM chat_ann_checkpoint")
            revision = indexed[3] + 1 if indexed else 1
            if needs_embedding_rebuild:
                connection.execute("DELETE FROM chat_message_embeddings")
                for conversation_id, serialized_messages in connection.execute(
                    "SELECT id, messages FROM conversations"
                ):
                    messages = json.loads(serialized_messages)
                    _index_messages(
                        connection,
                        conversation_id,
                        messages,
                        record_ann_changes=False,
                    )
            else:
                for vector_id, embedding in connection.execute(
                    "SELECT id, embedding FROM chat_message_embeddings"
                ).fetchall():
                    if isinstance(embedding, str):
                        connection.execute(
                            "UPDATE chat_message_embeddings SET embedding = ? WHERE id = ?",
                            (
                                _serialize_embedding(json.loads(embedding)),
                                vector_id,
                            ),
                        )
            connection.execute(
                """
                INSERT OR REPLACE INTO chat_embedding_state
                    (id, model, indexed, index_version, revision)
                VALUES (1, ?, 1, ?, ?)
                """,
                (_EMBEDDING_MODEL_NAME, _EMBEDDING_INDEX_VERSION, revision),
            )
    _get_ann_index(database_path)


def get_all_user_messages(database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        conversations = connection.execute(
            """
            SELECT title, messages
            FROM conversations
            ORDER BY rowid
            """
        ).fetchall()

    user_messages = []
    for title, serialized_messages in conversations:
        for message in json.loads(serialized_messages):
            content = message.get("content")
            if (
                message.get("role") == "user"
                and isinstance(content, str)
                and content.strip()
            ):
                user_messages.append((title, content.strip()))
    return user_messages


def create_conversation(database_path=DATABASE_PATH):
    conversation_id = uuid.uuid4().hex
    with _connect(database_path) as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, messages) VALUES (?, ?, ?)",
            (conversation_id, "New chat", "[]"),
        )
    return conversation_id


def save_conversation(conversation_id, messages, database_path=DATABASE_PATH):
    first_user_message = next(
        (
            message["content"].strip()
            for message in messages
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        "New chat",
    )
    title = (
        first_user_message[:57] + "..."
        if len(first_user_message) > 60
        else first_user_message
    )
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, messages = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title, json.dumps(messages), conversation_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Conversation not found: {conversation_id}")
        connection.execute(
            """
            UPDATE conversation_summaries SET stale = 1
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        _index_messages(connection, conversation_id, messages)
    _get_ann_index(database_path)


def _index_messages(
    connection,
    conversation_id,
    messages,
    *,
    record_ann_changes=True,
):
    existing_embeddings = {}
    for (
        vector_id,
        message_index,
        chunk_index,
        role,
        content,
        embedding,
    ) in connection.execute(
        """
        SELECT id, message_index, chunk_index, role, content, embedding
        FROM chat_message_embeddings
        WHERE conversation_id = ?
        """,
        (conversation_id,),
    ):
        existing_embeddings[(message_index, chunk_index, role, content)] = (
            vector_id,
            embedding,
        )

    chunks = [
        (message_index, chunk_index, message["role"], chunk)
        for message_index, message in enumerate(messages)
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
        for chunk_index, chunk in enumerate(_split_text(message["content"].strip()))
    ]
    pending = []
    for message_index, chunk_index, role, content in chunks:
        key = (message_index, chunk_index, role, content)
        if existing_embeddings.pop(key, None) is None:
            pending.append((message_index, chunk_index, role, content))

    new_vectors = (
        _embed_texts([content for _, _, _, content in pending]) if pending else []
    )
    new_embeddings = [
        (
            message_index,
            chunk_index,
            role,
            content,
            _serialize_embedding(vector),
        )
        for (message_index, chunk_index, role, content), vector in zip(
            pending, new_vectors, strict=True
        )
    ]
    removed_ids = [value[0] for value in existing_embeddings.values()]
    for vector_id in removed_ids:
        connection.execute(
            "DELETE FROM chat_message_embeddings WHERE id = ?", (vector_id,)
        )

    inserted_embeddings = []
    for message_index, chunk_index, role, content, embedding in new_embeddings:
        cursor = connection.execute(
            """
            INSERT INTO chat_message_embeddings
                (conversation_id, message_index, chunk_index, role, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                message_index,
                chunk_index,
                role,
                content,
                embedding,
            ),
        )
        inserted_embeddings.append((cursor.lastrowid, embedding))

    if record_ann_changes and (removed_ids or inserted_embeddings):
        revision = connection.execute(
            "SELECT revision FROM chat_embedding_state WHERE id = 1"
        ).fetchone()[0] + 1
        changes = [
            (revision, "remove", vector_id, None) for vector_id in removed_ids
        ]
        changes.extend(
            (revision, "add", vector_id, embedding)
            for vector_id, embedding in inserted_embeddings
        )
        connection.executemany(
            """
            INSERT INTO chat_ann_changes
                (revision, operation, vector_id, embedding)
            VALUES (?, ?, ?, ?)
            """,
            changes,
        )
        connection.execute(
            "UPDATE chat_embedding_state SET revision = ? WHERE id = 1",
            (revision,),
        )


def add_memory_claim(
    content,
    category,
    conversation_id,
    message_index,
    *,
    confidence=0.5,
    supersedes_id=None,
    database_path=DATABASE_PATH,
):
    content = content.strip()
    category = category.strip()
    if not content or not category:
        raise ValueError("Memory content and category must not be empty.")
    if not 0 <= confidence <= 1:
        raise ValueError("Memory confidence must be between 0 and 1.")

    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT title, messages FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        messages = json.loads(row[1])
        if not isinstance(message_index, int) or not 0 <= message_index < len(messages):
            raise ValueError("Memory source message does not exist.")
        source_message = messages[message_index]
        if source_message.get("role") != "user":
            raise ValueError("Memories must cite a user-authored source message.")
        source_excerpt = source_message.get("content")
        if not isinstance(source_excerpt, str) or not source_excerpt.strip():
            raise ValueError("Memory source message has no text.")
        if supersedes_id is not None:
            prior_claim = connection.execute(
                "SELECT category, status FROM memory_claims WHERE id = ?",
                (supersedes_id,),
            ).fetchone()
            if prior_claim is None or prior_claim[1] not in {
                "candidate",
                "confirmed",
                "disputed",
            }:
                raise ValueError(
                    f"Memory claim not found or not active: {supersedes_id}"
                )
            if prior_claim[0].casefold() != category.casefold():
                raise ValueError("A replacement memory must use the same category.")
        if supersedes_id is None:
            existing_claim = connection.execute(
                """
                SELECT id FROM memory_claims
                WHERE source_conversation_id = ?
                  AND source_message_index = ?
                  AND lower(content) = lower(?)
                  AND lower(category) = lower(?)
                  AND status != 'superseded'
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id, message_index, content, category),
            ).fetchone()
            if existing_claim is not None:
                return existing_claim[0]

        cursor = connection.execute(
            """
            INSERT INTO memory_claims
                (content, category, confidence, source_conversation_id,
                 source_message_index, source_excerpt, supersedes_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content,
                category,
                confidence,
                conversation_id,
                message_index,
                source_excerpt.strip(),
                supersedes_id,
            ),
        )
        _detect_memory_conflicts(connection, cursor.lastrowid)
        return cursor.lastrowid


_CONTRADICTORY_TERM_PAIRS = (
    ({"short", "concise", "brief"}, {"long", "detailed", "verbose"}),
    ({"early", "morning"}, {"late", "evening"}),
    ({"formal"}, {"casual", "informal"}),
)
_CONFLICT_STOP_WORDS = {
    "a", "an", "and", "are", "about", "be", "for", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "the", "this", "to", "user", "very",
}


def _detect_memory_conflicts(connection, claim_id):
    claim = connection.execute(
        """
        SELECT content, category, status FROM memory_claims WHERE id = ?
        """,
        (claim_id,),
    ).fetchone()
    if claim is None:
        raise ValueError(f"Memory claim not found: {claim_id}")
    content, category, status = claim
    if status not in {"candidate", "confirmed", "disputed"}:
        return
    tokens = set(re.findall(r"\w+", content.casefold())) - _CONFLICT_STOP_WORDS
    negated = bool(
        re.search(r"\b(?:not|never|don't|doesn't|dislike|hate|avoid)\b", content.casefold())
    )
    existing = connection.execute(
        """
        SELECT id, content FROM memory_claims
        WHERE lower(category) = lower(?)
          AND id != ?
          AND status IN ('candidate', 'confirmed', 'disputed')
        """,
        (category, claim_id),
    )
    for other_id, other_content in existing:
        other_tokens = (
            set(re.findall(r"\w+", other_content.casefold())) - _CONFLICT_STOP_WORDS
        )
        shared = tokens & other_tokens
        other_negated = bool(
            re.search(
                r"\b(?:not|never|don't|doesn't|dislike|hate|avoid)\b",
                other_content.casefold(),
            )
        )
        opposites = any(
            (bool(tokens & first) and bool(other_tokens & second))
            or (bool(tokens & second) and bool(other_tokens & first))
            for first, second in _CONTRADICTORY_TERM_PAIRS
        )
        negated_contradiction = negated != other_negated and len(shared) >= 2
        if opposites or negated_contradiction:
            low_id, high_id = sorted((claim_id, other_id))
            reason = (
                "The claims use potentially contradictory preference terms."
                if opposites
                else "One claim negates a similar statement."
            )
            connection.execute(
                """
                INSERT INTO memory_conflicts
                    (claim_a_id, claim_b_id, reason)
                VALUES (?, ?, ?)
                ON CONFLICT(claim_a_id, claim_b_id)
                DO UPDATE SET status = 'open', reason = excluded.reason
                """,
                (low_id, high_id, reason),
            )


def list_memory_conflicts(database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT memory_conflicts.id, memory_conflicts.claim_a_id,
                   claim_a.content, claim_a.status, memory_conflicts.claim_b_id,
                   claim_b.content, claim_b.status, memory_conflicts.reason,
                   memory_conflicts.status
            FROM memory_conflicts
            JOIN memory_claims AS claim_a
              ON claim_a.id = memory_conflicts.claim_a_id
            JOIN memory_claims AS claim_b
              ON claim_b.id = memory_conflicts.claim_b_id
            WHERE memory_conflicts.status = 'open'
            ORDER BY memory_conflicts.created_at DESC
            """
        ).fetchall()
    return [
        {
            "id": row[0],
            "claim_a_id": row[1],
            "claim_a": row[2],
            "claim_a_status": row[3],
            "claim_b_id": row[4],
            "claim_b": row[5],
            "claim_b_status": row[6],
            "reason": row[7],
            "status": row[8],
        }
        for row in rows
    ]


def resolve_memory_conflict(conflict_id, status, database_path=DATABASE_PATH):
    if status not in {"resolved", "dismissed"}:
        raise ValueError(f"Unsupported memory conflict status: {status!r}")
    with _connect(database_path) as connection:
        cursor = connection.execute(
            "UPDATE memory_conflicts SET status = ? WHERE id = ? AND status = 'open'",
            (status, conflict_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Open memory conflict not found: {conflict_id}")


def link_memory_claims(
    source_claim_id,
    target_claim_id,
    relation_type,
    database_path=DATABASE_PATH,
):
    relation_type = relation_type.strip()
    if not relation_type:
        raise ValueError("Memory relation type must not be empty.")
    if source_claim_id == target_claim_id:
        raise ValueError("A memory cannot be related to itself.")
    with _connect(database_path) as connection:
        claims = connection.execute(
            """
            SELECT id, status FROM memory_claims
            WHERE id IN (?, ?)
            """,
            (source_claim_id, target_claim_id),
        ).fetchall()
        if len(claims) != 2 or any(status != "confirmed" for _, status in claims):
            raise ValueError("Relations require two confirmed memory claims.")
        connection.execute(
            """
            INSERT OR IGNORE INTO memory_relations
                (source_claim_id, target_claim_id, relation_type)
            VALUES (?, ?, ?)
            """,
            (source_claim_id, target_claim_id, relation_type),
        )


def list_memory_relations(database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        return connection.execute(
            """
            SELECT memory_relations.source_claim_id, source.content,
                   memory_relations.relation_type, memory_relations.target_claim_id,
                   target.content
            FROM memory_relations
            JOIN memory_claims AS source
              ON source.id = memory_relations.source_claim_id
            JOIN memory_claims AS target
              ON target.id = memory_relations.target_claim_id
            ORDER BY memory_relations.created_at DESC
            """
        ).fetchall()


def list_memory_claims(database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT memory_claims.id, memory_claims.content, memory_claims.category,
                   memory_claims.status, memory_claims.confidence,
                   memory_claims.source_conversation_id,
                   memory_claims.source_message_index, memory_claims.source_excerpt,
                   memory_claims.valid_from, memory_claims.valid_until,
                   memory_claims.supersedes_id, conversations.title
            FROM memory_claims
            JOIN conversations
              ON conversations.id = memory_claims.source_conversation_id
            ORDER BY
                CASE memory_claims.status
                    WHEN 'candidate' THEN 0
                    WHEN 'confirmed' THEN 1
                    WHEN 'disputed' THEN 2
                    ELSE 3
                END,
                memory_claims.updated_at DESC, memory_claims.id DESC
            """
        ).fetchall()
    return [
        {
            "id": row[0],
            "content": row[1],
            "category": row[2],
            "status": row[3],
            "confidence": row[4],
            "source_conversation_id": row[5],
            "source_message_index": row[6],
            "source_excerpt": row[7],
            "valid_from": row[8],
            "valid_until": row[9],
            "supersedes_id": row[10],
            "source_title": row[11],
        }
        for row in rows
    ]


def save_conversation_summary(conversation_id, summary, database_path=DATABASE_PATH):
    if not summary.strip():
        raise ValueError("Conversation summary must not be empty.")
    with _connect(database_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        connection.execute(
            """
            INSERT INTO conversation_summaries
                (conversation_id, summary, stale, updated_at)
            VALUES (?, ?, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary = excluded.summary,
                stale = 0,
                updated_at = CURRENT_TIMESTAMP
            """,
            (conversation_id, summary.strip()),
        )


def list_conversation_summaries(database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT conversations.id, conversations.title,
                   conversation_summaries.summary,
                   conversation_summaries.stale,
                   conversation_summaries.updated_at
            FROM conversation_summaries
            JOIN conversations
              ON conversations.id = conversation_summaries.conversation_id
            ORDER BY conversations.updated_at DESC
            """
        ).fetchall()
    return [
        {
            "conversation_id": row[0],
            "title": row[1],
            "summary": row[2],
            "stale": bool(row[3]),
            "updated_at": row[4],
        }
        for row in rows
    ]


def update_memory_claim(
    claim_id, content, category, status, database_path=DATABASE_PATH
):
    if status not in {"candidate", "confirmed", "superseded", "disputed"}:
        raise ValueError(f"Unsupported memory status: {status!r}")
    if not content.strip() or not category.strip():
        raise ValueError("Memory content and category must not be empty.")
    with _connect(database_path) as connection:
        claim = connection.execute(
            "SELECT supersedes_id FROM memory_claims WHERE id = ?",
            (claim_id,),
        ).fetchone()
        if claim is None:
            raise ValueError(f"Memory claim not found: {claim_id}")
        supersedes_id = claim[0]
        if status == "confirmed" and supersedes_id is not None:
            cursor = connection.execute(
                """
                UPDATE memory_claims
                SET status = 'superseded',
                    valid_until = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status IN ('candidate', 'confirmed', 'disputed')
                """,
                (supersedes_id,),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"Memory claim not found or not active: {supersedes_id}"
                )
            connection.execute(
                "DELETE FROM memory_claim_embeddings WHERE claim_id = ?",
                (supersedes_id,),
            )
            connection.execute(
                """
                UPDATE memory_conflicts SET status = 'resolved'
                WHERE status = 'open'
                  AND (
                    (claim_a_id = ? AND claim_b_id = ?)
                    OR (claim_a_id = ? AND claim_b_id = ?)
                  )
                """,
                (supersedes_id, claim_id, claim_id, supersedes_id),
            )
        cursor = connection.execute(
            """
            UPDATE memory_claims
            SET content = ?, category = ?, status = ?,
                confidence = CASE
                    WHEN ? = 'confirmed' THEN MAX(confidence, 0.9)
                    ELSE confidence
                END,
                valid_until = CASE
                    WHEN ? IN ('superseded', 'disputed')
                    THEN COALESCE(valid_until, CURRENT_TIMESTAMP)
                    ELSE NULL
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                content.strip(),
                category.strip(),
                status,
                status,
                status,
                claim_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Memory claim not found: {claim_id}")
        connection.execute(
            "DELETE FROM memory_claim_embeddings WHERE claim_id = ?",
            (claim_id,),
        )
        if status == "confirmed":
            embedding = _serialize_embedding(_embed_texts([content.strip()])[0])
            connection.execute(
                """
                INSERT INTO memory_claim_embeddings (claim_id, embedding)
                VALUES (?, ?)
                """,
                (claim_id, embedding),
            )
        connection.execute(
            """
            DELETE FROM memory_conflicts
            WHERE (claim_a_id = ? OR claim_b_id = ?) AND status = 'open'
            """,
            (claim_id, claim_id),
        )
        _detect_memory_conflicts(connection, claim_id)


def delete_memory_claim(claim_id, database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        connection.execute(
            "UPDATE memory_claims SET supersedes_id = NULL WHERE supersedes_id = ?",
            (claim_id,),
        )
        connection.execute(
            """
            DELETE FROM memory_relations
            WHERE source_claim_id = ? OR target_claim_id = ?
            """,
            (claim_id, claim_id),
        )
        connection.execute(
            "DELETE FROM memory_conflicts WHERE claim_a_id = ? OR claim_b_id = ?",
            (claim_id, claim_id),
        )
        connection.execute(
            "DELETE FROM memory_claim_embeddings WHERE claim_id = ?",
            (claim_id,),
        )
        cursor = connection.execute("DELETE FROM memory_claims WHERE id = ?", (claim_id,))
        if cursor.rowcount != 1:
            raise ValueError(f"Memory claim not found: {claim_id}")


def delete_conversation(conversation_id, database_path=DATABASE_PATH):
    cache_key = str(Path(database_path).resolve())
    with _connect(database_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        connection.execute(
            """
            UPDATE memory_claims SET supersedes_id = NULL
            WHERE supersedes_id IN (
                SELECT id FROM memory_claims WHERE source_conversation_id = ?
            )
            """,
            (conversation_id,),
        )
        connection.execute(
            """
            DELETE FROM memory_relations
            WHERE source_claim_id IN (
                SELECT id FROM memory_claims WHERE source_conversation_id = ?
            ) OR target_claim_id IN (
                SELECT id FROM memory_claims WHERE source_conversation_id = ?
            )
            """,
            (conversation_id, conversation_id),
        )
        connection.execute(
            """
            DELETE FROM memory_conflicts
            WHERE claim_a_id IN (
                SELECT id FROM memory_claims WHERE source_conversation_id = ?
            ) OR claim_b_id IN (
                SELECT id FROM memory_claims WHERE source_conversation_id = ?
            )
            """,
            (conversation_id, conversation_id),
        )
        connection.execute(
            """
            DELETE FROM memory_claim_embeddings
            WHERE claim_id IN (
                SELECT id FROM memory_claims WHERE source_conversation_id = ?
            )
            """,
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM memory_claims WHERE source_conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM conversation_summaries WHERE conversation_id = ?",
            (conversation_id,),
        )
        _index_messages(connection, conversation_id, [])
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    _ann_indexes.pop(cache_key, None)
    _get_ann_index(database_path)


def _fts_query(query):
    words = []
    for word in re.findall(r"\w+", query.casefold()):
        if len(word) > 1 and word not in words:
            words.append(word)
    return " OR ".join(f'"{word}"' for word in words[:32])


def _vector_rank(query_embedding, index, candidate_count, database_path):
    vector = np.asarray(query_embedding, dtype=np.float32)
    if index.size <= _EXACT_SEARCH_MAX_VECTORS:
        with _connect(database_path) as connection:
            rows = connection.execute(
                "SELECT id, embedding FROM chat_message_embeddings"
            ).fetchall()
        scored = [
            (int(vector_id), float(np.dot(vector, _deserialize_embedding(embedding))))
            for vector_id, embedding in rows
        ]
        return [
            vector_id
            for vector_id, similarity in sorted(
                scored, key=lambda result: result[1], reverse=True
            )[:candidate_count]
            if similarity >= _MIN_SIMILARITY
        ]

    matches = index.search(vector, count=candidate_count)
    return [
        int(vector_id)
        for vector_id, distance in zip(
            matches.keys.tolist(),
            matches.distances.tolist(),
            strict=True,
        )
        if vector_id != 0 and 1.0 - float(distance) >= _MIN_SIMILARITY
    ]


def _rank_confirmed_claims(query_embedding, candidate_count, database_path):
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT memory_claims.id, memory_claim_embeddings.embedding
            FROM memory_claims
            JOIN memory_claim_embeddings
              ON memory_claim_embeddings.claim_id = memory_claims.id
            WHERE memory_claims.status = 'confirmed'
            """
        ).fetchall()
    scored = [
        (
            -int(claim_id),
            float(np.dot(query_embedding, _deserialize_embedding(embedding))),
        )
        for claim_id, embedding in rows
    ]
    return [
        claim_id
        for claim_id, similarity in sorted(
            scored, key=lambda result: result[1], reverse=True
        )[:candidate_count]
        if similarity >= _MIN_SIMILARITY
    ]


def _feedback_adjustment(query_hash, record_key, database_path):
    with _connect(database_path) as connection:
        helpful, total = connection.execute(
            """
            SELECT COALESCE(SUM(retrieval_metric_results.feedback), 0),
                   COUNT(retrieval_metric_results.feedback)
            FROM retrieval_metric_results
            JOIN retrieval_metrics
              ON retrieval_metrics.id = retrieval_metric_results.event_id
            WHERE retrieval_metrics.query_hash = ?
              AND retrieval_metric_results.record_key = ?
              AND retrieval_metric_results.feedback IS NOT NULL
            """,
            (query_hash, record_key),
        ).fetchone()
    if total < 3:
        return 1.0
    posterior = (2 * int(helpful) - int(total)) / (int(total) + 4)
    return 1.0 + 0.4 * posterior


def _expand_related_claims(results, database_path, *, max_hops=2, max_nodes=20):
    seed_claims = {
        int(result["feedback_key"].split(":", 1)[1]): (score, result)
        for score, result in results
        if result["kind"] == "claim"
    }
    known_ids = set(seed_claims)
    frontier = set(seed_claims)
    expanded = []
    for depth in range(1, max_hops + 1):
        if not frontier or len(known_ids) >= max_nodes:
            break
        placeholders = ", ".join("?" for _ in frontier)
        frontier_ids = tuple(frontier)
        with _connect(database_path) as connection:
            rows = connection.execute(
                """
                SELECT relation.source_claim_id, relation.target_claim_id,
                       relation.relation_type, claim.id, claim.content,
                       claim.category, claim.source_excerpt,
                       claim.source_conversation_id, claim.source_message_index,
                       claim.valid_from, conversations.title
                FROM memory_relations AS relation
                JOIN memory_claims AS claim
                  ON claim.id = CASE
                      WHEN relation.source_claim_id IN ({placeholders})
                      THEN relation.target_claim_id
                      ELSE relation.source_claim_id
                  END
                JOIN conversations
                  ON conversations.id = claim.source_conversation_id
                WHERE (
                    relation.source_claim_id IN ({placeholders})
                    OR relation.target_claim_id IN ({placeholders})
                )
                  AND claim.status = 'confirmed'
                LIMIT ?
                """.format(placeholders=placeholders),
                frontier_ids * 3 + (max_nodes,),
            ).fetchall()
        next_frontier = set()
        for (
            source_id, target_id, relation_type, claim_id, content, category,
            source_excerpt, conversation_id, message_index, valid_from, title,
        ) in rows:
            if claim_id in known_ids or len(known_ids) >= max_nodes:
                continue
            connected_id = source_id if target_id == claim_id else target_id
            parent = next(
                (
                    (score, result)
                    for score, result in expanded
                    if result["id"] == connected_id
                ),
                seed_claims.get(connected_id),
            )
            if parent is None:
                continue
            parent_score, _ = parent
            related = {
                "kind": "related_claim",
                "id": claim_id,
                "content": content,
                "category": category,
                "title": title,
                "role": "memory",
                "source_conversation_id": conversation_id,
                "source_message_index": message_index,
                "source_excerpt": source_excerpt,
                "date": valid_from,
                "relation_type": relation_type,
                "relation_depth": depth,
                "feedback_key": f"claim:{claim_id}",
            }
            known_ids.add(claim_id)
            next_frontier.add(claim_id)
            expanded.append((parent_score * (0.65**depth), related))
        frontier = next_frontier
    return expanded


def search_memory(
    query,
    history=None,
    limit=5,
    database_path=DATABASE_PATH,
    *,
    include_event_id=False,
    include_feedback_choices=False,
    retrieval_mode="hybrid",
    use_feedback=True,
):
    started_at = time.perf_counter()
    if not query.strip() or limit <= 0:
        if include_event_id and include_feedback_choices:
            return [], None, []
        if include_event_id:
            return [], None
        if include_feedback_choices:
            return [], []
        return []
    if retrieval_mode not in {"hybrid", "semantic", "keyword"}:
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode!r}")

    index, _ = _get_ann_index(database_path)
    with _connect(database_path) as connection:
        claim_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_claims
            JOIN memory_claim_embeddings
              ON memory_claim_embeddings.claim_id = memory_claims.id
            WHERE memory_claims.status = 'confirmed'
            """
        ).fetchone()[0]
    candidate_count = min(
        max(index.size, claim_count, 1), max(64, limit * 8), 500
    )
    retrieval_queries = [query.strip()]
    recent_context = []
    for turn in (history or [])[-4:]:
        if isinstance(turn, dict):
            role, content = turn.get("role"), turn.get("content")
            if role == "user" and isinstance(content, str) and content.strip():
                recent_context.append(content.strip())
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_message = turn[0]
            if isinstance(user_message, str) and user_message.strip():
                recent_context.append(user_message.strip())
    if recent_context:
        contextual_query = " ".join(recent_context[-2:] + [query.strip()])
        if contextual_query.casefold() != query.strip().casefold():
            retrieval_queries.append(contextual_query)

    rankings = []
    fts_query = _fts_query(query) if retrieval_mode != "semantic" else ""
    summary_conversations = []
    if fts_query:
        with _connect(database_path) as connection:
            summary_conversations = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT conversation_summaries.conversation_id
                    FROM conversation_summary_fts
                    JOIN conversation_summaries
                      ON conversation_summaries.id = conversation_summary_fts.rowid
                    WHERE conversation_summary_fts MATCH ?
                      AND conversation_summaries.stale = 0
                    ORDER BY rank
                    LIMIT 3
                    """,
                    (fts_query,),
                )
            ]
    if retrieval_mode != "keyword" and (index.size or claim_count):
        embeddings = _embed_texts([_QUERY_PREFIX + text for text in retrieval_queries])
        for embedding in embeddings:
            normalized = _normalize_embedding(embedding)
            if not any(normalized):
                continue
            if index.size:
                rankings.append(
                    (
                        "semantic",
                        _vector_rank(
                            normalized, index, candidate_count, database_path
                        ),
                    )
                )
            if retrieval_mode == "hybrid":
                rankings.append(
                    (
                        "semantic_claim",
                        _rank_confirmed_claims(
                            np.asarray(normalized, dtype=np.float32),
                            candidate_count,
                            database_path,
                        ),
                    )
                )
            if summary_conversations and index.size:
                placeholders = ", ".join("?" for _ in summary_conversations)
                with _connect(database_path) as connection:
                    summary_vectors = connection.execute(
                        """
                        SELECT id, embedding
                        FROM chat_message_embeddings
                        WHERE conversation_id IN ({placeholders})
                        """.format(placeholders=placeholders),
                        tuple(summary_conversations),
                    ).fetchall()
                summary_rank = sorted(
                    (
                        (
                            int(vector_id),
                            float(
                                np.dot(
                                    np.asarray(normalized, dtype=np.float32),
                                    _deserialize_embedding(serialized_embedding),
                                )
                            ),
                        )
                        for vector_id, serialized_embedding in summary_vectors
                    ),
                    key=lambda result: result[1],
                    reverse=True,
                )
                rankings.append(
                    (
                        "summary",
                        [
                            vector_id
                            for vector_id, similarity in summary_rank[:candidate_count]
                            if similarity >= _MIN_SIMILARITY
                        ],
                    )
                )

    if fts_query:
        with _connect(database_path) as connection:
            keyword_ids = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT rowid
                    FROM chat_memory_fts
                    WHERE chat_memory_fts MATCH ?
                      AND (
                          rowid > 0 OR EXISTS (
                              SELECT 1
                              FROM memory_claims
                              WHERE memory_claims.id = -chat_memory_fts.rowid
                                AND memory_claims.status = 'confirmed'
                          )
                      )
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, candidate_count),
                )
            ]
        if keyword_ids:
            rankings.append(("keyword", keyword_ids))

    fused_scores = {}
    has_named_term = any(
        match.start() > 0
        for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", query)
    )
    lexical_boost = 2 if (
        '"' in query
        or re.search(r"\d", query)
        or has_named_term
        or re.search(r"\b[A-Z]{2,}\b", query)
    ) else 1
    for ranking_type, ranking in rankings:
        seen = set()
        weight = lexical_boost if ranking_type == "keyword" else 1
        for rank, record_id in enumerate(ranking, start=1):
            if record_id not in seen:
                fused_scores[record_id] = (
                    fused_scores.get(record_id, 0.0)
                    + weight / (_RRF_K + rank)
                )
                seen.add(record_id)
    message_ids = [record_id for record_id in fused_scores if record_id > 0]
    claim_ids = [-record_id for record_id in fused_scores if record_id < 0]
    results = []
    with _connect(database_path) as connection:
        if message_ids:
            placeholders = ", ".join("?" for _ in message_ids)
            rows = connection.execute(
                """
                SELECT chat_message_embeddings.id,
                       conversations.title, chat_message_embeddings.role,
                       chat_message_embeddings.content,
                       chat_message_embeddings.conversation_id,
                       chat_message_embeddings.message_index,
                       conversations.updated_at
                FROM chat_message_embeddings
                JOIN conversations
                  ON conversations.id = chat_message_embeddings.conversation_id
                WHERE chat_message_embeddings.id IN ({placeholders})
                """.format(placeholders=placeholders),
                tuple(message_ids),
            ).fetchall()
            best_by_message = {}
            for vector_id, title, role, content, source_id, message_index, updated in rows:
                key = (source_id, message_index)
                result = {
                    "kind": "message",
                    "content": content,
                    "title": title,
                    "role": role,
                    "source_conversation_id": source_id,
                    "source_message_index": message_index,
                    "source_excerpt": content,
                    "date": updated,
                    "feedback_key": f"message:{vector_id}",
                }
                if key not in best_by_message or fused_scores[vector_id] > best_by_message[key][0]:
                    best_by_message[key] = (fused_scores[vector_id], result)
            results.extend(best_by_message.values())

        if claim_ids:
            placeholders = ", ".join("?" for _ in claim_ids)
            rows = connection.execute(
                """
                SELECT memory_claims.id, memory_claims.content,
                       memory_claims.category, memory_claims.source_excerpt,
                       memory_claims.source_conversation_id,
                       memory_claims.source_message_index,
                       memory_claims.valid_from, conversations.title
                FROM memory_claims
                JOIN conversations
                  ON conversations.id = memory_claims.source_conversation_id
                WHERE memory_claims.id IN ({placeholders})
                  AND memory_claims.status = 'confirmed'
                """.format(placeholders=placeholders),
                tuple(claim_ids),
            ).fetchall()
            for (
                claim_id, content, category, source_excerpt, source_id,
                message_index, valid_from, title,
            ) in rows:
                results.append(
                    (
                        fused_scores[-claim_id],
                        {
                            "kind": "claim",
                            "content": content,
                            "category": category,
                            "title": title,
                            "role": "memory",
                            "source_conversation_id": source_id,
                            "source_message_index": message_index,
                            "source_excerpt": source_excerpt,
                            "date": valid_from,
                            "id": claim_id,
                            "feedback_key": f"claim:{claim_id}",
                        },
                    )
                )
    results.extend(_expand_related_claims(results, database_path))
    query_hash = hashlib.sha256(
        " ".join(query.casefold().split()).encode("utf-8")
    ).hexdigest()
    results = [
        (
            score
            * (
                _feedback_adjustment(
                    query_hash, result["feedback_key"], database_path
                )
                if use_feedback
                else 1.0
            ),
            result,
        )
        for score, result in results
    ]
    results.sort(key=lambda result: result[0], reverse=True)
    selected_results = [result for _, result in results[:limit]]
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO retrieval_metrics
                (elapsed_ms, index_size, strategy, lexical_weight, result_count,
                 query_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (time.perf_counter() - started_at) * 1000,
                index.size,
                "keyword"
                if retrieval_mode == "keyword"
                else (
                    "exact"
                    if index.size <= _EXACT_SEARCH_MAX_VECTORS
                    else "hnsw"
                ),
                lexical_boost,
                len(selected_results),
                query_hash,
            ),
        )
        event_id = cursor.lastrowid
        connection.executemany(
            """
            INSERT INTO retrieval_metric_results (event_id, record_key, rank)
            VALUES (?, ?, ?)
            """,
            [
                (event_id, result["feedback_key"], rank)
                for rank, result in enumerate(selected_results, start=1)
            ],
        )
    feedback_choices = [
        (
            f"{result['kind']}: {result['title']} - {result['content'][:80]}",
            result["feedback_key"],
        )
        for result in selected_results
    ]
    if include_event_id:
        if include_feedback_choices:
            return selected_results, event_id, feedback_choices
        return selected_results, event_id
    if include_feedback_choices:
        return selected_results, feedback_choices
    return selected_results


def record_retrieval_feedback(
    event_id, record_key, helpful, database_path=DATABASE_PATH
):
    if not isinstance(helpful, bool):
        raise ValueError("Retrieval feedback must be a boolean.")
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE retrieval_metric_results
            SET feedback = ?
            WHERE event_id = ? AND record_key = ?
            """,
            (int(helpful), event_id, record_key),
        )
        if cursor.rowcount != 1:
            raise ValueError(
                f"Retrieval result not found: event={event_id}, record={record_key}"
            )


def evaluate_retrieval(
    cases,
    *,
    limit=5,
    database_path=DATABASE_PATH,
    modes=("semantic", "hybrid"),
):
    if limit <= 0:
        raise ValueError("Evaluation result limit must be positive.")
    if not cases:
        raise ValueError("Provide at least one labeled retrieval case.")
    if any(mode not in {"semantic", "keyword", "hybrid"} for mode in modes):
        raise ValueError("Evaluation modes must be semantic, keyword, or hybrid.")

    report = {}
    for mode in modes:
        recalls = []
        reciprocal_ranks = []
        latencies = []
        for case in cases:
            query = case.get("query")
            relevant_sources = case.get("relevant_sources")
            if (
                not isinstance(query, str)
                or not query.strip()
                or not isinstance(relevant_sources, (list, tuple, set))
                or not relevant_sources
            ):
                raise ValueError(
                    "Each evaluation case needs a non-empty query and relevant_sources."
                )
            if any(
                not isinstance(source, (list, tuple))
                or len(source) != 2
                or not isinstance(source[0], str)
                or not isinstance(source[1], int)
                or isinstance(source[1], bool)
                for source in relevant_sources
            ):
                raise ValueError(
                    "Relevant sources must be (conversation_id, message_index) pairs."
                )
            expected = {tuple(source) for source in relevant_sources}
            started = time.perf_counter()
            results = search_memory(
                query,
                limit=limit,
                database_path=database_path,
                retrieval_mode=mode,
                use_feedback=False,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            ranked_sources = [
                (result["source_conversation_id"], result["source_message_index"])
                for result in results
            ]
            hits = [source in expected for source in ranked_sources]
            recalls.append(
                len(set(ranked_sources) & expected) / len(expected)
            )
            reciprocal_ranks.append(
                next(
                    (1 / rank for rank, hit in enumerate(hits, start=1) if hit),
                    0.0,
                )
            )
        report[mode] = {
            "case_count": len(cases),
            f"recall_at_{limit}": sum(recalls) / len(recalls),
            "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "mean_latency_ms": sum(latencies) / len(latencies),
            "latencies_ms": latencies,
        }
    return report


def search_past_chats(query, limit=5, database_path=DATABASE_PATH):
    return [
        (result["title"], result["role"], result["content"])
        for result in search_memory(query, limit=limit, database_path=database_path)
        if result["kind"] == "message"
    ]


def list_conversations(database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, title, updated_at
            FROM conversations
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()
    return [
        (f"{title} ({updated_at})", conversation_id)
        for conversation_id, title, updated_at in rows
    ]


def load_conversation(conversation_id, database_path=DATABASE_PATH):
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT messages FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"Conversation not found: {conversation_id}")
    return json.loads(row[0])
