import logging
from collections.abc import AsyncIterator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if settings.is_sqlite else {},
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
logger = logging.getLogger(__name__)


def _sqlite_functions(dbapi_connection, _connection_record=None) -> None:
    from app.services.rag import cjk_index_text

    dbapi_connection.create_function("cjk_index_text", 1, cjk_index_text, deterministic=True)
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


if settings.is_sqlite:
    event.listen(engine.sync_engine, "connect", _sqlite_functions)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            from app.models import AuditLog

            # Release the failed business transaction before writing its denial.
            denied = [
                entry
                for entry in session.new
                if isinstance(entry, AuditLog) and entry.decision == "deny"
            ]
            await session.rollback()
            if denied:
                try:
                    session.add_all(denied)
                    await session.commit()
                except Exception:
                    logger.exception("Failed to persist denied-request audit")
                    await session.rollback()
            raise
        finally:
            from app.audit import DEFERRED_READ_AUDITS_KEY

            deferred = session.info.pop(DEFERRED_READ_AUDITS_KEY, [])
            if deferred:
                try:
                    await session.rollback()
                    async with SessionLocal() as audit_session:
                        audit_session.add_all(deferred)
                        await audit_session.commit()
                except Exception:
                    logger.exception("Failed to persist allowed-read audit")


async def init_db() -> None:
    from app import models  # noqa: F401

    async with engine.begin() as connection:
        if connection.dialect.name == "sqlite":
            await connection.execute(text("PRAGMA journal_mode=WAL"))
            # SQLite's legacy driver mode otherwise auto-commits DDL before index backfills.
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
        # pgvector is provided by the Compose image, but must be enabled in
        # each database before SQLAlchemy creates columns of type ``vector``.
        if connection.dialect.name == "postgresql":
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(Base.metadata.create_all)
        if connection.dialect.name == "postgresql":
            # create_all only installs indexes on new tables; existing deployments need this too.
            for index in Base.metadata.tables["document_chunks"].indexes:
                if index.name == "ix_document_chunks_lexical_gin":
                    await connection.run_sync(
                        lambda conn, idx=index: idx.create(conn, checkfirst=True)
                    )
        if connection.dialect.name == "sqlite":
            await init_sqlite_search(connection)


async def init_sqlite_search(connection) -> None:
    """Install FTS indexes and backfill existing chunks once on upgrade."""
    for suffix, tokenizer in (("fts", "unicode61"),):
        table = f"document_chunks_{suffix}"
        exists = await connection.scalar(
            text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :name"),
            {"name": table},
        )
        await connection.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5("
                f"content, content='document_chunks', content_rowid='rowid', tokenize='{tokenizer}')"
            )
        )
        await connection.execute(
            text(
                f"CREATE TRIGGER IF NOT EXISTS {table}_insert AFTER INSERT ON document_chunks BEGIN "
                f"INSERT INTO {table}(rowid, content) VALUES (new.rowid, new.content); END"
            )
        )
        await connection.execute(
            text(
                f"CREATE TRIGGER IF NOT EXISTS {table}_delete AFTER DELETE ON document_chunks BEGIN "
                f"INSERT INTO {table}({table}, rowid, content) "
                f"VALUES ('delete', old.rowid, old.content); END"
            )
        )
        await connection.execute(
            text(
                f"CREATE TRIGGER IF NOT EXISTS {table}_update AFTER UPDATE OF content ON document_chunks BEGIN "
                f"INSERT INTO {table}({table}, rowid, content) "
                f"VALUES ('delete', old.rowid, old.content); "
                f"INSERT INTO {table}(rowid, content) VALUES (new.rowid, new.content); END"
            )
        )
        if not exists:
            await connection.execute(text(f"INSERT INTO {table}({table}) VALUES ('rebuild')"))

    # Also register on externally supplied engines used by upgrades and tests.
    await connection.run_sync(lambda conn: _sqlite_functions(conn.connection.dbapi_connection))
    table = "document_chunks_cjk"
    exists = await connection.scalar(
        text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :name"), {"name": table}
    )
    await connection.execute(
        text(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5(content, tokenize='unicode61')"
        )
    )
    await connection.execute(
        text(
            f"CREATE TRIGGER IF NOT EXISTS {table}_insert AFTER INSERT ON document_chunks BEGIN "
            f"INSERT INTO {table}(rowid, content) VALUES (new.rowid, cjk_index_text(new.content)); END"
        )
    )
    await connection.execute(
        text(
            f"CREATE TRIGGER IF NOT EXISTS {table}_delete AFTER DELETE ON document_chunks BEGIN "
            f"DELETE FROM {table} WHERE rowid = old.rowid; END"
        )
    )
    await connection.execute(
        text(
            f"CREATE TRIGGER IF NOT EXISTS {table}_update AFTER UPDATE OF content ON document_chunks BEGIN "
            f"UPDATE {table} SET content = cjk_index_text(new.content) WHERE rowid = old.rowid; END"
        )
    )
    if not exists:
        await connection.execute(
            text(
                f"INSERT INTO {table}(rowid, content) SELECT rowid, cjk_index_text(content) FROM document_chunks"
            )
        )
    # The bigram index replaces the old trigram index; source chunks stay intact.
    for action in ("insert", "delete", "update"):
        await connection.execute(text(f"DROP TRIGGER IF EXISTS document_chunks_trigram_{action}"))
    await connection.execute(text("DROP TABLE IF EXISTS document_chunks_trigram"))
