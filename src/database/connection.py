"""PostgreSQL database connection pool and health check management.

Provides an asynchronous connection pool using asyncpg and SQLAlchemy async engine,
with resilient error recovery and health verification for the voice calling system.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

from src.config import Settings, get_settings
from src.logging import get_logger

logger = get_logger("database.connection")


class DatabasePool:
    """Manages asynchronous PostgreSQL connection pools and health status."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._pool: Optional[asyncpg.Pool] = None
        self._engine: Optional[AsyncEngine] = None
        self._locks: Dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

    def _get_lock(self) -> asyncio.Lock:
        """Return an asyncio.Lock tied to the current running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop not in self._locks:
            self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    @property
    def is_closed(self) -> bool:
        """True if the connection pool is not initialized or closed."""
        return self._pool is None or self._pool.is_closing()

    @property
    def dsn(self) -> str:
        """Return asyncpg-compatible DSN."""
        return self.settings.asyncpg_dsn

    @property
    def url(self) -> str:
        """Return SQLAlchemy-compatible async connection URL."""
        return self.settings.resolved_database_url

    async def get_pool(self) -> asyncpg.Pool:
        """Acquire or initialize the asyncpg connection pool."""
        current_loop = asyncio.get_running_loop()
        if self._pool is not None and not self._pool.is_closing():
            pool_loop = getattr(self._pool, "_loop", None)
            if pool_loop is current_loop and not pool_loop.is_closed():
                return self._pool
            else:
                self._pool = None

        lock = self._get_lock()
        async with lock:
            if self._pool is not None and not self._pool.is_closing():
                pool_loop = getattr(self._pool, "_loop", None)
                if pool_loop is current_loop and not pool_loop.is_closed():
                    return self._pool
                else:
                    self._pool = None

            logger.info(
                "Initializing asyncpg connection pool to PostgreSQL at %s:%s/%s",
                self.settings.postgres_host,
                self.settings.postgres_port,
                self.settings.postgres_db,
            )
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=2,
                max_size=10,
                command_timeout=10.0,
            )
        return self._pool

    async def get_engine(self) -> AsyncEngine:
        """Acquire or initialize the SQLAlchemy async engine."""
        if self._engine is not None:
            return self._engine

        lock = self._get_lock()
        async with lock:
            if self._engine is None:
                logger.info("Initializing SQLAlchemy async engine for %s", self.url)
                self._engine = create_async_engine(
                    self.url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                )
        return self._engine

    async def check_health(self, timeout: float = 2.0) -> bool:
        """Execute a lightweight read query to verify PostgreSQL health."""
        try:
            pool = await asyncio.wait_for(self.get_pool(), timeout=timeout)
            async with pool.acquire() as conn:
                val = await asyncio.wait_for(conn.fetchval("SELECT 1;"), timeout=timeout)
                return val == 1
        except Exception as exc:
            logger.warning("PostgreSQL health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Cleanly close connection pool and engine."""
        try:
            lock = self._get_lock()
            async with lock:
                if self._pool is not None:
                    try:
                        await self._pool.close()
                    except Exception as exc:
                        logger.debug("Error closing asyncpg pool: %s", exc)
                    self._pool = None

                if self._engine is not None:
                    try:
                        await self._engine.dispose()
                    except Exception as exc:
                        logger.debug("Error disposing SQLAlchemy engine: %s", exc)
                    self._engine = None
        except Exception:
            if self._pool is not None:
                try:
                    await self._pool.close()
                except Exception:
                    pass
                self._pool = None
            if self._engine is not None:
                try:
                    await self._engine.dispose()
                except Exception:
                    pass
                self._engine = None


# Singleton instance
_db_pool_instance: Optional[DatabasePool] = None


def get_db_pool(settings: Optional[Settings] = None) -> DatabasePool:
    """Acquire or create the shared DatabasePool instance."""
    global _db_pool_instance
    if _db_pool_instance is None:
        _db_pool_instance = DatabasePool(settings=settings)
    return _db_pool_instance


async def check_postgres_health(settings: Optional[Settings] = None) -> bool:
    """Convenience helper to check PostgreSQL availability."""
    pool = get_db_pool(settings)
    return await pool.check_health()
