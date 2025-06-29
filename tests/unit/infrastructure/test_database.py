"""Unit tests for database infrastructure.

Tests for database configuration, session management, and connection handling.
"""

from __future__ import annotations

import pytest
from flx_core.infrastructure.persistence.database import (
    DatabaseConfig,
    DatabaseManager,
    get_database_url,
)
from sqlalchemy.ext.asyncio import AsyncSession


class TestDatabaseConfig:
    """Test DatabaseConfig."""

    def test_database_config_defaults(self) -> None:
        """Test database config with default values."""
        config = DatabaseConfig()

        assert config.echo is False
        assert config.echo_pool is False
        assert config.pool_pre_ping is True
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_timeout == 30.0
        assert config.pool_recycle == 3600

    def test_database_config_custom(self) -> None:
        """Test database config with custom values."""
        config = DatabaseConfig(
            echo=True, pool_size=20, max_overflow=40, pool_timeout=60.0
        )

        assert config.echo is True
        assert config.pool_size == 20
        assert config.max_overflow == 40
        assert config.pool_timeout == 60.0

    def test_database_url_construction(self) -> None:
        """Test database URL construction."""
        # PostgreSQL
        url = get_database_url(
            driver="postgresql+asyncpg",
            host="localhost",
            port=5432,
            database="test_db",
            username="user",
            password="pass",
        )
        assert url == "postgresql+asyncpg://user:pass@localhost:5432/test_db"

        # SQLite
        url = get_database_url(driver="sqlite+aiosqlite", database=":memory:")
        assert url == "sqlite+aiosqlite:///:memory:"

        # Without password
        url = get_database_url(
            driver="postgresql+asyncpg",
            host="localhost",
            port=5432,
            database="test_db",
            username="user",
        )
        assert url == "postgresql+asyncpg://user@localhost:5432/test_db"


class TestDatabaseManager:
    """Test DatabaseManager."""

    @pytest.fixture
    async def db_manager(self, test_database: str) -> DatabaseManager:
        """Create database manager for tests."""
        config = DatabaseConfig(echo=False)
        manager = DatabaseManager(
            database_url=f"sqlite+aiosqlite:///{test_database}", config=config
        )
        await manager.initialize()
        yield manager
        await manager.dispose()

    async def test_database_manager_initialization(
        self, db_manager: DatabaseManager
    ) -> None:
        """Test database manager initialization."""
        assert db_manager.engine is not None
        assert db_manager.session_factory is not None
        assert db_manager.is_initialized is True

    async def test_get_session(self, db_manager: DatabaseManager) -> None:
        """Test getting database session."""
        async with db_manager.get_session() as session:
            assert isinstance(session, AsyncSession)
            assert session.is_active is True

            # Test basic query
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1

    async def test_create_tables(self, db_manager: DatabaseManager) -> None:
        """Test creating database tables."""
        # Tables should be created during initialization
        async with db_manager.get_session() as session:
            # Check if tables exist by querying metadata
            result = await session.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in result]

            # Should have some tables created
            assert len(tables) > 0

    async def test_health_check(self, db_manager: DatabaseManager) -> None:
        """Test database health check."""
        health = await db_manager.health_check()

        assert health["status"] == "healthy"
        assert health["database"] == "sqlite"
        assert "response_time_ms" in health
        assert health["response_time_ms"] > 0

    async def test_transaction_rollback(self, db_manager: DatabaseManager) -> None:
        """Test transaction rollback."""
        async with db_manager.get_session() as session:
            # Start transaction
            async with session.begin():
                # Execute some operation
                await session.execute("CREATE TABLE test_table (id INTEGER)")

                # Rollback by raising exception
                raise Exception("Test rollback")

        # Table should not exist after rollback
        async with db_manager.get_session() as session:
            result = await session.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
            )
            assert result.first() is None

    async def test_connection_pool_exhaustion(self, test_database: str) -> None:
        """Test connection pool behavior."""
        # Create manager with small pool
        config = DatabaseConfig(pool_size=2, max_overflow=1, pool_timeout=1.0)

        manager = DatabaseManager(
            database_url=f"sqlite+aiosqlite:///{test_database}", config=config
        )

        await manager.initialize()

        try:
            # Get multiple sessions
            sessions = []
            for _ in range(3):
                session = manager.get_session()
                sessions.append(session.__aenter__())

            # Should be able to get up to pool_size + max_overflow sessions
            for _i, session_coro in enumerate(sessions):
                session = await session_coro
                assert session is not None

        finally:
            # Cleanup
            for session in sessions:
                if hasattr(session, "__aexit__"):
                    await session.__aexit__(None, None, None)
            await manager.dispose()

    async def test_dispose(self, db_manager: DatabaseManager) -> None:
        """Test database manager disposal."""
        # Manager should be initialized
        assert db_manager.is_initialized is True

        # Dispose
        await db_manager.dispose()

        # Should be disposed
        assert db_manager.is_initialized is False

        # Getting session should raise error
        with pytest.raises(RuntimeError, match="not initialized"):
            async with db_manager.get_session():
                pass
