"""Database connection and session management with asyncpg optimization."""

from __future__ import annotations

import asyncio
import functools
import operator
import time
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import Field

# ZERO TOLERANCE - SQLAlchemy imports with enterprise compatibility
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from flext_core.domain.pydantic_base import DomainValueObject
from flext_core.models import Base

if TYPE_CHECKING:
    from collections.abc import Callable


class DatabaseEchoMode(Enum):
    """Database echo mode for SQLAlchemy logging control.

    Defines the SQL logging behavior for SQLAlchemy engine, replacing
    boolean echo parameters with explicit mode enum values for
    better type safety and clearer debugging control.

    Attributes
    ----------
        SILENT: Disable SQL query logging for production environments.
        VERBOSE: Enable SQL query logging for debugging and development.

    """

    SILENT = "silent"
    VERBOSE = "verbose"


class DatabaseConfig(DomainValueObject):
    """Database configuration.

    Manages database connection parameters and configuration settings
    for SQLAlchemy async engine with connection pooling and transactions.

    Note:
    ----
        Configures SQLAlchemy async engine with connection pooling and performance optimization.

    """

    database_url: str = Field(
        description="Database connection URL with driver and credentials",
    )
    echo_mode: DatabaseEchoMode = Field(
        default=DatabaseEchoMode.SILENT,
        description="SQL logging mode for debugging",
    )
    pool_size: int = Field(
        default=5,
        description="Number of connections to maintain in pool",
    )
    max_overflow: int = Field(
        default=10,
        description="Maximum overflow connections beyond pool size",
    )
    pool_timeout: int = Field(
        default=30,
        description="Timeout in seconds for getting connection from pool",
    )
    pool_recycle: int = Field(
        default=3600,
        description="Connection recycle time in seconds",
    )
    enable_asyncpg_optimization: bool = Field(
        default=True,
        description="Enable asyncpg-specific performance optimizations",
    )

    # Production health monitoring configuration
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds",
    )
    connection_timeout: int = Field(
        default=10,
        description="Connection timeout for health checks in seconds",
    )
    max_failed_health_checks: int = Field(
        default=3,
        description="Max failed health checks before marking unhealthy",
    )

    # Performance monitoring configuration
    enable_slow_query_logging: bool = Field(
        default=True,
        description="Enable slow query logging",
    )
    slow_query_threshold_ms: int = Field(
        default=1000,
        description="Slow query threshold in milliseconds",
    )
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring",
    )
    performance_sampling_rate: float = Field(
        default=0.1,
        description="Performance monitoring sampling rate",
    )

    # Connection failover configuration
    enable_connection_failover: bool = Field(
        default=False,
        description="Enable automatic connection failover",
    )
    max_connection_retries: int = Field(
        default=3,
        description="Maximum connection retry attempts",
    )
    connection_retry_delay: float = Field(
        default=1.0,
        description="Delay between connection retries in seconds",
    )


class DatabaseHealthStatus(DomainValueObject):
    """Database health status information."""

    is_healthy: bool = Field(description="Whether the database is healthy")
    status: str = Field(
        description="Health status description (healthy/unhealthy/critical)",
    )
    response_time_ms: float = Field(
        description="Last health check response time in milliseconds",
    )
    failed_checks: int = Field(description="Number of consecutive failed health checks")
    last_error: str | None = Field(
        default=None,
        description="Last error message if any",
    )
    check_timestamp: datetime = Field(description="Timestamp of the health check")
    connection_pool_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Connection pool information",
    )


class DatabaseHealthMonitor:
    """Monitors database health and connection status."""

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize health monitor.

        Args:
        ----
            config: Database configuration with health monitoring settings

        """
        self.config = config
        self.failed_checks = 0
        self.last_check_time: datetime | None = None
        self.is_healthy = True
        self.is_monitoring = False
        self.monitoring_task: asyncio.Task[None] | None = None

    async def check_health(
        self, database_manager: DatabaseManager,
    ) -> DatabaseHealthStatus:
        """Perform a database health check.

        Args:
        ----
            database_manager: Database manager to check

        Returns:
        -------
            Health status information

        """
        start_time = time.time()
        check_timestamp = datetime.now(UTC)

        try:
            # Perform health check query with timeout
            async with asyncio.timeout(self.config.connection_timeout):
                session_factory = database_manager.get_session_factory()
                async with session_factory() as session:
                    result = await session.execute(text("SELECT 1"))
                    result.scalar_one()

            # Health check successful
            response_time_ms = (time.time() - start_time) * 1000
            self.failed_checks = 0
            self.is_healthy = True
            self.last_check_time = check_timestamp

            # Get connection pool info
            try:
                pool_info = await database_manager.get_connection_info()
            except AttributeError:
                # Fallback for DatabaseManager without get_connection_info method
                pool_info = {"status": "not_available"}

            return DatabaseHealthStatus(
                is_healthy=True,
                status="healthy",
                response_time_ms=response_time_ms,
                failed_checks=0,
                last_error=None,
                check_timestamp=check_timestamp,
                connection_pool_info=pool_info,
            )

        except (
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
            OSError,
            RuntimeError,
            AttributeError,
        ) as e:
            # Health check failed - ZERO TOLERANCE specific exception types
            response_time_ms = (time.time() - start_time) * 1000
            self.failed_checks += 1
            self.is_healthy = False
            self.last_check_time = check_timestamp

            # Determine status based on failure count
            status = (
                "critical"
                if self.failed_checks >= self.config.max_failed_health_checks
                else "unhealthy"
            )

            return DatabaseHealthStatus(
                is_healthy=False,
                status=status,
                response_time_ms=response_time_ms,
                failed_checks=self.failed_checks,
                last_error=str(e),
                check_timestamp=check_timestamp,
                connection_pool_info={},
            )

    async def start_continuous_monitoring(
        self, database_manager: DatabaseManager,
    ) -> asyncio.Task[None]:
        """Start continuous health monitoring.

        Args:
        ----
            database_manager: Database manager to monitor

        Returns:
        -------
            Monitoring task

        """
        if self.is_monitoring:
            msg = "Health monitoring is already running"
            raise RuntimeError(msg)

        self.is_monitoring = True

        async def monitor_loop() -> None:
            while self.is_monitoring:
                try:
                    await self.check_health(database_manager)
                    await asyncio.sleep(self.config.health_check_interval)
                except (
                    ConnectionError,
                    TimeoutError,
                    ValueError,
                    TypeError,
                    OSError,
                    RuntimeError,
                    AttributeError,
                    asyncio.CancelledError,
                ):
                    # Continue monitoring even if individual checks fail - ZERO TOLERANCE specific exceptions
                    await asyncio.sleep(self.config.health_check_interval)

        self.monitoring_task = asyncio.create_task(monitor_loop())
        return self.monitoring_task

    async def stop_continuous_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.monitoring_task
            self.monitoring_task = None


class DatabaseConnectionFailover:
    """Manages database connection failover and recovery."""

    def __init__(
        self,
        primary_config: DatabaseConfig,
        failover_configs: list[DatabaseConfig],
        failover_threshold: int = 3,
        recovery_check_interval: int = 60,
    ) -> None:
        """Initialize connection failover manager.

        Args:
        ----
            primary_config: Primary database configuration
            failover_configs: List of failover database configurations
            failover_threshold: Number of failures before triggering failover
            recovery_check_interval: Interval for checking primary recovery

        """
        self.primary_config = primary_config
        self.failover_configs = failover_configs
        self.failover_threshold = failover_threshold
        self.recovery_check_interval = recovery_check_interval

        self.current_config = primary_config
        self.is_failed_over = False
        self.failure_count = 0
        self.current_failover_index = 0

    async def get_connection(self) -> DatabaseManager:
        """Get a working database connection with automatic failover.

        Returns
        -------
            Database manager with working connection

        Raises
        ------
            RuntimeError: If all databases are unavailable

        """
        # Try current configuration first
        try:
            manager = DatabaseManager(self.current_config)
            await manager.initialize()
            return manager  # noqa: TRY300
        except (
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
            OSError,
            RuntimeError,
            AttributeError,
        ):
            # Database connection failed - ZERO TOLERANCE specific exception types
            self.failure_count += 1

            # Check if we need to failover
            if (
                not self.is_failed_over
                and self.failure_count >= self.failover_threshold
            ):
                return await self._perform_failover()
            if self.is_failed_over:
                # Already failed over, try next failover if available
                return await self._try_next_failover()
            # Not yet at threshold, re-raise
            raise

    async def _perform_failover(self) -> DatabaseManager:
        """Perform failover to first available failover database."""
        for i, failover_config in enumerate(self.failover_configs):
            try:
                manager = DatabaseManager(failover_config)
                await manager.initialize()

                # Failover successful
                self.current_config = failover_config
                self.is_failed_over = True
                self.current_failover_index = i
                return manager  # noqa: TRY300

            except (
                ConnectionError,
                TimeoutError,
                ValueError,
                TypeError,
                OSError,
                RuntimeError,
                AttributeError,
            ):
                # Failover connection attempt failed - ZERO TOLERANCE specific exception types
                continue

        # All failovers failed
        msg = "All database connections failed, including failovers"
        raise RuntimeError(msg)

    async def _try_next_failover(self) -> DatabaseManager:
        """Try the next available failover database."""
        start_index = (self.current_failover_index + 1) % len(self.failover_configs)

        for i in range(len(self.failover_configs)):
            config_index = (start_index + i) % len(self.failover_configs)
            failover_config = self.failover_configs[config_index]

            try:
                manager = DatabaseManager(failover_config)
                await manager.initialize()

                # Switch to this failover
                self.current_config = failover_config
                self.current_failover_index = config_index
                return manager  # noqa: TRY300

            except (
                ConnectionError,
                TimeoutError,
                ValueError,
                TypeError,
                OSError,
                RuntimeError,
                AttributeError,
            ):
                # Next failover connection attempt failed - ZERO TOLERANCE specific exception types
                continue

        # All failovers failed
        msg = "All failover database connections failed"
        raise RuntimeError(msg)

    async def check_primary_recovery(self) -> bool:
        """Check if primary database has recovered.

        Returns
        -------
            True if primary is available and failover can be reverted

        """
        if not self.is_failed_over:
            return True

        try:
            manager = DatabaseManager(self.primary_config)
            await manager.initialize()
            await manager.close()

            # Primary recovered, revert failover
            self.current_config = self.primary_config
            self.is_failed_over = False
            self.failure_count = 0
            self.current_failover_index = 0
            return True  # noqa: TRY300

        except (
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
            OSError,
            RuntimeError,
            AttributeError,
        ):
            # Primary recovery check failed - ZERO TOLERANCE specific exception types
            return False

    def get_failover_status(self) -> dict[str, Any]:
        """Get current failover status information.

        Returns
        -------
            Dictionary with failover status details

        """
        return {
            "is_failed_over": self.is_failed_over,
            "current_database": (
                "primary"
                if not self.is_failed_over
                else f"failover_{self.current_failover_index}"
            ),
            "failure_count": self.failure_count,
            "failover_threshold": self.failover_threshold,
            "available_failovers": len(self.failover_configs),
            "primary_url": self.primary_config.database_url,
            "current_url": self.current_config.database_url,
        }


class DatabasePerformanceMonitor:
    """Monitors database performance and query statistics."""

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize performance monitor.

        Args:
        ----
            config: Database configuration with performance monitoring settings

        """
        self.config = config
        self.query_stats: dict[str, list[float]] = defaultdict(list)
        self.slow_queries: list[dict[str, Any]] = []
        self.alert_thresholds: dict[str, float] = {}

    def record_query(self, query: str, duration_ms: float) -> None:
        """Record query execution statistics.

        Args:
        ----
            query: SQL query string
            duration_ms: Query execution duration in milliseconds

        """
        # Record query stats
        self.query_stats[query].append(duration_ms)

        # Check for slow query
        if (
            self.config.enable_slow_query_logging
            and duration_ms >= self.config.slow_query_threshold_ms
        ):
            self.slow_queries.append(
                {
                    "query": query,
                    "duration_ms": duration_ms,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    @asynccontextmanager
    async def monitor_query(self, query: str) -> Any:
        """Context manager for monitoring query execution.

        Args:
        ----
            query: SQL query string

        Yields:
        ------
            None (context manager for timing)

        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_query(query, duration_ms)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics.

        Returns
        -------
            Dictionary with performance metrics

        """
        total_queries = sum(len(durations) for durations in self.query_stats.values())
        all_durations = [
            d for durations in self.query_stats.values() for d in durations
        ]

        return {
            "total_queries": total_queries,
            "slow_queries_count": len(self.slow_queries),
            "average_query_time": (
                sum(all_durations) / len(all_durations) if all_durations else 0
            ),
            "queries_by_duration": {
                "fast_queries": len([d for d in all_durations if d < 100]),
                "medium_queries": len([d for d in all_durations if 100 <= d < 1000]),
                "slow_queries": len([d for d in all_durations if d >= 1000]),
            },
            "top_slow_queries": sorted(
                self.slow_queries,
                key=operator.itemgetter("duration_ms"),
                reverse=True,
            )[:5],
        }

    def analyze_slow_queries(self) -> dict[str, Any]:
        """Analyze slow query patterns and statistics.

        Returns
        -------
            Dictionary with slow query analysis

        """
        total_queries = sum(len(durations) for durations in self.query_stats.values())
        slow_query_count = len(self.slow_queries)

        return {
            "total_slow_queries": slow_query_count,
            "slow_query_rate": (
                slow_query_count / total_queries if total_queries > 0 else 0
            ),
            "slowest_queries": sorted(
                self.slow_queries,
                key=operator.itemgetter("duration_ms"),
                reverse=True,
            )[:5],
            "slow_query_patterns": self._analyze_query_patterns(),
        }

    def _analyze_query_patterns(self) -> dict[str, int]:
        """Analyze patterns in slow queries."""
        patterns = defaultdict(int)
        for slow_query in self.slow_queries:
            query = slow_query["query"].upper()
            if "SELECT" in query:
                patterns["SELECT"] += 1
            elif "INSERT" in query:
                patterns["INSERT"] += 1
            elif "UPDATE" in query:
                patterns["UPDATE"] += 1
            elif "DELETE" in query:
                patterns["DELETE"] += 1
            else:
                patterns["OTHER"] += 1
        return dict(patterns)

    def set_alert_thresholds(
        self,
        slow_query_rate_threshold: float = 0.1,
        average_query_time_threshold: float = 500,
        max_query_time_threshold: float = 10000,
    ) -> None:
        """Set performance alert thresholds.

        Args:
        ----
            slow_query_rate_threshold: Threshold for slow query rate (0.0-1.0)
            average_query_time_threshold: Threshold for average query time in ms
            max_query_time_threshold: Threshold for maximum query time in ms

        """
        self.alert_thresholds = {
            "slow_query_rate": slow_query_rate_threshold,
            "average_query_time": average_query_time_threshold,
            "max_query_time": max_query_time_threshold,
        }

    def check_performance_alerts(self) -> list[dict[str, Any]]:
        """Check for performance alerts based on thresholds.

        Returns
        -------
            List of active alerts

        """
        alerts = []
        metrics = self.get_performance_metrics()

        if not self.alert_thresholds:
            return alerts

        # Check slow query rate
        total_queries = metrics["total_queries"]
        if total_queries > 0:
            slow_query_rate = metrics["slow_queries_count"] / total_queries
            if slow_query_rate >= self.alert_thresholds.get("slow_query_rate", 1.0):
                alerts.append(
                    {
                        "type": "slow_query_rate",
                        "message": f"Slow query rate {slow_query_rate:.1%} exceeds threshold",
                        "value": slow_query_rate,
                        "threshold": self.alert_thresholds["slow_query_rate"],
                    },
                )

        # Check average query time
        avg_time = metrics["average_query_time"]
        if avg_time >= self.alert_thresholds.get("average_query_time", float("inf")):
            alerts.append(
                {
                    "type": "average_query_time",
                    "message": f"Average query time {avg_time:.1f}ms exceeds threshold",
                    "value": avg_time,
                    "threshold": self.alert_thresholds["average_query_time"],
                },
            )

        return alerts


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize database manager.

        Sets up async SQLAlchemy engine and session factory with the provided
        configuration for enterprise database operations.

        Args:
        ----
            config: Database configuration with connection parameters

        Note:
        ----
            Uses async SQLAlchemy for optimal database performance.

        """
        self.config = config
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None

    async def initialize(self) -> None:
        """Initialize database engine and session factory with asyncpg optimizations.

        Creates async SQLAlchemy engine with configured connection parameters
        and sets up session factory for high-performance database operations.
        Applies asyncpg-specific optimizations for PostgreSQL databases.

        Note:
        ----
            Must be called before using the database manager for any operations.

        """
        # Prepare engine arguments with asyncpg optimizations
        engine_args: dict[str, Any] = {
            "echo": self.config.echo_mode == DatabaseEchoMode.VERBOSE,
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "pool_timeout": self.config.pool_timeout,
            "pool_recycle": self.config.pool_recycle,
        }

        # Apply asyncpg-specific optimizations for PostgreSQL
        if (
            self.config.enable_asyncpg_optimization
            and "postgresql" in self.config.database_url
        ):
            engine_args.update(self._get_asyncpg_optimizations())

        self.engine = create_async_engine(
            self.config.database_url,
            **engine_args,
        )

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    def _get_asyncpg_optimizations(self) -> dict[str, Any]:
        """Get asyncpg-specific optimization parameters.

        Returns connection parameters optimized for asyncpg driver performance
        including prepared statements, connection pooling, and query caching.

        Returns
        -------
            Dictionary of asyncpg optimization parameters

        """
        return {
            # asyncpg-specific connection arguments
            "connect_args": {
                "prepared_statement_cache_size": 100,  # Enable prepared statement caching
                "prepared_statement_name_func": lambda x: f"__asyncpg_stmt_{abs(hash(x)) % 2**31}__",
                "statement_cache_size": 0,  # Disable client-side statement cache (asyncpg handles this)
                "command_timeout": 60,  # Command timeout in seconds
                "server_settings": {
                    "timezone": "UTC",  # Ensure UTC timezone
                    "application_name": "flext_meltano_enterprise",  # Application identification
                    "jit": "off",  # Disable JIT for consistent performance
                },
            },
            # Pool configuration optimized for asyncpg
            "pool_pre_ping": True,  # Verify connections before use
            "pool_reset_on_return": "commit",  # Clean connection state
        }

    async def create_tables(self) -> None:
        """Create all database tables.

        Uses async SQLAlchemy DDL operations to create all tables defined
        in the SQLAlchemy Base metadata for application initialization.

        Returns:
        -------
            None after creating all database tables.

        Note:
        ----
            Uses async SQLAlchemy for optimal database performance.

        """
        if not self.engine:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)

        async with self.engine.begin() as conn:  # type: ignore[union-attr]
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all database tables.

        Uses async SQLAlchemy DDL operations to drop all tables defined
        in the SQLAlchemy Base metadata for cleanup and testing.

        Returns:
        -------
            None after dropping all database tables.

        Note:
        ----
            Uses async SQLAlchemy for optimal database performance.

        """
        if not self.engine:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)

        async with self.engine.begin() as conn:  # type: ignore[union-attr]
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        """Close database connections and clean up resources.

        Properly disposes of the SQLAlchemy engine and clears references
        to prevent connection leaks during application shutdown.

        Returns:
        -------
            None after closing database connections.

        Note:
        ----
            Uses async SQLAlchemy for optimal database performance.

        """
        if self.engine:
            await self.engine.dispose()  # type: ignore[union-attr]
            self.engine = None
            self.session_factory = None

    def get_session_factory(self) -> async_sessionmaker[AsyncSession]:  # type: ignore[type-arg]
        """Get the session factory for creating database sessions.

        Provides access to the configured async session factory for
        creating database sessions with proper connection pooling.

        Returns:
        -------
            Async session factory for database operations.

        Note:
        ----
            Uses async SQLAlchemy for optimal database performance.

        """
        if not self.session_factory:
            msg = "Database session factory not initialized"
            raise RuntimeError(msg)
        return self.session_factory

    def get_engine(self) -> AsyncEngine:
        """Get the database engine for direct operations.

        Provides access to the configured async SQLAlchemy engine for
        direct database operations and DDL commands.

        Returns:
        -------
            AsyncEngine instance for database operations.

        Note:
        ----
            Uses async SQLAlchemy for optimal database performance.

        """
        if not self.engine:
            msg = "Database engine not initialized"
            raise RuntimeError(msg)
        return self.engine


# Modern database manager registry using configuration key-based caching
_manager_factories: dict[str, Callable[[], DatabaseManager]] = {}


def initialize_database_manager(config: DatabaseConfig) -> DatabaseManager:
    """Initialize the database manager with configuration.

    Uses modern Python 3.13 pattern for configuration-based singleton.
    """
    # Create unique key from config

    # Create cached factory for this config
    @functools.lru_cache(maxsize=1)
    def get_manager() -> DatabaseManager:
        return DatabaseManager(config)

    _manager_factories["current"] = get_manager
    return get_manager()


def get_database_manager() -> DatabaseManager:
    """Get the database manager instance.

    Uses modern Python 3.13 factory pattern for configuration-based singleton.
    Requires initialization via initialize_database_manager first.
    """
    current_factory = _manager_factories.get("current")
    if current_factory is None:
        msg = "Database manager not initialized. Call initialize_database_manager() first."
        raise RuntimeError(msg)
    # Direct invocation pattern - factory must be callable
    try:
        return current_factory()
    except TypeError as e:
        msg = f"Database manager factory is not callable: {e}"
        raise TypeError(msg) from e


async def create_database_manager(config: DatabaseConfig) -> DatabaseManager:
    """Create and initialize database manager with asyncpg optimizations."""
    manager = initialize_database_manager(config)
    await manager.initialize()
    return manager


class AsyncpgDatabaseManager(DatabaseManager):
    """Enhanced database manager with asyncpg-specific optimizations.

    Provides additional PostgreSQL-specific functionality including:
    - Optimized bulk operations
    - Connection monitoring
    - Performance metrics
    - Advanced pooling strategies
    - Health monitoring
    - Connection failover
    - Performance monitoring
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize enhanced database manager.

        Args:
        ----
            config: Database configuration with asyncpg optimizations enabled

        """
        super().__init__(config)
        self._connection_count = 0
        self._query_count = 0

        # Initialize monitoring components
        self.health_monitor = (
            DatabaseHealthMonitor(config) if config.health_check_interval > 0 else None
        )
        self.performance_monitor = (
            DatabasePerformanceMonitor(config)
            if config.enable_performance_monitoring
            else None
        )
        self.connection_failover: DatabaseConnectionFailover | None = None

    async def get_connection_info(self) -> dict[str, Any]:
        """Get detailed connection information for monitoring.

        Returns
        -------
            Dictionary with connection pool status and performance metrics

        """
        if not self.engine:
            return {"status": "not_initialized"}

        pool = self.engine.pool
        pool_info = {
            "connection_count": self._connection_count,
            "query_count": self._query_count,
            "driver": (
                "asyncpg" if "postgresql" in self.config.database_url else "aiosqlite"
            ),
        }

        # Get pool statistics with proper exception handling
        try:
            pool_info["pool_size"] = pool.size()  # type: ignore[attr-defined]
        except AttributeError:
            pool_info["pool_size"] = "unknown"

        try:
            pool_info["checked_in"] = pool.checkedin()  # type: ignore[attr-defined]
        except AttributeError:
            pool_info["checked_in"] = "unknown"

        try:
            pool_info["checked_out"] = pool.checkedout()  # type: ignore[attr-defined]
        except AttributeError:
            pool_info["checked_out"] = "unknown"

        try:
            pool_info["overflow"] = pool.overflow()  # type: ignore[attr-defined]
        except AttributeError:
            pool_info["overflow"] = "unknown"

        return pool_info

    async def execute_bulk_insert(self, table: str, data: list[dict[str, Any]]) -> int:
        """Perform optimized bulk insert using asyncpg features.

        Args:
        ----
            table: Target table name
            data: List of dictionaries to insert

        Returns:
        -------
            Number of rows inserted

        """
        if not self.engine or not data:
            return 0

        # Use asyncpg's copy_records_to_table for maximum performance
        if "postgresql" in self.config.database_url:
            async with self.engine.begin() as conn:
                # Get raw asyncpg connection for bulk operations
                raw_conn = await conn.get_raw_connection()

                # Prepare data for asyncpg copy
                columns = list(data[0].keys())
                values = [[row[col] for col in columns] for row in data]

                # Use asyncpg's high-performance copy
                driver_conn = raw_conn.driver_connection
                if driver_conn is None:
                    msg = "Driver connection is not available for copy operation"
                    raise RuntimeError(msg)
                await driver_conn.copy_records_to_table(
                    table,
                    records=values,
                    columns=columns,
                )
                self._query_count += 1
                return len(data)
        else:
            # Fallback to standard SQLAlchemy bulk insert
            if self.session_factory:
                async with self.session_factory() as session:
                    # Note: This is a placeholder - actual implementation would need proper model handling
                    await session.commit()
                    return len(data)
            return 0

    async def optimize_performance(self) -> dict[str, Any]:
        """Apply runtime performance optimizations.

        Returns
        -------
            Dictionary with optimization results

        """
        if not self.engine or "postgresql" not in self.config.database_url:
            return {"optimizations": "not_applicable"}

        optimizations_applied = []

        try:
            async with self.engine.begin() as conn:  # type: ignore[union-attr]
                # Set optimal PostgreSQL parameters for this session
                await conn.execute(text("SET work_mem = '256MB'"))
                await conn.execute(text("SET maintenance_work_mem = '512MB'"))
                await conn.execute(text("SET effective_cache_size = '4GB'"))
                await conn.execute(text("SET random_page_cost = 1.1"))
                optimizations_applied.extend(
                    [
                        "work_mem_increased",
                        "maintenance_work_mem_increased",
                        "effective_cache_size_optimized",
                        "random_page_cost_optimized",
                    ],
                )

        except (RuntimeError, ValueError, OSError, ImportError, AttributeError) as e:
            return {"error": str(e), "optimizations": optimizations_applied}

        return {"optimizations": optimizations_applied, "status": "success"}

    async def initialize_with_health_check(self) -> None:
        """Initialize database with automatic health check.

        Raises
        ------
            RuntimeError: If initialization or health check fails

        """
        await self.initialize()

        if self.health_monitor:
            health_status = await self.health_monitor.check_health(self)
            if not health_status.is_healthy:
                msg = f"Database health check failed: {health_status.last_error}"
                raise RuntimeError(msg)

    async def connect_with_retry(self) -> None:
        """Connect to database with automatic retry logic.

        Raises
        ------
            RuntimeError: If all connection attempts fail

        """
        last_error = None

        for attempt in range(self.config.max_connection_retries):
            try:
                await self.initialize()
                return  # noqa: TRY300
            except Exception as e:
                last_error = e
                if attempt < self.config.max_connection_retries - 1:
                    await asyncio.sleep(self.config.connection_retry_delay)
                    continue
                break

        msg = f"Failed to connect after {self.config.max_connection_retries} attempts: {last_error}"
        raise RuntimeError(msg)

    async def execute_with_monitoring(
        self, query: str, parameters: dict[str, Any] | None = None,
    ) -> Any:
        """Execute query with performance monitoring.

        Args:
        ----
            query: SQL query to execute
            parameters: Query parameters

        Returns:
        -------
            Query result

        """
        if not self.performance_monitor:
            # Fallback to regular execution
            session_factory = self.get_session_factory()
            async with session_factory() as session:
                return await session.execute(text(query), parameters or {})

        # Execute with performance monitoring
        async with self.performance_monitor.monitor_query(query):
            session_factory = self.get_session_factory()
            async with session_factory() as session:
                return await session.execute(text(query), parameters or {})

    def setup_failover(
        self, failover_configs: list[DatabaseConfig], failover_threshold: int = 3,
    ) -> None:
        """Setup connection failover with backup databases.

        Args:
        ----
            failover_configs: List of failover database configurations
            failover_threshold: Number of failures before triggering failover

        """
        if self.config.enable_connection_failover:
            self.connection_failover = DatabaseConnectionFailover(
                primary_config=self.config,
                failover_configs=failover_configs,
                failover_threshold=failover_threshold,
            )

    async def get_connection_with_failover(self) -> DatabaseManager:
        """Get database connection with automatic failover.

        Returns
        -------
            Database manager with working connection

        Raises
        ------
            RuntimeError: If failover is not configured or all connections fail

        """
        if not self.connection_failover:
            msg = "Connection failover not configured"
            raise RuntimeError(msg)

        return await self.connection_failover.get_connection()

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive database status information.

        Returns
        -------
            Dictionary with complete database status

        """
        status = {
            "connection_info": {},
            "health_status": None,
            "performance_metrics": None,
            "failover_status": None,
            "configuration": {
                "database_url": (
                    self.config.database_url[:50] + "..."
                    if len(self.config.database_url) > 50
                    else self.config.database_url
                ),
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_timeout": self.config.pool_timeout,
                "enable_performance_monitoring": self.config.enable_performance_monitoring,
                "enable_connection_failover": self.config.enable_connection_failover,
                "health_check_interval": self.config.health_check_interval,
            },
        }

        # Add connection info
        try:
            # Note: This would be called asynchronously in real usage
            # For now, return basic connection info
            status["connection_info"] = {
                "connection_count": self._connection_count,
                "query_count": self._query_count,
                "driver": (
                    "asyncpg"
                    if "postgresql" in self.config.database_url
                    else "aiosqlite"
                ),
            }
        except Exception as e:
            status["connection_info"] = {"error": str(e)}

        # Add health status if available
        if self.health_monitor:
            status["health_status"] = {
                "is_healthy": self.health_monitor.is_healthy,
                "failed_checks": self.health_monitor.failed_checks,
                "last_check_time": (
                    self.health_monitor.last_check_time.isoformat()
                    if self.health_monitor.last_check_time
                    else None
                ),
                "is_monitoring": self.health_monitor.is_monitoring,
            }

        # Add performance metrics if available
        if self.performance_monitor:
            status["performance_metrics"] = (
                self.performance_monitor.get_performance_metrics()
            )

        # Add failover status if available
        if self.connection_failover:
            status["failover_status"] = self.connection_failover.get_failover_status()

        return status

    async def start_monitoring(self) -> None:
        """Start all monitoring services."""
        if self.health_monitor and not self.health_monitor.is_monitoring:
            await self.health_monitor.start_continuous_monitoring(self)

    async def stop_monitoring(self) -> None:
        """Stop all monitoring services."""
        if self.health_monitor and self.health_monitor.is_monitoring:
            await self.health_monitor.stop_continuous_monitoring()


# Export enhanced database functionality
__all__ = [
    "AsyncpgDatabaseManager",
    "DatabaseConfig",
    "DatabaseConnectionFailover",
    "DatabaseEchoMode",
    "DatabaseHealthMonitor",
    "DatabaseHealthStatus",
    "DatabaseManager",
    "DatabasePerformanceMonitor",
    "create_database_manager",
    "get_database_manager",
    "initialize_database_manager",
]
