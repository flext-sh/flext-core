"""Unified Unit of Work and Repository Factory implementation.

This module provides a centralized, high-performance Unit of Work (UoW)
and repository factory for the entire application. It adheres to strict
Dependency Inversion and Single Responsibility Principles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

# Auth models now unified in models.py
from flext_auth.models import Role  # Import domain model for typing

from flext_core.contracts.repository_contracts import (
    RepositoryInterface,
    UnitOfWorkInterface,
)
from flext_core.domain.entities import Pipeline, PipelineExecution, Plugin
from flext_core.infrastructure.persistence.models import (
    PipelineExecutionModel,
    PipelineModel,
    PluginModel,
    RoleModel,  # SQLAlchemy model now unified
)
from flext_core.infrastructure.persistence.repositories_core import (
    DomainSpecificRepository,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from typing import Self

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class UnitOfWork(UnitOfWorkInterface):
    """Concrete SQLAlchemy-based Unit of Work implementation.

    Manages the session, transaction, and repository instances for a
    single business transaction.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize SqlAlchemyUnitOfWork with database session.

        Args:
        ----
            session: AsyncSession for managing database transactions and repository instances.

        """
        self.session = session
        self._transaction_managed_externally = self.session.in_transaction()
        self._pipelines: RepositoryInterface | None = None
        self._executions: RepositoryInterface | None = None
        self._plugins: RepositoryInterface | None = None
        self._roles: RepositoryInterface | None = None
        self._event_bus = None
        self._entities_with_events: list[
            object
        ] = []  # Track entities with domain events

    @property
    def pipelines(self) -> RepositoryInterface:
        """Pipelines repository."""
        if self._pipelines is None:
            self._pipelines = self.get_repository(Pipeline, PipelineModel)
        return self._pipelines

    @property
    def executions(self) -> RepositoryInterface:
        """Pipeline executions repository."""
        if self._executions is None:
            self._executions = self.get_repository(
                PipelineExecution,
                PipelineExecutionModel,
            )
        return self._executions

    @property
    def plugins(self) -> RepositoryInterface:
        """Plugins repository."""
        if self._plugins is None:
            self._plugins = self.get_repository(Plugin, PluginModel)
        return self._plugins

    @property
    def roles(self) -> RepositoryInterface:
        """Roles repository."""
        if self._roles is None:
            self._roles = self.get_repository(Role, RoleModel)
        return self._roles

    async def __aenter__(self) -> Self:
        """Enter async context manager and begin transaction."""
        if not self._transaction_managed_externally:
            await self.session.begin()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit async context manager and handle transaction commit/rollback."""
        if not self._transaction_managed_externally:
            if exc_type:
                await self.rollback()
            else:
                await self.commit()
        await self.session.close()

    async def commit(self) -> None:
        """Commit the current transaction if not externally managed."""
        # Publish domain events before committing the transaction
        try:
            event_bus = self._event_bus
            if event_bus:
                await self._publish_domain_events()
        except AttributeError:
            pass

        if not self._transaction_managed_externally:
            await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction if not externally managed."""
        if not self._transaction_managed_externally:
            await self.session.rollback()

    def get_repository(
        self,
        entity_class: type[Any],
        model_class: type[Any],
        id_field: str = "id",
    ) -> Any:
        """Create and return a domain-specific repository."""
        repository: Any = DomainSpecificRepository(
            session=self.session,
            entity_class=entity_class,  # type: ignore[arg-type]
            model_class=model_class,
            id_field=id_field,
        )
        return repository

    async def test_connection(self) -> bool:
        """Test database connection health.

        Performs a simple database health check to ensure the connection
        is active and operational. This method executes a lightweight
        query to validate database connectivity.

        Returns:
        -------
            bool: True if connection is healthy, False otherwise

        Note:
        ----
            Uses a simple SELECT 1 query for minimal database impact
            during health checks.

        """
        try:
            from sqlalchemy import text

            # Execute a simple query to test connection
            result = await self.session.execute(text("SELECT 1"))
            return result.scalar() == 1
        except (RuntimeError, ValueError, OSError, ImportError, AttributeError):
            # Any exception indicates connection failure
            return False

    def set_event_bus(self, event_bus: object) -> None:
        """Set the event bus for domain event publishing."""
        self._event_bus = event_bus

    def track_entity_with_events(self, entity: object) -> None:
        """Track an entity that might have domain events."""
        if entity not in self._entities_with_events:
            self._entities_with_events.append(entity)

    async def _publish_domain_events(self) -> None:
        """Collect and publish domain events from all entities in the session."""
        # Get all entities from the session identity map
        for entity in self.session.identity_map.all_states():
            try:
                uncommitted_events = entity.object.uncommitted_events
                if uncommitted_events:
                    # Publish all uncommitted events
                    for event in uncommitted_events:
                        await self._event_bus.publish(event)
                    # Clear the events after publishing
                    entity.object.mark_events_as_committed()
            except AttributeError:
                continue

        # Also check tracked entities
        for entity in self._entities_with_events:
            try:
                uncommitted_events = entity.uncommitted_events
                if uncommitted_events:
                    # Publish all uncommitted events
                    for event in uncommitted_events:
                        await self._event_bus.publish(event)
                    # Clear the events after publishing
                    entity.mark_events_as_committed()
            except AttributeError:
                continue


class UnitOfWorkFactory:
    """Factory for creating UnitOfWork instances."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Initialize UnitOfWorkFactory with session factory.

        Args:
        ----
            session_factory: Factory for creating AsyncSession instances.

        """
        self._session_factory = session_factory

    async def create(self) -> UnitOfWork:
        """Create a UoW instance directly."""
        return UnitOfWork(self._session_factory())

    async def __call__(self) -> AsyncGenerator[UnitOfWork]:
        """Create a UoW instance within an async generator context.

        Note: This pattern has limitations with exception handling.
        For better exception handling, use create() method directly.
        """
        uow = UnitOfWork(self._session_factory())
        yield uow
