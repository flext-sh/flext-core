"""SQLAlchemy models for FLEXT Core entities including Authentication."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as POSTGRES_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import CHAR, TypeDecorator

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect
    from sqlalchemy.sql.type_api import TypeEngine


class UniversalUUID(TypeDecorator[UUID]):
    """Database-agnostic UUID column type.

    Uses native UUID for PostgreSQL and stores as string for other databases.
    This ensures compatibility across SQLite (for tests) and PostgreSQL (production).
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(POSTGRES_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(
        self, value: UUID | None, dialect: Dialect,
    ) -> UUID | str | None:
        if value is None or dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(
        self, value: UUID | str | None, _dialect: Dialect,
    ) -> UUID | None:
        if value is None:
            return value
        if not isinstance(value, UUID):
            return UUID(value)
        return value


class Base(DeclarativeBase):
    """Base model for all SQLAlchemy entities.

    Implements the central framework component with specific functionality
    following established architectural patterns.

    Architecture: Enterprise Patterns
    Standards: SOLID principles, clean code

    Attributes:
    ----------
        created_at (Mapped[datetime]): Timestamp when record was created.
        updated_at (Mapped[datetime]): Timestamp when record was last updated.

    Methods:
    -------
        No public methods - provides standard audit fields.

    Examples:
    --------
        Typical usage of the class:

        ```python
        # This is a base class - inherit from it for entities
        class MyEntity(Base):
            __tablename__ = "my_entity"
            id: Mapped[int] = mapped_column(primary_key=True)
        ```

    See Also:
    --------
        - [Architecture Documentation](../../docs/architecture/index.md)
        - [Design Patterns](../../docs/architecture/001-clean-architecture-ddd.md)

    Note:
    ----
        This class provides domain-driven design patterns for database entities.

    """

    # Standard audit fields for all entities
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class PipelineModel(Base):
    """SQLAlchemy model for Pipeline entity.

    Represents a data pipeline definition in the database, storing pipeline
    configuration, scheduling information, and execution parameters. This model
    maps the domain Pipeline entity to the database schema.

    The model includes comprehensive pipeline metadata including:
    - Pipeline identity and naming
    - Scheduling configuration for automated runs
    - Execution constraints (timeouts, concurrency, retries)
    - Environment variables for runtime configuration
    - Audit trail for changes
    - Relationships to steps and execution history

    Attributes:
    ----------
        id: UUID primary key for the pipeline
        name: Unique pipeline name (max 100 chars)
        description: Optional detailed pipeline description
        environment_variables: JSON dict of environment variables
        schedule_expression: Cron expression for scheduled runs
        timezone: Timezone for schedule interpretation (default UTC)
        max_concurrent_executions: Maximum parallel executions allowed
        timeout_seconds: Optional execution timeout in seconds
        retry_attempts: Number of retry attempts on failure
        retry_delay_seconds: Delay between retry attempts
        is_active: Whether the pipeline is enabled
        created_by: User who created the pipeline
        updated_by: User who last updated the pipeline
        steps: Related pipeline steps ordered by execution
        executions: Related execution history records

    Examples:
    --------
        Creating a new pipeline record:

        ```python
        pipeline = PipelineModel(
            id=uuid4(),
            name="daily-sales-etl",
            description="Extract daily sales data to warehouse",
            schedule_expression="0 2 * * *",  # 2 AM daily
            max_concurrent_executions=1,
            is_active=True
        )
        session.add(pipeline)
        ```

    See Also:
    --------
        - [Architecture Documentation](../../docs/architecture/index.md)
        - [Domain-Driven Design Patterns](../../docs/architecture/001-clean-architecture-ddd.md)

    Note:
    ----
        This model follows the repository pattern with SQLAlchemy declarative
        mapping for clean separation of domain and persistence concerns.

    """

    __tablename__ = "pipelines"

    # Identity
    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    # Pipeline definition
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    environment_variables: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    # Schedule configuration
    schedule_expression: Mapped[str | None] = mapped_column(String(100), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="UTC")

    # Execution configuration
    max_concurrent_executions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    timeout_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    retry_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_delay_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=30.0,
    )

    # State
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Audit fields
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    steps: Mapped[list[PipelineStepModel]] = relationship(
        "PipelineStepModel",
        back_populates="pipeline",
        cascade="all, delete-orphan",
        order_by="PipelineStepModel.order",
    )
    executions: Mapped[list[PipelineExecutionModel]] = relationship(
        "PipelineExecutionModel",
        back_populates="pipeline",
        cascade="all, delete-orphan",
    )


class PipelineStepModel(Base):
    """SQLAlchemy model for pipeline step configuration.

    Represents an individual step within a pipeline, defining which plugin
    to execute, its configuration, and dependencies on other steps. Steps
    are ordered within a pipeline and can have complex dependency graphs.

    This model uses a composite primary key of (pipeline_id, step_id) to
    ensure unique step identifiers within each pipeline while allowing
    step reuse across pipelines.

    Attributes:
    ----------
        pipeline_id: UUID foreign key to parent pipeline
        step_id: String identifier unique within the pipeline
        plugin_id: UUID foreign key to the plugin to execute
        order: Integer defining execution order (for non-dependent steps)
        configuration: JSON dict of plugin-specific configuration
        depends_on: JSON list of step_ids this step depends on
        pipeline: Relationship to parent PipelineModel
        plugin: Relationship to associated PluginModel

    Examples:
    --------
        Creating a pipeline step:

        ```python
        step = PipelineStepModel(
            pipeline_id=pipeline.id,
            step_id="extract-customers",
            plugin_id=tap_postgres_plugin.id,
            order=1,
            configuration={
                "table_name": "customers",
                "select_columns": ["id", "name", "email"]
            },
            depends_on=[]
        )
        session.add(step)
        ```

    See Also:
    --------
        - [Pipeline Architecture](../../docs/architecture/index.md)
        - [Plugin System Design](../../docs/architecture/003-plugin-system.md)

    Note:
    ----
        Steps are executed based on their dependency graph, with the order
        field used only to sequence steps that have no dependencies.

    """

    __tablename__ = "pipeline_steps"

    # Composite primary key
    pipeline_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("pipelines.id"),
        primary_key=True,
    )
    step_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Step definition
    plugin_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("plugins.id"),
        nullable=False,
    )
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    depends_on: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Relationships
    pipeline: Mapped[PipelineModel] = relationship(
        "PipelineModel",
        back_populates="steps",
    )
    plugin: Mapped[PluginModel] = relationship(
        "PluginModel",
        back_populates="pipeline_steps",
    )


class PipelineExecutionModel(Base):
    """SQLAlchemy model for pipeline execution history.

    Tracks individual pipeline execution runs including status, timing,
    resource usage, and output data. Each execution is uniquely identified
    by a UUID and has a sequential execution number within its pipeline.

    This model provides comprehensive execution tracking for:
    - Execution lifecycle (pending, running, completed, failed)
    - Performance metrics (CPU, memory, duration)
    - Input/output data capture
    - Error tracking and debugging information
    - Audit trail of who triggered the execution

    Attributes:
    ----------
        id: UUID primary key for the execution
        pipeline_id: UUID foreign key to the pipeline
        execution_number: Sequential number within the pipeline
        triggered_by: User or system that triggered execution
        trigger_type: How execution was triggered (manual, scheduled, webhook)
        status: Current execution status (pending, running, completed, failed)
        started_at: Timestamp when execution started
        completed_at: Timestamp when execution completed
        input_data: JSON dict of input parameters
        output_data: JSON dict of execution results
        error_message: Error details if execution failed
        log_messages: JSON array of log entries
        cpu_usage: CPU utilization percentage
        memory_usage: Memory usage in MB
        duration_seconds: Total execution time in seconds
        pipeline: Relationship to parent PipelineModel

    Examples:
    --------
        Recording a pipeline execution:

        ```python
        execution = PipelineExecutionModel(
            id=uuid4(),
            pipeline_id=pipeline.id,
            execution_number=next_number,
            triggered_by="scheduler",
            trigger_type="scheduled",
            status="pending",
            input_data={"date": "2024-01-01"}
        )
        session.add(execution)
        ```

    See Also:
    --------
        - [Execution Monitoring](../../docs/monitoring/index.md)
        - [Pipeline Architecture](../../docs/architecture/004-orchestration-layer.md)

    Note:
    ----
        Execution records are immutable once completed to maintain
        audit trail integrity.

    """

    __tablename__ = "pipeline_executions"

    # Identity
    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True)
    pipeline_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("pipelines.id"),
        nullable=False,
    )
    execution_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Execution metadata
    triggered_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    trigger_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="manual",
    )

    # Execution state
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Execution data
    input_data: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    output_data: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_messages: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Resource usage
    cpu_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_usage: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    pipeline: Mapped[PipelineModel] = relationship(
        "PipelineModel",
        back_populates="executions",
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "pipeline_id",
            "execution_number",
            name="uq_pipeline_execution_number",
        ),
    )


class PluginModel(Base):
    """SQLAlchemy model for plugin registry.

    Represents a reusable plugin component that can be used in pipeline steps.
    Plugins encapsulate specific data integration functionality such as
    extractors (taps), loaders (targets), transformers, or orchestrators.

    Each plugin has a unique name and contains metadata about its installation,
    configuration schema, and documentation. Plugins are versioned and can be
    shared across multiple pipelines.

    Attributes:
    ----------
        id: UUID primary key for the plugin
        name: Unique plugin identifier (e.g., "tap-postgres")
        plugin_type: Category of plugin (extractor, loader, transformer, etc.)
        namespace: Python namespace for the plugin package
        pip_url: Optional pip installation URL or package name
        configuration: JSON dict of default configuration settings
        version: Semantic version of the plugin
        description: Human-readable plugin description
        documentation_url: Link to plugin documentation
        keywords: JSON array of searchable keywords
        pipeline_steps: Relationship to steps using this plugin

    Examples:
    --------
        Registering a new plugin:

        ```python
        plugin = PluginModel(
            id=uuid4(),
            name="tap-postgres",
            plugin_type="extractor",
            namespace="tap_postgres",
            pip_url="pipelinewise-tap-postgres==1.8.0",
            configuration={
                "host": "localhost",
                "port": 5432,
                "default_replication_method": "INCREMENTAL"
            },
            version="1.8.0",
            keywords=["postgres", "database", "sql", "extractor"]
        )
        session.add(plugin)
        ```

    See Also:
    --------
        - [Plugin Architecture](../../docs/architecture/003-plugin-system.md)
        - [Plugin Development Guide](../../docs/development/plugin-guide.md)

    Note:
    ----
        Plugin configurations are merged with step-specific overrides
        during pipeline execution.

    """

    __tablename__ = "plugins"

    # Identity
    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    plugin_type: Mapped[str] = mapped_column(String(50), nullable=False)
    namespace: Mapped[str] = mapped_column(String(100), nullable=False)
    variant: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Plugin details
    pip_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    documentation_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Installation and status tracking
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="inactive")
    metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Audit fields
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.now, onupdate=datetime.now)

    # Relationships
    pipeline_steps: Mapped[list[PipelineStepModel]] = relationship(
        "PipelineStepModel",
        back_populates="plugin",
    )


# === AUTHENTICATION MODELS - ZERO TOLERANCE MIGRATION FROM flext_auth ===


class UserModel(Base):
    """User database model for authentication and authorization.

    SQLAlchemy model representing users in the authentication system with
    comprehensive fields for identity management, security, and auditing.
    """

    __tablename__ = "auth_users"

    id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        primary_key=True,
        default=uuid4,
    )
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    user_roles: Mapped[list[UserRoleModel]] = relationship(
        "UserRoleModel",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class RoleModel(Base):
    """Role database model for role-based access control."""

    __tablename__ = "auth_roles"

    id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    role_permissions: Mapped[list[RolePermissionModel]] = relationship(
        "RolePermissionModel",
        back_populates="role",
        cascade="all, delete-orphan",
    )
    user_roles: Mapped[list[UserRoleModel]] = relationship(
        "UserRoleModel",
        back_populates="role",
        cascade="all, delete-orphan",
    )


class PermissionModel(Base):
    """Permission database model for granular access control."""

    __tablename__ = "auth_permissions"

    id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    resource: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    role_permissions: Mapped[list[RolePermissionModel]] = relationship(
        "RolePermissionModel",
        back_populates="permission",
        cascade="all, delete-orphan",
    )


class UserRoleModel(Base):
    """User-Role association model for many-to-many relationships."""

    __tablename__ = "auth_user_roles"

    user_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("auth_users.id"),
        primary_key=True,
    )
    role_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("auth_roles.id"),
        primary_key=True,
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    user: Mapped[UserModel] = relationship("UserModel", back_populates="user_roles")
    role: Mapped[RoleModel] = relationship("RoleModel", back_populates="user_roles")


class RolePermissionModel(Base):
    """Role-Permission association model for many-to-many relationships."""

    __tablename__ = "auth_role_permissions"

    role_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("auth_roles.id"),
        primary_key=True,
    )
    permission_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("auth_permissions.id"),
        primary_key=True,
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    role: Mapped[RoleModel] = relationship(
        "RoleModel",
        back_populates="role_permissions",
    )
    permission: Mapped[PermissionModel] = relationship(
        "PermissionModel",
        back_populates="role_permissions",
    )
