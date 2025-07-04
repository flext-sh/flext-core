"""Core domain entities for FLEXT framework."""

from datetime import datetime

# Python < 3.11 compatibility for datetime.UTC
try:
    from datetime import UTC
except ImportError:
    import datetime
    UTC = datetime.UTC
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from flext_core.domain.base import (
    PipelineId,
    PluginId,
    create_pipeline_id,
    create_plugin_id,
)


class PipelineName(BaseModel):
    """Pipeline name value object with validation."""

    value: str = Field(..., min_length=1, max_length=100, description="Pipeline name")

    @field_validator("value")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate pipeline name format."""
        if not v.replace("-", "").replace("_", "").replace(".", "").isalnum():
            msg = "Pipeline name must contain only alphanumeric characters, hyphens, underscores, and dots"
            raise ValueError(
                msg
            )
        return v

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"PipelineName('{self.value}')"


class ExecutionStatus(StrEnum):
    """Pipeline execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

    def is_terminal(self) -> bool:
        """Check if status is terminal (execution finished)."""
        return self in {self.SUCCESS, self.FAILED, self.CANCELLED, self.TIMEOUT}

    def is_active(self) -> bool:
        """Check if execution is currently active."""
        return self in {self.PENDING, self.RUNNING}


class Pipeline(BaseModel):
    """Core pipeline entity."""

    id: PipelineId = Field(
        default_factory=create_pipeline_id, description="Unique pipeline identifier"
    )
    name: PipelineName = Field(..., description="Pipeline name")
    description: str = Field(
        default="", max_length=500, description="Pipeline description"
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Pipeline configuration"
    )
    tags: list[str] = Field(default_factory=list, description="Pipeline tags")
    is_active: bool = Field(default=True, description="Whether pipeline is active")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    model_config: ClassVar = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            PipelineId: str,
        },
    }

    def update_configuration(self, config: dict[str, Any]) -> None:
        """Update pipeline configuration."""
        self.configuration.update(config)
        self.updated_at = datetime.now(UTC)

    def add_tag(self, tag: str) -> None:
        """Add tag to pipeline."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now(UTC)

    def remove_tag(self, tag: str) -> None:
        """Remove tag from pipeline."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now(UTC)

    def deactivate(self) -> None:
        """Deactivate pipeline."""
        self.is_active = False
        self.updated_at = datetime.now(UTC)

    def activate(self) -> None:
        """Activate pipeline."""
        self.is_active = True
        self.updated_at = datetime.now(UTC)


class PipelineStep(BaseModel):
    """Pipeline step entity for defining pipeline execution steps."""

    id: str = Field(
        default_factory=lambda: f"step_{datetime.now(UTC).timestamp()}",
        description="Unique step identifier",
    )
    name: str = Field(..., min_length=1, max_length=100, description="Step name")
    description: str = Field(default="", max_length=500, description="Step description")
    step_type: str = Field(
        ..., description="Type of step (extract, transform, load, etc.)"
    )
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Step configuration"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Step dependencies (other step IDs)"
    )
    order: int = Field(default=0, description="Execution order within pipeline")
    timeout_seconds: int = Field(default=3600, description="Step timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    is_enabled: bool = Field(default=True, description="Whether step is enabled")

    model_config: ClassVar = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        },
    }

    def enable(self) -> None:
        """Enable step execution."""
        self.is_enabled = True

    def disable(self) -> None:
        """Disable step execution."""
        self.is_enabled = False

    def add_dependency(self, step_id: str) -> None:
        """Add step dependency."""
        if step_id not in self.dependencies:
            self.dependencies.append(step_id)

    def remove_dependency(self, step_id: str) -> None:
        """Remove step dependency."""
        if step_id in self.dependencies:
            self.dependencies.remove(step_id)


class PipelineExecution(BaseModel):
    """Pipeline execution entity for tracking execution state and metrics."""

    id: str = Field(
        default_factory=lambda: f"exec_{datetime.now(UTC).timestamp()}",
        description="Unique execution identifier",
    )
    pipeline_id: PipelineId = Field(..., description="Associated pipeline identifier")
    status: ExecutionStatus = Field(
        default=ExecutionStatus.PENDING, description="Current execution status"
    )

    # Execution timing
    started_at: datetime | None = Field(
        default=None, description="Execution start timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Execution completion timestamp"
    )
    duration_seconds: float | None = Field(
        default=None, description="Execution duration in seconds"
    )

    # Execution context
    triggered_by: str | None = Field(
        default=None, description="Who/what triggered the execution"
    )
    execution_context: dict[str, Any] = Field(
        default_factory=dict, description="Execution context data"
    )

    # Results and metrics
    result_data: dict[str, Any] = Field(
        default_factory=dict, description="Execution result data"
    )
    error_message: str | None = Field(
        default=None, description="Error message if execution failed"
    )
    logs: list[str] = Field(default_factory=list, description="Execution logs")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )

    model_config: ClassVar = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            PipelineId: str,
        },
    }

    def start(self, triggered_by: str | None = None) -> None:
        """Start execution."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.now(UTC)
        self.triggered_by = triggered_by

    def complete_success(self, result_data: dict[str, Any] | None = None) -> None:
        """Mark execution as successfully completed."""
        self.status = ExecutionStatus.SUCCESS
        self.completed_at = datetime.now(UTC)
        if result_data:
            self.result_data.update(result_data)
        self._calculate_duration()

    def complete_failure(self, error_message: str) -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error_message = error_message
        self._calculate_duration()

    def cancel(self) -> None:
        """Cancel execution."""
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.now(UTC)
        self._calculate_duration()

    def add_log(self, message: str) -> None:
        """Add log message to execution."""
        timestamp = datetime.now(UTC).isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    def _calculate_duration(self) -> None:
        """Calculate execution duration."""
        if self.started_at and self.completed_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()

    @property
    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.status == ExecutionStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """Check if execution is completed (success or failure)."""
        return self.status.is_terminal()

    @property
    def was_successful(self) -> bool:
        """Check if execution completed successfully."""
        return self.status == ExecutionStatus.SUCCESS


class Plugin(BaseModel):
    """Plugin entity for managing FLEXT plugins."""

    id: PluginId = Field(
        default_factory=lambda: create_plugin_id(
            f"plugin_{datetime.now(UTC).timestamp()}"
        ),
        description="Unique plugin identifier",
    )
    name: str = Field(..., min_length=1, max_length=100, description="Plugin name")
    version: str = Field(default="1.0.0", description="Plugin version")
    description: str = Field(
        default="", max_length=500, description="Plugin description"
    )
    plugin_type: str = Field(
        ..., description="Type of plugin (tap, target, transform, etc.)"
    )
    author: str = Field(default="", description="Plugin author")
    license: str = Field(default="", description="Plugin license")

    # Plugin metadata
    configuration_schema: dict[str, Any] = Field(
        default_factory=dict, description="Configuration schema"
    )
    settings: dict[str, Any] = Field(
        default_factory=dict, description="Plugin settings"
    )
    capabilities: list[str] = Field(
        default_factory=list, description="Plugin capabilities"
    )

    # Plugin state
    is_installed: bool = Field(default=False, description="Whether plugin is installed")
    is_enabled: bool = Field(default=True, description="Whether plugin is enabled")
    installation_path: str | None = Field(
        default=None, description="Plugin installation path"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    model_config: ClassVar = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            PluginId: str,
        },
    }

    def install(self, installation_path: str) -> None:
        """Mark plugin as installed."""
        self.is_installed = True
        self.installation_path = installation_path
        self.updated_at = datetime.now(UTC)

    def uninstall(self) -> None:
        """Mark plugin as uninstalled."""
        self.is_installed = False
        self.installation_path = None
        self.updated_at = datetime.now(UTC)

    def enable(self) -> None:
        """Enable plugin."""
        self.is_enabled = True
        self.updated_at = datetime.now(UTC)

    def disable(self) -> None:
        """Disable plugin."""
        self.is_enabled = False
        self.updated_at = datetime.now(UTC)

    def update_settings(self, settings: dict[str, Any]) -> None:
        """Update plugin settings."""
        self.settings.update(settings)
        self.updated_at = datetime.now(UTC)

    def add_capability(self, capability: str) -> None:
        """Add capability to plugin."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            self.updated_at = datetime.now(UTC)
