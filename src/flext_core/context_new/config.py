"""Context configuration model (new).

Single-class module: defines `FlextContextConfig` only.
"""

from __future__ import annotations

from pydantic import Field

from flext_core.constants import FlextConstants
from flext_core.models import FlextModels


class FlextContextConfig(FlextModels.BaseConfig):
    """Typed configuration for context system behavior."""

    environment: FlextConstants.Config.ConfigEnvironment = Field(
        default=FlextConstants.Config.ConfigEnvironment.DEVELOPMENT
    )
    log_level: FlextConstants.Config.LogLevel = Field(
        default=FlextConstants.Config.LogLevel.DEBUG
    )

    enable_correlation_tracking: bool = Field(default=True)
    enable_service_context: bool = Field(default=True)
    enable_performance_tracking: bool = Field(default=True)
    context_propagation_enabled: bool = Field(default=True)
    max_context_depth: int = Field(default=20, ge=0)
    context_serialization_enabled: bool = Field(default=True)
    context_cleanup_enabled: bool = Field(default=True)
    enable_nested_contexts: bool = Field(default=True)
