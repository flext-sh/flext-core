"""Shared models for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides basic shared models that are technology-agnostic
and serve as the foundation for domain models across FLEXT projects.

ONLY ABSTRACT/GENERIC MODELS - No technology-specific implementations.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import Field

# ==============================================================================
# HEALTH AND MONITORING MODELS - GENERIC
# ==============================================================================


class HealthStatus(StrEnum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Component health information."""

    component_name: str = Field(description="Component name")
    status: HealthStatus = Field(description="Health status")
    message: str | None = Field(default=None, description="Health message")
    last_check: str | None = Field(
        default=None,
        description="Last health check timestamp",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health details",
    )


# ==============================================================================
# PLUGIN METADATA MODELS - GENERIC
# ==============================================================================


class PluginMetadata(BaseModel):
    """Plugin metadata for registration and discovery."""

    name: str = Field(description="Plugin name")
    version: str | None = Field(default=None, description="Plugin version")
    author: str | None = Field(default=None, description="Plugin author")
    description: str | None = Field(default=None, description="Plugin description")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Plugin capabilities",
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Plugin requirements",
    )
    config_schema: dict[str, Any] | None = Field(
        default=None,
        description="Configuration schema",
    )


# ==============================================================================
# BASIC RESULT MODELS - GENERIC
# ==============================================================================


class ValidationResult(BaseModel):
    """Generic validation result."""

    is_valid: bool = Field(description="Validation status")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Validation details",
    )


# ==============================================================================
# EXPORTS - ONLY ABSTRACT/GENERIC MODELS
# ==============================================================================

__all__ = [
    "ComponentHealth",
    # Health models
    "HealthStatus",
    # Plugin models
    "PluginMetadata",
    # Result models
    "ValidationResult",
]
