"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from ..pydantic import FlextModelsPydantic


class ProjectMetadataContract(FlextModelsPydantic.BaseModel):
    """Frozen declaration base for owned project metadata."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
    )


class PyprojectIngressContract(ProjectMetadataContract):
    """Frozen declaration base for standards-owned TOML tables."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(
            frozen=True, extra="ignore", populate_by_name=True
        )
    )


# NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): one non-part,
# declaration-only owner replaces the method-bearing split model hierarchy.
