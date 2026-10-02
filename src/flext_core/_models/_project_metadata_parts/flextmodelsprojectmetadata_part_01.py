"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._models.pydantic import FlextModelsPydantic


class FlextModelsProjectMetadataContract(FlextModelsPydantic.BaseModel):
    """Frozen declaration base for owned project metadata."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
    )


# NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): one non-part,
# declaration-only owner replaces the method-bearing split model hierarchy.
# The standards-owned TOML ingress base lives in part_05 (one class per
# module, NS-000).
