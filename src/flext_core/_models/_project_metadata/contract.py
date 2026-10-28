"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._models import FlextModelsPydantic


class FlextModelsProjectMetadataContract(FlextModelsPydantic.BaseModel):
    """Frozen declaration base for owned project metadata."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
    )


__all__: list[str] = ["FlextModelsProjectMetadataContract"]
