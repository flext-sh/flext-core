"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._models._project_metadata_parts import (
    flextmodelsprojectmetadata_part_01 as _01,
)
from flext_core._models.pydantic import FlextModelsPydantic

FlextModelsProjectMetadataContract = _01.FlextModelsProjectMetadataContract


class FlextModelsPyprojectIngressContract(FlextModelsProjectMetadataContract):
    """Frozen declaration base for standards-owned TOML tables."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(
            frozen=True,
            extra="ignore",
            populate_by_name=True,
        )
    )


__all__: list[str] = ["FlextModelsPyprojectIngressContract"]
