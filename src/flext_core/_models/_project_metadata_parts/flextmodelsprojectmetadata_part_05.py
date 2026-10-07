"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_01 import (  # ruff: ignore[line-too-long] -- the dotted module path is a single identifier chain; it cannot be wrapped without renaming the module
    FlextModelsProjectMetadataContract,
)
from flext_core._models.pydantic import FlextModelsPydantic


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
