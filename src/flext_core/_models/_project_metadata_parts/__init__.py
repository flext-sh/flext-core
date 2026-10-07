# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Project Metadata Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
        FlextModelsProjectMetadataFields,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
        FlextModelsProjectMetadataAggregates,
        FlextModelsProjectMetadataContract,
        FlextModelsProjectMetadataDocument,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_05 import (
        FlextModelsPyprojectIngressContract,
    )


__all__: tuple[str, ...] = (
    "FlextModelsProjectMetadataAggregates",
    "FlextModelsProjectMetadataContract",
    "FlextModelsProjectMetadataDocument",
    "FlextModelsProjectMetadataFields",
    "FlextModelsPyprojectIngressContract",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsProjectMetadataAggregates": ".flextmodelsprojectmetadata_part_04",
        "FlextModelsProjectMetadataContract": ".flextmodelsprojectmetadata_part_04",
        "FlextModelsProjectMetadataDocument": ".flextmodelsprojectmetadata_part_04",
        "FlextModelsProjectMetadataFields": ".flextmodelsprojectmetadata_part_03",
        "FlextModelsPyprojectIngressContract": ".flextmodelsprojectmetadata_part_05",
    }),
    public_exports=__all__,
)
