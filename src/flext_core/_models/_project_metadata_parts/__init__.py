# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Project Metadata Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_01 import (
        FlextModelsProjectMetadataContract,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_02 import (
        FlextModelsProjectMetadataFields,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
        FlextModelsProjectMetadataAggregates,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
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

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextmodelsprojectmetadata_part_01": (
                "FlextModelsProjectMetadataContract",
            ),
            ".flextmodelsprojectmetadata_part_02": (
                "FlextModelsProjectMetadataFields",
            ),
            ".flextmodelsprojectmetadata_part_03": (
                "FlextModelsProjectMetadataAggregates",
            ),
            ".flextmodelsprojectmetadata_part_04": (
                "FlextModelsProjectMetadataDocument",
            ),
            ".flextmodelsprojectmetadata_part_05": (
                "FlextModelsPyprojectIngressContract",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
