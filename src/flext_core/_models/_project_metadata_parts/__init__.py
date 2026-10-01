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
        ProjectMetadataContract,
        PyprojectIngressContract,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_02 import (
        ProjectMetadataFields,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
        ProjectMetadataAggregates,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
        ProjectMetadataDocument,
    )


__all__: tuple[str, ...] = (
    "ProjectMetadataAggregates",
    "ProjectMetadataContract",
    "ProjectMetadataDocument",
    "ProjectMetadataFields",
    "PyprojectIngressContract",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextmodelsprojectmetadata_part_01": (
                "ProjectMetadataContract",
                "PyprojectIngressContract",
            ),
            ".flextmodelsprojectmetadata_part_02": ("ProjectMetadataFields",),
            ".flextmodelsprojectmetadata_part_03": ("ProjectMetadataAggregates",),
            ".flextmodelsprojectmetadata_part_04": ("ProjectMetadataDocument",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
