# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Project Metadata Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._project_metadata.aggregates import (
        FlextModelsProjectMetadataAggregates,
    )
    from flext_core._models._project_metadata.contract import (
        FlextModelsProjectMetadataContract,
    )
    from flext_core._models._project_metadata.document import (
        FlextModelsProjectMetadataDocument,
    )
    from flext_core._models._project_metadata.pyproject_ingress import (
        FlextModelsPyprojectIngressContract,
    )


__all__: tuple[str, ...] = (
    "FlextModelsProjectMetadataAggregates",
    "FlextModelsProjectMetadataContract",
    "FlextModelsProjectMetadataDocument",
    "FlextModelsPyprojectIngressContract",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsProjectMetadataAggregates": ".aggregates",
        "FlextModelsProjectMetadataContract": ".contract",
        "FlextModelsProjectMetadataDocument": ".document",
        "FlextModelsPyprojectIngressContract": ".pyproject_ingress",
    }),
    public_exports=__all__,
)[0]
