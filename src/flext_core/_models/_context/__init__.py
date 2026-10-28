# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for context.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._context.data import FlextModelsContextData
    from flext_core._models._context.export import FlextModelsContextExport
    from flext_core._models._context.metadata import FlextModelsContextMetadata
    from flext_core._models._context.scope import FlextModelsContextScope
    from flext_core._models._context.tokens import FlextModelsContextTokens

__all__: tuple[str, ...] = (
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextScope",
    "FlextModelsContextTokens",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsContextData": ".data",
        "FlextModelsContextExport": ".export",
        "FlextModelsContextMetadata": ".metadata",
        "FlextModelsContextScope": ".scope",
        "FlextModelsContextTokens": ".tokens",
    }),
    public_exports=__all__,
)
