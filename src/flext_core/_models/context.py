"""Context management patterns — MRO-composed facade.

This module composes all context model sub-namespaces via MRO into a single
FlextModelsContext class. Access nested classes via FlextModels.* aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextModelsContextData,
    FlextModelsContextExport,
    FlextModelsContextMetadata,
    FlextModelsContextProxyVar,
    FlextModelsContextScope,
    FlextModelsContextTokens,
)


class FlextModelsContext(
    FlextModelsContextTokens,
    FlextModelsContextProxyVar,
    FlextModelsContextData,
    FlextModelsContextExport,
    FlextModelsContextScope,
    FlextModelsContextMetadata,
):
    """Context management pattern container — MRO facade.

    Composes all context model namespaces. Nested classes accessible via
    FlextModels.* aliases (e.g. m.ContextData, m.StructlogProxyContextVar).
    """


__all__ = ["FlextModelsContext"]
