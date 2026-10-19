"""Context management patterns — MRO-composed facade.

This module composes all context model sub-namespaces via MRO into a single
FlextModelsContext class. Access nested classes via FlextModels.* aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import t
from flext_core._models._context import (
    FlextModelsContextData,
    FlextModelsContextExport,
    FlextModelsContextMetadata,
    FlextModelsContextTokens,
)
from flext_core._models._context._scope import FlextModelsContextScope


class FlextModelsContext(
    FlextModelsContextTokens,
    FlextModelsContextData,
    FlextModelsContextExport,
    FlextModelsContextScope,
    FlextModelsContextMetadata,
):
    """Context management pattern container — MRO facade.

    Composes all context model namespaces. Nested classes accessible via
    FlextModels.* aliases (e.g. m.ContextData and m.StructlogProxyToken).
    """


__all__: t.MutableSequenceOf[str] = ["FlextModelsContext"]
