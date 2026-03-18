"""Context management patterns — MRO-composed facade.

This module composes all context model sub-namespaces via MRO into a single
FlextModelsContext class. Access nested classes via FlextModels.* aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._context._data import FlextModelsContextData
from flext_core._models._context._export import FlextModelsContextExport
from flext_core._models._context._metadata import FlextModelsContextMetadata
from flext_core._models._context._proxy_var import FlextModelsContextProxyVar
from flext_core._models._context._scope import FlextModelsContextScope
from flext_core._models._context._tokens import FlextModelsContextTokens


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
