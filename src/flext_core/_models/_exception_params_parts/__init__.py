# AUTO-GENERATED FILE — Regenerate with: make gen
"""Exception Params Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._models._exception_params_parts.flextmodelsexceptionparams_part_03 import (
        FlextModelsExceptionParams as FlextModelsExceptionParams,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextmodelsexceptionparams_part_03": ("FlextModelsExceptionParams",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
