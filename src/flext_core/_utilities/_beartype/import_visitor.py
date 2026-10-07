"""Import discipline enforcement — blacklist + alias rebind + library owners.

The IMPORT_BLACKLIST implementation lives in its own module
(one top-level class per module, NS-000); this visitor facade binds the
family predicates to their canonical implementations.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models import FlextModelsEnforcement
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities import (
    FlextUtilitiesBeartypeAliasVisitor,
    FlextUtilitiesBeartypeLibraryVisitor,
)
from flext_core._utilities._beartype._import_blacklist_visitor import (
    _ImportBlacklistVisitor,
)


class FlextUtilitiesBeartypeImportVisitor:
    """IMPORT_BLACKLIST + ALIAS_REBIND + LIBRARY_IMPORT visitor facade."""

    @staticmethod
    def v_import_blacklist(
        params: FlextModelsEnforcement.ImportBlacklistParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
        """IMPORT_BLACKLIST — concrete-class / pydantic consumer-import discipline.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        return _ImportBlacklistVisitor.v_import_blacklist(params, target)

    @staticmethod
    def v_alias_rebind(
        params: FlextModelsEnforcement.AliasRebindParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
        """ALIAS_REBIND.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        return FlextUtilitiesBeartypeAliasVisitor.v_alias_rebind(params, target)

    @staticmethod
    def v_compatibility_alias(
        params: FlextModelsEnforcement.CompatibilityAliasParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
        """COMPATIBILITY_ALIAS.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        return FlextUtilitiesBeartypeAliasVisitor.v_compatibility_alias(params, target)

    @staticmethod
    def v_library_import(
        params: FlextModelsEnforcement.LibraryImportParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
        """LIBRARY_IMPORT.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        return FlextUtilitiesBeartypeLibraryVisitor.v_library_import(params, target)


__all__: list[str] = ["FlextUtilitiesBeartypeImportVisitor"]
