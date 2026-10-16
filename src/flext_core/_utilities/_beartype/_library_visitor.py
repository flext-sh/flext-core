"""Library abstraction owner enforcement visitor.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c
from flext_core._models import FlextModelsEnforcement
from flext_core._typings import FlextTypingBase
from flext_core._utilities._beartype import (
    FlextUtilitiesBeartypeHelpers,
    FlextUtilitiesBeartypeModuleSource,
)

_NO_VIOLATION: FlextTypingBase.StrMapping | None = None


class FlextUtilitiesBeartypeLibraryVisitor:
    """LIBRARY_IMPORT visitor — §2.7 library abstraction owner enforcement."""

    @staticmethod
    def v_library_import(
        params: FlextModelsEnforcement.LibraryImportParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
        """LIBRARY_IMPORT — §2.7 library abstraction owner enforcement (Phase 3 hook).

        A namespace member whose runtime origin is an owned library only
        violates when its module-level binding is not proven to derive from
        the owner project's imports: bindings rooted in owner imports are
        facade provenance (legal), while direct imports, aliased imports,
        and dynamic ``__import__`` acquisitions stay violations.
        Rule-selected roots use the same owner/provenance boundary, regardless
        of whether the declaring module is public or private.

        Returns:
            The resulting ``t.StrMapping | None``.

        Raises:
            ValueError: If a selected library has no canonical owner.

        """
        owners = c.ENFORCEMENT_LIBRARY_OWNERS
        unknown = set(params.modules).difference(owners)
        if unknown:
            msg = f"Selected libraries have no canonical owner: {sorted(unknown)}"
            raise ValueError(msg)
        module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        package = target.__module__.split(".")[0].replace("_", "-")
        candidates = tuple(
            (name, origin_root, owners[origin_root])
            for name, value in vars(module).items()
            if (origin := FlextUtilitiesBeartypeHelpers.object_module_name_for(value))
            is not None
            and (origin_root := origin.split(".")[0]) in owners
            and (not params.modules or origin_root in params.modules)
            and owners[origin_root] != package
        )
        if not candidates:
            return _NO_VIOLATION
        try:
            tree = FlextUtilitiesBeartypeModuleSource.parse_module(module)
        except (OSError, TypeError, SyntaxError):
            tree = None
        for name, origin_root, owner in candidates:
            owner_root = owner.replace("-", "_")
            if FlextUtilitiesBeartypeModuleSource.owner_derived(tree, name, owner_root):
                continue
            return {"lib": origin_root, "owner": owner, "package": package}
        return _NO_VIOLATION


__all__: list[str] = ["FlextUtilitiesBeartypeLibraryVisitor"]
