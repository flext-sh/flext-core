"""Library abstraction owner enforcement visitor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .helpers import FlextUtilitiesBeartypeHelpers as _ubh

# mro-qtjb (flext-core): t at runtime — module-level _NO_VIOLATION annotation
# is evaluated at runtime by beartype claw instrumentation.
from flext_core._typings.base import FlextTypingBase as t

if TYPE_CHECKING:
    from flext_core._protocols.enforcement import FlextProtocolsEnforcement as pe

_NO_VIOLATION: t.StrMapping | None = None


class FlextUtilitiesBeartypeLibraryVisitor:
    """LIBRARY_IMPORT visitor — §2.7 library abstraction owner enforcement."""

    @staticmethod
    def v_library_import(
        params: pe.LibraryImportParams, target: type
    ) -> t.StrMapping | None:
        """LIBRARY_IMPORT — §2.7 library abstraction owner enforcement (Phase 3 hook)."""
        if not params.library_owners:
            return _NO_VIOLATION
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        module_name = getattr(target, "__module__", "") or ""
        package = module_name.split(".")[0].replace("_", "-")
        for value in vars(module).values():
            origin = _ubh.object_module_name_for(value)
            if origin is None:
                continue
            origin_root = origin.split(".")[0]
            owner = params.library_owners.get(origin_root)
            if owner is None or package == owner:
                continue
            return {"lib": origin_root, "owner": owner, "package": package}
        return _NO_VIOLATION


__all__: list[str] = ["FlextUtilitiesBeartypeLibraryVisitor"]
