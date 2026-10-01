"""Library abstraction owner enforcement visitor."""

from __future__ import annotations

from ..._constants.enforcement import FlextConstantsEnforcement as c
from ..._models.enforcement import FlextModelsEnforcement as me
from ..._typings.base import FlextTypingBase as t
from .helpers import FlextUtilitiesBeartypeHelpers as _ubh
from .module_source import FlextUtilitiesBeartypeModuleSource

_NO_VIOLATION: t.StrMapping | None = None


class FlextUtilitiesBeartypeLibraryVisitor:
    """LIBRARY_IMPORT visitor — §2.7 library abstraction owner enforcement."""

    @staticmethod
    def v_library_import(
        params: me.LibraryImportParams,
        target: type,
    ) -> t.StrMapping | None:
        """LIBRARY_IMPORT — §2.7 library abstraction owner enforcement (Phase 3 hook).

        A namespace member whose runtime origin is an owned library only
        violates when its module-level binding is not proven to derive from
        the owner project's imports: bindings rooted in owner imports are
        facade provenance (legal), while direct imports, aliased imports,
        and dynamic ``__import__`` acquisitions stay violations.
        """
        _ = params
        owners = c.ENFORCEMENT_LIBRARY_OWNERS
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        package = target.__module__.split(".")[0].replace("_", "-")
        candidates = tuple(
            (name, origin_root, owners[origin_root])
            for name, value in vars(module).items()
            if (origin := _ubh.object_module_name_for(value)) is not None
            and (origin_root := origin.split(".")[0]) in owners
            and owners[origin_root] != package
        )
        if not candidates:
            return _NO_VIOLATION
        try:
            tree = FlextUtilitiesBeartypeModuleSource.parse(module)
        except (OSError, TypeError, SyntaxError):
            tree = None
        for name, origin_root, owner in candidates:
            owner_root = owner.replace("-", "_")
            if FlextUtilitiesBeartypeModuleSource.owner_derived(tree, name, owner_root):
                continue
            return {"lib": origin_root, "owner": owner, "package": package}
        return _NO_VIOLATION


__all__: list[str] = ["FlextUtilitiesBeartypeLibraryVisitor"]
