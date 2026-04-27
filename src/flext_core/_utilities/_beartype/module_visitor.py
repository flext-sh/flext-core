"""Module-level introspection — LOC ceiling enforcement."""

from __future__ import annotations

import inspect
from pathlib import Path

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers

_NO_VIOLATION: t.StrMapping | None = None


class FlextUtilitiesBeartypeModuleVisitor:
    """LOC_CAP + DUPLICATE_SYMBOL visitors."""

    @staticmethod
    def v_loc_cap(
        params: me.LocCapParams,
        target: type,
    ) -> t.StrMapping | None:
        """LOC_CAP — module logical-LOC ceiling (AGENTS.md §3.1)."""
        module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        try:
            source_lines, _start = inspect.getsourcelines(module)
        except (OSError, TypeError):
            return _NO_VIOLATION
        source = "".join(source_lines)
        src_file = inspect.getsourcefile(module) or ""
        loc = sum(
            1
            for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if loc > params.max_logical_loc:
            return {
                "file": Path(src_file).name,
                "loc": str(loc),
                "cap": str(params.max_logical_loc),
            }
        return _NO_VIOLATION

    @staticmethod
    def v_duplicate_symbol(
        _params: me.DuplicateSymbolParams,
        _target: type,
    ) -> t.StrMapping | None:
        """DUPLICATE_SYMBOL — workspace cross-project SSOT (Phase 3 hook).

        Implementation lives in the workspace walker, not the per-class
        runtime hook — needs the cross-project symbol index that only the
        walker can build. Returns None at runtime.
        """
        return _NO_VIOLATION
