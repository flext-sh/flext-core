"""Module-level introspection — LOC ceiling, class census, alias shims."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

from .helpers import FlextUtilitiesBeartypeHelpers

_NO_VIOLATION: t.StrMapping | None = None
_MODULE_EXEMPT_FILES: frozenset[str] = frozenset({
    "__init__.py",
    "__main__.py",
    "conftest.py",
})


def _is_synthetic_parametrized_type(value: object) -> bool:
    """Return True for synthetic ``Foo[int]`` specializations.

    Pydantic generic BaseModel subclasses leak parameterized specializations
    into the defining module's namespace (e.g. ``FlextInfraServiceBase[bool]``).
    These are not source-level declarations and must not count toward module
    class caps or backwards-compat alias rules.
    """
    return isinstance(value, type) and "[" in getattr(value, "__qualname__", "")


class FlextUtilitiesBeartypeModuleVisitor:
    """LOC_CAP + MODULE_ALIAS + DUPLICATE_SYMBOL visitors."""

    @staticmethod
    def v_loc_cap(
        params: me.LocCapParams,
        target: type,
    ) -> t.StrMapping | None:
        """LOC_CAP — module logical-LOC ceiling + top-level class census (§3.1)."""
        module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = inspect.getsourcefile(module) or ""
        filename = Path(src_file).name
        if params.max_top_level_classes:
            package = module.__name__.split(".")[0]
            if not package.startswith("flext_") or filename.startswith("_"):
                return _NO_VIOLATION
            try:
                source_lines, _start = inspect.getsourcelines(module)
            except (OSError, TypeError):
                return _NO_VIOLATION
            source = "".join(source_lines)
            try:
                tree = ast.parse(source, filename=src_file)
            except SyntaxError:
                return _NO_VIOLATION
            top_level_class_count = sum(
                1
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and not any(
                    base.id == "Warning"
                    for base in node.bases
                    if isinstance(base, ast.Name)
                )
            )
            if top_level_class_count > params.max_top_level_classes:
                return {
                    "file": filename,
                    "count": str(top_level_class_count),
                    "cap": str(params.max_top_level_classes),
                }
            return _NO_VIOLATION
        try:
            source_lines, _start = inspect.getsourcelines(module)
        except (OSError, TypeError):
            return _NO_VIOLATION
        source = "".join(source_lines)
        loc = sum(
            1
            for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if loc > params.max_logical_loc:
            return {
                "file": filename,
                "loc": str(loc),
                "cap": str(params.max_logical_loc),
            }
        return _NO_VIOLATION

    @staticmethod
    def v_module_alias(
        params: me.AliasRebindParams,
        target: type,
    ) -> t.StrMapping | None:
        """MODULE_ALIAS — module-level CapWords compat alias / nested-class hoist."""
        if params.expected_form != "no_module_compat_alias":
            return _NO_VIOLATION
        module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = inspect.getsourcefile(module) or ""
        filename = Path(src_file).name
        package = module.__name__.split(".")[0]
        if (
            not package.startswith("flext_")
            or filename.startswith("_")
            or filename in c.ENFORCEMENT_CANONICAL_FILES
            or filename in _MODULE_EXEMPT_FILES
        ):
            return _NO_VIOLATION
        return next(
            (
                {"alias": name, "target": value.__name__, "file": filename}
                for name, value in vars(module).items()
                if FlextUtilitiesBeartypeModuleVisitor.is_module_alias_candidate(
                    name,
                    value,
                )
            ),
            _NO_VIOLATION,
        )

    @staticmethod
    def is_module_alias_candidate(
        name: str,
        value: object,
    ) -> bool:
        """Return True when a module-level symbol looks like a compat alias."""
        if not isinstance(value, type):
            return False
        if _is_synthetic_parametrized_type(value):
            return False
        if not name[:1].isupper() or name.isupper():
            return False
        if value.__name__ == name and "." not in getattr(value, "__qualname__", ""):
            return False
        origin = FlextUtilitiesBeartypeHelpers.object_module_name_for(value) or ""
        return origin.startswith("flext_")

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
