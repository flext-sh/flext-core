"""Result-flow and DI modernizer transformer.

Applies conservative AST rewrites for result-flow and dependency-injection
anti-patterns:

- ``raise ValueError("msg")`` inside functions that already use the ``r``
  result alias → ``return r[str].fail("msg", error_code="RESULT_VALUE_ERROR")``.
- ``from dependency_injector import containers, providers`` →
  ``from flext_core.di import containers, providers``.
- ``from dependency_injector.containers import DeclarativeContainer`` →
  ``from flext_core.di.containers import DeclarativeContainer``.
- ``from dependency_injector.providers import Factory, Singleton`` →
  ``from flext_core.di.providers import Factory, Singleton``.

The transformer is intentionally conservative: it only rewrites code when the
result is unambiguous and records every change.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c
from flext_infra.transformers._rewrite import FlextInfraSourceRewriter
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorResultDiModernizer(FlextInfraRopeTransformer):
    """AST-driven transformer for result-flow and DI anti-patterns."""

    _description = (
        "migrate result-flow and dependency-injector patterns to FLEXT canonical forms"
    )

    _CORE_PKG: ClassVar[str] = c.Infra.PKG_CORE_UNDERSCORE
    _DI_FACADE: ClassVar[str] = "flext_core.di"
    _VALUE_ERROR_CODE: ClassVar[str] = "RESULT_VALUE_ERROR"

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply result-flow and DI modernizations to source text."""
        self.changes.clear()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, list(self.changes)

        has_result_alias = self._module_imports_result_alias(tree)
        visitor = self._ResultDiVisitor(source, has_result_alias=has_result_alias)
        visitor.visit(tree)

        if not visitor.rewrites:
            return source, list(self.changes)

        updated = FlextInfraSourceRewriter.apply_rewrites(source, visitor.rewrites)
        for change in visitor.changes:
            self._record_change(change)

        return updated, list(self.changes)

    @classmethod
    def _module_imports_result_alias(cls, tree: ast.AST) -> bool:
        """Return whether the module imports ``r`` from ``flext_core``."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != cls._CORE_PKG:
                continue
            if any(alias.name == "r" for alias in node.names):
                return True
        return False

    class _ResultDiVisitor(FlextInfraSourceRewriter):
        """Collect rewrites for result-flow and DI anti-patterns."""

        def __init__(self, source: str, *, has_result_alias: bool) -> None:
            super().__init__(source)
            self._has_result_alias = has_result_alias

        @override
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            """Canonicalize dependency_injector imports."""
            module = node.module
            if module == "dependency_injector":
                self._rewrite_di_root_import(node)
            elif module == "dependency_injector.containers":
                self._rewrite_di_submodule_import(
                    node, f"{self._di_facade()}.containers", "DeclarativeContainer"
                )
            elif module == "dependency_injector.providers":
                self._rewrite_di_submodule_import(
                    node, f"{self._di_facade()}.providers", "Factory", "Singleton"
                )
            self.generic_visit(node)

        def _rewrite_di_root_import(self, node: ast.ImportFrom) -> None:
            """Rewrite ``from dependency_injector import ...``."""
            names = [alias.name for alias in node.names]
            if not all(name in {"containers", "providers"} for name in names):
                return
            new_text = f"from {self._di_facade()} import {', '.join(names)}\n"
            self.append_rewrite(
                node, new_text, "Rewrote dependency_injector import to flext_core.di"
            )

        def _rewrite_di_submodule_import(
            self, node: ast.ImportFrom, new_module: str, *expected_names: str
        ) -> None:
            """Rewrite a submodule import when all names are expected."""
            names = [alias.name for alias in node.names]
            if not all(name in expected_names for name in names):
                return
            as_clauses = []
            for alias in node.names:
                if alias.asname is not None:
                    as_clauses.append(f"{alias.name} as {alias.asname}")
                else:
                    as_clauses.append(alias.name)
            new_text = f"from {new_module} import {', '.join(as_clauses)}\n"
            self.append_rewrite(
                node, new_text, f"Rewrote {node.module} import to {new_module}"
            )

        @override
        def visit_Raise(self, node: ast.Raise) -> None:
            """Rewrite ``raise ValueError("msg")`` to ``return r[str].fail(...)``."""
            if not self._has_result_alias:
                self.generic_visit(node)
                return

            exc = node.exc
            if not isinstance(exc, ast.Call):
                self.generic_visit(node)
                return
            if not isinstance(exc.func, ast.Name) or exc.func.id != "ValueError":
                self.generic_visit(node)
                return
            if len(exc.args) != 1 or not isinstance(exc.args[0], ast.Constant):
                self.generic_visit(node)
                return
            message = exc.args[0].value
            if not isinstance(message, str):
                self.generic_visit(node)
                return

            new_text = (
                f'return r[str].fail("{message}", error_code="{self._error_code()}")'
            )
            self.append_rewrite(
                node, new_text, "Replaced raise ValueError with r[str].fail result"
            )
            self.generic_visit(node)

        @staticmethod
        def _di_facade() -> str:
            """Return canonical DI facade module path."""
            return FlextInfraRefactorResultDiModernizer._DI_FACADE

        @staticmethod
        def _error_code() -> str:
            """Return error code used for value-error failures."""
            return FlextInfraRefactorResultDiModernizer._VALUE_ERROR_CODE


__all__: list[str] = ["FlextInfraRefactorResultDiModernizer"]
