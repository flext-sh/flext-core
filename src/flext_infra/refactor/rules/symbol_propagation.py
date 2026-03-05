"""Rule that propagates refactor API/module renames across callsites."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import libcst as cst
from libcst.metadata import MetadataWrapper, QualifiedNameProvider

from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.symbol_propagator import (
    FlextInfraRefactorSymbolPropagator,
)


class FlextInfraRefactorSymbolPropagationRule(FlextInfraRefactorRule):
    """Apply declarative module/symbol renames for workspace-wide propagation."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        target_modules_raw = self.config.get("target_modules", [])
        module_renames_raw = self.config.get("module_renames", {})
        symbol_renames_raw = self.config.get("import_symbol_renames", {})

        target_modules = {
            str(item)
            for item in cast("list[object]", target_modules_raw)
            if isinstance(item, str)
        }
        module_renames = {
            str(k): str(v)
            for k, v in cast("dict[object, object]", module_renames_raw).items()
            if isinstance(k, str) and isinstance(v, str)
        }
        symbol_renames = {
            str(k): str(v)
            for k, v in cast("dict[object, object]", symbol_renames_raw).items()
            if isinstance(k, str) and isinstance(v, str)
        }

        if not target_modules and not module_renames and not symbol_renames:
            return tree, []

        transformer = FlextInfraRefactorSymbolPropagator(
            target_modules=target_modules,
            module_renames=module_renames,
            import_symbol_renames=symbol_renames,
        )
        wrapper = MetadataWrapper(tree)
        return wrapper.visit(transformer), transformer.changes


class FlextInfraRefactorSignaturePropagator(cst.CSTTransformer):
    """Apply declarative signature migrations to callsites."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(
        self,
        *,
        migrations: list[dict[str, object]],
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize transformer state for declarative signature migrations."""
        self._migrations = migrations
        self._on_change = on_change
        self.changes: list[str] = []

    @override
    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.BaseExpression:
        qualified_names = {
            item.name
            for item in self.get_metadata(
                QualifiedNameProvider,
                original_node.func,
                default=set(),
            )
        }

        simple_name = self._simple_callable_name(original_node.func)
        result_call = updated_node

        for migration in self._migrations:
            if not self._matches_migration(
                migration,
                qualified_names=qualified_names,
                simple_name=simple_name,
            ):
                continue

            migration_id = str(migration.get("id", "signature-migration"))
            keyword_renames = self._keyword_renames(migration)
            remove_keywords = self._remove_keywords(migration)
            add_keywords = self._add_keywords(migration)

            next_args: list[cst.Arg] = []
            changed = False
            seen_keyword_names: set[str] = set()

            for arg in list(result_call.args):
                keyword_name = self._keyword_name(arg)
                if keyword_name is not None and keyword_name in remove_keywords:
                    changed = True
                    self._record_change(
                        f"[{migration_id}] Removed keyword: {keyword_name}"
                    )
                    continue

                if keyword_name is not None and keyword_name in keyword_renames:
                    renamed = keyword_renames[keyword_name]
                    next_args.append(arg.with_changes(keyword=cst.Name(renamed)))
                    seen_keyword_names.add(renamed)
                    changed = True
                    self._record_change(
                        f"[{migration_id}] Renamed keyword: {keyword_name} -> {renamed}"
                    )
                    continue

                next_args.append(arg)
                if keyword_name is not None:
                    seen_keyword_names.add(keyword_name)

            for key, value_literal in add_keywords.items():
                if key in seen_keyword_names:
                    continue
                next_args.append(
                    cst.Arg(
                        value=self._literal_expr(value_literal), keyword=cst.Name(key)
                    )
                )
                changed = True
                self._record_change(
                    f"[{migration_id}] Added keyword: {key}={value_literal}"
                )

            if changed:
                result_call = result_call.with_changes(args=tuple(next_args))

        return result_call

    def _matches_migration(
        self,
        migration: dict[str, object],
        *,
        qualified_names: set[str],
        simple_name: str | None,
    ) -> bool:
        target_qualified = {
            item
            for item in cast(
                "list[object]", migration.get("target_qualified_names", [])
            )
            if isinstance(item, str)
        }
        target_simple = {
            item
            for item in cast("list[object]", migration.get("target_simple_names", []))
            if isinstance(item, str)
        }

        if target_qualified and qualified_names.intersection(target_qualified):
            return True
        return bool(
            simple_name is not None and target_simple and simple_name in target_simple
        )

    def _keyword_renames(self, migration: dict[str, object]) -> dict[str, str]:
        raw = migration.get("keyword_renames", {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(k): str(v)
            for k, v in cast("dict[object, object]", raw).items()
            if isinstance(k, str) and isinstance(v, str)
        }

    def _remove_keywords(self, migration: dict[str, object]) -> set[str]:
        raw = migration.get("remove_keywords", [])
        if not isinstance(raw, list):
            return set()
        return {item for item in cast("list[object]", raw) if isinstance(item, str)}

    def _add_keywords(self, migration: dict[str, object]) -> dict[str, str]:
        raw = migration.get("add_keywords", {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(k): str(v)
            for k, v in cast("dict[object, object]", raw).items()
            if isinstance(k, str) and isinstance(v, str)
        }

    def _keyword_name(self, arg: cst.Arg) -> str | None:
        if arg.keyword is None:
            return None
        return arg.keyword.value

    def _simple_callable_name(self, expr: cst.BaseExpression) -> str | None:
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return expr.attr.value
        return None

    def _literal_expr(self, value: str) -> cst.BaseExpression:
        lowered = value.strip().lower()
        if lowered == "true":
            return cst.Name("True")
        if lowered == "false":
            return cst.Name("False")
        if lowered == "none":
            return cst.Name("None")
        if value.isdigit():
            return cst.Integer(value)
        if value.startswith('"') and value.endswith('"'):
            return cst.SimpleString(value)
        if value.startswith("'") and value.endswith("'"):
            return cst.SimpleString(value)
        return cst.Name(value)

    def _record_change(self, message: str) -> None:
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)


class FlextInfraRefactorSignaturePropagationRule(FlextInfraRefactorRule):
    """Apply declarative signature migrations in a generic, workspace-safe way."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        migrations_raw = self.config.get("signature_migrations", [])
        migrations: list[dict[str, object]] = []
        for item in cast("list[object]", migrations_raw):
            if not isinstance(item, dict):
                continue
            typed_item = cast("dict[str, object]", item)
            enabled_raw = typed_item.get("enabled", True)
            if not bool(enabled_raw):
                continue
            migrations.append(typed_item)
        if not migrations:
            return tree, []

        transformer = FlextInfraRefactorSignaturePropagator(migrations=migrations)
        wrapper = MetadataWrapper(tree)
        return wrapper.visit(transformer), transformer.changes


__all__ = [
    "FlextInfraRefactorSignaturePropagationRule",
    "FlextInfraRefactorSignaturePropagator",
    "FlextInfraRefactorSymbolPropagationRule",
]
