"""Migrate canonical alias imports from flext_core to the owning project.

libcst-based transformer implementing ENFORCE-080: when a project re-exports a
canonical alias locally (c/m/p/t/u), consumers inside that project must import
it from the local facade instead of from flext_core.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

import libcst as cst

from flext_infra import c, m
from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.rope_source import FlextInfraUtilitiesRopeSource
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class _CstImportHelpers:
    """Static libcst helpers for reading and building import statements."""

    # flext-j47u: keep CST rendering typed; Rope remains the semantic source.
    @staticmethod
    def dotted_name(module: cst.BaseExpression | None) -> str | None:
        """Return a dotted name for a libcst import expression."""
        if module is None:
            return None
        parts: list[str] = []
        current: cst.BaseExpression = module
        while isinstance(current, cst.Attribute):
            parts.append(current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.append(current.value)
            return ".".join(reversed(parts))
        return None

    @staticmethod
    def import_aliases(node: cst.ImportFrom) -> tuple[cst.ImportAlias, ...]:
        """Return plain import aliases, ignoring ``*`` imports."""
        if isinstance(node.names, cst.ImportStar):
            return ()
        return tuple(node.names)

    @staticmethod
    def make_import_alias(display: str) -> cst.ImportAlias:
        """Build a libcst ImportAlias from ``Name`` or ``LongName as alias``."""
        if " as " in display:
            name, alias = display.split(" as ", maxsplit=1)
            return cst.ImportAlias(
                name=cst.Name(name), asname=cst.AsName(name=cst.Name(alias))
            )
        return cst.ImportAlias(name=cst.Name(display))

    @staticmethod
    def module_expression(module: str) -> cst.Attribute | cst.Name:
        """Build a typed libcst expression for a dotted module name."""
        parts = module.split(".")
        expression: cst.Attribute | cst.Name = cst.Name(parts[0])
        for part in parts[1:]:
            expression = cst.Attribute(value=expression, attr=cst.Name(part))
        return expression

    @classmethod
    def insert_local_imports(
        cls, tree: cst.Module, imports_to_add: dict[str, dict[str, str]]
    ) -> cst.Module:
        """Prepend newly required local alias imports after __future__/docstring."""
        if not imports_to_add:
            return tree

        new_stmts: list[cst.SimpleStatementLine] = []
        for module in sorted(imports_to_add):
            aliases = [
                cls.make_import_alias(display)
                for _bound, display in sorted(imports_to_add[module].items())
            ]
            new_stmts.append(
                cst.SimpleStatementLine(
                    body=[
                        cst.ImportFrom(
                            module=cls.module_expression(module), names=aliases
                        )
                    ]
                )
            )

        insert_pos = 0
        for idx, stmt in enumerate(tree.body):
            if idx != insert_pos:
                break
            if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
                body_stmt = stmt.body[0]
                if isinstance(body_stmt, cst.ImportFrom):
                    mod = cls.dotted_name(body_stmt.module)
                    if mod == "__future__":
                        insert_pos = idx + 1
                        continue
                if isinstance(body_stmt, cst.Expr) and isinstance(
                    body_stmt.value, cst.SimpleString
                ):
                    insert_pos = idx + 1
                    continue
            break

        new_body = list(tree.body)
        for offset, stmt in enumerate(new_stmts):
            new_body.insert(insert_pos + offset, stmt)
        return tree.with_changes(body=new_body)


class _TypeCheckingContext:
    """Mixin tracking whether we are inside an ``if TYPE_CHECKING:`` block."""

    _tc_stack: list[bool]

    def _enter_if(self, node: cst.If) -> bool:
        """Return True and push when this is a TYPE_CHECKING branch."""
        is_tc = isinstance(node.test, cst.Name) and node.test.value == "TYPE_CHECKING"
        self._tc_stack.append(is_tc)
        return is_tc

    def _leave_if(self) -> None:
        self._tc_stack.pop()

    def _in_type_checking(self) -> bool:
        return any(self._tc_stack)


class _CollectExistingLocal(cst.CSTVisitor, _TypeCheckingContext):
    """Collect bound names already imported from the project's local facades."""

    def __init__(self, import_root: str) -> None:
        self.import_root = import_root
        self.existing_local: dict[str, set[str]] = {}
        self._tc_stack = []

    @override
    def visit_If(self, node: cst.If) -> bool:
        self._enter_if(node)
        return True

    @override
    def leave_If(self, original_node: cst.If) -> None:
        self._leave_if()

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if self._in_type_checking():
            return
        module = _CstImportHelpers.dotted_name(node.module)
        if module is None or not (
            module == self.import_root or module.startswith(f"{self.import_root}.")
        ):
            return
        for alias in _CstImportHelpers.import_aliases(node):
            bound = alias.evaluated_alias or alias.evaluated_name
            self.existing_local.setdefault(module, set()).add(bound)


class _AliasMigrationTransformer(cst.CSTTransformer, _TypeCheckingContext):
    """Rewrite flext_core canonical-alias imports to local project facades."""

    def __init__(
        self,
        *,
        import_root: str,
        local_aliases: frozenset[str],
        existing_local: dict[str, set[str]],
        alias_to_module: t.StrMapping,
        record_change: t.Infra.ChangeCallback,
    ) -> None:
        self._import_root = import_root
        self._local_aliases = local_aliases
        self._existing_local = existing_local
        self._alias_to_module = alias_to_module
        self._record_change = record_change
        self.imports_to_add: dict[str, dict[str, str]] = {}
        self.changes: list[str] = []
        self._tc_stack = []

    @override
    def visit_If(self, node: cst.If) -> bool:
        self._enter_if(node)
        return True

    @override
    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        _ = original_node
        self._leave_if()
        return updated_node

    @override
    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        _ = original_node
        module = _CstImportHelpers.dotted_name(updated_node.module)
        if (
            module is None
            or not module.startswith(c.Infra.PKG_CORE_UNDERSCORE)
            or self._in_type_checking()
        ):
            return updated_node

        kept: list[cst.ImportAlias] = []
        aliases = _CstImportHelpers.import_aliases(updated_node)
        for alias in aliases:
            bound = alias.evaluated_alias or alias.evaluated_name
            if bound not in self._local_aliases:
                kept.append(alias)
                continue

            suffix = self._alias_to_module.get(bound)
            if suffix is None:
                kept.append(alias)
                continue

            local_module = (
                self._import_root
                if self._import_root == c.Infra.DIR_TESTS
                else f"{self._import_root}.{suffix}"
            )
            display = bound
            if bound not in self._existing_local.get(local_module, set()):
                self.imports_to_add.setdefault(local_module, {})[bound] = display
            message = f"Migrated {display} from flext_core to {local_module}"
            self.changes.append(message)
            if self._record_change is not None:
                self._record_change(message)

        if not kept:
            return cst.RemoveFromParent()
        if len(kept) == len(aliases):
            return updated_node
        return updated_node.with_changes(
            names=[
                cst.ImportAlias(name=alias.name, asname=alias.asname) for alias in kept
            ]
        )


class FlextInfraRefactorProjectAliasMigrator(FlextInfraRopeTransformer):
    """Rewrite ``from flext_core import c`` to ``from <proj>.constants import c``."""

    _description = "rewrite foreign canonical alias imports to local project facade"
    _ALIAS_TO_LOCAL_MODULE: ClassVar[t.StrMapping] = c.Infra.FAMILY_PUBLIC_MODULES

    def __init__(
        self,
        *,
        file_path: Path | None = None,
        project_alias_owners: t.StrSequenceMapping | None = None,
        current_project: str = "",
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize with project -> local aliases SSOT from flext-core.

        Args:
            file_path: Optional file path used to infer the owning project.
            project_alias_owners: SSOT mapping project package -> local aliases.
                Defaults to ``c.ENFORCEMENT_PROJECT_ALIAS_OWNERS``.
            current_project: Explicit project package; overrides inference.
            on_change: Optional callback invoked for each recorded change.

        """
        super().__init__(on_change=on_change)
        owners = (
            project_alias_owners
            if project_alias_owners is not None
            else c.ENFORCEMENT_PROJECT_ALIAS_OWNERS
        )
        self._project_alias_owners = dict(owners)
        self._file_path = file_path
        self._explicit_project = current_project

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply alias migration to source text."""
        self.changes.clear()
        if self._file_path is not None and (
            self._is_private_facade_implementation(self._file_path)
            or FlextInfraUtilitiesRopeSource.looks_like_facade_file(
                file_path=self._file_path, source=source
            )
        ):
            return source, []
        context = self._resolve_context(
            file_path=self._file_path, current_project=self._explicit_project
        )
        if (
            not context.policy_owner
            or context.policy_owner == c.Infra.PKG_CORE_UNDERSCORE
        ):
            return source, []
        local_aliases = self._project_alias_owners.get(context.policy_owner)
        if not local_aliases:
            return source, []

        try:
            tree = cst.parse_module(source)
        except cst.ParserSyntaxError as exc:
            msg = f"canonical alias parse failed: {exc!s}"
            raise ValueError(msg) from exc

        collector = _CollectExistingLocal(context.import_root)
        tree.visit(collector)

        transformer = _AliasMigrationTransformer(
            import_root=context.import_root,
            local_aliases=frozenset(local_aliases),
            existing_local=collector.existing_local,
            alias_to_module=self._ALIAS_TO_LOCAL_MODULE,
            record_change=self._record_change,
        )
        new_tree = tree.visit(transformer)
        if not transformer.changes:
            return source, []

        final_tree = _CstImportHelpers.insert_local_imports(
            new_tree, transformer.imports_to_add
        )
        return final_tree.code, list(self.changes)

    @staticmethod
    def _resolve_context(
        *, file_path: Path | None, current_project: str
    ) -> m.Infra.AliasMigrationContext:
        """Resolve explicit or path-backed alias migration context."""
        if current_project:
            return m.Infra.AliasMigrationContext(
                policy_owner=current_project, import_root=current_project
            )
        if file_path is None:
            return m.Infra.AliasMigrationContext(policy_owner="", import_root="")
        context: m.Infra.AliasMigrationContext = (
            FlextInfraUtilitiesDiscovery.alias_migration_context(file_path)
        )
        return context

    @staticmethod
    def _is_private_facade_implementation(file_path: Path) -> bool:
        """Return whether ``file_path`` implements a project facade namespace."""
        family_dirs = frozenset(c.Infra.FAMILY_DIRECTORIES.values())
        return bool(family_dirs.intersection(file_path.parts))


__all__: list[str] = ["FlextInfraRefactorProjectAliasMigrator"]
