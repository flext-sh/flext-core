"""Nested class reference propagation transformer — rope-based implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c, u
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import m, t


class FlextInfraNestedClassPropagationTransformer(FlextInfraRopeTransformer):
    """Propagate import and name references after classes are nested into namespaces.

    After class nesting, ``from module import OldClass`` becomes
    ``from module import Namespace`` and bare ``OldClass`` references
    become ``Namespace.OldClass``.
    """

    def __init__(
        self,
        class_renames: t.StrMapping,
        policy_context: t.Infra.PolicyContext | None = None,
        class_families: t.StrMapping | None = None,
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize with class rename mappings and optional policy context."""
        super().__init__(on_change=on_change)
        self._class_renames = class_renames
        self._policy_context = policy_context
        self._class_families = class_families or {}

    _description = "nested class propagation"

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply reference propagation to in-memory source without persisting."""
        self.changes.clear()
        updated = source
        for old_name, rename_to in self._class_renames.items():
            if not self._should_propagate(old_name, "propagate_imports"):
                continue
            if self._blocked_by_prefix(old_name):
                continue
            rename_parts = rename_to.split(".")
            namespace = rename_parts[0]
            aliases = self._find_import_aliases(updated, old_name=old_name)
            updated = self._rewrite_import(
                updated, old_name=old_name, namespace=namespace
            )
            if self._should_propagate(old_name, "propagate_name_references"):
                updated = self._qualify_name_references(
                    updated, old_name=old_name, qualified=rename_to
                )
                updated = self._qualify_return_annotations(
                    updated, old_name=old_name, qualified=rename_to
                )
                nested_name = rename_parts[-1]
                for alias_name in aliases:
                    qualified_alias = f"{alias_name}.{nested_name}"
                    updated = self._qualify_alias_references(
                        updated, alias_name=alias_name, qualified=qualified_alias
                    )
                    updated = self._qualify_return_annotations(
                        updated, old_name=alias_name, qualified=qualified_alias
                    )
            if self._should_propagate(old_name, "propagate_attribute_references"):
                updated = self._qualify_attribute_references(
                    updated,
                    old_name=old_name,
                    rename_parts=rename_parts,
                    aliases=aliases,
                )
        return updated, list(self.changes)

    def _rewrite_import(self, source: str, *, old_name: str, namespace: str) -> str:
        """Rewrite ``from mod import OldName`` to ``from mod import Namespace``."""
        pattern = c.Infra.compile_import_namespace_rewrite(old_name)
        new_source: str = pattern.sub(rf"\g<1>{namespace}\2", source)
        if new_source != source:
            self._record_change(f"Rewired import: {old_name} -> {namespace}")
        return new_source

    def _qualify_name_references(
        self, source: str, *, old_name: str, qualified: str
    ) -> str:
        """Replace bare ``OldName`` with ``Namespace.Nested`` in non-definition sites."""
        pattern = c.Infra.compile_bare_qualify_allowing_call(old_name)
        replacement_result = pattern.subn(qualified, source)
        new_source: str = replacement_result[0]
        count = replacement_result[1]
        if count > 0 and new_source != source:
            self._record_change(f"Qualified reference: {old_name} -> {qualified}")
        return new_source

    def _find_import_aliases(self, source: str, *, old_name: str) -> t.StrSequence:
        """Collect local aliases bound from ``from x import OldName as Alias``."""
        pattern = c.Infra.compile_import_alias_finder(old_name)
        return [match.group(1) for match in pattern.finditer(source)]

    def _qualify_alias_references(
        self, source: str, *, alias_name: str, qualified: str
    ) -> str:
        """Replace bare alias usage with ``Alias.Nested`` outside import lines."""
        pattern = c.Infra.compile_alias_qualify(alias_name)
        lines = source.splitlines(keepends=True)
        changed = False
        rewritten: list[str] = []
        for line in lines:
            if line.lstrip().startswith("from ") and " import " in line:
                rewritten.append(line)
                continue
            new_line, count = pattern.subn(qualified, line)
            changed = changed or count > 0
            rewritten.append(new_line)
        new_source = "".join(rewritten)
        if changed and new_source != source:
            self._record_change(
                f"Qualified alias reference: {alias_name} -> {qualified}"
            )
        return new_source

    def _qualify_return_annotations(
        self, source: str, *, old_name: str, qualified: str
    ) -> str:
        """Replace ``-> OldName`` with ``-> Namespace.OldName`` in signatures."""
        pattern = c.Infra.compile_flext_prefixed_annotation("->", old_name)
        replacement_result = pattern.subn(rf"\1{qualified}", source)
        new_source: str = replacement_result[0]
        count = replacement_result[1]
        if count > 0 and new_source != source:
            self._record_change(
                f"Qualified return annotation: {old_name} -> {qualified}"
            )
        return new_source

    def _qualify_attribute_references(
        self,
        source: str,
        *,
        old_name: str,
        rename_parts: t.StrSequence,
        aliases: t.StrSequence,
    ) -> str:
        """Qualify attribute-style references like ``module.OldName``."""
        attr_pattern = c.Infra.compile_attribute_qualify(old_name)
        namespace = rename_parts[0]
        alias_names = set(aliases)
        suffix = ".".join(rename_parts)

        def qualify_reference(match: t.Infra.RegexMatch) -> str:
            qualifier = match.group(1)
            if qualifier == namespace or qualifier in alias_names:
                return match.group(0)
            return f"{qualifier}.{suffix}"

        new_source: str = attr_pattern.sub(qualify_reference, source)
        if new_source != source:
            self._record_change(
                f"Qualified attribute reference: .{old_name} -> .{suffix}"
            )
        return new_source

    def _should_propagate(self, symbol_name: str, policy_key: str) -> bool:
        """Check policy for a specific propagation mode."""
        policy: m.Infra.ClassNestingPolicy | None = u.Infra.policy_for_symbol(
            policy_context=self._policy_context,
            symbol_families=self._class_families,
            symbol_name=symbol_name,
        )
        if policy is None:
            return True
        if policy_key == "propagate_imports":
            propagate_imports: bool = policy.propagate_imports
            return propagate_imports
        if policy_key == "propagate_name_references":
            propagate_name_references: bool = policy.propagate_name_references
            return propagate_name_references
        if policy_key == "propagate_attribute_references":
            propagate_attribute_references: bool = policy.propagate_attribute_references
            return propagate_attribute_references
        return False

    def _blocked_by_prefix(self, symbol_name: str) -> bool:
        """Check if symbol is blocked by prefix policy."""
        policy: m.Infra.ClassNestingPolicy | None = u.Infra.policy_for_symbol(
            policy_context=self._policy_context,
            symbol_families=self._class_families,
            symbol_name=symbol_name,
        )
        if policy is None:
            return False
        return any(symbol_name.startswith(p) for p in policy.blocked_reference_prefixes)


__all__: list[str] = ["FlextInfraNestedClassPropagationTransformer"]
