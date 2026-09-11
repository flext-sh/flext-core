"""Tier 0 import transformer — extracted concern of FlextInfraTransformerTier0ImportFixer."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_infra import m, u

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_infra import t


class FlextInfraTier0TransformerMixin:
    """Carrier for the rope-based Tier 0 import ``Transformer``.

    Composed into FlextInfraTransformerTier0ImportFixer via inheritance so the
    nested ``Transformer`` class stays reachable as
    ``FlextInfraTransformerTier0ImportFixer.Transformer``.
    """

    class Transformer:
        """Rewrite Tier 0 imports via rope to remove circularity."""

        _CLASS_IMPORTS_MAP: ClassVar[t.StrMapping] = {
            "u": "flext_core",
            "FlextUtilitiesGuardsTypeCore": "flext_core_type_core",
            "FlextUtilitiesGuards": "flext_core",
            "FlextUtilitiesGuardsType": "flext_core_type",
            "FlextUtilitiesCache": "flext_core",
            "FlextUtilitiesMapper": "flext_core",
            "FlextUtilitiesModel": "flext_core",
        }

        def __init__(
            self,
            *,
            analysis: m.Infra.Tier0ImportAnalysis,
            alias_to_submodule: t.StrMapping,
            core_package: str,
        ) -> None:
            """Initialize transformer with analysis and insertion settings."""
            self.analysis = analysis
            self._package_name = analysis.package_name
            self._core_package = core_package
            self._root_remove = (
                set(analysis.category_b)
                | set(analysis.category_c)
                | set(analysis.category_d)
            )
            self._core_pending = set(analysis.category_b)
            self._type_checking_pending = set(analysis.category_c)
            self._direct_pending: MutableMapping[str, t.Infra.StrSet] = {}
            for a in sorted(analysis.category_d):
                sub = alias_to_submodule.get(a, analysis.alias_to_module.get(a, ""))
                if sub:
                    self._direct_pending.setdefault(sub, set()).add(a)
            self._missing_classes: t.Infra.StrSet = set()
            self._changes: t.MutableSequenceOf[str] = []

        @property
        def changes(self) -> t.StrSequence:
            """Recorded transformation changes."""
            return self._changes

        def transform(
            self, rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
        ) -> t.Infra.TransformResult:
            """Apply tier 0 import fixes via rope."""
            source = resource.read()
            self._detect_missing_classes(source)
            if self._root_remove:
                u.Infra.remove_import_names(
                    rope_project,
                    resource,
                    self._package_name,
                    sorted(self._root_remove),
                    apply=True,
                )
            if self._core_pending:
                u.Infra.add_import(
                    rope_project,
                    resource,
                    self._core_package,
                    sorted(self._core_pending),
                    apply=True,
                )
                self._core_pending.clear()
            for sub, aliases in sorted(self._direct_pending.items()):
                if aliases:
                    u.Infra.add_import(
                        rope_project,
                        resource,
                        f"{self._package_name}.{sub}",
                        sorted(aliases),
                        apply=True,
                    )
            for name in sorted(self._missing_classes):
                mod = self._CLASS_IMPORTS_MAP[name]
                u.Infra.add_import(rope_project, resource, mod, [name], apply=True)
            if self._type_checking_pending:
                self._add_type_checking_block(rope_project, resource)
            final = resource.read()
            return final, list(self._changes)

        def _detect_missing_classes(self, source: str) -> None:
            """Find class names used but not imported."""
            self._missing_classes.clear()
            for name in self._CLASS_IMPORTS_MAP:
                if name in source and f"import {name}" not in source:
                    self._missing_classes.add(name)

        def _add_type_checking_block(
            self, rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
        ) -> None:
            """Add TYPE_CHECKING block with pending type-only imports."""
            source = resource.read()
            names = ", ".join(sorted(self._type_checking_pending))
            tc_import = f"from {self._package_name} import {names}"
            if "from typing import TYPE_CHECKING" not in source:
                u.Infra.add_import(
                    rope_project, resource, "typing", ["TYPE_CHECKING"], apply=True
                )
                self._changes.append("Added 'from typing import TYPE_CHECKING'")
                source = resource.read()
            if "if TYPE_CHECKING:" not in source:
                block = f"\nif TYPE_CHECKING:\n    {tc_import}\n"
                lines = source.splitlines(keepends=True)
                idx = u.Infra.find_import_insert_position(lines)
                lines.insert(idx, block)
                new_source = "".join(lines)
                resource.write(new_source)
                self._changes.append(
                    f"Added TYPE_CHECKING block for {self._package_name}"
                )


__all__: list[str] = ["FlextInfraTier0TransformerMixin"]
