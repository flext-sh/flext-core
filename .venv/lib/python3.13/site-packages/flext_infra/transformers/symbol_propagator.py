"""Workspace-wide symbol propagation transformer — rope-based implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorSymbolPropagator(FlextInfraRopeTransformer):
    """Propagate import/symbol renames safely using rope import manipulation.

    Handles three kinds of rename:
    1. Module renames (``from old_module`` -> ``from new_module``)
    2. Import symbol renames (``import OldName`` -> ``import NewName``)
    3. Local reference propagation (bare ``OldName`` -> ``NewName`` in body)
    """

    _description = "symbol propagator"

    def __init__(
        self,
        *,
        target_modules: t.Infra.StrSet,
        module_renames: t.StrMapping,
        import_symbol_renames: t.StrMapping,
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize symbol propagation configuration."""
        super().__init__(on_change=on_change)
        self._target_modules = target_modules
        self._module_renames = module_renames
        self._import_symbol_renames = import_symbol_renames

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply module/symbol renames to in-memory source."""
        self.changes.clear()
        updated = source
        # Phase 1: Rename module paths in from-imports
        for old_module, new_module in self._module_renames.items():
            updated = self._rename_module_in_imports(
                updated, old_module=old_module, new_module=new_module
            )

        # Phase 2: Rename imported symbols in target modules
        local_renames: t.MutableStrMapping = {}
        for old_name, new_name in self._import_symbol_renames.items():
            updated, renamed = self._rename_import_symbol(
                updated, old_name=old_name, new_name=new_name
            )
            if renamed:
                local_renames[old_name] = new_name

        # Phase 3: Propagate local reference renames
        for old_name, new_name in local_renames.items():
            updated = self._propagate_local_rename(
                updated, old_name=old_name, new_name=new_name
            )
        return updated, list(self.changes)

    def _rename_module_in_imports(
        self, source: str, *, old_module: str, new_module: str
    ) -> str:
        """Replace ``from old_module import ...`` with ``from new_module import ...``."""
        pattern = c.Infra.compile_from_module_rename(old_module)
        replacement_result = pattern.subn(rf"\g<1>{new_module}\2", source)
        new_source: str = replacement_result[0]
        count = replacement_result[1]
        if count > 0 and new_source != source:
            self._record_change(f"Renamed import module: {old_module} -> {new_module}")
            return new_source
        return source

    def _rename_import_symbol(
        self, source: str, *, old_name: str, new_name: str
    ) -> tuple[str, bool]:
        """Rename symbol in import statement within target modules."""
        # Match the symbol in from-import lines for any target module
        for target_module in self._target_modules:
            pattern = c.Infra.compile_import_symbol_rename(target_module, old_name)
            new_source: str = pattern.sub(rf"\g<1>{new_name}", source)
            if new_source != source:
                self._record_change(
                    f"Renamed imported symbol: {old_name} -> {new_name}"
                )
                return new_source, True
        return source, False

    def _propagate_local_rename(
        self, source: str, *, old_name: str, new_name: str
    ) -> str:
        """Replace bare references to old_name with new_name in non-import lines."""
        pattern = c.Infra.compile_bare_reference_rename(old_name)
        replacement_result = pattern.subn(new_name, source)
        new_source: str = replacement_result[0]
        count = replacement_result[1]
        if count > 0 and new_source != source:
            self._record_change(
                f"Propagated local symbol rename: {old_name} -> {new_name}"
            )
            return new_source
        return source


__all__: list[str] = ["FlextInfraRefactorSymbolPropagator"]
