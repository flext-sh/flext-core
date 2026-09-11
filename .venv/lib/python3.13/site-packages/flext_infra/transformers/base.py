"""Base class for rope-based transformers with change-tracking."""

from __future__ import annotations

import re
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraChangeTrackingTransformer:
    """Mixin providing change-tracking with optional callback.

    Provides ``changes`` list and ``_record_change`` method.
    Subclasses that need a rope ``transform()`` contract should
    inherit :class:`FlextInfraRopeTransformer` instead.
    """

    def __init__(self, *, on_change: t.Infra.ChangeCallback = None) -> None:
        """Initialize change tracking with an optional callback."""
        self._on_change = on_change
        self.changes: t.MutableSequenceOf[str] = []

    def _record_change(self, message: str) -> None:
        """Record change."""
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)


class FlextInfraRopeTransformer(FlextInfraChangeTrackingTransformer):
    """Base for all rope transformers — tracks changes and invokes callback.

    Subclasses that follow the ``read → apply_to_source → write`` pattern
    need only implement ``apply_to_source`` and set ``_description``.
    The default ``transform()`` handles the boilerplate.
    """

    _description: str = "transformation"

    @abstractmethod
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply transformation to in-memory source."""
        ...

    @staticmethod
    def _ensure_u_import(source: str) -> str:
        """Ensure ``from <core_pkg> import u`` is present in source text.

        The facades are resolved at call time: this module is imported while
        ``flext_infra.utilities`` is still initializing, so a module-level
        ``from flext_infra import u`` would bind the parent CLI facade.
        """
        from flext_infra import c, u

        core_pkg = c.Infra.PKG_CORE_UNDERSCORE
        pkg_match = re.search(
            rf"^from\s+{re.escape(core_pkg)}\s+import\s+([^\n]+)", source, re.MULTILINE
        )
        if pkg_match:
            names = pkg_match.group(1).strip()
            name_set = {n.strip() for n in names.split(",")}
            if "u" in name_set:
                return source
            new_names = names + ", u"
            return source[: pkg_match.start(1)] + new_names + source[pkg_match.end(1) :]
        lines = source.splitlines(keepends=True)
        insert_idx = u.Infra.find_import_insert_position(lines, past_existing=False)
        lines.insert(insert_idx, f"from {core_pkg} import u\n")
        return "".join(lines)

    def transform(
        self, rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.TransformResult:
        """Read → apply_to_source → write if changed. Override for custom logic."""
        _ = rope_project
        source = resource.read()
        updated, changes = self.apply_to_source(source)
        if updated != source and changes:
            resource.write(updated)
        return updated, changes


__all__: list[str] = [
    "FlextInfraChangeTrackingTransformer",
    "FlextInfraRopeTransformer",
]
