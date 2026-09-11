"""Deprecated class remover transformer — rope-based legacy cleanup.

Removes classes marked as deprecated by name pattern or deprecation
warning usage in __init__, using regex analysis on source text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorDeprecatedRemover:
    """Remove classes marked as deprecated by name or warning usage."""

    def __init__(
        self,
        changes: t.MutableSequenceOf[str] | None = None,
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize change sinks used by the transformer."""
        self.changes: t.MutableSequenceOf[str] = changes if changes is not None else []
        self._on_change = on_change

    def apply_to_source(self, source: str) -> str:
        """Remove deprecated classes from source text."""
        result = source
        for match in reversed(list(c.Infra.CLASS_BLOCK_RE.finditer(source))):
            class_body = match.group(1)
            class_name = match.group(2)
            if self._is_deprecated(class_name, class_body):
                self._record_change(f"Removed deprecated class: {class_name}")
                result = result[: match.start()] + result[match.end() :]
        return result

    def _is_deprecated(self, class_name: str, class_body: str) -> bool:
        """Check if a class is deprecated by name or __init__ warning."""
        if "deprecated" in class_name.lower():
            return True
        return c.Infra.DUNDER_INIT in class_body and bool(
            c.Infra.DEPRECATION_WARN_RE.search(class_body)
        )

    def _record_change(self, message: str) -> None:
        """Record change."""
        self.changes.append(message)
        if self._on_change is not None:
            self._on_change(message)


__all__: list[str] = ["FlextInfraRefactorDeprecatedRemover"]
