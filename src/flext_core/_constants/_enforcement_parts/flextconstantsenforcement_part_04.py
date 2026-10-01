"""Rule metadata enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from enum import StrEnum


class FlextConstantsEnforcementRules:
    """Rule categories and AST hook sentinels of the runtime engine."""

    class EnforcementCategory(StrEnum):
        """Rule category — dispatches engine behaviour per row."""

        ATTR = "attr"
        FIELD = "field"
        MODEL_CLASS = "model_class"
        NAMESPACE = "namespace"
        PROTOCOL_TREE = "protocol_tree"

    class EnforceAstHookSymbol(StrEnum):
        """AST identifier names matched by A-PT enforcement hooks."""

        CAST_CALL = "cast"
        """ENFORCE-039: ``ast.Name.id`` matched as the ``typing.cast`` call."""


__all__: list[str] = ["FlextConstantsEnforcementRules"]
