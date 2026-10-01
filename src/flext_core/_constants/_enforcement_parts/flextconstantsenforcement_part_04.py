"""Rule metadata enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar


class FlextConstantsEnforcementRules:
    """Rule categories and AST hook sentinels of the runtime engine."""

    class EnforcementCategory(StrEnum):
        """Rule category — dispatches engine behaviour per row."""

        ATTR = "attr"
        FIELD = "field"
        MODEL_CLASS = "model_class"
        NAMESPACE = "namespace"
        PROTOCOL_TREE = "protocol_tree"

    # --- ENFORCE-039 / 041 / 043 / 044 detection inputs ---
    # Centralized SSOT for the AST-name / path / builtin sentinels consumed by the
    # corresponding ``check_<tag>`` predicates on ``FlextUtilitiesBeartypeEngine``.

    class EnforceAstHookSymbol(StrEnum):
        """AST identifier names matched by A-PT enforcement hooks."""

        CAST_CALL = "cast"
        """ENFORCE-039: ``ast.Name.id`` matched as the ``typing.cast`` call."""

        MODEL_REBUILD_ATTR = "model_rebuild"
        """ENFORCE-041: ``ast.Attribute.attr`` matched as ``BaseModel.model_rebuild``."""

    ENFORCE_FLEXT_CORE_PATH_MARKERS: ClassVar[frozenset[str]] = frozenset({
        "flext_core",
        "flext-core",
    })
    """Path fragments identifying flext-core source files (ENFORCE-039 exemption)."""

    ENFORCE_NON_WORKSPACE_PATH_MARKERS: ClassVar[frozenset[str]] = frozenset({
        "/usr/lib/",
        "/usr/local/lib/",
        "dist-packages",
        "site-packages",
    })
    """Filesystem path fragments identifying third-party source."""

    ENFORCE_PRIVATE_PROBE_BUILTINS: ClassVar[frozenset[str]] = frozenset({
        "getattr",
        "hasattr",
        "setattr",
    })
    """ENFORCE-044: builtins that probe attributes by name."""


__all__: list[str] = ["FlextConstantsEnforcementRules"]
