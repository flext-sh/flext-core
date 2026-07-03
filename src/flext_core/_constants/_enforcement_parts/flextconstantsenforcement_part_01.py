"""Enforcement enum constants for FlextConstantsEnforcement."""

from __future__ import annotations

from enum import StrEnum, unique


class FlextConstantsEnforcementEnums:
    """Canonical enforcement enum namespace."""

    class EnforcementMode(StrEnum):
        """Controls whether violations raise, warn, or are ignored."""

        OFF = "off"
        STRICT = "strict"
        WARN = "warn"

    class EnforcementLayer(StrEnum):
        """Facade layer where a violation is raised."""

        CONSTANTS = "Constants"
        MODEL = "Model"
        NAMESPACE = "namespace"
        PROTOCOLS = "Protocols"
        TYPES = "Types"
        UTILITIES = "Utilities"

    class EnforcementSeverity(StrEnum):
        """Severity label attached to every violation."""

        BEST_PRACTICES = "best practices"
        HARD_RULES = "HARD rules"
        NAMESPACE_RULES = "namespace rules"

    @unique
    class EnforcementPredicateKind(StrEnum):
        """Generic detection predicate keyed by AST/typing shape."""

        ALIAS_REBIND = "alias_rebind"
        ATTR_SHAPE = "attr_shape"
        CLASS_PLACEMENT = "class_placement"
        CLASSVAR_CONSTANT = "classvar_constant"
        COMPATIBILITY_ALIAS = "compatibility_alias"
        DEPRECATED_SYNTAX = "deprecated_syntax"
        DUPLICATE_SYMBOL = "duplicate_symbol"
        FIELD_SHAPE = "field_shape"
        FOREIGN_CANONICAL_ALIAS_IMPORT = "foreign_canonical_alias_import"
        IMPORT_BLACKLIST = "import_blacklist"
        LIBRARY_IMPORT = "library_import"
        LOC_CAP = "loc_cap"
        LOOSE_SYMBOL = "loose_symbol"
        METHOD_SHAPE = "method_shape"
        MODEL_CONFIG = "model_config"
        MODULE_ALIAS = "module_alias"
        MRO_SHAPE = "mro_shape"
        PROTOCOL_TREE = "protocol_tree"
        WRAPPER = "wrapper"


__all__ = ["FlextConstantsEnforcementEnums"]
