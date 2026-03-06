"""Constants namespace for flext_infra.refactor."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum, auto
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar


class FlextInfraRefactorConstants:
    """Shared constants for refactor engine modules."""

    # -- Runtime aliases -----------------------------------------------------

    RUNTIME_ALIAS_NAMES: ClassVar[frozenset[str]] = frozenset({
        "c",
        "m",
        "r",
        "t",
        "u",
        "p",
        "d",
        "e",
        "h",
        "s",
        "x",
    })

    # -- Fix action categories -----------------------------------------------

    LEGACY_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "remove",
        "inline_and_remove",
        "remove_and_update_refs",
        "keep_try_only",
    })

    IMPORT_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "replace_with_alias",
        "hoist_to_module_top",
    })

    CLASS_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "reorder_methods",
    })

    MRO_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "remove_inheritance_keep_class",
        "fix_mro_redeclaration",
    })

    PROPAGATION_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "propagate_symbol_renames",
        "rename_imported_symbols",
        "propagate_signature_migrations",
    })

    PATTERN_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "convert_dict_to_mapping_annotations",
        "remove_redundant_casts",
    })

    FUTURE_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "ensure_future_annotations",
    })

    FUTURE_CHECKS: ClassVar[frozenset[str]] = frozenset({
        "missing_future_import",
    })

    # -- Family / MRO constants ----------------------------------------------

    FAMILY_SUFFIXES: ClassVar[MappingProxyType[str, str]] = MappingProxyType({
        "c": "Constants",
        "t": "Types",
        "p": "Protocols",
        "m": "Models",
        "u": "Utilities",
    })
    """Facade family letter → class suffix mapping."""

    FAMILY_FILES: ClassVar[dict[str, str]] = {
        "c": "*constants.py",
        "t": "*typings.py",
        "p": "*protocols.py",
        "m": "*models.py",
        "u": "*utilities.py",
    }
    """Facade family letter → file glob mapping."""

    # -- Project classification constants ------------------------------------

    DOMAIN_PACKAGES: ClassVar[frozenset[str]] = frozenset({
        "flext-ldap",
        "flext-ldif",
        "flext-db-oracle",
        "flext-oracle-wms",
        "flext-oracle-oic",
    })
    """Known domain-layer packages."""

    PLATFORM_PACKAGES: ClassVar[frozenset[str]] = frozenset({
        "flext-cli",
        "flext-meltano",
        "flext-api",
        "flext-auth",
        "flext-web",
        "flext-grpc",
    })
    """Known platform-layer packages."""

    INTEGRATION_CLASS_PREFIXES: ClassVar[tuple[str, ...]] = (
        "FlextTap",
        "FlextTarget",
        "FlextDbt",
    )
    """Class name prefixes that identify integration projects."""

    # -- Scanner / nesting constants -----------------------------------------

    CONFIDENCE_TO_SCORE: ClassVar[Mapping[str, float]] = MappingProxyType({
        "high": 0.95,
        "medium": 0.75,
        "low": 0.55,
    })
    """Confidence level → numeric score mapping for violations."""

    CONFIDENCE_RANKS: ClassVar[dict[str, int]] = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }
    """Confidence level → priority rank mapping."""

    REQUIRED_CLASS_TARGETS: ClassVar[tuple[str, ...]] = (
        "TimeoutEnforcer",
        "CircuitBreakerManager",
    )
    """Class names always required in scanner output."""

    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]+")
    """Pattern to split class name fragments."""

    # -- Analysis constants --------------------------------------------------

    MAPPINGS_RELATIVE_PATH: ClassVar[Path] = (
        Path("rules") / "class-nesting-mappings.yml"
    )
    """Relative path from the refactor package to the nesting mappings YAML."""

    VIOLATION_PATTERNS: ClassVar[Mapping[str, re.Pattern[str]]] = MappingProxyType({
        "container_invariance": re.compile(
            r"\bdict\s*\[\s*str\s*,\s*t\.(?:Container|ContainerValue)\s*\]"
        ),
        "redundant_cast": re.compile(r"\bcast\s*\(\s*[\"'][^\"']+[\"']\s*,"),
        "direct_submodule_import": re.compile(
            r"\bfrom\s+flext_core\.[\w\.]+\s+import\b"
        ),
        "legacy_typing_mapping": re.compile(
            r"\bfrom\s+typing\s+import\s+.*\bMapping\b"
        ),
        "runtime_alias_violation": re.compile(
            r"\bfrom\s+flext_core\s+import\s+(?!.*\b(?:c|m|r|t|u|p|d|e|h|s|x)\b).*"
        ),
    })
    """Regex patterns for violation analysis."""

    MODEL_TOKENS: ClassVar[tuple[str, ...]] = (
        "model",
        "schema",
        "entity",
        "pydantic",
        "dataclass",
    )
    """Tokens indicating model-related code."""

    DECORATOR_TOKENS: ClassVar[tuple[str, ...]] = (
        "decorator",
        "inject",
        "provide",
    )
    """Tokens indicating decorator-related code."""

    DISPATCHER_TOKENS: ClassVar[tuple[str, ...]] = (
        "dispatcher",
        "dispatch",
        "command",
        "query",
        "event",
    )
    """Tokens indicating dispatcher-related code."""

    NAMESPACE_PREFIXES: ClassVar[Mapping[str, str]] = MappingProxyType({
        "utility": "FlextUtilities",
        "models": "FlextModels",
        "decorators": "FlextDecorators",
        "dispatcher": "FlextDispatcher",
    })
    """Namespace → class prefix mapping for violation classification."""

    CLASSIFICATION_PRIORITY: ClassVar[tuple[str, ...]] = (
        "dispatcher",
        "decorators",
        "models",
        "utility",
    )
    """Priority order for violation classification."""

    # -- Enums ---------------------------------------------------------------

    class MethodCategory(StrEnum):
        """Categorias de metodos para ordenacao."""

        MAGIC = auto()
        PROPERTY = auto()
        STATIC = auto()
        CLASS = auto()
        PUBLIC = auto()
        PROTECTED = auto()
        PRIVATE = auto()


__all__ = ["FlextInfraRefactorConstants"]
