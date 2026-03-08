"""Constants namespace for flext_infra.refactor."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, TypeAlias


class FlextInfraRefactorConstants:
    """Shared constants for refactor engine modules."""

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
    CLASS_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({"reorder_methods"})
    MRO_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "remove_inheritance_keep_class",
        "fix_mro_redeclaration",
        "migrate_to_class_mro",
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
        "ensure_future_annotations"
    })
    FUTURE_CHECKS: ClassVar[frozenset[str]] = frozenset({"missing_future_import"})
    MRO_TARGETS: ClassVar[frozenset[str]] = frozenset({"constants", "typings", "all"})
    "Accepted target arguments for MRO migration runs."
    MRO_SCAN_DIRECTORIES: ClassVar[tuple[str, ...]] = (
        "src",
        "examples",
        "scripts",
        "tests",
    )
    "Directories scanned for constants modules in each project."
    MRO_CONSTANTS_FILE_NAMES: ClassVar[frozenset[str]] = frozenset({
        "constants.py",
        "_constants.py",
    })
    "Canonical constants module file names."
    MRO_CONSTANTS_DIRECTORY: ClassVar[str] = "constants"
    "Canonical constants package directory name."
    MRO_TYPINGS_FILE_NAMES: ClassVar[frozenset[str]] = frozenset({
        "typings.py",
        "_typings.py",
    })
    "Canonical typings module file names."
    MRO_TYPINGS_DIRECTORY: ClassVar[str] = "typings"
    "Canonical typings package directory name."
    DEFAULT_CONSTANTS_CLASS: ClassVar[str] = "FlextConstants"
    "Fallback constants class name when none exists in module."
    DEFAULT_TYPES_CLASS: ClassVar[str] = "FlextTypes"
    "Fallback types class name when none exists in module."
    CONSTANTS_FILE_GLOB: ClassVar[str] = "constants.py"
    "Constants module glob scanned by the migration scanner."
    CONSTANTS_CLASS_SUFFIX: ClassVar[str] = "Constants"
    "Class-name suffix used to identify constants facades."
    FINAL_ANNOTATION_NAME: ClassVar[str] = "Final"
    "Annotation marker used to detect module-level constants."
    CONSTANT_PATTERN_REGEX: ClassVar[str] = "^_*[A-Z][A-Z0-9_]*$"
    "Naming pattern for module-level constant candidates."
    CONSTANT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(CONSTANT_PATTERN_REGEX)
    "Compiled naming pattern for module-level constant candidates."
    DEFAULT_FACADE_ALIAS: ClassVar[str] = "c"
    "Default facade alias inserted during import rewrite."
    MRO_CLASS_TEMPLATE: ClassVar[str] = "class {class_name}:\n    pass\n"
    "Template used to create a new constants facade class."
    FAMILY_SUFFIXES: ClassVar[Mapping[str, str]] = {
        "c": "Constants",
        "t": "Types",
        "p": "Protocols",
        "m": "Models",
        "u": "Utilities",
    }
    "Facade family letter → class suffix mapping."
    FAMILY_FILES: ClassVar[dict[str, str]] = {
        "c": "*constants.py",
        "t": "*typings.py",
        "p": "*protocols.py",
        "m": "*models.py",
        "u": "*utilities.py",
    }
    "Facade family letter → file glob mapping."
    DOMAIN_PACKAGES: ClassVar[frozenset[str]] = frozenset({
        "flext-ldap",
        "flext-ldif",
        "flext-db-oracle",
        "flext-oracle-wms",
        "flext-oracle-oic",
    })
    "Known domain-layer packages."
    PLATFORM_PACKAGES: ClassVar[frozenset[str]] = frozenset({
        "flext-cli",
        "flext-meltano",
        "flext-api",
        "flext-auth",
        "flext-web",
        "flext-grpc",
    })
    "Known platform-layer packages."
    INTEGRATION_CLASS_PREFIXES: ClassVar[tuple[str, ...]] = (
        "FlextTap",
        "FlextTarget",
        "FlextDbt",
    )
    "Class name prefixes that identify integration projects."
    CONFIDENCE_TO_SCORE: ClassVar[Mapping[str, float]] = {
        "high": 0.95,
        "medium": 0.75,
        "low": 0.55,
    }
    "Confidence level → numeric score mapping for violations."
    CONFIDENCE_RANKS: ClassVar[dict[str, int]] = {"low": 0, "medium": 1, "high": 2}
    "Confidence level → priority rank mapping."
    REQUIRED_CLASS_TARGETS: ClassVar[tuple[str, ...]] = (
        "TimeoutEnforcer",
        "CircuitBreakerManager",
    )
    "Class names always required in scanner output."
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]+")
    "Pattern to split class name fragments."
    MAPPINGS_RELATIVE_PATH: ClassVar[Path] = (
        Path("rules") / "class-nesting-mappings.yml"
    )
    "Relative path from the refactor package to the nesting mappings YAML."
    VIOLATION_PATTERNS: ClassVar[Mapping[str, re.Pattern[str]]] = {
        "container_invariance": re.compile(
            "\\bdict\\s*\\[\\s*str\\s*,\\s*t\\.(?:Container|ContainerValue)\\s*\\]"
        ),
        "redundant_cast": re.compile("\\bcast\\s*\\(\\s*[\\\"'][^\\\"']+[\\\"']\\s*,"),
        "direct_submodule_import": re.compile(
            "\\bfrom\\s+flext_core\\.[\\w\\.]+\\s+import\\b"
        ),
        "legacy_typing_mapping": re.compile(
            "\\bfrom\\s+typing\\s+import\\s+.*\\bMapping\\b"
        ),
        "runtime_alias_violation": re.compile(
            "\\bfrom\\s+flext_core\\s+import\\s+(?!.*\\b(?:c|m|r|t|u|p|d|e|h|s|x)\\b).*"
        ),
    }
    "Regex patterns for violation analysis."
    MODEL_TOKENS: ClassVar[tuple[str, ...]] = (
        "model",
        "schema",
        "entity",
        "pydantic",
        "dataclass",
    )
    "Tokens indicating model-related code."
    DECORATOR_TOKENS: ClassVar[tuple[str, ...]] = ("decorator", "inject", "provide")
    "Tokens indicating decorator-related code."
    DISPATCHER_TOKENS: ClassVar[tuple[str, ...]] = (
        "dispatcher",
        "dispatch",
        "command",
        "query",
        "event",
    )
    "Tokens indicating dispatcher-related code."
    NAMESPACE_PREFIXES: ClassVar[Mapping[str, str]] = {
        "utility": "FlextUtilities",
        "models": "FlextModels",
        "decorators": "FlextDecorators",
        "dispatcher": "FlextDispatcher",
    }
    "Namespace → class prefix mapping for violation classification."
    CLASSIFICATION_PRIORITY: ClassVar[tuple[str, ...]] = (
        "dispatcher",
        "decorators",
        "models",
        "utility",
    )
    "Priority order for violation classification."
    CAST_ARITY: int = 2
    "Expected number of arguments for typing.cast calls."
    MIN_PATH_DEPTH: int = 2
    "Minimum relative path depth for module prefix detection."

    class MethodCategory:
        MAGIC: ClassVar[str] = "magic"
        PROPERTY: ClassVar[str] = "property"
        STATIC: ClassVar[str] = "static"
        CLASS: ClassVar[str] = "class"
        PUBLIC: ClassVar[str] = "public"
        PROTECTED: ClassVar[str] = "protected"
        PRIVATE: ClassVar[str] = "private"

    PROJECT_KIND_VALUES: ClassVar[frozenset[str]] = frozenset({
        "core",
        "domain",
        "platform",
        "integration",
        "app",
    })
    ProjectKind: TypeAlias = str


__all__ = ["FlextInfraRefactorConstants"]
