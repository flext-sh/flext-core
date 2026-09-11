"""Constants namespace for flext_infra.refactor."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum, unique
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from flext_core import c
from flext_infra._constants.base import FlextInfraConstantsBase as cb
from flext_infra._constants.namespace import FlextInfraConstantsNamespace

if TYPE_CHECKING:
    from flext_infra import t


def _build_namespace_file_to_family(
    mapping: Sequence[tuple[str, Sequence[str]]],
) -> t.StrMapping:
    """Build file name → family alias mapping from (alias, file_names) pairs."""
    result: dict[str, str] = {}
    for alias, file_names in mapping:
        for file_name in file_names:
            result[file_name] = alias
    return MappingProxyType(result)


def _build_namespace_family_expected_alias(
    mapping: Sequence[tuple[str, Sequence[str]]], suffixes: t.StrMapping
) -> t.MappingKV[str, t.StrPair]:
    """Build file name → (alias, suffix) mapping from family specs."""
    result: dict[str, t.StrPair] = {}
    for alias, file_names in mapping:
        for file_name in file_names:
            result[file_name] = (alias, suffixes[alias])
    return MappingProxyType(result)


class FlextInfraConstantsRefactor(FlextInfraConstantsNamespace):
    """Shared constants for refactor modules."""

    RK_REFACTOR: Final[str] = "refactor"
    RK_PROJECT_SCAN_DIRS: Final[str] = "project_scan_dirs"
    RK_IGNORE_PATTERNS: Final[str] = "ignore_patterns"
    RK_FILE_EXTENSIONS: Final[str] = "file_extensions"
    RK_FORBIDDEN_IMPORTS: Final[str] = "forbidden_imports"
    RK_REDUNDANT_TYPE_TARGETS: Final[str] = "redundant_type_targets"
    RK_TARGET_MODULES: Final[str] = "target_modules"
    RK_MODULE_RENAMES: Final[str] = "module_renames"
    RK_IMPORT_SYMBOL_RENAMES: Final[str] = "import_symbol_renames"
    RK_SIGNATURE_MIGRATIONS: Final[str] = "signature_migrations"
    RK_METHOD_ORDER: Final[str] = "method_order"
    RK_ORDER: Final[str] = "order"
    RK_TIER0_MODULES: Final[str] = "tier0_modules"
    RK_CORE_ALIASES: Final[str] = "core_aliases"
    RK_CORE_PACKAGE: Final[str] = "core_package"
    RK_ALIAS_TO_SUBMODULE: Final[str] = "alias_to_submodule"
    RK_QUALITY_GATES: Final[str] = "quality_gates"
    RK_EXPECTED_BASE_CHAIN: Final[str] = "expected_base_chain"
    RK_HELPER_NAME: Final[str] = "helper_name"
    RK_CONFIDENCE_THRESHOLD: Final[str] = "confidence_threshold"
    RK_ALLOW_ALIASES: Final[str] = "allow_aliases"
    RK_ALLOW_TARGET_SUFFIXES: Final[str] = "allow_target_suffixes"
    RK_TARGET_NAME: Final[str] = "target_name"
    RK_IMPORTS_RESOLVE: Final[str] = "imports_resolve"
    RK_FLEXT_VALID: Final[str] = "flext_valid"
    RK_LSP_DIAGNOSTICS_CLEAN: Final[str] = "lsp_diagnostics_clean"
    CLASS_NESTING_MAPPINGS_FILENAME: Final[str] = "class-nesting-mappings.yml"
    CLASS_NESTING_POLICY_FILENAME: Final[str] = "class-policy-v2.yml"
    REFACTOR_CONFIG_KEYS: Final[t.StrSequence] = (
        RK_PROJECT_SCAN_DIRS,
        RK_IGNORE_PATTERNS,
        RK_FILE_EXTENSIONS,
    )
    """Allowed keys under the ``refactor`` config scope."""

    TYPING_DEFINITION_FILES: Final[frozenset[str]] = frozenset({
        "typings.py",
        "_typings",
        "protocols.py",
        "_protocols",
    })
    TYPING_INLINE_UNION_CANONICAL_MAP: Final[t.MappingKV[frozenset[str], str]] = (
        MappingProxyType({
            frozenset({"str", "int", "float", "bool"}): "t.Primitives",
            frozenset({"int", "float"}): "t.Numeric",
            frozenset({"str", "int", "float", "bool", "datetime"}): "t.Scalar",
            frozenset({
                "str",
                "int",
                "float",
                "bool",
                "datetime",
                "Path",
            }): "t.JsonValue",
        })
    )

    @unique
    class RefactorRuleKind(StrEnum):
        """Canonical executable text-rule kinds."""

        FUTURE_ANNOTATIONS = "future_annotations"
        LEGACY_REMOVAL = "legacy_removal"
        IMPORT_MODERNIZER = "import_modernizer"
        CLASS_RECONSTRUCTOR = "class_reconstructor"
        PATTERN_CORRECTIONS = "pattern_corrections"
        TYPING_UNIFICATION = "typing_unification"
        TYPING_ANNOTATION_FIX = "typing_annotation_fix"
        TIER0_IMPORT_FIX = "tier0_import_fix"
        SYMBOL_PROPAGATION = "symbol_propagation"
        SIGNATURE_PROPAGATION = "signature_propagation"

    @unique
    class RefactorFileRuleKind(StrEnum):
        """Canonical executable Rope-backed file-rule kinds."""

        CLASS_NESTING = "class_nesting"

    RULE_MATCHERS_BY_KIND: Final[
        t.MappingKV[
            RefactorRuleKind,
            tuple[
                tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]],
                ...,
            ],
        ]
    ] = MappingProxyType({
        RefactorRuleKind.FUTURE_ANNOTATIONS: (
            (
                frozenset({"ensure_future_annotations"}),
                frozenset({"missing_future_import"}),
                frozenset(),
                frozenset(),
            ),
        ),
        RefactorRuleKind.LEGACY_REMOVAL: (
            (
                frozenset({
                    "remove",
                    "inline_and_remove",
                    "remove_and_update_refs",
                    "keep_try_only",
                }),
                frozenset(),
                frozenset(),
                frozenset(),
            ),
        ),
        RefactorRuleKind.IMPORT_MODERNIZER: (
            (
                frozenset({"replace_with_alias", "hoist_to_module_top"}),
                frozenset(),
                frozenset(),
                frozenset(),
            ),
        ),
        RefactorRuleKind.CLASS_RECONSTRUCTOR: (
            (frozenset({"reorder_methods"}), frozenset(), frozenset(), frozenset()),
        ),
        RefactorRuleKind.PATTERN_CORRECTIONS: (
            (
                frozenset({
                    "convert_dict_to_mapping_annotations",
                    "fix_silent_failure_sentinels",
                }),
                frozenset(),
                frozenset(),
                frozenset(),
            ),
            (
                frozenset({"remove_redundant_casts"}),
                frozenset(),
                frozenset(),
                frozenset({RK_REDUNDANT_TYPE_TARGETS}),
            ),
        ),
        RefactorRuleKind.TYPING_UNIFICATION: (
            (frozenset({"unify_typings"}), frozenset(), frozenset(), frozenset()),
        ),
        RefactorRuleKind.TYPING_ANNOTATION_FIX: (
            (
                frozenset({"replace_object_annotations", "remove_unused_models"}),
                frozenset(),
                frozenset(),
                frozenset(),
            ),
        ),
        RefactorRuleKind.TIER0_IMPORT_FIX: (
            (frozenset({"fix_tier0_imports"}), frozenset(), frozenset(), frozenset()),
        ),
        RefactorRuleKind.SYMBOL_PROPAGATION: (
            (
                frozenset({"propagate_symbol_renames"}),
                frozenset(),
                frozenset({RK_IMPORT_SYMBOL_RENAMES}),
                frozenset(),
            ),
            (
                frozenset({"rename_imported_symbols"}),
                frozenset(),
                frozenset(),
                frozenset(),
            ),
        ),
        RefactorRuleKind.SIGNATURE_PROPAGATION: (
            (
                frozenset({"propagate_signature_migrations"}),
                frozenset(),
                frozenset(),
                frozenset({RK_SIGNATURE_MIGRATIONS}),
            ),
        ),
    })
    FILE_RULE_MATCHERS_BY_KIND: Final[
        t.MappingKV[
            RefactorFileRuleKind,
            tuple[
                tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]],
                ...,
            ],
        ]
    ] = MappingProxyType({
        RefactorFileRuleKind.CLASS_NESTING: (
            (frozenset({"nest_classes"}), frozenset(), frozenset(), frozenset()),
        )
    })
    RULE_TABLE_HEADERS: Final[t.StrSequence] = (
        cb.RK_ID,
        cb.NAME,
        cb.RK_DESCRIPTION,
        cb.RK_ENABLED,
        cb.RK_SEVERITY,
    )
    FLEXT_CONSTANTS_FILE_NAMES: Final[frozenset[str]] = frozenset({
        "constants.py",
        "_constants.py",
    })
    "Canonical constants module file names."
    FLEXT_CONSTANTS_DIRECTORY: Final[str] = "constants"
    "Canonical constants package directory name."
    FLEXT_TYPINGS_FILE_NAMES: Final[frozenset[str]] = frozenset({
        "typings.py",
        "_typings.py",
    })
    "Canonical typings module file names."
    FLEXT_TYPINGS_DIRECTORY: Final[str] = "typings"
    "Canonical typings package directory name."
    FLEXT_PROTOCOLS_FILE_NAMES: Final[frozenset[str]] = frozenset({
        "protocols.py",
        "_protocols.py",
    })
    "Canonical protocols module file names."
    FLEXT_PROTOCOLS_DIRECTORY: Final[str] = "protocols"
    "Canonical protocols package directory name."
    FLEXT_PROTOCOLS_DIRECTORIES: Final[frozenset[str]] = frozenset({
        FLEXT_PROTOCOLS_DIRECTORY,
        f"_{FLEXT_PROTOCOLS_DIRECTORY}",
    })
    "Sanctioned protocol package directory names (public and private)."
    FLEXT_MODELS_FILE_NAMES: Final[frozenset[str]] = frozenset({"models.py"})
    "Canonical models module file names."
    FLEXT_MODELS_DIRECTORY: Final[str] = "models"
    "Canonical models package directory name."
    FLEXT_MODELS_DIRECTORIES: Final[frozenset[str]] = frozenset({
        FLEXT_MODELS_DIRECTORY,
        f"_{FLEXT_MODELS_DIRECTORY}",
    })
    "Sanctioned model package directory names (public and private)."
    FLEXT_UTILITIES_FILE_NAMES: Final[frozenset[str]] = frozenset({
        "utilities.py",
        "_utilities.py",
    })
    "Canonical utilities module file names."
    FLEXT_UTILITIES_DIRECTORY: Final[str] = "utilities"
    "Canonical utilities package directory name."
    CONSTANTS_CLASS_SUFFIX: Final[str] = "Constants"
    "Class-name suffix used to identify constants facades."
    CONSTANT_PATTERN: Final[t.RegexPattern] = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
    "Compiled naming pattern for module-level constant candidates."
    FAMILY_SUFFIXES: Final[t.StrMapping] = MappingProxyType({
        "c": "Constants",
        "t": "Types",
        "p": "Protocols",
        "m": "Models",
        "u": "Utilities",
    })
    "Facade family letter → class suffix mapping."
    FAMILY_DIRECTORIES: Final[t.StrMapping] = MappingProxyType({
        "c": "_constants",
        "t": "_typings",
        "p": "_protocols",
        "m": "_models",
        "u": "_utilities",
    })
    "Facade family letter → subdirectory name mapping."
    FAMILY_FILES: Final[t.StrMapping] = MappingProxyType({
        "c": "*constants.py",
        "t": "*typings.py",
        "p": "*protocols.py",
        "m": "*models.py",
        "u": "*utilities.py",
    })
    "Facade family letter → file glob mapping."
    FAMILY_PUBLIC_MODULES: Final[t.StrMapping] = MappingProxyType({
        "c": "constants",
        "m": "models",
        "p": "protocols",
        "t": "typings",
        "u": "utilities",
    })
    "Facade family letter → public facade module suffix mapping."
    NAMESPACE_FILE_TO_FAMILY: Final[t.StrMapping] = _build_namespace_file_to_family((
        ("c", tuple(FLEXT_CONSTANTS_FILE_NAMES)),
        ("t", tuple(FLEXT_TYPINGS_FILE_NAMES)),
        ("p", tuple(FLEXT_PROTOCOLS_FILE_NAMES)),
        ("m", tuple(FLEXT_MODELS_FILE_NAMES)),
        ("u", tuple(FLEXT_UTILITIES_FILE_NAMES)),
    ))
    "Canonical facade file name → family alias mapping."
    NAMESPACE_FAMILY_EXPECTED_ALIAS: Final[t.MappingKV[str, t.StrPair]] = (
        _build_namespace_family_expected_alias(
            (
                ("c", tuple(FLEXT_CONSTANTS_FILE_NAMES)),
                ("t", tuple(FLEXT_TYPINGS_FILE_NAMES)),
                ("p", tuple(FLEXT_PROTOCOLS_FILE_NAMES)),
                ("m", tuple(FLEXT_MODELS_FILE_NAMES)),
                ("u", tuple(FLEXT_UTILITIES_FILE_NAMES)),
            ),
            FAMILY_SUFFIXES,
        )
    )
    "Canonical facade file name → expected (alias, suffix) pair."
    FLEXT_FAMILIES: Final[frozenset[str]] = frozenset({"c", "t", "p", "m", "u"})
    "All FLEXT families."
    FLEXT_FAMILY_PACKAGE_DIRS: Final[t.StrMapping] = MappingProxyType({
        "c": "flext_core/constants.py",
        "t": "flext_core/typings.py",
        "p": "flext_core/protocols.py",
        "m": "flext_core/models",
        "u": "flext_core/_utilities",
    })
    "Family letter → relative package dir/file."
    FLEXT_FAMILY_FACADE_MODULES: Final[t.StrMapping] = MappingProxyType({
        "c": "flext_core/constants.py",
        "t": "flext_core/typings.py",
        "p": "flext_core/protocols.py",
        "m": "flext_core/models.py",
        "u": "flext_core/utilities.py",
    })
    "Family letter → facade module path."
    DOMAIN_PACKAGES: Final[frozenset[str]] = frozenset({
        "flext-ldap",
        "flext-ldif",
        "flext-db-oracle",
        "flext-oracle-wms",
        "flext-oracle-oic",
    })
    "Known domain-layer packages."
    PLATFORM_PACKAGES: Final[frozenset[str]] = frozenset({
        "flext-cli",
        "flext-meltano",
        "flext-api",
        "flext-auth",
        "flext-web",
        "flext-grpc",
    })
    "Known platform-layer packages."
    INTEGRATION_CLASS_PREFIXES: Final[tuple[str, ...]] = (
        "FlextTap",
        "FlextTarget",
        "FlextDbt",
    )
    "Class name prefixes that identify integration projects."
    CONFIDENCE_TO_SCORE: Final[t.MappingKV[str, float]] = MappingProxyType({
        "high": 0.95,
        "medium": 0.75,
        "low": 0.55,
    })
    "Confidence level → numeric score mapping for violations."
    CONFIDENCE_RANKS: Final[t.IntMapping] = MappingProxyType({
        "low": 0,
        "medium": 1,
        "high": 2,
    })
    "Confidence level → priority rank mapping."
    REQUIRED_CLASS_TARGETS: Final[t.StrSequence] = (
        "TimeoutEnforcer",
        "CircuitBreakerManager",
    )
    "Class names always required in scanner output."
    CLASS_PATTERN: Final[t.RegexPattern] = re.compile(r"[^A-Za-z0-9]+")
    "Pattern to split class name fragments."
    MAPPINGS_RELATIVE_PATH: Final[Path] = Path("rules") / "class-nesting-mappings.yml"
    "Relative path from the refactor package to the nesting mappings YAML."
    MODEL_TOKENS: Final[t.StrSequence] = (
        "model",
        "schema",
        "entity",
        "pydantic",
        "dataclass",
    )
    "Tokens indicating model-related code."
    DECORATOR_TOKENS: Final[t.StrSequence] = ("decorator", "inject", "provide")
    "Tokens indicating decorator-related code."
    DISPATCHER_TOKENS: Final[t.StrSequence] = (
        "dispatcher",
        "dispatch",
        "command",
        "query",
        "event",
    )
    "Tokens indicating dispatcher-related code."
    NAMESPACE_PREFIXES: Final[t.StrMapping] = MappingProxyType({
        "utility": "FlextUtilities",
        "models": "FlextModels",
        "decorators": "d",
        "dispatcher": "FlextDispatcher",
    })
    "Namespace → class prefix mapping for violation classification."
    CLASSIFICATION_PRIORITY: Final[t.StrSequence] = (
        "dispatcher",
        "decorators",
        "models",
        "utility",
    )
    "Priority order for violation classification."
    MIN_PATH_DEPTH: int = 2
    "Minimum relative path depth for module prefix detection."
    NAMESPACE_CONSTANT_PATTERN: Final[t.RegexPattern] = re.compile(
        r"^_?[A-Z][A-Z0-9_]+$"
    )
    "Regex: namespace constant candidate names."
    CLASSVAR_EXEMPT_NAMES: Final[frozenset[str]] = c.ENFORCEMENT_CLASSVAR_EXEMPT_NAMES
    "ClassVar attribute names that are framework idioms and stay in place (SSOT: flext-core)."
    CLASSVAR_ALLOWED_CALLS: Final[frozenset[str]] = frozenset({
        "Path",
        "PurePath",
        "PosixPath",
        "WindowsPath",
        "frozenset",
        "tuple",
        "dict",
        "list",
        "set",
        "MappingProxyType",
    })
    "Canonical factory calls allowed as ClassVar default values."
    NAMESPACE_MIN_ALIAS_LENGTH: Final[int] = 2
    FACADE_ALIAS_RE: Final[t.RegexPattern] = re.compile(
        r"^(\w)\b[^=]*=\s*(\w+)", re.MULTILINE
    )
    "Matches ``m = FlextFooModels`` alias assignments in facade files."

    RUNTIME_ALIAS_SRC_DEPTH_MIN: Final[int] = 2
    "Minimum relative path depth for a root ``src/`` facade."
    RUNTIME_ALIAS_SRC_DEPTH_EXACT: Final[int] = 3
    "Exact relative path depth for a root ``src/<pkg>/<facade>.py`` file."
    RUNTIME_ALIAS_NON_ROOT_DIRS: Final[frozenset[str]] = frozenset({
        "tests",
        "examples",
        "scripts",
    })
    "Top-level directories that may contain facade-style alias files."
    RUNTIME_ALIAS_NON_ROOT_DEPTH_EXACT: Final[int] = 2
    "Exact relative path depth for a top-level ``tests|examples|scripts/<file>.py``."
    RUNTIME_ALIAS_PARTS_SKIP: Final[frozenset[str]] = frozenset({
        "_parts",
        "_root_typing_parts",
    })
    "Path fragments that disqualify a file from root-facade alias detection."

    # --- Detector regex constants ---
    ASSIGN_RE: Final[t.RegexPattern] = re.compile(r"^([A-Z_]\w*)\s*[:=]", re.MULTILINE)
    "Matches top-level UPPER_CASE assignments for loose constant detection."
    LOGGER_ASSIGN_RE: Final[t.RegexPattern] = re.compile(
        r"^([A-Za-z_]\w*)\s*[:=]\s*(?:(?:\w+\.)*)?"
        r"(?:fetch_logger|create_module_logger|get_logger|logging\.getLogger)\s*\(",
        re.IGNORECASE | re.MULTILINE,
    )
    "Matches top-level logger assignments created outside namespace classes."
    PEP695_RE: Final[t.RegexPattern] = re.compile(r"^type\s+(\w+)\s*=", re.MULTILINE)
    "Matches PEP 695 type alias definitions."
    TYPEALIAS_ANNOT_RE: Final[t.RegexPattern] = re.compile(
        r"^(\w+)\s*:\s*(?:\w+\.)*TypeAlias\s*=", re.MULTILINE
    )
    "Matches TypeAlias annotation syntax for typing alias detection."
    TYPING_FACTORY_ASSIGN_RE: Final[t.RegexPattern] = re.compile(
        r"^(\w+)\s*=\s*(?:(?:\w+\.)*)?"
        r"(?:TypeVar|ParamSpec|TypeVarTuple|NewType)\s*\(",
        re.MULTILINE,
    )
    "Matches TypeVar/ParamSpec/TypeVarTuple/NewType assignments."
    COMPAT_ALIAS_RE: Final[t.RegexPattern] = re.compile(
        r"^([A-Z]\w+)\s*=\s*([A-Z]\w+)\s*$", re.MULTILINE
    )
    "Matches compatibility alias assignments (CapitalName = CapitalName)."
    COMPAT_SKIP_NAMES: Final[frozenset[str]] = frozenset({
        "__all__",
        "__version__",
        "__version_info__",
    })
    "Names to skip during compatibility alias detection."
    ENFORCEMENT_CANONICAL_ALIASES: Final[frozenset[str]] = (
        c.ENFORCEMENT_CANONICAL_ALIASES
    )
    "Canonical short aliases exposed by FLEXT facades (SSOT: flext-core)."
    ENFORCEMENT_PROJECT_ALIAS_OWNERS: Final[t.StrSequenceMapping] = (
        c.ENFORCEMENT_PROJECT_ALIAS_OWNERS
    )
    "Project package → canonical aliases it re-exports locally (SSOT: flext-core)."
    # flext-j47u: consume core enforcement data through its exact canonical alias.
    ENFORCEMENT_LIBRARY_OWNERS: Final[t.StrMapping] = c.ENFORCEMENT_LIBRARY_OWNERS
    "External library → project that owns its abstraction facade (SSOT: flext-core)."
    FUTURE_ANNOTATIONS_RE: Final[t.RegexPattern] = re.compile(
        r"^from\s+__future__\s+import\s+annotations\b", re.MULTILINE
    )
    "Matches 'from __future__ import annotations' import statement."
    ONLY_DOCSTRING_RE: Final[t.RegexPattern] = re.compile(
        r'^("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*$'
    )
    "Matches files that contain only a module docstring."
    MIN_METHODS_FOR_REORDER: Final[int] = 2
    "Minimum method count before class method reordering is attempted."

    # --- Class nesting refactor constants (was: class ClassNesting) ---
    NESTING_COERCE_KEYS: Final[t.StrSequence] = (
        cb.RK_LOOSE_NAME,
        RK_HELPER_NAME,
        cb.RK_TARGET_NAMESPACE,
        RK_TARGET_NAME,
        cb.RK_REWRITE_SCOPE,
        cb.RK_CONFIDENCE,
    )
    "Keys to coerce from string to typed values in nesting mappings."
    NESTING_SECTION_KEYS: Final[t.StrSequence] = (
        cb.RK_CLASS_NESTING,
        cb.RK_HELPER_CONSOLIDATION,
    )
    "Top-level section keys in class nesting YAML configs."

    # --- Method category StrEnum (was: plain class MethodCategory) ---
    @unique
    class MethodCategory(StrEnum):
        """Canonical method category identifiers for FLEXT reordering."""

        MAGIC = "magic"
        PROPERTY = "property"
        STATIC = "static"
        CLASS = "class"
        PUBLIC = "public"
        PROTECTED = "protected"
        PRIVATE = "private"

    # --- Scan constants (was: class Scan) ---
    SCAN_ALLOWED_TOP_LEVEL: Final[frozenset[str]] = frozenset({
        "__all__",
        "__version__",
        "__version_info__",
    })
    "Top-level names allowed without namespace classification."
    NAMESPACE_PRIVATE_BASE_MODULE: Final[str] = "_base.py"
    "Private base module name allowed to host private FLEXT base contracts."
    NAMESPACE_PRIVATE_BASE_CLASS_SUFFIXES: Final[frozenset[str]] = frozenset({
        "Base",
        "Mixin",
        "Typing",
    })
    "Allowed suffixes for multiple private classes in a private base module."
    NAMESPACE_PYTEST_MODULE_PREFIX: Final[str] = "test_"
    "Pytest module prefix exempt from production loose-object structure checks."
    NAMESPACE_PYTEST_MODULE_SUFFIXES: Final[frozenset[str]] = frozenset({
        "_test.py",
        "_tests.py",
    })
    "Pytest module suffixes exempt from production loose-object structure checks."

    # --- Census mode StrEnum (was: class Census plain strings) ---
    @unique
    class CensusMode(StrEnum):
        """Canonical census usage mode identifiers."""

        ALIAS_FLAT = "alias_flat"
        "Usage via u.method_name (flat alias)."
        ALIAS_NS = "alias_namespaced"
        "Usage via u.ClassName.method_name (namespaced)."
        DIRECT = "direct"
        "Usage via FlextUtilitiesXxx.method_name (direct)."

    ACCESSOR_WARNING_PREFIXES: Final[frozenset[str]] = frozenset({
        "get_",
        "set_",
        "is_",
    })
    "Public accessor name prefixes that should be renamed (drop the prefix or use a canonical verb)."

    # --- Symbol/identifier patterns ---
    IDENTIFIER_PATTERN: Final[t.RegexPattern] = re.compile(r"\b[A-Za-z_]\w*\b")
    "Regex: Python identifier word boundary match."

    # --- Import bypass pattern (for transformer matching) ---
    IMPORT_BYPASS_RE: Final[t.RegexPattern] = re.compile(
        r"^try:\n"
        r"(    from .+\n)"
        r"except ImportError:\n"
        r"    from .+\n",
        re.MULTILINE,
    )
    "Regex: try/except ImportError import bypass block (strict form)."

    # --- Deprecated class pattern ---
    CLASS_BLOCK_RE: Final[t.RegexPattern] = re.compile(
        r"^(class\s+(\w+)\b[^\n]*:\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*)", re.MULTILINE
    )
    "Regex: full class block including body lines."
    DEPRECATION_WARN_RE: Final[t.RegexPattern] = re.compile(r"\.warn\s*\(")
    "Regex: deprecation warning call site (.warn())."

    # --- Lazy import fixer ---
    DEF_ASYNC_CLASS_RE: Final[t.RegexPattern] = re.compile(
        r"^(?:def |async def |class )", re.MULTILINE
    )
    "Regex: top-level def/async def/class keyword (for lazy import detection)."


__all__: list[str] = ["FlextInfraConstantsRefactor"]
