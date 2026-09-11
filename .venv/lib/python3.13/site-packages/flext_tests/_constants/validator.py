"""Architecture validator constants for flext-tests (data-only facade)."""

from __future__ import annotations

import re
from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextTestsConstantsValidator:
    """Architecture validator rule, message, regex, and layer constants."""

    @unique
    class ValidatorSeverity(StrEnum):
        """Violation severity levels."""

        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"

    VALIDATOR_RULE_IMPORT_001: Final[t.StrPair] = (
        "HIGH",
        "Lazy import (not at module top)",
    )
    VALIDATOR_RULE_IMPORT_002: Final[t.StrPair] = (
        "HIGH",
        "TYPE_CHECKING block detected",
    )
    VALIDATOR_RULE_IMPORT_003: Final[t.StrPair] = (
        "HIGH",
        "try/except ImportError pattern",
    )
    VALIDATOR_RULE_IMPORT_004: Final[t.StrPair] = ("CRITICAL", "sys.path manipulation")
    VALIDATOR_RULE_IMPORT_005: Final[t.StrPair] = (
        "MEDIUM",
        "Direct technology import (should use facade)",
    )
    VALIDATOR_RULE_IMPORT_006: Final[t.StrPair] = (
        "HIGH",
        "Non-root import from flext-* package",
    )
    VALIDATOR_RULE_TYPE_001: Final[t.StrPair] = ("CRITICAL", "type suppression comment")
    VALIDATOR_RULE_TYPE_002: Final[t.StrPair] = ("CRITICAL", "wildcard type annotation")
    VALIDATOR_RULE_TYPE_003: Final[t.StrPair] = ("MEDIUM", "Unapproved  usage")
    VALIDATOR_RULE_TYPE_004: Final[t.StrPair] = (
        "CRITICAL",
        "legacy typing factory or generic syntax",
    )
    VALIDATOR_RULE_TYPE_005: Final[t.StrPair] = (
        "CRITICAL",
        "legacy typing annotation form",
    )
    VALIDATOR_RULE_TYPE_006: Final[t.StrPair] = (
        "CRITICAL",
        "forbidden object annotation",
    )
    VALIDATOR_RULE_TYPE_007: Final[t.StrPair] = ("HIGH", "bool-returning is_* helper")
    VALIDATOR_RULE_TEST_001: Final[t.StrPair] = ("HIGH", "monkeypatch usage detected")
    VALIDATOR_RULE_TEST_002: Final[t.StrPair] = (
        "HIGH",
        "Mock/MagicMock usage detected",
    )
    VALIDATOR_RULE_TEST_003: Final[t.StrPair] = (
        "HIGH",
        "@patch decorator usage detected",
    )
    VALIDATOR_RULE_CONFIG_001: Final[t.StrPair] = (
        "CRITICAL",
        "mypy ignore_errors = true",
    )
    VALIDATOR_RULE_CONFIG_002: Final[t.StrPair] = (
        "HIGH",
        "Custom ruff ignore beyond approved list",
    )
    VALIDATOR_RULE_CONFIG_003: Final[t.StrPair] = (
        "MEDIUM",
        "disallow_incomplete_defs = false",
    )
    VALIDATOR_RULE_CONFIG_004: Final[t.StrPair] = ("MEDIUM", "warn_return_any = false")
    VALIDATOR_RULE_CONFIG_005: Final[t.StrPair] = ("LOW", "reportPrivateUsage = false")
    VALIDATOR_RULE_BYPASS_001: Final[t.StrPair] = ("MEDIUM", "noqa comment detected")
    VALIDATOR_RULE_BYPASS_002: Final[t.StrPair] = (
        "LOW",
        "pragma: no cover (unapproved)",
    )
    VALIDATOR_RULE_BYPASS_003: Final[t.StrPair] = (
        "HIGH",
        "Exception swallowing (bare except or pass)",
    )
    VALIDATOR_RULE_LAYER_001: Final[t.StrPair] = (
        "CRITICAL",
        "Lower layer importing upper layer",
    )
    VALIDATOR_RULE_MD_001: Final[t.StrPair] = (
        "CRITICAL",
        "Python syntax error in markdown code block",
    )
    VALIDATOR_RULE_MD_002: Final[t.StrPair] = (
        "CRITICAL",
        "Forbidden typing import in markdown code block",
    )
    VALIDATOR_RULE_MD_003: Final[t.StrPair] = (
        "MEDIUM",
        "Missing future annotations in markdown code block",
    )
    VALIDATOR_RULE_MD_004: Final[t.StrPair] = (
        "HIGH",
        "Forbidden type annotation in markdown code block",
    )
    VALIDATOR_RULE_MD_005: Final[t.StrPair] = (
        "HIGH",
        "Forbidden Any annotation in markdown code block",
    )

    VALIDATOR_MSG_LAYER_VIOLATION: Final[str] = (
        "'{current}' L{current_level} -> '{imported}' L{imported_level}"
    )
    VALIDATOR_MSG_CONFIG_IGNORE: Final[str] = (
        "ignore_errors = true for module '{module}'"
    )
    VALIDATOR_MSG_TEST_MONKEYPATCH: Final[str] = (
        "monkeypatch usage in function '{func}'"
    )
    VALIDATOR_MSG_TYPE_ANY_ARG: Final[str] = "wildcard type in argument '{arg}'"
    VALIDATOR_MSG_TYPE_ANY_RETURN: Final[str] = "wildcard type in return type"
    VALIDATOR_MSG_TYPE_LEGACY_FACTORY: Final[str] = (
        "legacy typing factory or generic syntax '{name}'"
    )
    VALIDATOR_MSG_TYPE_LEGACY_ANNOTATION: Final[str] = (
        "legacy typing annotation '{name}'"
    )
    VALIDATOR_MSG_TYPE_OBJECT_ANNOTATION: Final[str] = (
        "forbidden object annotation in {location}"
    )
    VALIDATOR_MSG_TYPE_BOOL_IS_HELPER: Final[str] = (
        "bool-returning is_* helper '{name}'"
    )
    VALIDATOR_MSG_BYPASS_BARE_EXCEPT: Final[str] = "bare except"
    VALIDATOR_MSG_BYPASS_TRIVIAL_BODY: Final[str] = "except with a trivial body"
    VALIDATOR_MSG_MD_SYNTAX: Final[str] = "SyntaxError in code block: {msg}"
    VALIDATOR_MSG_MD_FORBIDDEN_IMPORT: Final[str] = "Forbidden import: {import_name}"
    VALIDATOR_MSG_MD_MISSING_FUTURE: Final[str] = (
        "Missing: from __future__ import annotations"
    )
    VALIDATOR_MSG_MD_FORBIDDEN_ANNOTATION: Final[str] = (
        "Forbidden annotation: {annotation} (use t.* contracts)"
    )
    VALIDATOR_TYPE_IGNORE_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"#\s*type:\s*ignore"
    )
    VALIDATOR_NOQA_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"#\s*noqa", re.IGNORECASE
    )
    VALIDATOR_PRAGMA_NO_COVER_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"#\s*pragma:\s*no\s*cover", re.IGNORECASE
    )
    VALIDATOR_CAST_USAGE_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"\b(?:typing\.)?cast\s*\("
    )
    VALIDATOR_LEGACY_FACTORY_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"\b(?P<name>ParamSpec|TypeVar)\s*\("
    )
    VALIDATOR_TYPE_ALIAS_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r":\s*(?:typing\.)?TypeAlias\b"
    )
    VALIDATOR_GENERIC_BASE_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*class\s+\w+\s*\([^)]*\b(?:typing\.)?Generic\s*\["
    )
    VALIDATOR_LEGACY_ANNOTATION_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"(?:[:(,]\s*|->\s*)(?:typing\.)?(?P<name>Dict|FrozenSet|List|Optional|Set|Tuple|TypeAliasType|TypeGuard|Union)\b"
    )
    VALIDATOR_FUNCTION_DEF_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*(?:async\s+def|def)\s+(?P<name>[A-Za-z_]\w*)"
    )
    VALIDATOR_ANY_ARG_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"(?P<arg>[A-Za-z_]\w*)\s*:\s*(?:typing\.)?Any\b"
    )
    VALIDATOR_ANY_RETURN_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"->\s*(?:typing\.)?Any\b"
    )
    VALIDATOR_ANY_VAR_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*[A-Za-z_]\w*\s*:\s*(?:typing\.)?Any\b"
    )
    VALIDATOR_OBJECT_ARG_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"(?P<arg>[A-Za-z_]\w*)\s*:\s*(?:builtins\.)?object\b"
    )
    VALIDATOR_OBJECT_RETURN_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"->\s*(?:builtins\.)?object\b"
    )
    VALIDATOR_OBJECT_VAR_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*[A-Za-z_]\w*\s*:\s*(?:builtins\.)?object\b"
    )
    VALIDATOR_BOOL_IS_HELPER_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*(?:async\s+def|def)\s+(?P<name>is_[A-Za-z_]\w*)\b.*->\s*bool\b"
    )
    VALIDATOR_INDENTED_IMPORT_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]+(?:from\s+\S+\s+import\b|import\s+\S+)"
    )
    VALIDATOR_IMPORT_ERROR_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*except\b.*\b(?:ImportError|ModuleNotFoundError)\b.*:\s*(?:#.*)?$"
    )
    VALIDATOR_BARE_EXCEPT_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*except\s*:\s*(?:#.*)?$"
    )
    VALIDATOR_EXCEPT_HEADER_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^(?P<indent>[ \t]*)except\b.*:\s*(?:#.*)?$"
    )
    VALIDATOR_PASS_OR_ELLIPSIS_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^(?:pass|\.\.\.)\s*(?:#.*)?$"
    )
    VALIDATOR_SYS_PATH_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"\bsys\.path(?:\s*\[|\.(?:append|extend|insert|pop|remove)\s*\()"
    )
    VALIDATOR_FLEXT_FROM_IMPORT_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*from\s+(?P<module>flext_[A-Za-z0-9_.]+)\s+import\b"
    )
    VALIDATOR_FLEXT_IMPORT_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*import\s+(?P<module>flext_[A-Za-z0-9_.]+)\b"
    )
    VALIDATOR_FROM_IMPORT_LINE_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*from\s+(?P<module>\.*[A-Za-z_][\w.]*)\s+import\s+(?P<names>[^#]+)"
    )
    VALIDATOR_IMPORT_LINE_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*import\s+(?P<modules>[^#]+)"
    )
    VALIDATOR_MONKEYPATCH_ACCESS_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"\bmonkeypatch\.(?P<attr>[A-Za-z_]\w*)\b"
    )
    VALIDATOR_MOCK_CALL_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"\b(?P<name>AsyncMock|MagicMock|Mock|PropertyMock)\s*\("
    )
    VALIDATOR_PATCH_DECORATOR_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^[ \t]*@(?:[A-Za-z_]\w*\.)*patch(?:\b|\s*\(|\.)"
    )
    VALIDATOR_MD_OPTION_DOCS: Final[str] = "--markdown-docs"
    VALIDATOR_MD_NOTEST_MARKER: Final[str] = "notest"
    VALIDATOR_MD_PYTHON_BLOCK_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"^```(?P<info>python\S*(?:\s+notest)?)\s*$\n(?P<code>.*?)^```\s*$",
        re.MULTILINE | re.DOTALL,
    )
    VALIDATOR_MD_OBJECT_ANNOTATION_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"(?::\s*object\b|->.*\bobject\b)"
    )
    VALIDATOR_MD_ANY_ANNOTATION_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        r"(?::\s*Any\b|->.*\bAny\b)"
    )
    VALIDATOR_MD_FUTURE_ANNOTATIONS_MARKER: Final[str] = (
        "from __future__ import annotations"
    )
    VALIDATOR_MD_TYPING_IMPORT_PREFIX: Final[str] = "from typing import"
    VALIDATOR_MD_FORBIDDEN_TYPING_NAMES: Final[frozenset[str]] = frozenset({
        "Any",
        "Optional",
        "Union",
    })

    ENFORCEMENT_WORKSPACE_MARKERS: Final[t.StrSequence] = (
        "AGENTS.md",
        "flext-core",
        "flext-tests",
    )
    VALIDATOR_EXCLUDE_PATTERNS: Final[t.StrSequence] = (
        "**/.venv/**",
        "**/venv/**",
        "**/__pycache__/**",
        "**/build/**",
        "**/dist/**",
        "**/.git/**",
        "**/htmlcov/**",
        "**/*.pyc",
        "**/.hypothesis/**",
    )
    VALIDATOR_INCLUDE_PATTERNS: Final[t.StrSequence] = ("**/*.py",)
    VALIDATOR_IMPORTS_KEY: Final[str] = "imports"
    VALIDATOR_TYPES_KEY: Final[str] = "types"
    VALIDATOR_TESTS_KEY: Final[str] = "tests"
    VALIDATOR_CONFIG_KEY: Final[str] = "settings"
    VALIDATOR_BYPASS_KEY: Final[str] = "bypass"
    VALIDATOR_LAYER_KEY: Final[str] = "layer"
    VALIDATOR_MARKDOWN_KEY: Final[str] = "markdown"
    VALIDATOR_APPROVED_CAST_SERVICE_PATTERN: Final[str] = "service\\.py$"
    VALIDATOR_APPROVED_CAST_CONTAINER_PATTERN: Final[str] = "container\\.py$"
    VALIDATOR_APPROVED_PRAGMA_PATTERN: Final[str] = "__init__\\.py$"
    VALIDATOR_APPROVED_INTERNAL_INIT_PATTERN: Final[str] = "_[^/]+/__init__\\.py$"
    VALIDATOR_APPROVED_CAST_SERVICE_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        VALIDATOR_APPROVED_CAST_SERVICE_PATTERN
    )
    VALIDATOR_APPROVED_CAST_CONTAINER_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        VALIDATOR_APPROVED_CAST_CONTAINER_PATTERN
    )
    VALIDATOR_APPROVED_PRAGMA_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        VALIDATOR_APPROVED_PRAGMA_PATTERN
    )
    VALIDATOR_APPROVED_INTERNAL_INIT_RE: ClassVar[t.Infra.RegexPattern] = re.compile(
        VALIDATOR_APPROVED_INTERNAL_INIT_PATTERN
    )
    VALIDATOR_APPROVED_PATH_REGEX_BY_PATTERN: ClassVar[
        t.MappingKV[str, t.Infra.RegexPattern]
    ] = MappingProxyType({
        VALIDATOR_APPROVED_CAST_SERVICE_PATTERN: VALIDATOR_APPROVED_CAST_SERVICE_RE,
        VALIDATOR_APPROVED_CAST_CONTAINER_PATTERN: VALIDATOR_APPROVED_CAST_CONTAINER_RE,
        VALIDATOR_APPROVED_PRAGMA_PATTERN: VALIDATOR_APPROVED_PRAGMA_RE,
        VALIDATOR_APPROVED_INTERNAL_INIT_PATTERN: VALIDATOR_APPROVED_INTERNAL_INIT_RE,
    })
    VALIDATOR_APPROVED_CAST_PATTERNS: Final[t.StrSequence] = (
        VALIDATOR_APPROVED_CAST_SERVICE_PATTERN,
        VALIDATOR_APPROVED_CAST_CONTAINER_PATTERN,
    )
    VALIDATOR_LEGACY_FACTORY_NAMES: Final[frozenset[str]] = frozenset({
        "ParamSpec",
        "TypeAlias",
        "TypeVar",
    })
    VALIDATOR_LEGACY_BASE_NAMES: Final[frozenset[str]] = frozenset({"Generic"})
    VALIDATOR_LEGACY_ANNOTATION_NAMES: Final[frozenset[str]] = frozenset({
        "Dict",
        "FrozenSet",
        "List",
        "Optional",
        "Set",
        "Tuple",
        "TypeAliasType",
        "TypeGuard",
        "Union",
    })
    VALIDATOR_APPROVED_PRAGMA_PATTERNS: Final[t.StrSequence] = (
        VALIDATOR_APPROVED_PRAGMA_PATTERN,
    )
    VALIDATOR_APPROVED_MOCK_NAMES: Final[frozenset[str]] = frozenset({
        "Mock",
        "MagicMock",
        "AsyncMock",
        "PropertyMock",
    })
    VALIDATOR_APPROVED_INTERNAL_INIT_PATTERNS: Final[t.StrSequence] = (
        VALIDATOR_APPROVED_INTERNAL_INIT_PATTERN,
    )
    LAYER_CONSTANTS: Final[int] = 0
    LAYER_TYPINGS: Final[int] = 0
    LAYER_PROTOCOLS: Final[int] = 0
    LAYER_CONFIG: Final[int] = 1
    LAYER_RUNTIME: Final[int] = 2
    LAYER_EXCEPTIONS: Final[int] = 3
    LAYER_RESULT: Final[int] = 3
    LAYER_LOGGINGS: Final[int] = 4
    LAYER_MODELS: Final[int] = 5
    LAYER_UTILITIES: Final[int] = 5
    LAYER_MIXINS: Final[int] = 5
    LAYER_CONTAINER: Final[int] = 6
    LAYER_SERVICE: Final[int] = 6
    LAYER_CONTEXT: Final[int] = 6
    LAYER_HANDLERS: Final[int] = 7
    LAYER_DISPATCHER: Final[int] = 8
    LAYER_REGISTRY: Final[int] = 8
    LAYER_DECORATORS: Final[int] = 9


__all__: list[str] = ["FlextTestsConstantsValidator"]
