"""Lazy-init constants for the codegen package."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraConstantsCodegenLazy:
    """Lazy-init and export-policy constants for codegen."""

    MAX_LINE_LENGTH: Final[int] = 88
    "Maximum line length for generated import lines."
    MAX_ALIAS_LENGTH: Final[int] = 2
    "Maximum length of a public facade alias."
    AUTOGEN_HEADER: Final[str] = "# AUTO-GENERATED FILE — Regenerate with: make gen"
    "Header prepended to every auto-generated ``__init__.py`` file."
    AUTOGEN_HEADERS: Final[tuple[str, str]] = (
        AUTOGEN_HEADER,
        "# @generated AUTO-GENERATED FILE — Regenerate with: make gen",
    )
    "Current and former generated initializer headers accepted during migration."
    ROOT_EXPORTS_FILENAME: Final[str] = "_exports.py"
    "Root public ABI contract module consumed by lazy-init planning."
    ROOT_EXPORTS_DIR: Final[str] = "_constants"
    "Directory under each package where lazy-init registries must live."
    GENERATED_EXPORT_SIDECAR_RE: Final[t.RegexPattern] = re.compile(
        r"^(?:_exports(?:_lazy(?:_part_[0-9]+)?)?|_lazy_exports)\.py$"
    )
    "Regex matching every generated lazy-export sidecar filename "
    "(``_exports.py``, ``_exports_lazy.py``, ``_exports_lazy_part_N.py``, "
    "``_lazy_exports.py``); legacy variants are excluded from discovery and cleanup."
    OBSOLETE_ROOT_SUPPORT_NAMES: Final[frozenset[str]] = frozenset({
        "_root_exports",
        "_root_exports_parts",
        "_root_typing",
        "_root_typing_parts",
    })
    "Closed set of retired root registry module and package names."
    ROOT_TEMPLATE_RUNTIME_IMPORTS: Final[frozenset[str]] = frozenset({
        "build_lazy_import_map",
        "install_lazy_exports",
    })
    "Names bound eagerly by the canonical root initializer template."
    TEST_ONLY_SOURCE_MODULE_RE: Final[t.RegexPattern] = re.compile(
        r"^(?:_?test(?:_[A-Za-z0-9_]+)?|[A-Za-z0-9_]+_tests?)\.py$"
    )
    "Test-module filenames forbidden from installable package export maps."
    INIT_PY: Final[str] = "__init__.py"
    "Standard Python package initializer filename."
    # flext-wkii.17.26 (codex): cleanup is the only owner of retired init artifacts.
    OBSOLETE_GENERATED_INIT_FILES: Final[t.StrSequence] = ("__unit__.py",)
    "Generated initializer artifacts removed during every codegen pass."
    INIT_PYI: Final[str] = "__init__.pyi"
    "Typing stub paired with generated thin package initializers."
    ROOT_PUBLIC_EXPORTS_SUFFIX: Final[str] = "_PUBLIC_EXPORTS"
    "Suffix for tuple constants that declare frozen public root exports."
    ALL_SCAN_PATTERNS: Final[t.StrSequence] = (
        "src/**/__init__.py",
        "tests/**/__init__.py",
        "examples/**/__init__.py",
        "scripts/**/__init__.py",
    )
    "Glob patterns for all directories the lazy-init generator scans."
    NON_PUBLIC_LAZY_ROOTS: Final[frozenset[str]] = frozenset({
        "examples",
        "scripts",
        "tests",
    })
    "Root import surfaces generated as private lazy plumbing, not public ABI."
    # flext-pulj (codex): pytest must register fixture plugins before importing
    # them, so their private package initializer is always side-effect free.
    # Real cycle exceptions are the bootstrap packages imported while
    # ``flext_core.lazy`` initializes; importing them with a lazy facade would
    # re-enter the partially-initialized module and fail.
    BOOTSTRAP_CYCLE_EXCEPTION_SEGMENTS: Final[frozenset[str]] = frozenset({
        "_lazy_parts",
        "_typings",
    })
    "Package segments whose initializer must remain empty to avoid bootstrap cycles."

    # The generated bootstrap opens with `from flext_core.lazy import ...`, so a
    # package that `flext_core.lazy` itself reaches at module scope cannot carry
    # one: importing it would re-enter the module that is still initializing and
    # fail with "cannot import name 'build_lazy_import_map' from partially
    # initialized module 'flext_core.lazy'". `flext_core.lazy` pulls
    # `._lazy_parts`, which pulls `._typings`, which reaches the other private
    # facets, so the whole private surface of the bootstrap-owning distribution
    # keeps side-effect-free initializers. Private packages of every OTHER
    # distribution import the bootstrap normally and are unaffected.
    LAZY_BOOTSTRAP_ROOT_PACKAGE: Final[str] = "flext_core"

    BARE_IMPORT_FROM_RE: Final[t.RegexPattern] = re.compile(
        r"^from\s+import\s", re.MULTILINE
    )
    "Regex: malformed ``from import`` statement (missing module name)."

    LINT_TOOLS: Final[t.StrSequencePairTuple] = (
        # Ruff runs with NO --select override: the project's pyproject.toml
        # (select=ALL + narrow whitelist + preview) is the ONLY rule policy.
        ("ruff", ("ruff", "check", "{file}", "--no-fix")),
        ("pyright", ("pyright", "{file}")),
        ("mypy", ("mypy", "{file}", "--no-error-summary")),
        ("pyrefly", ("pyrefly", "check", "{file}")),
    )
    "Lint tool names and their CLI command templates for validation."
    LOCAL_INFERRED_SEGMENTS: Final[frozenset[str]] = frozenset({
        "_constants",
        "_exceptions",
        "_models",
        "_protocols",
        "_typings",
        "_utilities",
        "constants",
        "exceptions",
        "models",
        "protocols",
        "typings",
        "utilities",
        "services",
        "docs",
        "tools",
    })
    "Module segments recognized as local inferred imports in lazy-load chain."
    PUBLIC_ROOT_MODULE_EXPORTS: Final[frozenset[str]] = frozenset()
    "Internal child packages exported at the root as module objects only."
    INFRA_ONLY_EXPORTS: Final[frozenset[str]] = frozenset({
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
        "pytest_addoption",
        "pytest_collect_file",
        "pytest_collection_modifyitems",
        "pytest_configure",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
        "pytest_sessionfinish",
        "pytest_sessionstart",
        "pytest_terminal_summary",
        "pytest_warning_recorded",
    })
    "Exports excluded from package __init__.py auto-export."
    PUBLISHED_ALL_EXCLUDE: Final[frozenset[str]] = frozenset({
        "build_lazy_import_map",
        "lazy",
        "normalize_lazy_imports",
    })
    # flext-pulj (codex): these remain direct inline lazy imports without
    # widening the explicit wildcard contract or requiring root sidecars.
    "Public-module symbols withheld from generated root-facade __all__."
    PUBLIC_ROOT_ALIAS_ORDER: Final[t.StrSequence] = (
        "c",
        "t",
        "p",
        "m",
        "u",
        "d",
        "e",
        "h",
        "r",
        "s",
        "x",
        "infra",
        "main",
    )
    "Canonical dependency order for public aliases and operational entry points."
    # flext-wkii.17 (Codex): static analyzers bind local facade classes, not rebinds.
    PUBLIC_ROOT_TYPING_FACADE_SUFFIXES: Final[t.MappingKV[str, str]] = (
        MappingProxyType({
            "c": "Constants",
            "t": "Types",
            "p": "Protocols",
            "m": "Models",
            "u": "Utilities",
            "s": "ServiceBase",
        })
    )
    "Named local facade suffixes used by generated TYPE_CHECKING aliases."
    ROOT_WRAPPER_SEGMENTS: Final[frozenset[str]] = frozenset({
        "docs",
        "src",
        "tests",
        "examples",
        "scripts",
    })
    "Directory segments recognized as project-root wrapper paths."
    DUPLICATE_CLASS_MIN_LEN: Final[int] = 8
    "Minimum class-name length for workspace-wide duplicate detection."
    TEST_RUNTIME_ALIAS_TARGETS: Final[t.MappingKV[str, t.StrPair]] = MappingProxyType({
        "c": ("flext_tests", "c"),
        "d": ("flext_tests", "d"),
        "e": ("flext_tests", "e"),
        "h": ("flext_tests", "h"),
        "m": ("flext_tests", "m"),
        "p": ("flext_tests", "p"),
        "r": ("flext_tests", "r"),
        "s": ("flext_tests", "s"),
        "t": ("flext_tests", "t"),
        "td": ("flext_tests", "td"),
        "tf": ("flext_tests", "tf"),
        "tk": ("flext_tests", "tk"),
        "tm": ("flext_tests", "tm"),
        "tv": ("flext_tests", "tv"),
        "u": ("flext_tests", "u"),
        "x": ("flext_tests", "x"),
    })
    "Mapping of test-only aliases to flext-tests runtime targets."


__all__: list[str] = ["FlextInfraConstantsCodegenLazy"]
