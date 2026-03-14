"""Centralized constants for the codegen package.

All constants used across codegen modules are defined here to avoid
duplication and ensure single-source-of-truth for configuration values.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import Final


class FlextInfraCodegenConstants:
    """Namespace for all codegen-related constants."""

    EXCLUDED_PROJECTS: Final[frozenset[str]] = frozenset({"flexcore"})
    "Projects excluded from all codegen operations (Go/Python hybrid)."
    TYPEVAR_CALLABLES: Final[frozenset[str]] = frozenset({
        "TypeVar",
        "ParamSpec",
        "TypeVarTuple",
    })
    "Callable names that create type variables (for standalone detection)."
    SRC_MODULES: Final[tuple[tuple[str, str, str, str], ...]] = (
        ("constants.py", "Constants", "FlextConstants", "Constants"),
        ("typings.py", "Types", "FlextTypes", "Type aliases"),
        ("protocols.py", "Protocols", "FlextProtocols", "Protocol definitions"),
        ("models.py", "Models", "FlextModels", "Domain models"),
        ("utilities.py", "Utilities", "FlextUtilities", "Utility functions"),
    )
    "Base module definitions for src/: (filename, class_suffix, base_class, docstring)."
    TESTS_MODULES: Final[tuple[tuple[str, str, str, str], ...]] = (
        ("constants.py", "Constants", "FlextTestsConstants", "Test constants"),
        ("typings.py", "Types", "FlextTestsTypes", "Test type aliases"),
        ("protocols.py", "Protocols", "FlextTestsProtocols", "Test protocols"),
        ("models.py", "Models", "FlextTestsModels", "Test models"),
        ("utilities.py", "Utilities", "FlextTestsUtilities", "Test utilities"),
    )
    "Base module definitions for tests/: (filename, class_suffix, base_class, docstring)."
    VIOLATION_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\[(?P<rule>NS-\d{3})-\d{3}\]\s+(?P<module>[^:]+):(?P<line>\d+)\s+\u2014\s+(?P<message>.+)",
    )
    "Regex to parse violation strings: [NS-00X-NNN] path:line — message."
    ALIAS_TO_SUFFIX: Final[dict[str, str]] = {
        "c": "Constants",
        "d": "Decorators",
        "e": "Exceptions",
        "h": "Handlers",
        "m": "Models",
        "p": "Protocols",
        "r": "Result",
        "s": "Service",
        "t": "Types",
        "u": "Utilities",
        "x": "Mixins",
    }
    "Single-letter alias → class suffix mapping for lazy-init generation."
    SKIP_MODULES: Final[frozenset[str]] = frozenset({
        "__future__",
        "typing",
        "collections.abc",
        "abc",
    })
    "Modules to skip when deriving lazy import mappings."
    SKIP_STDLIB: Final[frozenset[str]] = frozenset({
        "sys",
        "importlib",
        "typing",
        "collections",
        "abc",
    })
    "Stdlib modules to skip in lazy-init import derivation."
    MAX_LINE_LENGTH: Final[int] = 88
    "Maximum line length for generated import lines."
    AUTOGEN_HEADER: Final[str] = (
        "# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.\n"
        "# Regenerate with: make codegen\n"
        "#"
    )
    "Header prepended to every auto-generated ``__init__.py`` file."
    ALL_SCAN_PATTERNS: Final[tuple[str, ...]] = (
        "src/**/__init__.py",
        "tests/**/__init__.py",
        "examples/**/__init__.py",
        "scripts/**/__init__.py",
    )
    "Glob patterns for all directories the lazy-init generator scans."

    class QualityGate:
        """Constants used by constants quality gate analysis/reporting."""

        REPORT_DIR: Final[str] = ".reports/codegen/constants-quality-gate"
        PASS_VERDICTS: Final[tuple[str, ...]] = ("PASS", "CONDITIONAL_PASS")
        RULE_KEYS: Final[tuple[str, ...]] = ("NS-000", "NS-001", "NS-002")
        CHECK_NAMESPACE_COMPLIANCE: Final[str] = "namespace_compliance"
        CHECK_MRO_VALIDITY: Final[str] = "mro_validity"
        CHECK_IMPORT_RESOLUTION: Final[str] = "import_resolution"
        CHECK_LAYER_COMPLIANCE: Final[str] = "layer_compliance"
        CHECK_DUPLICATION_REDUCTION: Final[str] = "duplication_reduction"
        CHECK_TYPE_SAFETY: Final[str] = "type_safety"
        CHECK_LINT_CLEAN: Final[str] = "lint_clean"
        CHECK_BASELINE_LOAD: Final[str] = "baseline_load"
