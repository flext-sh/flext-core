"""CLI-related constants for flext-infra.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraConstantsCli:
    """Shared CLI flag vocabularies and route tables."""

    PROCESS_TIMEOUT_EXIT_CODE: Final[int] = 124
    "Exit code emitted by the canonical wall-time limiter."
    PROCESS_COMMAND_NOT_FOUND_EXIT_CODE: Final[int] = 127
    "POSIX shell exit code for an executable that could not be located or started."
    PROCESS_SIGNAL_EXIT_OFFSET: Final[int] = 128
    "POSIX shell offset used to encode a terminating signal."
    PROCESS_EXIT_ERROR_CODE: Final[str] = "EXTERNAL_PROCESS_EXIT"
    "Stable result error code for non-zero external process exits."

    _SHARED_BOOL_FLAG_VALUES: Final[tuple[str, ...]] = (
        "--apply",
        "--check",
        "--check-only",
        "--dry-run",
        "--fail-fast",
        "--verbose",
        "--quiet",
        "--no-fail",
        "--typings",
        "--apply-typings",
        "--no-pip-check",
        "--skip-check",
        "--skip-comments",
        "--audit",
        "--rewrite-constraints",
    )
    _SHARED_VALUE_FLAG_VALUES: Final[tuple[str, ...]] = (
        "--checks",
        "--docstring-min",
        "--workspace",
        "--projects",
        "--project",
        "--module",
        "--namespace",
        "--gates",
        "--what",
        "--format",
        "--output",
        "--report",
        "--output-dir",
        "--json-output",
        "--reports-dir",
        "--ruff-args",
        "--pyright-args",
        "--operation",
        "--branch",
        "--base",
        "--signer",
        "--allowed-signers",
        "--expected-gates",
        "--commit-sha",
        "--output",
    )
    SHARED_BOOL_FLAGS: Final[frozenset[str]] = frozenset(_SHARED_BOOL_FLAG_VALUES)
    SHARED_VALUE_FLAGS: Final[frozenset[str]] = frozenset(_SHARED_VALUE_FLAG_VALUES)
    CLI_GROUP_DESCRIPTIONS: Final[t.StrMapping] = MappingProxyType({
        "check": "Lint gates and pyrefly settings management",
        "codegen": "Code generation and workspace standardization",
        "validate": "Infrastructure validators and diagnostics",
        "deps": "Dependency detection and modernization",
        "docs": "Documentation audit, fix, build, generate, validate",
        "github": "GitHub workflow synchronization and linting",
        "maintenance": "Python version enforcement",
        "refactor": "Declarative refactoring and modernization",
        "release": "Release orchestration",
        "workspace": "Workspace detection and orchestration",
    })


__all__: tuple[str, ...] = ("FlextInfraConstantsCli",)
