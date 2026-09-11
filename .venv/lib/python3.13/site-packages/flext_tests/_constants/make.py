"""Make command framework constants for flext_tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextTestsConstantsMake:
    """Constants for the generic Make command registry."""

    MAKE_DEFAULT_COMMAND: Final[str] = "all"
    MAKE_HEADER_START: Final[str] = "/// flext-command"
    MAKE_HEADER_END: Final[str] = "///"
    MAKE_DISPATCH_ENV: Final[str] = "FLEXT_COMMAND_DISPATCHED"
    MAKE_DISPATCH_ENV_VALUE: Final[str] = "Y"
    MAKE_DISPATCH_VERB_ENV: Final[str] = "FLEXT_COMMAND_VERB"
    MAKE_DISPATCH_WHAT_ENV: Final[str] = "FLEXT_COMMAND_WHAT"
    MAKE_DISPATCH_PATH_ENV: Final[str] = "FLEXT_COMMAND_PATH"
    MAKE_SURFACE_VALIDATE_ENV: Final[str] = "FLEXT_SURFACE_VALIDATE"
    MAKE_APPLY_PARAM: Final[str] = "APPLY"
    MAKE_WHAT_PARAM: Final[str] = "WHAT"
    MAKE_HELP_PARAM: Final[str] = "HELP"
    MAKE_OPTIONS_PARAM: Final[str] = "OPTIONS"
    MAKE_PYTHONPATH_ENV: Final[str] = "PYTHONPATH"
    MAKE_COMMAND_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".sh"})
    MAKE_IGNORED_COMMAND_DIRS: Final[frozenset[str]] = frozenset({
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "github",
        "lib",
        "maintenance",
    })
    MAKE_MUTATION_REQUIRED_PARAMS: Final[frozenset[str]] = frozenset({MAKE_APPLY_PARAM})
    MAKE_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "Y", "YES", "TRUE"})
    MAKE_SAFE_PROBE_VALUES: Final[t.MappingKV[str, str]] = {
        "APPLY": "N",
        "CHECK_GATES": "lint",
        "DEPS_REPORT": "0",
        "DOCS_PHASE": "validate",
        "DRY_RUN": "1",
        "MESSAGE": "chore: surface validation",
        "PYTEST_ARGS": "-q",
        "TAG": "surface-validation",
        "VALIDATE_SCOPE": "project",
    }


__all__: list[str] = ["FlextTestsConstantsMake"]
