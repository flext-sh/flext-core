"""Centralized constants for the core subpackage."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final


class FlextInfraCoreConstants:
    """Core infrastructure constants."""

    EXEMPT_FILENAMES: Final[frozenset[str]] = frozenset({
        "__init__.py",
        "conftest.py",
        "__main__.py",
    })
    EXEMPT_PREFIXES: Final[frozenset[str]] = frozenset({"test_", "_"})
    ALIAS_NAMES: Final[frozenset[str]] = frozenset({
        "c",
        "t",
        "m",
        "p",
        "u",
        "r",
        "d",
        "e",
        "h",
        "s",
        "x",
        "tc",
    })
    DUNDER_ALLOWED: Final[frozenset[str]] = frozenset({"__all__", "__version__"})
    TYPEVAR_CALLABLES: Final[frozenset[str]] = frozenset({
        "TypeVar",
        "ParamSpec",
        "TypeVarTuple",
    })
    ENUM_BASES: Final[frozenset[str]] = frozenset({"StrEnum", "Enum", "IntEnum"})
    COLLECTION_CALLS: Final[frozenset[str]] = frozenset({
        "frozenset",
        "tuple",
        "dict",
        "list",
    })
    SKILLS_DIR: Final[Path] = Path(".claude/skills")
    REPORT_DEFAULT: Final[str] = ".claude/skills/{skill}/report.json"
    BASELINE_DEFAULT: Final[str] = ".claude/skills/{skill}/baseline.json"
    CACHE_TTL_SECONDS: Final[int] = 300
    MISSING_IMPORT_RE: Final[re.Pattern[str]] = re.compile(
        r"Cannot find module `([^`]+)` \[missing-import\]",
    )
    MYPY_HINT_RE: Final[re.Pattern[str]] = re.compile(
        r"note:\s+(?:hint|note):\s+.*?`(types-\S+)`",
    )
    MYPY_STUB_RE: Final[re.Pattern[str]] = re.compile(
        r"Library stubs not installed for ['\"](\S+?)['\"]",
    )
    INTERNAL_PREFIXES: Final[tuple[str, ...]] = ("flext_", "flext-")


class FlextInfraSharedInfraConstants:
    """Shared infrastructure constants consumed by flext_infra.constants."""

    class Files:
        PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
        MAKEFILE_FILENAME: Final[str] = "Makefile"
        BASE_MK: Final[str] = "base.mk"
        GO_MOD: Final[str] = "go.mod"
        GITMODULES: Final[str] = ".gitmodules"
        GITIGNORE: Final[str] = ".gitignore"
        INIT_PY: Final[str] = "__init__.py"

    class Git:
        DIR: Final[str] = ".git"
        ORIGIN: Final[str] = "origin"
        MAIN: Final[str] = "main"
        HEAD: Final[str] = "HEAD"

    class Packages:
        CORE: Final[str] = "flext-core"
        CORE_UNDERSCORE: Final[str] = "flext_core"
        ROOT: Final[str] = "flext"

    class Extensions:
        PYTHON: Final[str] = ".py"
        PYTHON_GLOB: Final[str] = "*.py"

    class Directories:
        TESTS: Final[str] = "tests"
        EXAMPLES: Final[str] = "examples"
        SCRIPTS: Final[str] = "scripts"
        TYPINGS: Final[str] = "typings"
        DOCS: Final[str] = "docs"
        BUILD: Final[str] = "build"
        DIST: Final[str] = "dist"
        SITE: Final[str] = "site"

    class Timeouts:
        DEFAULT: Final[int] = 300
        SHORT: Final[int] = 60
        MEDIUM: Final[int] = 120
        LONG: Final[int] = 600
        CI: Final[int] = 900

    class Paths:
        WORKSPACE_MARKERS: Final[frozenset[str]] = frozenset({
            ".git",
            "Makefile",
            "pyproject.toml",
        })
        VENV_BIN_REL: Final[str] = ".venv/bin"
        DEFAULT_SRC_DIR: Final[str] = "src"


__all__ = ["FlextInfraCoreConstants", "FlextInfraSharedInfraConstants"]
