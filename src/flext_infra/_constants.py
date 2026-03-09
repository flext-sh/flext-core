"""Shared constants for flext_infra constants facade."""

from __future__ import annotations

from typing import Final


class FlextInfraSharedConstants:
    """Shared infrastructure constants extracted from main constants facade."""

    class Files:
        """File-related constants."""

        PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
        MAKEFILE_FILENAME: Final[str] = "Makefile"
        BASE_MK: Final[str] = "base.mk"
        GO_MOD: Final[str] = "go.mod"
        GITMODULES: Final[str] = ".gitmodules"
        GITIGNORE: Final[str] = ".gitignore"
        INIT_PY: Final[str] = "__init__.py"

    class Git:
        """Git-related constants."""

        DIR: Final[str] = ".git"
        ORIGIN: Final[str] = "origin"
        MAIN: Final[str] = "main"
        HEAD: Final[str] = "HEAD"

    class Packages:
        """Package naming constants used across infra modules."""

        CORE: Final[str] = "flext-core"
        CORE_UNDERSCORE: Final[str] = "flext_core"
        ROOT: Final[str] = "flext"

    class Extensions:
        """File extension constants."""

        PYTHON: Final[str] = ".py"
        PYTHON_GLOB: Final[str] = "*.py"

    class Directories:
        """Common directory name constants."""

        TESTS: Final[str] = "tests"
        EXAMPLES: Final[str] = "examples"
        SCRIPTS: Final[str] = "scripts"
        TYPINGS: Final[str] = "typings"
        DOCS: Final[str] = "docs"
        BUILD: Final[str] = "build"
        DIST: Final[str] = "dist"
        SITE: Final[str] = "site"

    class Timeouts:
        """Standard timeout values in seconds."""

        DEFAULT: Final[int] = 300
        SHORT: Final[int] = 60
        MEDIUM: Final[int] = 120
        LONG: Final[int] = 600
        CI: Final[int] = 900

    class Paths:
        """Path resolution constants for workspace navigation."""

        WORKSPACE_MARKERS: Final[frozenset[str]] = frozenset({
            ".git",
            "Makefile",
            "pyproject.toml",
        })
        VENV_BIN_REL: Final[str] = ".venv/bin"
        DEFAULT_SRC_DIR: Final[str] = "src"


__all__ = ["FlextInfraSharedConstants"]
