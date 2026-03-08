"""Centralized constants for the deps subpackage."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final


class FlextInfraDepsConstants:
    """Deps infrastructure constants."""

    PYRIGHT_BASE_ROOT: Final[list[str]] = [
        "scripts",
        "src",
        "typings",
        "typings/generated",
    ]
    MYPY_BASE_ROOT: Final[list[str]] = ["typings", "typings/generated", "src"]
    PYRIGHT_BASE_PROJECT: Final[list[str]] = [
        ".",
        "src",
        "tests",
        "examples",
        "scripts",
        "../typings",
        "../typings/generated",
    ]
    MYPY_BASE_PROJECT: Final[list[str]] = [
        ".",
        "../typings",
        "../typings/generated",
        "src",
    ]
    GIT_REF_RE: Final[re.Pattern[str]] = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$"
    )
    GITHUB_REPO_URL_RE: Final[re.Pattern[str]] = re.compile(
        "^(?:git@github\\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\\.git)?|https://github\\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\\.git)?)$"
    )
    PEP621_PATH_RE: Final[re.Pattern[str]] = re.compile("@\\s*(?:file:)?(?P<path>.+)$")
    SKIP_DIRS: Final[frozenset[str]] = frozenset({
        ".archive",
        ".claude.disabled",
        ".flext-deps",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".sisyphus",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "site",
        "vendor",
    })
    DEP_NAME_RE: Final[re.Pattern[str]] = re.compile("^\\s*([A-Za-z0-9_.-]+)")
    RECENT_LINES_FOR_MARKER: Final[int] = 3
    RECENT_LINES_FOR_DEV_DEP: Final[int] = 4
    FLEXT_DEPS_DIR: Final[str] = ".flext-deps"
    PEP621_PATH_DEP_RE: Final[re.Pattern[str]] = re.compile(
        "^(?P<name>[A-Za-z0-9_.-]+)\\s*@\\s*(?:file:(?://)?)?(?P<path>.+)$"
    )
    PEP621_NAME_RE: Final[re.Pattern[str]] = re.compile(
        "^\\s*(?P<name>[A-Za-z0-9_.-]+)"
    )
    PYTEST_STANDARD_MARKERS: tuple[str, ...] = (
        "unit: unit tests",
        "integration: integration tests",
        "performance: performance and benchmark tests",
        "slow: slow-running tests",
        "docker: tests requiring Docker",
        "e2e: end-to-end integration tests",
        "edge_cases: edge case tests",
        "stress: stress tests",
        "resilience: resilience tests",
    )
    PYTEST_STANDARD_ADDOPTS: tuple[str, ...] = ("--strict-markers",)
    MYPY_PLUGINS: tuple[str, ...] = ("pydantic.mypy",)
    MYPY_DISABLED_ERROR_CODES: tuple[str, ...] = ("prop-decorator",)
    MYPY_BOOLEAN_SETTINGS: tuple[tuple[str, bool], ...] = (
        ("ignore_missing_imports", True),
        ("namespace_packages", True),
        ("explicit_package_bases", True),
    )
    PYDANTIC_MYPY_SETTINGS: tuple[tuple[str, bool], ...] = (
        ("init_forbid_extra", True),
        ("init_typed", True),
        ("warn_required_dynamic_aliases", True),
    )
    PYRIGHT_STRICT_SETTINGS: tuple[tuple[str, str], ...] = (
        ("pythonVersion", "3.13"),
        ("pythonPlatform", "Linux"),
        ("typeCheckingMode", "strict"),
    )
    PYREFLY_STRICT_ERRORS: tuple[str, ...] = (
        "deprecated",
        "redundant-cast",
        "implicit-abstract-class",
        "implicit-any",
        "implicitly-defined-attribute",
        "missing-override-decorator",
        "missing-source",
        "not-required-key-access",
        "open-unpacking",
        "protocol-implicitly-defined-attribute",
        "unannotated-attribute",
        "unannotated-parameter",
        "unannotated-return",
    )
    PYREFLY_DISABLED_ERRORS: tuple[str, ...] = ("bad-override",)
    TOMLSORT_DEFAULTS: tuple[tuple[str, object], ...] = (
        ("all", True),
        ("in_place", True),
        ("sort_first", ["project", "build-system", "tool", "dependency-groups"]),
    )
    YAMLFIX_DEFAULTS: tuple[tuple[str, object], ...] = (
        ("line_length", 88),
        ("preserve_quotes", True),
        ("whitelines", 1),
        ("section_whitelines", 1),
        ("explicit_start", False),
    )
    BANNER: str = "# [MANAGED] FLEXT pyproject standardization\n# Sections with [MANAGED] are enforced by flext_infra.deps.modernizer.\n# Sections with [CUSTOM] are project-specific extension points.\n# Sections with [AUTO] are derived from workspace layout and dependencies.\n"
    COMMENT_MARKERS: tuple[tuple[str, str], ...] = (
        ("[build-system]", "# [MANAGED] build system"),
        ("[project]", "# [CUSTOM] project metadata"),
        ("[tool.poetry.group.dev.dependencies]", "# [CUSTOM] poetry dev extensions"),
        ("[tool.deptry]", "# [MANAGED] deptry"),
        ("[tool.ruff]", "# [MANAGED] ruff"),
        ("[tool.codespell]", "# [MANAGED] codespell"),
        ("[tool.tomlsort]", "# [MANAGED] tomlsort"),
        ("[tool.yamlfix]", "# [MANAGED] yamlfix"),
        ("[tool.pytest]", "# [MANAGED] pytest"),
        ("[tool.coverage]", "# [MANAGED] coverage"),
        ("[tool.mypy]", "# [MANAGED] mypy"),
        ("[tool.pydantic-mypy]", "# [MANAGED] pydantic-mypy"),
        ("[tool.pyrefly]", "# [MANAGED] pyrefly"),
        ("[tool.pyright]", "# [MANAGED] pyright"),
    )
    RUFF_SHARED_TEMPLATE: str = '# [MANAGED] FLEXT shared Ruff baseline\n# This file is generated by flext_infra.deps.modernizer when missing.\n# Projects may add per-project overrides in pyproject.toml [tool.ruff*].\n\nline-length = 88\nindent-width = 4\ntarget-version = "py313"\n\n[lint]\nselect = ["E", "F", "I", "UP", "B", "SIM", "C4", "PT", "RUF"]\nignore = ["E501"]\n\n[lint.isort]\ncombine-as-imports = true\nforce-single-line = false\nsplit-on-trailing-comma = true\n'
    MIN_ARGV: int = 2
    DEFAULT_MODULE_TO_TYPES_PACKAGE: Mapping[str, str] = {
        "yaml": "types-pyyaml",
        "ldap3": "types-ldap3",
        "redis": "types-redis",
        "requests": "types-requests",
        "setuptools": "types-setuptools",
        "toml": "types-toml",
        "dateutil": "types-python-dateutil",
        "psutil": "types-psutil",
        "psycopg2": "types-psycopg2",
        "protobuf": "types-protobuf",
        "pyyaml": "types-pyyaml",
        "decorator": "types-decorator",
        "jsonschema": "types-jsonschema",
        "openpyxl": "types-openpyxl",
        "xlrd": "types-xlrd",
    }
    "Default mapping from module name to ``types-*`` stub package."


__all__ = ["FlextInfraDepsConstants"]
