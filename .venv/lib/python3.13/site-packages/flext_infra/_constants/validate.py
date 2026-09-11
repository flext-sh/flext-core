"""Centralized constants for the core subpackage."""

from __future__ import annotations

import re
from enum import IntEnum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraConstantsSharedInfra:
    """Shared infrastructure constants consumed by flext_infra.constants."""

    @unique
    class ScriptExitCode(IntEnum):
        """Canonical exit codes for infra-owned validation scripts."""

        PASS = 0
        FAIL = 1
        USAGE = 2
        INFRA = 3

    EXEMPT_FILENAMES: Final[frozenset[str]] = frozenset({
        "__init__.py",
        "conftest.py",
        "__main__.py",
    })
    EXEMPT_PREFIXES: Final[frozenset[str]] = frozenset({"test_", "_"})
    FACADE_MODULE_DEPTH: Final[int] = 3
    "Relative path part count for root facade modules (src/<pkg>/<file>.py)."
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
    CLASSVAR_ANNOTATION_NAMES: Final[frozenset[str]] = frozenset({"ClassVar"})
    "Names treated as class-variable constant annotations."
    COLLECTION_CALLS: Final[frozenset[str]] = frozenset({
        "frozenset",
        "tuple",
        "dict",
        "list",
    })
    SKILLS_DIR: Final[Path] = Path(".agents/skills")
    BASELINE_DEFAULT: Final[str] = ".agents/skills/{skill}/baseline.json"
    SCRIPT_EXIT_CODE_VALUES: Final[frozenset[int]] = frozenset(
        int(item) for item in ScriptExitCode
    )
    SCRIPT_HEADER_MAX_LINES: Final[int] = 10
    SCRIPT_MIN_CODE_LINES: Final[int] = 20
    SKILL_REPORT_VALIDATED_TOP_DIRS: Final[frozenset[str]] = frozenset({"."})
    SKILL_REPORT_SKIPPED_TOP_DIRS: Final[frozenset[str]] = frozenset({
        "evidence",
        "plans",
        "drafts",
        "validation",
        "dependencies",
    })
    SKILL_REPORT_SKIPPED_FILES: Final[frozenset[str]] = frozenset({".gitkeep"})
    SKILL_OWNER_MARKER_RE: Final[t.RegexPattern] = re.compile(
        r"^# Owner-Skill:\s+(.agents/skills/([a-z0-9][-a-z0-9]*)/SKILL\.md)\s*$"
    )
    SKILL_REPORT_ARTIFACT_NAME_RE: Final[t.RegexPattern] = re.compile(
        r"^[a-z][-a-z0-9]*--[a-z]+--[a-z][-a-z0-9]*\.[a-z]+$"
    )
    SKILL_REPORT_ARTIFACT_SKILL_RE: Final[t.RegexPattern] = re.compile(
        r"^[a-z][-a-z0-9]*$"
    )
    SKILL_REPORT_ARTIFACT_SLUG_INVALID_RE: Final[t.RegexPattern] = re.compile(
        r"[^a-z0-9-]+"
    )
    SKILL_REPORT_ARTIFACT_MULTI_DASH_RE: Final[t.RegexPattern] = re.compile(r"-+")
    SKILL_REPORTS_PATH_RE: Final[t.RegexPattern] = re.compile(r"\.reports/([^\s\"']+)")
    SKILL_BASH_EXIT_RE: Final[t.RegexPattern] = re.compile(r"^\s*exit\s+(\d+)")
    SKILL_INTERACTIVE_PY_RE: Final[t.RegexPattern] = re.compile(r"\binput\s*\(")
    SKILL_INTERACTIVE_SH_RE: Final[t.RegexPattern] = re.compile(
        r"\bread\s+-p\b|\bselect\s+\w+\s+in\b|\bdialog\b|\bwhiptail\b"
    )
    SKILL_INTERACTIVE_GATE_RE: Final[t.RegexPattern] = re.compile(r"--interactive")
    SKILL_VALIDATOR_NAME_RE: Final[t.RegexPattern] = re.compile(
        r"^(enforce|check|validate|test|verify|audit|lint|scan)[-_]"
    )
    SKILL_FIXER_NAME_RE: Final[t.RegexPattern] = re.compile(
        r"^(fix|autofix|repair|correct|reorder|refactor|standardize)[-_]"
    )
    MISSING_IMPORT_RE: Final[t.RegexPattern] = re.compile(
        r"Cannot find module `([^`]+)` \[missing-import\]"
    )
    MYPY_HINT_RE: Final[t.RegexPattern] = re.compile(
        r'note:\s+(?:hint|note):\s+(?:["`].*?\bpip\s+install\s+|install\s+stub\s+package\s+["`]?)'
        r'([A-Za-z0-9][A-Za-z0-9_.-]*)["`]?',
        re.IGNORECASE,
    )
    MYPY_STUB_RE: Final[t.RegexPattern] = re.compile(
        r"Library stubs not installed for ['\"](\S+?)['\"]"
    )
    INTERNAL_PREFIXES: Final[tuple[str, ...]] = ("flext_", "flext-")
    METADATA_TOMLLIB_MODULES: Final[frozenset[str]] = frozenset({"tomllib"})
    METADATA_ALLOWLIST_PATH_MARKERS: Final[t.StrSequence] = (
        "flext-core/src/flext_core/_utilities/project_metadata.py",
        "flext-infra/src/flext_infra/_utilities/iteration.py",
        "flext-infra/src/flext_infra/__version__.py",
    )
    METADATA_TARGET_SCOPE_MARKERS: Final[t.StrSequence] = (
        "flext-infra/src/flext_infra/",
    )

    # --- Integration baseline discovery ---
    # Ordered preference used to derive one repository's integration baseline
    # from live Git. A provider default is a fallback ordering, never the
    # answer: repositories under the same provider legitimately integrate on
    # different branches, so the published remote-tracking branch decides.
    #
    # This is the built-in ordering of conventional names only. The governing
    # value is `codegen.branch_policy.integration_branch_preference`, which a
    # workspace declares for itself — a fleet that integrates on a versioned
    # line names it there rather than asking for a constant here. Product- and
    # release-specific names do not belong in this tuple.
    INTEGRATION_BRANCH_PREFERENCE: Final[tuple[str, ...]] = ("develop", "dev", "main")

    # --- File names (was: class Files) ---
    PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
    MAKEFILE_FILENAME: Final[str] = "Makefile"
    GITMODULES: Final[str] = ".gitmodules"
    # Why: conform .gitmodules merge classifies sections via these patterns;
    # they belong beside GITMODULES on c.Infra, not as leaf re.compile copies.
    GITMODULE_SECTION_RE: Final[t.RegexPattern] = re.compile(
        r'(?m)^\[submodule "[^"]+"\]\s*$'
    )
    "``.gitmodules`` submodule section header at line start."
    GITMODULE_PATH_RE: Final[t.RegexPattern] = re.compile(
        r"(?m)^[ \t]*path[ \t]*=[ \t]*(.+?)[ \t]*$"
    )
    "``.gitmodules`` path assignment value inside a submodule section."
    FOLLOW_SUPERPROJECT_BRANCH: Final[str] = "."
    GITIGNORE: Final[str] = ".gitignore"
    PRE_COMMIT_CONFIG_FILENAME: Final[str] = ".pre-commit-config.yaml"
    "Hook-config projection whose presence decides whether a checkout runs hooks."
    BEADS_CONFIG_RELPATH: Final[str] = ".beads/config.yaml"
    BEADS_METADATA_RELPATH: Final[str] = ".beads/metadata.json"
    BEADS_OVERRIDE_RELPATH: Final[str] = "config/beads.yaml"
    "Generated project-owned Beads configuration paths."
    GITIGNORE_DERIVED_SECTION_NAME: Final[str] = "Derived build and tool artifacts"
    "Heading of the trailing .gitignore section holding derived artifacts."
    GITIGNORE_MANAGED_SECTION_NAME: Final[str] = "Tracked managed artifacts"
    "Heading of the trailing .gitignore section that re-allows managed files."
    GITIGNORE_LAYOUT_SECTION_NAME: Final[str] = "Project layout exceptions"
    GITIGNORE_PROJECT_SECTION_NAME: Final[str] = (
        "Project ignore patterns (config/*.yaml ManagedArtifacts.Gitignore)"
    )
    "Heading of the trailing .gitignore section holding layout-SSOT additions."
    MANAGED_FILE_POLICY_DELEGATED: Final[str] = "delegated"
    MANAGED_FILE_POLICY_FULL: Final[str] = "full"
    "Managed-file policy whose artifact is generated per checkout, not committed."
    INIT_PY: Final[str] = "__init__.py"
    API_PY: Final[str] = "api.py"
    CONSTANTS_PY: Final[str] = "constants.py"
    MODELS_PY: Final[str] = "models.py"
    RESULT_PY: Final[str] = "result.py"
    UTILITIES_PY: Final[str] = "utilities.py"
    TYPINGS_PY: Final[str] = "typings.py"
    PROTOCOLS_PY: Final[str] = "protocols.py"
    PY_TYPED: Final[str] = "py.typed"

    # --- Git constants (was: class Git) ---
    GIT_DIR: Final[str] = ".git"
    GIT_ORIGIN: Final[str] = "origin"
    GIT_MAIN: Final[str] = "main"
    GIT_HEAD: Final[str] = "HEAD"
    GIT_MERGE_HEAD: Final[str] = "MERGE_HEAD"

    # --- Package name prefixes (was: class Packages) ---
    PKG_CORE: Final[str] = "flext-core"
    PKG_CORE_UNDERSCORE: Final[str] = "flext_core"
    PKG_TESTS_UNDERSCORE: Final[str] = "flext_tests"
    PKG_INFRA_UNDERSCORE: Final[str] = "flext_infra"
    PKG_ROOT: Final[str] = "flext"
    PKG_PREFIX_HYPHEN: Final[str] = "flext-"
    PKG_PREFIX_UNDERSCORE: Final[str] = "flext_"

    # --- Dunder names (was: class Dunders) ---
    DUNDER_ALL: Final[str] = "__all__"
    DUNDER_VERSION: Final[str] = "__version__"
    DUNDER_INIT: Final[str] = "__init__"
    DUNDER_PYCACHE: Final[str] = "__pycache__"

    # --- File extensions (was: class Extensions) ---
    EXT_PYTHON: Final[str] = ".py"
    EXT_PYTHON_GLOB: Final[str] = "*.py"

    # --- Directory names (was: class Directories) ---
    DIR_TESTS: Final[str] = "tests"
    DIR_EXAMPLES: Final[str] = "examples"
    DIR_SCRIPTS: Final[str] = "scripts"
    TIER_WHITELIST_NON_RUNTIME_DIR_PARTS: Final[frozenset[str]] = frozenset({
        DIR_TESTS,
        DIR_EXAMPLES,
        DIR_SCRIPTS,
    })
    TIER_WHITELIST_SETTINGS_MODULE_LIBRARIES: Final[frozenset[str]] = frozenset({
        "pydantic_settings"
    })
    DIR_TYPINGS: Final[str] = "typings"
    DIR_DOCS: Final[str] = "docs"
    DIR_BUILD: Final[str] = "build"
    DIR_SITE: Final[str] = "site"

    # --- Timeout values in seconds (was: class Timeouts) ---
    TIMEOUT_DEFAULT: Final[int] = 300
    TIMEOUT_SHORT: Final[int] = 60
    TIMEOUT_SHORT_POLL: Final[int] = 2
    "Bounded wait proving a child is still blocked on a held owner lock."
    TIMEOUT_MEDIUM: Final[int] = 120
    TIMEOUT_LONG: Final[int] = 600
    TIMEOUT_CI: Final[int] = 900

    # --- Path constants (was: class Paths) ---
    VENV_BIN_REL: Final[str] = ".venv/bin"
    DEFAULT_SRC_DIR: Final[str] = "src"


__all__: list[str] = ["FlextInfraConstantsSharedInfra"]
