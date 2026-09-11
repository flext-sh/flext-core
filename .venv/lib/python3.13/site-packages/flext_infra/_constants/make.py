"""Make-related constants for flext-infra project.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraConstantsMake:
    """Make-related constants for Makefile generation and CLI routing."""

    # Why: conform Makefile policy classifies declarations via these patterns;
    # they belong on c.Infra, not as leaf module re.compile copies.
    MAKE_ASSIGNMENT_RE: Final[t.RegexPattern] = re.compile(
        r"^[A-Za-z_][A-Za-z0-9_]*\s*(?::?:|\?|\+)?="
    )
    "GNU Make variable assignment at column 0 (``=``, ``:=``, ``::=``, ``?=``, ``+=``)."
    MAKE_DIRECTIVE_RE: Final[t.RegexPattern] = re.compile(
        r"^(?:export|unexport|override|include|-include|sinclude|vpath)\b"
    )
    "GNU Make directives that scope or include a declaration rather than define a target."
    MAKE_CONDITIONAL_RE: Final[t.RegexPattern] = re.compile(
        r"^(?:else\b|endif\b|ifeq\b|ifneq\b|ifdef\b|ifndef\b)"
    )
    "GNU Make conditional control flow; structural, never a target declaration."

    VERB_CHECK: Final[str] = "check"
    VERB_CLEAN: Final[str] = "clean"
    VERB_VALIDATE: Final[str] = "validate"
    VERB_PUBLISH: Final[str] = "publish"
    VERB_RUN: Final[str] = "run"
    VERB_CHECKS: Final[str] = "checks"

    # --- Canonical make contract constants (was: class Make) ---

    CLI_GROUP_CHECK: Final[str] = "check"
    CLI_GROUP_CODEGEN: Final[str] = "codegen"
    CLI_GROUP_DEPS: Final[str] = "deps"
    CLI_GROUP_DOCS: Final[str] = "docs"
    CLI_GROUP_GITHUB: Final[str] = "github"
    CLI_GROUP_MAINTENANCE: Final[str] = "maintenance"
    CLI_GROUP_REFACTOR: Final[str] = "refactor"
    CLI_GROUP_RELEASE: Final[str] = "release"
    CLI_ROUTE_RELEASE: Final[str] = "release run"
    CLI_GROUP_VALIDATE: Final[str] = "validate"
    CLI_ROUTE_MAINTENANCE: Final[str] = "maintenance run"
    CLI_GROUP_WORKSPACE: Final[str] = "workspace"
    CLI_GROUPS_TRANSLATING_WHAT: Final[frozenset[str]] = frozenset({
        CLI_GROUP_CHECK,
        CLI_GROUP_VALIDATE,
        CLI_GROUP_CODEGEN,
    })
    "Groups whose --what maps onto a selector instead of a subcommand option."
    MYPY_MEMORY_LIMIT_MB_ENV: Final[str] = "MYPY_MEMORY_LIMIT_MB"
    MYPY_MEMORY_LIMIT_MB_DEFAULT: Final[int] = 6144
    MYPY_TIMEOUT_SECONDS_ENV: Final[str] = "MYPY_TIMEOUT_SECONDS"
    MYPY_TIMEOUT_SECONDS_DEFAULT: Final[int] = 600
    MYPY_TIMEOUT_GRACE_SECONDS: Final[int] = 10
    PRLIMIT_COMMAND: Final[str] = "prlimit"
    PRLIMIT_ADDRESS_SPACE_OPTION: Final[str] = "--as"
    TIMEOUT_COMMAND: Final[str] = "timeout"
    TIMEOUT_KILL_AFTER_SECONDS: Final[int] = 5
    CHECK_GATES_VARIABLE: Final[str] = "CHECK_GATES"
    "Make variable carrying the gate selection."
    # The BUILT-IN check vocabulary: read-only gates this package implements.
    # It is the BASE of the vocabulary, never the whole of it -- a project
    # declares its own gates in config and they are unioned in by
    # MakeSpec.check_gates_allowed. `format` is NOT here: it rewrites files, so
    # it is owned by `make fmt APPLY=Y` / `make fix APPLY=Y`
    # (PROJECT_CHECK_GATES_FIXABLE_VALUES) and a read-only verb must never
    # invoke it.
    PROJECT_CHECK_GATES_ALLOWED_VALUES: Final[tuple[str, ...]] = (
        "lint",
        "pyrefly",
        "mypy",
        "pyright",
        "security",
        "markdown",
        "smells",
        "direnv",
        "duplication",
    )
    # The gates CI=N owns: the type checkers only. They are the slow, whole-
    # program analyses, so CI=Y runs the strict complement of this set -- ruff
    # lint included -- and the two contexts can never overlap nor leave a gate
    # unowned. An unset CI runs every allowed gate.
    PROJECT_CHECK_GATES_LOCAL_VALUES: Final[tuple[str, ...]] = ("pyrefly", "mypy")
    PROJECT_CHECK_GATES_DEFAULT_VALUES: Final[tuple[str, ...]] = (
        PROJECT_CHECK_GATES_ALLOWED_VALUES
    )
    # flext-38p39: the gates that can repair what they report. `make fix APPLY=Y`
    # routes through `check run --fix`, which without a selector would execute
    # every gate -- including pyright and mypy, which fix nothing and cost ~37s,
    # timing the verb out. Formatting is NOT here: `format` belongs to
    # `make fmt` alone -- fix repairs findings, fmt rewrites style.
    # `format` is deliberately absent even though its gate reports can_fix:
    # verbs own tools by intent. `fmt` owns formatting, `fix` repairs findings,
    # `check` is read-only -- so `format` appears in NO check vocabulary,
    # neither ALLOWED nor FIXABLE.
    # `lint` is absent for the same ownership reason: ruff mutation belongs to
    # fmt/fix's ruff stage, and this set must stay DISJOINT from the CI=N gates
    # (lint, pyrefly) so `make fix` under the local token resolves to a
    # documented no-op instead of colliding with the pre-push gate set.
    PROJECT_CHECK_GATES_FIXABLE_VALUES: Final[tuple[str, ...]] = ("markdown", "smells")
    PROJECT_CHECK_GATES_ALLOWED: Final[str] = ",".join(
        PROJECT_CHECK_GATES_ALLOWED_VALUES
    )
    PROJECT_CHECK_GATES_DEFAULT: Final[str] = ",".join(
        PROJECT_CHECK_GATES_DEFAULT_VALUES
    )
    PROJECT_VALIDATE_GATES_ALLOWED: Final[str] = "complexity,docstring"
    ORCHESTRATED_PROJECT_VERBS: Final[t.StrSequence] = (
        "build",
        "check",
        "clean",
        "docs",
        "fmt",
        "fix",
        "scan",
        "test",
        "val",
    )
    ORCHESTRATOR_REMOVE_ENV_KEYS: Final[t.StrSequence] = (
        "GNUMAKEFLAGS",
        "MAKEFLAGS",
        "MAKEFILES",
        "MAKELEVEL",
        "MAKEOVERRIDES",
        "MISE_DIR",
        "MISE_EXE",
        "MISE_ORIG_PATH",
        "MISE_SESSION",
        "MISE_SHELL",
        "MISE_SHIMS",
        "MISE_VERBOSE",
        "MFLAGS",
        "MYPYPATH",
        "PROJECT",
        "PROJECTS",
        "PYTHONPATH",
        "UV_PROJECT",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
        # flext-zykgh: WHAT is a dispatch selector owned by each Makefile, exactly
        # like PROJECT/PROJECTS above. The root translates `WHAT=<gate>` into
        # `--make-arg CHECK_GATES=<gate>` before delegating, but the inherited
        # WHAT then overrode that intent in the member, which dispatched
        # `_builtin_check_<gate>` — a target the standalone profile never emits
        # because it selects gates through CHECK_GATES. The member failed with
        # "No rule to make target" in ~34ms while reporting "0 errors", a
        # missing-target failure disguised as a clean gate run.
        "WHAT",
    )
    "Environment keys removed before project-level make orchestration."
    ORCHESTRATOR_ENV_NO_COLOR: Final[str] = "NO_COLOR"
    ORCHESTRATOR_ENV_PATH: Final[str] = "PATH"
    ORCHESTRATOR_ENV_PYTHONPATH: Final[str] = "PYTHONPATH"
    ORCHESTRATOR_ENV_PYTHONDONTWRITEBYTECODE: Final[str] = "PYTHONDONTWRITEBYTECODE"
    ORCHESTRATOR_ENV_PATH_SEPARATOR: Final[str] = ":"
    ORCHESTRATOR_ENV_MISE_SHIMS: Final[str] = "MISE_SHIMS"
    ORCHESTRATOR_ENV_WORKSPACE_MISE_SHIMS: Final[str] = "WORKSPACE_MISE_SHIMS"
    PYTEST_ENV_ARGS: Final[str] = "FLEXT_PYTEST_ARGS_RAW"
    PYTEST_ENV_DIAG: Final[str] = "FLEXT_PYTEST_DIAG_RAW"
    PYTEST_ENV_FAIL_FAST: Final[str] = "FLEXT_PYTEST_FAIL_FAST_RAW"
    PYTEST_ENV_FILE: Final[str] = "FLEXT_PYTEST_FILE_RAW"
    PYTEST_ENV_FILES: Final[str] = "FLEXT_PYTEST_FILES_RAW"
    PYTEST_ENV_MATCH: Final[str] = "FLEXT_PYTEST_MATCH_RAW"
    PYTEST_ENV_REPORTS: Final[str] = "FLEXT_PYTEST_REPORTS_RAW"
    PYTEST_ENV_TARGET: Final[str] = "FLEXT_PYTEST_TARGET_RAW"
    PYTEST_ENV_VERBOSE: Final[str] = "FLEXT_PYTEST_VERBOSE_RAW"
    PYTEST_ENV_WHAT: Final[str] = "FLEXT_PYTEST_WHAT_RAW"
    PYTEST_ENV_COV: Final[str] = "FLEXT_PYTEST_COV_RAW"
    PYTEST_ENV_PROFILE: Final[str] = "FLEXT_PYTEST_PROFILE_RAW"
    PYTEST_ENV_CI: Final[str] = "CI"
    PYTEST_INHERITED_ENV_REMOVE_KEYS: Final[t.StrSequence] = (
        "PYTEST_ADDOPTS",
        "PYTHONPATH",
    )
    PROJECT_VARIABLE_DEFAULTS: Final[t.StrPairSequence] = (
        ("PYTEST_ARGS", ""),
        ("DEPENDENCY", ""),
        ("DIAG", "0"),
        (CHECK_GATES_VARIABLE, ""),
        ("VALIDATE_GATES", ""),
        ("SCOPE", "project"),
        ("NAMESPACE", ""),
        ("GATES", ""),
        ("PROPAGATE", ""),
        ("FIX", ""),
        ("FILE", ""),
        ("FILES", ""),
        ("CHANGED_ONLY", ""),
        ("MATCH", ""),
        ("COV", ""),
        ("RUFF_ARGS", ""),
        ("PYRIGHT_ARGS", ""),
        ("CHECK_ONLY", ""),
        ("FAIL_FAST", ""),
        ("VERBOSE", ""),
    )
    WORKSPACE_VARIABLE_DEFAULTS: Final[t.StrPairSequence] = (
        ("PROJECT", ""),
        ("PROJECTS", ""),
        ("WHAT", ""),
        ("PYTEST_ARGS", ""),
        ("DEPENDENCY", ""),
        ("VALIDATE_SCOPE", "all"),
        ("FAIL_FAST", ""),
        ("JOBS", ""),
        (CHECK_GATES_VARIABLE, ""),
        ("MYPY_MEMORY_LIMIT_MB", str(MYPY_MEMORY_LIMIT_MB_DEFAULT)),
        ("MYPY_TIMEOUT_SECONDS", str(MYPY_TIMEOUT_SECONDS_DEFAULT)),
        ("VALIDATE_GATES", ""),
        ("SCOPE", "project"),
        ("NAMESPACE", ""),
        ("GATES", ""),
        ("PROPAGATE", ""),
        ("FIX", ""),
        ("FILE", ""),
        ("FILES", ""),
        ("CHANGED_ONLY", ""),
        ("MATCH", ""),
        ("COV", ""),
        ("RUFF_ARGS", ""),
        ("PYRIGHT_ARGS", ""),
        ("CHECK_ONLY", ""),
        ("DRY_RUN", ""),
        ("MESSAGE", ""),
        ("PR_TITLE", ""),
        ("INDEX", ""),
        ("DEPS_REPORT", "1"),
        ("VERBOSE", ""),
    )
    PROJECT_CORE_VERBS: Final[t.StrPairSequence] = (
        ("boot", "Install dependencies and hooks"),
        ("build", "Build distributable artifacts"),
        ("check", "Run lint gates (CHECK_GATES= to select)"),
        (
            "fix-enforcement",
            "Auto-fix enforcement violations (APPLY=1, PROJECTS=..., RULES=...)",
        ),
        ("scan", "Run all security checks"),
        ("fmt", "Run all formatting"),
        ("docs", "Run docs (WHAT= to select)"),
        ("test", "Run bounded pytest (FILE=/MATCH= selectors)"),
        ("val", "Run validate gates (FIX=1 to auto-fix)"),
        ("clean", "Clean build/test/type artifacts"),
    )
    PROJECT_DAEMON_VERBS: Final[t.StrPairSequence] = (
        ("daemon-start", "Start all daemons (mypy + pyright)"),
        ("daemon-stop", "Stop all daemons"),
        ("daemon-status", "Show status of all daemons"),
        ("daemon-restart", "Restart all daemons"),
    )
    PROJECT_OPTION_LINES: Final[t.StrSequence] = (
        f"CHECK_GATES={PROJECT_CHECK_GATES_ALLOWED}",
        f"MYPY_MEMORY_LIMIT_MB={MYPY_MEMORY_LIMIT_MB_DEFAULT}  Mypy address-space cap",
        f"MYPY_TIMEOUT_SECONDS={MYPY_TIMEOUT_SECONDS_DEFAULT}  Mypy wall-time cap",
        f"VALIDATE_GATES={PROJECT_VALIDATE_GATES_ALLOWED}",
        "FILE=src/foo.py             Single file for check/fmt/test",
        'FILES="a.py b.py"          Multiple files for check/fmt; test rejects it',
        "CHANGED_ONLY=1              Git-changed Python files for check",
        "CHECK_ONLY=1                Dry-run format/check (no writes)",
        'RUFF_ARGS="--select E501"   Extra args for ruff check',
        'PYRIGHT_ARGS="--level basic" Extra args for pyright',
        "PYTEST_ARGS=<value>         Rejected; use FILE, MATCH, or WHAT",
        "DEPENDENCY=<distribution>   Select one package for deps WHAT=upgrade",
        "MATCH=test_name             Alias for pytest -k",
        "FAIL_FAST=1                 Add -x to pytest",
        "DIAG=1                      Emit extended pytest diagnostics",
        "FIX=1                       Auto-fix supported gates",
        "APPLY=1                     Apply enforcement fixes (default dry-run)",
        "PROJECTS=p1,p2              Scope fix-enforcement to projects",
        "RULES=ENFORCE-XXX,...       Scope fix-enforcement to rules",
        "VERBOSE=1                   Show executed commands",
    )
    # Phase-set per verb for legacy CLI helpers. Make routing is owned by
    # the registry discovered from scripts/cmd through flext-tests.
    WHAT_PHASES: Final[t.MappingKV[str, frozenset[str]]] = MappingProxyType({
        "boot": frozenset({"imp", "stat", "submodules", "sync", "venv"}),
        "build": frozenset({
            "constraints",
            "docs",
            "gen",
            "mod",
            "stubs",
            "sync",
            "up",
        }),
        "check": frozenset({
            "boundary",
            "coordination",
            "cqrs",
            "fmt",
            "format",
            "lint",
            "loc-cap",
            "markdown",
            "mypy",
            "pol",
            "pyre",
            "pyrefly",
            "pyright",
            "scan",
            "silent-failure",
            "types",
        }),
        "ship": frozenset({"pr", "push", "rel", "save", "tag"}),
        "test": frozenset({"all"}),
        "val": frozenset({"all", "project", "workspace"}),
    })
    STANDALONE_BOOTSTRAP_VERBS: Final[t.StrPairSequence] = (
        ("venv", "Create virtual environment"),
        ("setup", "Full standalone setup"),
        ("help", "Show this help"),
    )
    STANDALONE_POST_SETUP_VERBS: Final[str] = (
        "check, test, fmt, build, val, clean, docs"
    )
    PROJECT_SELECTION_CONFLICT_ERROR: Final[str] = (
        "ERROR: Cannot use PROJECT and PROJECTS together"
    )
    PROJECT_SELECTION_CONFLICT_HINT: Final[str] = (
        'Use PROJECT=<name> or PROJECTS="proj-a proj-b"'
    )
    PROJECT_SELECTION_EMPTY_ERROR: Final[str] = "ERROR: no projects selected"
    PROJECT_SELECTION_EMPTY_HINT: Final[str] = (
        'Use PROJECT=<name> or PROJECTS="proj-a proj-b"'
    )
    WORKSPACE_BOOT_HINT: Final[str] = "make boot"
    SAVE_USAGE: Final[str] = "make save MESSAGE='chore: your message'"
    FORWARD_MODE_VALUE: Final[str] = "value"
    FORWARD_MODE_ENABLED: Final[str] = "enabled"
    CHECK_FORWARD_ARGS: Final[t.StrPairSequence] = (
        ("CHECK_GATES", FORWARD_MODE_VALUE),
        ("FILE", FORWARD_MODE_VALUE),
        ("FILES", FORWARD_MODE_VALUE),
        ("CHANGED_ONLY", FORWARD_MODE_ENABLED),
        ("FIX", FORWARD_MODE_ENABLED),
        ("RUFF_ARGS", FORWARD_MODE_VALUE),
        ("PYRIGHT_ARGS", FORWARD_MODE_VALUE),
        ("CHECK_ONLY", FORWARD_MODE_ENABLED),
    )
    TEST_FORWARD_ARGS: Final[t.StrPairSequence] = (
        ("FILE", FORWARD_MODE_VALUE),
        ("MATCH", FORWARD_MODE_VALUE),
        ("VERBOSE", FORWARD_MODE_ENABLED),
    )
    VALIDATE_FORWARD_ARGS: Final[t.StrPairSequence] = (
        ("FIX", FORWARD_MODE_ENABLED),
        ("VALIDATE_GATES", FORWARD_MODE_VALUE),
    )


__all__: list[str] = ["FlextInfraConstantsMake"]
