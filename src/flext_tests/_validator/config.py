"""Config validation for FLEXT architecture.

Detects pyproject.toml violations: mypy ignore_errors, weakened type settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from flext_core.result import r
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.utilities import u


class FlextValidatorConfig:
    """Config validation methods for FlextTestsValidator.

    Uses c.Tests.Validator for constants and m.Tests.Validator for models.
    """

    @classmethod
    def scan(
        cls,
        files: list[Path],
        approved_exceptions: dict[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Scan pyproject.toml files for config violations.

        Args:
            files: List of pyproject.toml files to scan
            approved_exceptions: Dict mapping rule IDs to list of approved file patterns

        Returns:
            FlextResult with ScanResult containing all violations found

        """
        violations: list[m.Tests.Validator.Violation] = []
        approved = approved_exceptions or {}

        for file_path in files:
            # Only scan pyproject.toml files
            if file_path.name != "pyproject.toml":
                continue
            file_violations = cls._scan_file(file_path, approved)
            violations.extend(file_violations)

        return r[m.Tests.Validator.ScanResult].ok(
            m.Tests.Validator.ScanResult.create(
                validator_name=c.Tests.Validator.Defaults.VALIDATOR_CONFIG,
                files_scanned=len(files),
                violations=violations,
            ),
        )

    @classmethod
    def validate(
        cls,
        pyproject_path: Path,
        approved_exceptions: dict[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Validate a single pyproject.toml file.

        Args:
            pyproject_path: Path to pyproject.toml file
            approved_exceptions: Dict mapping rule IDs to list of approved file patterns

        Returns:
            FlextResult with ScanResult containing all violations found

        """
        return cls.scan([pyproject_path], approved_exceptions)

    @classmethod
    def _scan_file(
        cls,
        file_path: Path,
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Scan a single pyproject.toml for config violations."""
        violations: list[m.Tests.Validator.Violation] = []

        try:
            content = file_path.read_text(encoding="utf-8")
            data = tomllib.loads(content)
        except (OSError, tomllib.TOMLDecodeError):
            return violations

        lines = content.splitlines()

        # Check mypy settings
        violations.extend(cls._check_mypy_settings(file_path, data, lines, approved))

        # Check ruff settings
        violations.extend(cls._check_ruff_settings(file_path, data, lines, approved))

        # Check pyright settings
        violations.extend(cls._check_pyright_settings(file_path, data, lines, approved))

        return violations

    @classmethod
    def _create_config_violation(
        cls,
        file_path: Path,
        line_number: int,
        rule_id: str,
        code_snippet: str,
        extra_desc: str = "",
    ) -> m.Tests.Validator.Violation:
        """Create a config violation (config files have no lines list)."""
        severity, desc = c.Tests.Validator.Rules.get(rule_id)
        description = f"{desc}: {extra_desc}" if extra_desc else desc
        return m.Tests.Validator.Violation(
            file_path=file_path,
            line_number=line_number,
            rule_id=rule_id,
            severity=severity,
            description=description,
            code_snippet=code_snippet,
        )

    @classmethod
    def _check_mypy_settings(
        cls,
        file_path: Path,
        data: dict[str, object],
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Check mypy configuration for violations."""
        violations: list[m.Tests.Validator.Violation] = []

        # Type annotations for .get() results to help pyright inference
        tool_data_raw: object = data.get("tool", {})
        if not isinstance(tool_data_raw, dict):
            return violations
        tool_data: dict[str, object] = tool_data_raw
        mypy_config_raw: object = tool_data.get("mypy", {})
        if not isinstance(mypy_config_raw, dict):
            return violations
        mypy_config: dict[str, object] = mypy_config_raw

        # Check global ignore_errors
        if (
            not u.Tests.Validator.is_approved("CONFIG-001", file_path, approved)
            and mypy_config.get("ignore_errors") is True
        ):
            line_num = u.Tests.Validator.find_line_number(lines, "ignore_errors")
            violations.append(
                cls._create_config_violation(
                    file_path,
                    line_num,
                    "CONFIG-001",
                    "ignore_errors = true",
                    "(global)",
                ),
            )

        # Check per-module overrides
        # Type annotations for .get() results to help pyright inference
        overrides_raw: object = mypy_config.get("overrides", [])
        if not isinstance(overrides_raw, list):
            return violations
        overrides: list[object] = overrides_raw
        for override in overrides:
            if not isinstance(override, dict):
                continue
            # Type narrowing: override is dict[str, object]
            override_dict: dict[str, object] = override
            module_raw = override_dict.get("module", "unknown")
            module: str = str(module_raw) if module_raw is not None else "unknown"
            is_approved = u.Tests.Validator.is_approved(
                "CONFIG-001",
                file_path,
                approved,
            )
            ignore_errors_raw = override_dict.get("ignore_errors")
            if ignore_errors_raw is True and not is_approved:
                line_num = u.Tests.Validator.find_line_number(
                    lines,
                    f'module = "{module}"',
                )
                violations.append(
                    cls._create_config_violation(
                        file_path,
                        line_num,
                        "CONFIG-001",
                        f"ignore_errors = true (module: {module})",
                        c.Tests.Validator.Messages.CONFIG_IGNORE.format(
                            module=module,
                        ),
                    ),
                )

        # Check disallow_incomplete_defs
        if (
            not u.Tests.Validator.is_approved("CONFIG-003", file_path, approved)
            and mypy_config.get("disallow_incomplete_defs") is False
        ):
            line_num = u.Tests.Validator.find_line_number(
                lines,
                "disallow_incomplete_defs",
            )
            violations.append(
                cls._create_config_violation(
                    file_path,
                    line_num,
                    "CONFIG-003",
                    "disallow_incomplete_defs = false",
                ),
            )

        # Check warn_return_any
        if (
            not u.Tests.Validator.is_approved("CONFIG-004", file_path, approved)
            and mypy_config.get("warn_return_any") is False
        ):
            line_num = u.Tests.Validator.find_line_number(lines, "warn_return_any")
            violations.append(
                cls._create_config_violation(
                    file_path,
                    line_num,
                    "CONFIG-004",
                    "warn_return_any = false",
                ),
            )

        return violations

    @classmethod
    def _check_ruff_settings(
        cls,
        file_path: Path,
        data: dict[str, object],
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Check ruff configuration for violations."""
        if u.Tests.Validator.is_approved("CONFIG-002", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        tool_data = data.get("tool", {})
        if not isinstance(tool_data, dict):
            return violations
        # Type annotations for .get() results to help pyright inference
        # tool_data.get() returns object, need type narrowing
        ruff_config_raw = tool_data.get("ruff", {})
        # Type narrowing: isinstance check ensures ruff_config_raw is dict
        if not isinstance(ruff_config_raw, dict):
            return violations
        # Type narrowing: ruff_config_raw is dict after isinstance check
        # Use cast to help pyright infer type (dict[str, object] is compatible)
        ruff_config: dict[str, object] = cast("dict[str, object]", ruff_config_raw)  # type: ignore[assignment]
        lint_config_raw = ruff_config.get("lint", {})
        if not isinstance(lint_config_raw, dict):
            return violations
        # Type narrowing: lint_config_raw is dict after isinstance check
        lint_config: dict[str, object] = cast("dict[str, object]", lint_config_raw)

        # Check for custom ignores beyond approved list
        ignores_raw = lint_config.get("ignore", [])
        if isinstance(ignores_raw, list):
            # Type narrowing: ignores_raw is list[object], need to check each item
            approved_ignores = c.Tests.Validator.Approved.RUFF_IGNORES
            # Type narrowing: ignores_raw is list after isinstance check
            ignores_list: list[object] = cast("list[object]", ignores_raw)
            for ignore_raw in ignores_list:
                # Type narrowing: ignore_raw is object, convert to str for comparison
                ignore_str: str = str(ignore_raw)
                if ignore_str not in approved_ignores:
                    line_num = u.Tests.Validator.find_line_number(lines, ignore_str)
                    violations.append(
                        cls._create_config_violation(
                            file_path,
                            line_num,
                            "CONFIG-002",
                            f'"{ignore_str}"',
                            c.Tests.Validator.Messages.CONFIG_RUFF.format(
                                code=ignore_str
                            ),
                        ),
                    )

        return violations

    @classmethod
    def _check_pyright_settings(
        cls,
        file_path: Path,
        data: dict[str, object],
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Check pyright configuration for violations."""
        violations: list[m.Tests.Validator.Violation] = []

        tool_data = data.get("tool", {})
        if not isinstance(tool_data, dict):
            return violations
        # Type narrowing: tool_data.get() returns object, need to check type
        pyright_config_raw = tool_data.get("pyright", {})
        # Type narrowing: isinstance check ensures pyright_config_raw is dict
        if not isinstance(pyright_config_raw, dict):
            return violations
        # Type narrowing: pyright_config_raw is dict after isinstance check
        # Use cast to help pyright infer type (dict[str, object] is compatible)
        pyright_config: dict[str, object] = cast(
            "dict[str, object]", pyright_config_raw
        )  # type: ignore[assignment]

        # Check reportPrivateUsage
        if (
            not u.Tests.Validator.is_approved("CONFIG-005", file_path, approved)
            and pyright_config.get("reportPrivateUsage") is False
        ):
            line_num = u.Tests.Validator.find_line_number(lines, "reportPrivateUsage")
            violations.append(
                cls._create_config_violation(
                    file_path,
                    line_num,
                    "CONFIG-005",
                    "reportPrivateUsage = false",
                ),
            )

        return violations


__all__ = ["FlextValidatorConfig"]
