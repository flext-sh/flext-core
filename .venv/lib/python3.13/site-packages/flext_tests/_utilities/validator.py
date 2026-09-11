"""Validator utilities mixin for flext-tests."""

from __future__ import annotations

from collections.abc import Callable, MutableSequence
from pathlib import Path
from typing import ClassVar

import flext_tests.constants as tests_constants
import flext_tests.models as tests_models
import flext_tests.typings as tests_typings
from flext_tests import p, r


class FlextTestsValidatorUtilitiesMixin:
    """Validator utilities for architecture validation."""

    # NOTE (multi-agent): validator_rule/path_pattern_matches/layer_dict moved
    # here from c.Tests constants facet (declaration purity, mro-i6nq.11).
    @staticmethod
    def validator_rule(rule_id: str) -> tests_typings.t.StrPair:
        """Resolve one validator rule (severity, description) by rule id."""
        attr_name = "VALIDATOR_RULE_" + rule_id.replace("-", "_")
        rule: tests_typings.t.StrPair = getattr(tests_constants.c.Tests, attr_name)
        return rule

    @staticmethod
    def path_pattern_matches(value: str, pattern: str) -> bool:
        """Check whether one validator path pattern matches value."""
        compiled = tests_constants.c.Tests.VALIDATOR_APPROVED_PATH_REGEX_BY_PATTERN.get(
            pattern
        )
        return compiled.search(value) is not None if compiled else False

    @staticmethod
    def layer_dict() -> tests_typings.t.IntMapping:
        """Return the architecture layer hierarchy as a mapping."""
        constants = tests_constants.c.Tests
        return {
            "constants": constants.LAYER_CONSTANTS,
            "typings": constants.LAYER_TYPINGS,
            "protocols": constants.LAYER_PROTOCOLS,
            "settings": constants.LAYER_CONFIG,
            "runtime": constants.LAYER_RUNTIME,
            "exceptions": constants.LAYER_EXCEPTIONS,
            "result": constants.LAYER_RESULT,
            "loggings": constants.LAYER_LOGGINGS,
            "models": constants.LAYER_MODELS,
            "utilities": constants.LAYER_UTILITIES,
            "mixins": constants.LAYER_MIXINS,
            "container": constants.LAYER_CONTAINER,
            "service": constants.LAYER_SERVICE,
            "context": constants.LAYER_CONTEXT,
            "handlers": constants.LAYER_HANDLERS,
            "dispatcher": constants.LAYER_DISPATCHER,
            "registry": constants.LAYER_REGISTRY,
            "decorators": constants.LAYER_DECORATORS,
        }

    @staticmethod
    def create_violation(
        file_path: Path,
        line_number: int,
        rule_id: str,
        lines: tests_typings.t.StrSequence,
        extra_desc: str = "",
    ) -> tests_models.m.Tests.Violation:
        """Create a violation model using c.Tests.

        Args:
            file_path: Path to file with violation
            line_number: Line number of violation (1-indexed)
            rule_id: Rule identifier (e.g., "IMPORT-001")
            lines: File content as list of lines
            extra_desc: Optional extra description

        Returns:
            r[TEntity]: Result containing created entity or error
            Violation model instance

        """
        severity, desc = FlextTestsValidatorUtilitiesMixin.validator_rule(rule_id)
        description = f"{desc}: {extra_desc}" if extra_desc else desc
        line = lines[line_number - 1] if line_number <= len(lines) else ""
        return tests_models.m.Tests.Violation(
            file_path=file_path,
            line_number=line_number,
            rule_id=rule_id,
            severity=severity,
            description=description,
            code_snippet=line.strip(),
        )

    @staticmethod
    def find_line_number(lines: tests_typings.t.StrSequence, pattern: str) -> int:
        """Find line number containing pattern."""
        for i, line in enumerate(lines, start=1):
            if pattern in line:
                return i
        return 1

    @staticmethod
    def split_import_targets(value: str) -> tests_typings.t.StrSequence:
        """Normalize one import target list into canonical imported names."""
        cleaned = value.split("#", maxsplit=1)[0].replace("(", " ").replace(")", " ")
        targets: list[str] = []
        for raw_target in cleaned.split(","):
            target = raw_target.split(" as ", maxsplit=1)[0].strip()
            if target:
                targets.append(target)
        return tuple(targets)

    @staticmethod
    def approved(
        rule_id: str,
        file_path: Path,
        approved: tests_typings.t.MappingKV[str, tests_typings.t.StrSequence],
        extra_patterns: tests_typings.t.StrSequence = (),
    ) -> bool:
        """Check if file is approved for this rule.

        Args:
            rule_id: Rule identifier (e.g., "IMPORT-001")
            file_path: Path to file being checked
            approved: Dict mapping rule IDs to list of approved file patterns
            extra_patterns: Additional canonical patterns to honor for this scan

        Returns:
            r[TEntity]: Result containing created entity or error
            True if file matches any approved pattern for this rule

        """
        patterns = tuple(approved.get(rule_id, ())) + tuple(extra_patterns)
        file_str = str(file_path)
        return any(
            FlextTestsValidatorUtilitiesMixin.path_pattern_matches(file_str, pattern)
            for pattern in patterns
        )

    @staticmethod
    def code_match(line: str, pattern: tests_typings.t.Infra.RegexPattern) -> bool:
        """Check if one pattern match appears outside quoted string literals.

        Args:
            line: Source code line
            pattern: Compiled regex pattern to search

        Returns:
            r[TEntity]: Result containing created entity or error
            True if the first match is not inside a quoted string literal

        """
        match = pattern.search(line)
        if match is None:
            return False
        pos = match.start()
        in_single = False
        in_double = False
        in_triple_single = False
        in_triple_double = False
        i = 0
        while i < pos:
            if line[i : i + 3] == '"""' and (not in_single) and (not in_triple_single):
                in_triple_double = not in_triple_double
                i += 3
                continue
            if line[i : i + 3] == "'''" and (not in_double) and (not in_triple_double):
                in_triple_single = not in_triple_single
                i += 3
                continue
            if (
                line[i] == '"'
                and (not in_single)
                and (not in_triple_single)
                and (not in_triple_double)
            ):
                in_double = not in_double
            elif (
                line[i] == "'"
                and (not in_double)
                and (not in_triple_single)
                and (not in_triple_double)
            ):
                in_single = not in_single
            i += 1
        return not (in_single or in_double or in_triple_single or in_triple_double)

    @staticmethod
    def real_comment(line: str, pattern: tests_typings.t.Infra.RegexPattern) -> bool:
        """Check if pattern match is in a real comment, not inside a string.

        Used by validators to avoid false positives from patterns appearing
        in docstrings or string literals.

        Args:
            line: Source code line
            pattern: Compiled regex pattern to search

        Returns:
            r[TEntity]: Result containing created entity or error
            True if pattern appears in real code comment (after #),
            not inside a string literal (single/double/triple quoted)

        """
        return FlextTestsValidatorUtilitiesMixin.code_match(line, pattern)

    @staticmethod
    def except_block_only_pass(
        lines: tests_typings.t.StrSequence, line_number: int
    ) -> bool:
        """Check whether one ``except`` block body contains only pass or ellipsis."""
        header_index = line_number - 1
        if header_index < 0 or header_index >= len(lines):
            return False
        header_line = lines[header_index]
        header_match = tests_constants.c.Tests.VALIDATOR_EXCEPT_HEADER_RE.match(
            header_line
        )
        if header_match is None:
            return False
        trailing = header_line.rsplit(":", maxsplit=1)[-1].strip()
        if (
            trailing
            and tests_constants.c.Tests.VALIDATOR_PASS_OR_ELLIPSIS_RE.match(trailing)
            is not None
        ):
            return True
        header_indent = len(header_match.group("indent").expandtabs())
        body_lines: list[str] = []
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            current_indent = len(line.expandtabs()) - len(line.lstrip().expandtabs())
            if current_indent <= header_indent:
                break
            if stripped.startswith("#"):
                continue
            body_lines.append(stripped)
        return (
            len(body_lines) == 1
            and tests_constants.c.Tests.VALIDATOR_PASS_OR_ELLIPSIS_RE.match(
                body_lines[0]
            )
            is not None
        )

    # NOTE (multi-agent): scanner behavior moved here from _validator/models.py
    # (declaration purity - a models-named module must not hold scan behavior;
    # mro-i6nq.11). Scanners inherit ValidatorScannerMixin.
    @staticmethod
    def validator_run_scan(
        *,
        files: tests_typings.t.SequenceOf[Path],
        approved_exceptions: tests_typings.t.MappingKV[str, tests_typings.t.StrSequence]
        | None,
        validator_name: str,
        scan_file: Callable[
            [Path, tests_typings.t.MappingKV[str, tests_typings.t.StrSequence]],
            tests_typings.t.SequenceOf[tests_models.m.Tests.Violation],
        ],
    ) -> p.Result[tests_models.m.Tests.ScanResult]:
        """Run one validator scan across files and build a ScanResult."""
        violations: MutableSequence[tests_models.m.Tests.Violation] = []
        approved = approved_exceptions or {}
        for file_path in files:
            violations.extend(scan_file(file_path, approved))
        return r[tests_models.m.Tests.ScanResult].ok(
            tests_models.m.Tests.ScanResult(
                validator_name=validator_name,
                files_scanned=len(files),
                violations=violations,
            )
        )

    class ValidatorScannerMixin:
        """MRO mixin: validator classes inherit scan(...) for free.

        Each consumer declares _VALIDATOR_KEY and a _scan_file
        classmethod; scan delegates to validator_run_scan.
        """

        _VALIDATOR_KEY: ClassVar[str]

        @classmethod
        def _scan_file(
            cls,
            file_path: Path,
            approved: tests_typings.t.MappingKV[str, tests_typings.t.StrSequence],
        ) -> tests_typings.t.SequenceOf[tests_models.m.Tests.Violation]:
            """Subclass MUST override: scan one file and yield violations."""
            raise NotImplementedError

        @classmethod
        def scan(
            cls,
            files: tests_typings.t.SequenceOf[Path],
            approved_exceptions: tests_typings.t.MappingKV[
                str, tests_typings.t.StrSequence
            ]
            | None = None,
        ) -> p.Result[tests_models.m.Tests.ScanResult]:
            """Scan files for violations using the consumer's _scan_file."""
            return FlextTestsValidatorUtilitiesMixin.validator_run_scan(
                files=files,
                approved_exceptions=approved_exceptions,
                validator_name=cls._VALIDATOR_KEY,
                scan_file=cls._scan_file,
            )
