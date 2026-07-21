"""Behavioral contract tests for the flext-tests architecture validator (tv).

These tests assert the OBSERVABLE PUBLIC CONTRACT of the ``tv`` validator facade
and the ``m.Tests.ScanResult`` / ``m.Tests.Violation`` models: the ``r[T]``
outcome of each validation verb, the ``passed <=> no violations`` invariant,
the public shape of every emitted violation, aggregation semantics of
``tv.all``, model-factory invariants, and idempotence. Nothing here reaches
into private attributes or internal collaborators.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_tests import m, tm, tv
from tests import c


class TestsFlextCoreArchitecture:
    """Public-behavior tests for the architecture validator facade."""

    @staticmethod
    def _core_package() -> Path:
        repo_root = Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
        return repo_root / c.Tests.SRC_DIR / c.Tests.CORE_PACKAGE_DIR

    @staticmethod
    def _pyproject() -> Path:
        repo_root = Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
        return repo_root / c.Tests.PYPROJECT_FILENAME

    def test_each_validator_verb_succeeds_and_self_identifies(self) -> None:
        """Each validator returns a scan named for its invoked public verb."""
        for verb in c.Tests.VALIDATOR_METHODS:
            scan = tm.ok(getattr(tv, verb)(self._core_package()))
            tm.that(scan.validator_name, eq=verb)
            tm.that(scan.files_scanned, is_=int, gte=1)

    @pytest.mark.parametrize("verb", c.Tests.VALIDATOR_METHODS)
    def test_passed_flag_is_true_exactly_when_no_violations(self, verb: str) -> None:
        """A validator passes exactly when its scan emits no violations."""
        scan = tm.ok(getattr(tv, verb)(self._core_package()))
        tm.that(scan.passed, eq=not scan.violations)

    @pytest.mark.parametrize("verb", c.Tests.VALIDATOR_METHODS)
    def test_every_emitted_violation_exposes_valid_public_fields(
        self, verb: str
    ) -> None:
        """Every violation exposes severity, location, identity, and description."""
        scan = tm.ok(getattr(tv, verb)(self._core_package()))
        valid_severities = frozenset(c.Tests.ValidatorSeverity)
        for violation in scan.violations:
            tm.that(violation.severity in valid_severities, eq=True)
            tm.that(violation.line_number, is_=int, gte=1)
            tm.that(bool(violation.rule_id), eq=True)
            tm.that(bool(violation.description), eq=True)
            tm.that(violation.file_path.suffix, eq=".py")

    def test_all_aggregates_union_of_individual_validators(self) -> None:
        """The all verb aggregates individual findings and maximum scan count."""
        package = self._core_package()
        individual = [
            tm.ok(getattr(tv, verb)(package)) for verb in c.Tests.VALIDATOR_METHODS
        ]
        aggregate = tm.ok(tv.all(package))

        tm.that(aggregate.validator_name, eq="all")
        tm.that(
            len(aggregate.violations),
            eq=sum(len(scan.violations) for scan in individual),
        )
        tm.that(
            aggregate.files_scanned, eq=max(scan.files_scanned for scan in individual)
        )
        tm.that(aggregate.passed, eq=not aggregate.violations)

    def test_all_with_pyproject_option_still_returns_passed_invariant(self) -> None:
        """Pyproject-aware aggregation preserves the passed invariant."""
        package = self._core_package()
        scan = tm.ok(
            tv.all(
                package,
                options=tv.AllValidationOptions(pyproject_path=self._pyproject()),
            )
        )
        tm.that(scan.validator_name, eq="all")
        tm.that(scan.passed, eq=not scan.violations)

    def test_validate_config_scans_pyproject_and_reports_result(self) -> None:
        """Config validation reports scan count and the passed invariant."""
        scan = tm.ok(tv.validate_config(self._pyproject()))
        tm.that(scan.passed, eq=not scan.violations)
        tm.that(scan.files_scanned, is_=int, gte=0)

    def test_non_python_file_path_scans_nothing_and_passes(self) -> None:
        """A non-Python path produces an empty successful import scan."""
        # A non-".py" file has no source to inspect: the contract is an empty,
        # passing scan rather than an error.
        scan = tm.ok(tv.imports(self._pyproject()))
        tm.that(scan.files_scanned, eq=0)
        tm.that(scan.violations, empty=True)
        tm.that(scan.passed, eq=True)

    @pytest.mark.parametrize("verb", c.Tests.VALIDATOR_METHODS)
    def test_validation_is_idempotent(self, verb: str) -> None:
        """Repeated validation returns identical public scan outcomes."""
        package = self._core_package()
        first = tm.ok(getattr(tv, verb)(package))
        second = tm.ok(getattr(tv, verb)(package))
        tm.that(second.passed, eq=first.passed)
        tm.that(len(second.violations), eq=len(first.violations))
        tm.that(second.files_scanned, eq=first.files_scanned)

    def test_all_validation_options_public_defaults(self) -> None:
        """All-validation options expose their documented public defaults."""
        options = tv.AllValidationOptions()
        tm.that(options.include_tests_validation, eq=False)
        tm.that(bool(options.exclude_patterns), eq=True)
        tm.that(options.pyproject_path, none=True)

    def test_scan_result_create_derives_passed_from_violations(self) -> None:
        """ScanResult derives passed state directly from its violations."""
        empty_scan = m.Tests.ScanResult(
            validator_name="unit", files_scanned=3, violations=[]
        )
        tm.that(empty_scan.passed, eq=True)

        violation = m.Tests.Violation(
            file_path=Path("offender.py"),
            line_number=7,
            rule_id="RULE_X",
            severity=c.Tests.ValidatorSeverity.HIGH,
            description="something went wrong",
        )
        failing_scan = m.Tests.ScanResult(
            validator_name="unit", files_scanned=3, violations=[violation]
        )
        tm.that(failing_scan.passed, eq=False)
        tm.that(len(failing_scan.violations), eq=1)

    def test_scan_result_model_dump_roundtrips_public_state(self) -> None:
        """ScanResult validation round-trips every public field and invariant."""
        violation = m.Tests.Violation(
            file_path=Path("offender.py"),
            line_number=7,
            rule_id="RULE_X",
            severity=c.Tests.ValidatorSeverity.MEDIUM,
            description="detail",
        )
        original = m.Tests.ScanResult(
            validator_name="roundtrip", files_scanned=2, violations=[violation]
        )
        restored = m.Tests.ScanResult.model_validate(
            original.model_dump(exclude={"passed"})
        )
        tm.that(restored.validator_name, eq="roundtrip")
        tm.that(restored.files_scanned, eq=2)
        tm.that(restored.passed, eq=False)
        tm.that(len(restored.violations), eq=1)

    @pytest.mark.parametrize(
        ("raw_severity", "expected"),
        [
            ("critical", c.Tests.ValidatorSeverity.CRITICAL),
            ("HIGH", c.Tests.ValidatorSeverity.HIGH),
            ("Medium", c.Tests.ValidatorSeverity.MEDIUM),
            ("low", c.Tests.ValidatorSeverity.LOW),
        ],
    )
    def test_violation_coerces_severity_string_to_enum(
        self, raw_severity: str, expected: c.Tests.ValidatorSeverity
    ) -> None:
        """Violation validation normalizes severity strings to public enums."""
        violation = m.Tests.Violation(
            file_path=Path("x.py"),
            line_number=1,
            rule_id="R",
            severity=raw_severity,
            description="d",
        )
        tm.that(violation.severity, eq=expected)
