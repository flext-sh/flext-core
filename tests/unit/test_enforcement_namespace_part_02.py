"""Namespace enforcement tests — part 02 (run_layer, constant rules, mode dispatch).

Behavioral contract of ``FlextUtilitiesEnforcement`` exercised through its public
surface only:

* ``run_layer(target, layer)`` — the ``__init_subclass__`` hook. Observable
  behavior is the *warnings it emits* (or does not emit) for a class.
* ``check(target, *, layer=...)`` — returns an ``m.Report`` whose public
  ``violations`` carry ``rule_id`` / ``severity`` / ``layer`` / ``qualname`` /
  ``agents_md_anchor`` / ``message``.
* ``emit(report, *, mode=...)`` — dispatches on the enforcement mode: silent when
  OFF, warns when WARN, raises ``TypeError`` when STRICT.

No test inspects private attributes, patches internals, or asserts on
implementation-only shapes (``__qualname__`` internals, exact warning counts).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest

from flext_core import c, m
from flext_core._constants.enforcement import FlextMroViolation, FlextSmellViolation
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
from tests.unit._enforcement_support import make_class
from tests.utilities import u

type WarningRecords = list[warnings.WarningMessage]
type ClassFactory = Callable[[], type]


def _synthetic(name: str, body: dict[str, object], module: str | None = None) -> type:
    """Build a synthetic class, optionally overriding its owning module."""
    cls = make_class(name, body)
    if module is not None:
        cls.__module__ = module
    return cls


def _local_constants_class() -> type:
    """Return a genuinely function-local class (has ``<locals>`` qualname)."""

    class FlextLocalConstants:
        ITEMS: list[str] = ["a"]  # violating shape, but function-local

    return FlextLocalConstants


def _run_layer_records(target: type, layer: str) -> WarningRecords:
    """Drive ``run_layer`` and return the warnings it emits (public behavior)."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        FlextUtilitiesEnforcement.run_layer(target, layer)
    return list(recorded)


def _bad_constant_report() -> p.Report:
    """Build a non-empty constants report for mode-dispatch tests."""
    return u.check(
        make_class("FlextSyntheticCli", {"GROUPS": frozenset({"foo"})}),
        layer="constants",
    )


class TestsFlextCoreEnforcementNamespacePart02:
    __test__ = True

    # ------------------------------------------------------------------ #
    # run_layer — emitted warnings are the observable contract           #
    # ------------------------------------------------------------------ #

    def test_run_layer_warns_on_mutable_constant_with_smell_category(self) -> None:
        """A constants class with a mutable ``list`` field is flagged.

        The caller observes ``FlextMroViolation``-family warnings whose messages
        name the offending rules (``const_mutable`` and ``ENFORCE-079``).
        """
        bad = make_class(
            "FlextSyntheticConstants",
            {"ITEMS": ["a"], "__annotations__": {"ITEMS": list[str]}},
        )

        recorded = _run_layer_records(bad, "constants")

        assert recorded, "expected run_layer to emit at least one warning"
        assert all(issubclass(rec.category, FlextMroViolation) for rec in recorded), (
            "every emitted warning must be from the FLEXT violation family"
        )
        texts = [str(rec.message) for rec in recorded]
        assert any("[const_mutable]" in text for text in texts)
        assert any("[ENFORCE-079]" in text for text in texts)

    @pytest.mark.parametrize(
        ("case", "factory"),
        [
            ("function_local", _local_constants_class),
            (
                "tests_qualified_exempt",
                lambda: _synthetic(
                    "TestsFlextSyntheticConstants",
                    {"ITEMS": ["a"], "__annotations__": {"ITEMS": list[str]}},
                    module="tests.unit.synthetic",
                ),
            ),
            ("clean_class", lambda: make_class("FlextSyntheticCleanConstants", {})),
        ],
    )
    def test_run_layer_stays_silent_for_exempt_or_clean_classes(
        self, case: str, factory: ClassFactory
    ) -> None:
        """Function-local, tests-qualified, and clean classes emit no warnings."""
        target = factory()

        recorded = _run_layer_records(target, "constants")

        assert recorded == [], f"{case} should not raise enforcement warnings"

    # ------------------------------------------------------------------ #
    # check — Report.violations public contract                          #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        ("name", "body"),
        [
            (
                "classvar_constant",
                {
                    "GROUPS": frozenset({"foo"}),
                    "__annotations__": {"GROUPS": "ClassVar[frozenset[str]]"},
                },
            ),
            ("implicit_constant", {"GROUPS": frozenset({"foo"})}),
        ],
    )
    def test_check_flags_constant_declared_outside_constants(
        self, name: str, body: dict[str, object]
    ) -> None:
        """UPPER_CASE constants outside ``_constants`` yield an ENFORCE-079 violation.

        The full public violation contract is asserted, not just its presence.
        """
        bad = make_class("FlextSyntheticCli", body)

        report = u.check(bad)

        matches = [v for v in report.violations if v.rule_id == "ENFORCE-079"]
        assert matches, f"{name}: expected an ENFORCE-079 violation"
        violation = matches[0]
        assert "Constant 'GROUPS' declared" in violation.message
        assert violation.qualname == "FlextSyntheticCli"
        assert violation.agents_md_anchor
        assert violation.severity
        assert violation.layer

    @pytest.mark.parametrize(
        ("case", "body", "module"),
        [
            (
                "inside_constants_module",
                {"GROUPS": frozenset({"foo"})},
                "flext_infra._constants.refactor",
            ),
            (
                "classvar_inside_constants_module",
                {
                    "GROUPS": frozenset({"foo"}),
                    "__annotations__": {"GROUPS": "ClassVar[frozenset[str]]"},
                },
                "flext_infra._constants.refactor",
            ),
            (
                "framework_idiom_names",
                {
                    "model_config": {"title": "x"},
                    "logger": None,
                    "__annotations__": {
                        "model_config": "ClassVar[dict[str, str]]",
                        "logger": "ClassVar[object | None]",
                    },
                },
                None,
            ),
            (
                "lowercase_name_not_a_constant",
                {
                    "groups": frozenset({"foo"}),
                    "__annotations__": {"groups": "ClassVar[frozenset[str]]"},
                },
                None,
            ),
        ],
    )
    def test_check_exempts_permitted_constant_shapes(
        self, case: str, body: dict[str, object], module: str | None
    ) -> None:
        """Constants inside ``_constants``, framework idioms, and lowercase names pass."""
        good = _synthetic("FlextSyntheticExempt", body, module=module)

        report = u.check(good)

        assert not any(v.rule_id == "ENFORCE-079" for v in report.violations), (
            f"{case} must not raise ENFORCE-079"
        )

    def test_check_clean_class_reports_no_violations(self) -> None:
        """A structurally clean class produces an empty report."""
        clean = make_class("FlextSyntheticCleanConstants", {})

        report = u.check(clean, layer="constants")

        assert report.empty
        assert report.violations == []

    # ------------------------------------------------------------------ #
    # emit — mode dispatch is the observable contract                    #
    # ------------------------------------------------------------------ #

    def test_emit_raises_type_error_in_strict_mode(self) -> None:
        """STRICT mode turns a violation report into a raised ``TypeError``."""
        report = _bad_constant_report()
        assert not report.empty

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(TypeError, match="ENFORCE-079"):
                FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.STRICT)

    def test_emit_stays_silent_in_off_mode(self) -> None:
        """OFF mode neither warns nor raises for a non-empty report."""
        report = _bad_constant_report()
        assert not report.empty

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.OFF)

        assert recorded == []

    def test_emit_warns_in_warn_mode(self) -> None:
        """WARN mode surfaces the violation as a ``FlextSmellViolation`` warning."""
        report = _bad_constant_report()
        assert not report.empty

        with pytest.warns(FlextSmellViolation, match="ENFORCE-079"):
            FlextUtilitiesEnforcement.emit(report, mode=c.EnforcementMode.WARN)

    def test_emit_is_a_noop_for_empty_report(self) -> None:
        """An empty report never warns or raises regardless of mode."""
        empty_report = u.check(make_class("FlextSyntheticCleanConstants", {}))
        assert empty_report.empty

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.emit(empty_report, mode=c.EnforcementMode.STRICT)

        assert recorded == []
