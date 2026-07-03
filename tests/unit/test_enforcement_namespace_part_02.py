"""Namespace enforcement tests — part 02 (run_layer and constant rules)."""

from __future__ import annotations

import warnings

from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
from tests.constants import c
from tests.unit._enforcement_support import make_class
from tests.utilities import u


class TestsFlextEnforcementNamespacePart02:
    __test__ = False

    def test_run_layer_emits_mro_violation_under_default_warn_mode(self) -> None:
        """``run_layer`` gates on ``c.ENFORCEMENT_NAMESPACE_MODE`` (Final, WARN).

        Coverage note: the OFF/STRICT branches of this gate cannot be driven
        directly — ``run_layer`` takes no ``mode`` parameter and the module
        constant is ``Final`` (mutating globals is forbidden). Mode dispatch
        itself is fully covered by the explicit ``emit(mode=...)`` tests in
        ``test_enforcement_reports.py``; ``run_layer`` delegates to ``emit``
        with the namespace-mode constant asserted here as precondition.
        """
        assert c.ENFORCEMENT_NAMESPACE_MODE is c.EnforcementMode.WARN
        bad = make_class(
            "FlextSyntheticConstants",
            {"ITEMS": ["a"], "__annotations__": {"ITEMS": list[str]}},
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.run_layer(bad, "constants")
        assert len(recorded) == 2
        texts = [str(r.message) for r in recorded]
        assert any("[const_mutable]" in text for text in texts)
        assert any("[ENFORCE-079]" in text for text in texts)

    def test_run_layer_skips_function_local_classes(self) -> None:
        class FlextLocalConstants:
            ITEMS: list[str] = ["a"]  # violating shape, but function-local

        assert "<locals>" in FlextLocalConstants.__qualname__
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.run_layer(FlextLocalConstants, "constants")
        assert recorded == []

    def test_run_layer_exempts_tests_qualified_classes(self) -> None:
        fake = type(
            "TestsFlextSyntheticConstants",
            (),
            {"ITEMS": ["a"], "__annotations__": {"ITEMS": list[str]}},
        )
        fake.__qualname__ = "TestsFlextSyntheticConstants"
        fake.__module__ = "tests.unit.synthetic"
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.run_layer(fake, "constants")
        assert recorded == []

    def test_run_layer_silent_for_clean_class(self) -> None:
        clean = make_class("FlextSyntheticCleanConstants", {})
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            FlextUtilitiesEnforcement.run_layer(clean, "constants")
        assert recorded == []

    def test_classvar_constant_outside_constants_emits_enforce_079(self) -> None:
        """ClassVar UPPER_CASE attributes outside _constants trigger ENFORCE-079."""
        bad = make_class(
            "FlextSyntheticCli",
            {
                "GROUPS": frozenset({"foo"}),
                "__annotations__": {"GROUPS": "ClassVar[frozenset[str]]"},
            },
        )
        report = u.check(bad)
        assert any(
            "Constant 'GROUPS' declared" in v.message and v.rule_id == "ENFORCE-079"
            for v in report.violations
        )

    def test_classvar_constant_inside_constants_is_exempt(self) -> None:
        """ClassVar UPPER_CASE attributes inside _constants modules are allowed."""
        good = make_class(
            "FlextSyntheticConstantsRefactor",
            {
                "GROUPS": frozenset({"foo"}),
                "__annotations__": {"GROUPS": "ClassVar[frozenset[str]]"},
            },
        )
        good.__module__ = "flext_infra._constants.refactor"
        report = u.check(good)
        assert not any(v.rule_id == "ENFORCE-079" for v in report.violations)

    def test_classvar_constant_exempt_model_config_and_logger(self) -> None:
        """model_config and logger ClassVar names are framework idioms."""
        good = make_class(
            "FlextSyntheticModel",
            {
                "model_config": {"title": "x"},
                "logger": None,
                "__annotations__": {
                    "model_config": "ClassVar[dict[str, str]]",
                    "logger": "ClassVar[object | None]",
                },
            },
        )
        report = u.check(good)
        assert not any(v.rule_id == "ENFORCE-079" for v in report.violations)

    def test_classvar_constant_lowercase_name_skipped(self) -> None:
        """Non-UPPER_CASE ClassVar attributes are not treated as constants."""
        good = make_class(
            "FlextSyntheticCli",
            {
                "groups": frozenset({"foo"}),
                "__annotations__": {"groups": "ClassVar[frozenset[str]]"},
            },
        )
        report = u.check(good)
        assert not any(v.rule_id == "ENFORCE-079" for v in report.violations)

    def test_implicit_constant_outside_constants_emits_enforce_079(self) -> None:
        """UPPER_CASE constant-like attributes without ClassVar also trigger ENFORCE-079."""
        bad = make_class(
            "FlextSyntheticCli",
            {"GROUPS": frozenset({"foo"})},
        )
        report = u.check(bad)
        assert any(
            "Constant 'GROUPS' declared" in v.message and v.rule_id == "ENFORCE-079"
            for v in report.violations
        )

    def test_implicit_constant_inside_constants_is_exempt(self) -> None:
        """Implicit constants inside _constants modules are allowed."""
        good = make_class(
            "FlextSyntheticConstantsRefactor",
            {"GROUPS": frozenset({"foo"})},
        )
        good.__module__ = "flext_infra._constants.refactor"
        report = u.check(good)
        assert not any(v.rule_id == "ENFORCE-079" for v in report.violations)
