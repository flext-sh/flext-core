"""Namespace enforcement tests."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

from flext_core._constants.enforcement import FlextMroViolation
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
from tests.constants import c
from tests.models import TestsFlextModelsMixins
from tests.unit._enforcement_support import make_class
from tests.utilities import u


class TestsFlextEnforcementNamespace:
    def test_private_underscore_class_skipped(self) -> None:
        """Underscore-prefixed classes are implementation details, not facades."""

        class _PrivateHelper:
            pass

        report = u.check(_PrivateHelper)
        assert all(v.layer != "namespace" for v in report.violations)

    def test_generic_bracket_specialization_skipped(self) -> None:
        """Synthetic ``Foo[int]``-style names are Pydantic/Generic artifacts."""
        # Build a synthetic target with bracketed name — mimicking what Pydantic
        # generates for parameterized generic specializations.
        fake = type("Foo[int]", (), {})
        report = u.check(fake)
        assert all(v.layer != "namespace" for v in report.violations)

    def test_pydantic_generic_parametrized_subclass_does_not_count(
        self,
        tmp_path: Path,
    ) -> None:
        """Pydantic leaks ``Base[int]`` into the base module during subclassing.

        These synthetic specializations must not count as extra top-level
        classes or backwards-compat aliases for the base class module.
        """
        package = tmp_path / "src" / "demo_pkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "base.py").write_text(
            "from __future__ import annotations\n\n"
            "from pydantic import BaseModel\n\n"
            "class DemoServiceBase[T](BaseModel):\n"
            "    pass\n",
            encoding="utf-8",
        )
        (package / "consumer.py").write_text(
            "from __future__ import annotations\n\n"
            "from demo_pkg.base import DemoServiceBase\n\n"
            "class DemoConsumer(DemoServiceBase[bool]):\n"
            "    pass\n",
            encoding="utf-8",
        )

        src_path = str(tmp_path / "src")
        sys.path.insert(0, src_path)
        try:
            spec = importlib.util.spec_from_file_location(
                "demo_pkg.base",
                package / "base.py",
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Force creation of the parametrized specialization in the base module.
            importlib.import_module("demo_pkg.consumer")

            report = u.check(module.DemoServiceBase)
            namespace_violations = [
                v for v in report.violations if v.layer == "namespace"
            ]
            assert not namespace_violations
        finally:
            sys.path.remove(src_path)
            sys.modules.pop("demo_pkg", None)
            sys.modules.pop("demo_pkg.base", None)
            sys.modules.pop("demo_pkg.consumer", None)

    def test_inner_class_qualname_exempts_prefix_check(self) -> None:
        """Classes with ``.`` in qualname (nested) skip class_prefix."""
        # Simulate a top-level class' inner class via a synthetic target whose
        # qualname signals nesting without being function-local.
        fake = type("InnerNs", (), {})
        fake.__qualname__ = "Outer.InnerNs"  # signals nested position
        fake.__module__ = "nonexistent_project"
        report = u.check(fake)
        assert not any(
            "class name missing project prefix" in v.message for v in report.violations
        )

    def test_facade_root_exempt(self) -> None:
        """Classes in ENFORCEMENT_NAMESPACE_FACADE_ROOTS skip prefix rule."""
        fake = type("FlextModels", (), {})  # literal root name
        fake.__module__ = "anything"
        report = u.check(fake)
        assert not any(
            "class name missing project prefix" in v.message for v in report.violations
        )

    def test_flext_core_override_returns_flext(self) -> None:
        """flext_core is the single src package that maps to ``Flext``."""
        project = FlextUtilitiesEnforcement._project(FlextUtilitiesEnforcement)
        assert project is not None
        prefix, _namespace = project
        assert prefix == "Flext"

    def test_tests_module_gets_tests_prefix_composition(self) -> None:
        """Classes in ``tests.*`` carry ``Tests`` + project prefix (e.g. TestsFlext)."""
        report = u.check(TestsFlextModelsMixins)
        namespace_msgs = [
            v.message
            for v in report.violations
            if v.layer == "namespace" and "class name" in v.message
        ]
        # The class name IS "TestsFlextModelsMixins" which starts with
        # "TestsFlext" — the composed prefix — so no class_prefix violation.
        assert not namespace_msgs

    def test_project_class_stem_override_controls_class_prefix(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sample"
        package = root / "src" / "xmlapi"
        package.mkdir(parents=True)
        (root / "pyproject.toml").write_text(
            """
[project]
name = "xml-api"
version = "0.1.0"
license = "MIT"

[tool.flext.project]
class_stem_override = "XmlAPI"
""".strip(),
            encoding="utf-8",
        )
        module_path = package / "__init__.py"
        module_path.write_text("class XmlAPIModels:\n    pass\n", encoding="utf-8")
        spec = importlib.util.spec_from_file_location(
            "xmlapi_override_sample", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        report = u.check(module.XmlAPIModels)

        assert not any(
            "class name missing project prefix" in v.message for v in report.violations
        )

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
        assert all(r.category is FlextMroViolation for r in recorded)
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
        clean = make_class(
            "FlextSyntheticCleanConstants",
            {"items": ("a",), "__annotations__": {"items": tuple[str, ...]}},
        )
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
