"""Behavioral tests for namespace + constant enforcement.

Every assertion here is over the OBSERVABLE public contract:

- ``u.check(target)`` -> ``m.Report`` with public ``violations`` (each carrying
  ``message`` / ``rule_id``), and
- ``u.run_layer(target, layer)`` -> emitted warnings.

No private attribute/method is inspected; the class-prefix policy (e.g. the
``flext_core`` -> ``Flext`` mapping) is asserted through the violations a caller
actually observes, not through internal resolver internals.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from typing import TYPE_CHECKING, ClassVar

from tests import c, u
from tests.unit._enforcement_support import make_class

if TYPE_CHECKING:
    from pathlib import Path

import pytest

_PREFIX_FRAGMENT = "class name missing project prefix"


def _synthetic(name: str, *, qualname: str, module: str) -> type:
    """Build a bare target with controlled qualname/module for enforcement."""
    target = type(name, (), {})
    target.__qualname__ = qualname
    target.__module__ = module
    return target


class TestsFlextCoreEnforcementNamespace:
    """Public-contract tests for the namespace enforcement checker."""

    __test__ = True

    @pytest.mark.parametrize(
        ("name", "qualname", "module"),
        [
            pytest.param(
                "_PrivateHelper",
                "_PrivateHelper",
                "flext_core.demo",
                id="underscore-private-class",
            ),
            pytest.param(
                "Foo[int]",
                "Foo[int]",
                "flext_core.demo",
                id="generic-bracket-specialization",
            ),
            pytest.param(
                "InnerNs",
                "Outer.InnerNs",
                "nonexistent_project",
                id="nested-inner-class-qualname",
            ),
            pytest.param(
                "FlextModels", "FlextModels", "anything", id="facade-root-name"
            ),
            pytest.param(
                "FlextWidget",
                "FlextWidget",
                "flext_core.demo",
                id="correct-flext-prefix",
            ),
        ],
    )
    def test_exempt_or_compliant_targets_report_no_prefix_violation(
        self, name: str, qualname: str, module: str
    ) -> None:
        """Exempt shapes and correctly-prefixed classes yield no prefix violation."""
        target = _synthetic(name, qualname=qualname, module=module)

        report = u.check(target)

        assert not any(_PREFIX_FRAGMENT in v.message for v in report.violations)

    def test_flext_core_class_without_prefix_is_flagged_for_flext(self) -> None:
        """A ``flext_core`` class lacking the prefix is told to start with ``Flext``.

        This is the observable side of the ``flext_core -> Flext`` project
        mapping: the caller sees a violation demanding the ``Flext`` prefix.
        """
        target = _synthetic("Widget", qualname="Widget", module="flext_core.demo")

        report = u.check(target)

        prefix_msgs = [
            v.message for v in report.violations if _PREFIX_FRAGMENT in v.message
        ]
        assert prefix_msgs
        assert all('"Flext"' in msg for msg in prefix_msgs)

    def test_tests_module_class_with_composed_prefix_is_compliant(self) -> None:
        """Classes under ``tests.*`` are compliant with the ``TestsFlext`` prefix."""
        target = _synthetic(
            "TestsFlextModelsMixins",
            qualname="TestsFlextModelsMixins",
            module="tests._models.mixins",
        )

        report = u.check(target)

        assert not any(_PREFIX_FRAGMENT in v.message for v in report.violations)

    def test_pydantic_generic_specialization_does_not_count_as_namespace_violation(
        self, tmp_path: Path
    ) -> None:
        """Pydantic's synthetic ``Base[int]`` leak must not add prefix violations."""
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
                "demo_pkg.base", package / "base.py"
            )
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            importlib.import_module("demo_pkg.consumer")

            report = u.check(module.DemoServiceBase)

            # The pydantic-generated ``DemoServiceBase[bool]`` specialization must
            # not surface as an extra top-level class or backwards-compat alias.
            assert not any(
                "[bool]" in v.message or "backwards-compat" in v.message
                for v in report.violations
            )
        finally:
            sys.path.remove(src_path)
            sys.modules.pop("demo_pkg", None)
            sys.modules.pop("demo_pkg.base", None)
            sys.modules.pop("demo_pkg.consumer", None)

    def test_project_class_stem_override_controls_required_prefix(
        self, tmp_path: Path
    ) -> None:
        """``[tool.flext.project] class_stem_override`` sets the accepted prefix."""
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
        spec = importlib.util.spec_from_file_location("xmlapi", module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)

            report = u.check(module.XmlAPIModels)

            assert not any(_PREFIX_FRAGMENT in v.message for v in report.violations)
        finally:
            sys.modules.pop("xmlapi", None)

    def test_run_layer_emits_warnings_for_mutable_constant_under_warn_mode(
        self,
    ) -> None:
        """Under WARN mode ``run_layer`` warns on a mutable list constant.

        Precondition: the namespace mode constant is WARN, so ``run_layer``
        surfaces violations as warnings rather than raising.
        """
        assert c.ENFORCEMENT_NAMESPACE_MODE is c.EnforcementMode.WARN
        bad = make_class(
            "FlextSyntheticConstants",
            {"ITEMS": ["a"], "__annotations__": {"ITEMS": list[str]}},
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            u.run_layer(bad, "constants")

        texts = [str(r.message) for r in recorded]
        assert len(recorded) == 2
        assert any("[const_mutable]" in text for text in texts)
        assert any("[ENFORCE-079]" in text for text in texts)

    def test_run_layer_stays_silent_for_clean_class(self) -> None:
        """``run_layer`` emits nothing for a constant-free class."""
        target = make_class("FlextSyntheticCleanConstants", {})

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            u.run_layer(target, "constants")

        assert recorded == []

    def test_run_layer_exempts_tests_qualified_classes(self) -> None:
        """A ``Tests``-qualified class with a violating shape is still exempt."""
        target = type(
            "TestsFlextSyntheticConstants",
            (),
            {"ITEMS": ["a"], "__annotations__": {"ITEMS": list[str]}},
        )
        target.__qualname__ = "TestsFlextSyntheticConstants"
        target.__module__ = "tests.unit.synthetic"

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            u.run_layer(target, "constants")

        assert recorded == []

    def test_run_layer_skips_function_local_classes(self) -> None:
        """Function-local classes (``<locals>`` qualname) are never enforced."""

        class FlextLocalConstants:
            ITEMS: ClassVar[list[str]] = ["a"]  # violating shape, but function-local

        assert "<locals>" in FlextLocalConstants.__qualname__

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            u.run_layer(FlextLocalConstants, "constants")

        assert recorded == []

    @pytest.mark.parametrize(
        ("name", "body", "module_override", "expect_enforce_079"),
        [
            pytest.param(
                "FlextSyntheticCli",
                {
                    "GROUPS": frozenset({"foo"}),
                    "__annotations__": {"GROUPS": "ClassVar[frozenset[str]]"},
                },
                None,
                True,
                id="classvar-constant-outside-constants",
            ),
            pytest.param(
                "FlextSyntheticConstantsRefactor",
                {
                    "GROUPS": frozenset({"foo"}),
                    "__annotations__": {"GROUPS": "ClassVar[frozenset[str]]"},
                },
                "flext_infra._constants.refactor",
                False,
                id="classvar-constant-inside-constants",
            ),
            pytest.param(
                "FlextSyntheticModel",
                {
                    "model_config": {"title": "x"},
                    "logger": None,
                    "__annotations__": {
                        "model_config": "ClassVar[dict[str, str]]",
                        "logger": "ClassVar[object | None]",
                    },
                },
                None,
                False,
                id="framework-idiom-classvars",
            ),
            pytest.param(
                "FlextSyntheticCli",
                {
                    "groups": frozenset({"foo"}),
                    "__annotations__": {"groups": "ClassVar[frozenset[str]]"},
                },
                None,
                False,
                id="lowercase-classvar-not-a-constant",
            ),
            pytest.param(
                "FlextSyntheticCli",
                {"GROUPS": frozenset({"foo"})},
                None,
                True,
                id="implicit-constant-outside-constants",
            ),
            pytest.param(
                "FlextSyntheticConstantsRefactor",
                {"GROUPS": frozenset({"foo"})},
                "flext_infra._constants.refactor",
                False,
                id="implicit-constant-inside-constants",
            ),
        ],
    )
    def test_constant_placement_governs_enforce_079(
        self,
        name: str,
        body: dict[str, object],
        module_override: str | None,
        expect_enforce_079: bool,
    ) -> None:
        """ENFORCE-079 fires only for UPPER_CASE constants outside ``_constants``."""
        target = make_class(name, body)
        if module_override is not None:
            target.__module__ = module_override

        report = u.check(target)

        emitted = any(
            v.rule_id == "ENFORCE-079" and "Constant 'GROUPS' declared" in v.message
            for v in report.violations
        )
        assert emitted is expect_enforce_079
