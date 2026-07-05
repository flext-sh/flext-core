"""Namespace enforcement tests — part 01 (identity and prefix rules)."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING

from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
from tests.utilities import u

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextEnforcementNamespacePart01:
    __test__ = False

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
        target = type("TestsFlextModelsMixins", (), {})
        target.__module__ = "tests._models.mixins"
        report = u.check(target)
        namespace_msgs = [
            v.message
            for v in report.violations
            if v.layer == "namespace" and "class name" in v.message
        ]
        # The class name IS "TestsFlextModelsMixins" which starts with
        # "TestsFlext" — the composed prefix — so no class_prefix violation.
        assert not namespace_msgs

    def test_project_class_stem_override_controls_class_prefix(
        self,
        tmp_path: Path,
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
            "xmlapi_override_sample",
            module_path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        report = u.check(module.XmlAPIModels)

        assert not any(
            "class name missing project prefix" in v.message for v in report.violations
        )
