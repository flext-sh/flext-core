"""Namespace enforcement tests — part 01 (identity and prefix rules).

Behavioral tests for the public ``FlextUtilitiesEnforcement.check`` contract
exposed via ``u.check``. Every assertion targets the returned ``Report`` public
surface (``.violations`` / ``.messages`` / ``.empty`` / ``in``) — never private
helpers of the enforcement engine.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING

import pytest

from tests import u

if TYPE_CHECKING:
    from pathlib import Path

_MISSING_PREFIX = "class name missing project prefix"


class TestsFlextCoreEnforcementNamespacePart01:
    """Public-contract tests for the namespace-prefix enforcement rules."""

    __test__ = True

    def test_private_underscore_class_has_no_namespace_violation(self) -> None:
        """Underscore-prefixed classes are implementation details, not facades."""

        class _PrivateHelper:
            pass

        _PrivateHelper.__module__ = "flext_core.x"

        report = u.check(_PrivateHelper)

        assert report.empty
        assert _MISSING_PREFIX not in report

    def test_generic_bracket_specialization_has_no_namespace_violation(self) -> None:
        """Synthetic ``Foo[int]``-style names are Pydantic/Generic artifacts."""
        fake = type("Foo[int]", (), {})
        fake.__module__ = "flext_core.x"

        report = u.check(fake)

        assert report.empty
        assert _MISSING_PREFIX not in report

    def test_pydantic_generic_parametrized_subclass_produces_no_violation(
        self, tmp_path: Path
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
                "demo_pkg.base", package / "base.py"
            )
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Force creation of the parametrized specialization in the base module.
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

    def test_inner_class_qualname_exempts_prefix_check(self) -> None:
        """Classes with ``.`` in qualname (nested) skip class_prefix."""
        fake = type("InnerNs", (), {})
        fake.__qualname__ = "Outer.InnerNs"  # signals nested position
        fake.__module__ = "flext_core.x"

        report = u.check(fake)

        assert _MISSING_PREFIX not in report

    def test_facade_root_name_exempts_prefix_check(self) -> None:
        """Classes named as facade roots (e.g. ``FlextModels``) skip prefix rule."""
        fake = type("FlextModels", (), {})
        fake.__module__ = "flext_core.x"

        report = u.check(fake)

        assert _MISSING_PREFIX not in report

    def test_flext_core_class_missing_prefix_is_flagged(self) -> None:
        """flext_core is the src package mapped to the ``Flext`` prefix.

        A concrete, non-underscore top-level class in a ``flext_core.*`` module
        that lacks the ``Flext`` prefix must produce a ``Namespace``-layer
        violation whose message names the required prefix.
        """
        offender = type("Widget", (), {})
        offender.__module__ = "flext_core.something"

        report = u.check(offender)

        assert report  # truthy: at least one violation
        assert len(report) >= 1
        assert _MISSING_PREFIX in report
        prefix_violations = [
            v for v in report.violations if _MISSING_PREFIX in v.message
        ]
        assert prefix_violations
        violation = prefix_violations[0]
        assert violation.layer == "Namespace"
        assert violation.qualname == "Widget"
        assert '"Flext"' in violation.message

    def test_flext_core_class_with_prefix_is_clean(self) -> None:
        """A properly ``Flext``-prefixed flext_core class raises no violation."""
        compliant = type("FlextWidget", (), {})
        compliant.__module__ = "flext_core.something"

        report = u.check(compliant)

        assert report.empty
        assert _MISSING_PREFIX not in report

    @pytest.mark.parametrize(
        ("class_name", "flagged"),
        [
            ("TestsFlextModelsMixins", False),
            ("TestsFlextRunner", False),
            ("Wrong", True),
            ("FlextModelsMixins", True),
        ],
    )
    def test_tests_module_requires_tests_prefix_composition(
        self, class_name: str, *, flagged: bool
    ) -> None:
        """Classes in ``tests.*`` must carry the composed ``TestsFlext`` prefix."""
        target = type(class_name, (), {})
        target.__module__ = "tests._models.mixins"

        report = u.check(target)

        assert (_MISSING_PREFIX in report) is flagged

    def test_project_class_stem_override_controls_class_prefix(
        self, tmp_path: Path
    ) -> None:
        """``class_stem_override`` in pyproject drives the required class prefix."""
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
        spec.loader.exec_module(module)

        report = u.check(module.XmlAPIModels)

        assert _MISSING_PREFIX not in report
