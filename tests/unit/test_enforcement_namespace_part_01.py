"""Namespace enforcement tests — part 01 (identity and prefix rules).

Behavioral tests for the public ``FlextUtilitiesEnforcement.check`` contract
exposed via ``u.check``. Every assertion targets the returned ``Report`` public
surface (``.violations`` / ``.messages`` / ``.empty`` / ``in``) — never private
helpers of the enforcement engine.
"""

from __future__ import annotations

import pytest

from tests.utilities import u

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
