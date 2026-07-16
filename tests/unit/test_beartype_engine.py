"""Behavioral tests for the FlextUtilitiesBeartypeEngine public facade.

Asserts the OBSERVABLE CONTRACT of the engine's structural predicate methods
that the split domain modules do not exercise: symbol-placement predicates
(``defined_inside`` / ``defined_in_function_scope``) and attribute-acceptance
classification (``attr_accept_public`` / ``attr_accept_utility`` /
``attr_accept_constants``). Every assertion targets a public return value; no
private attribute or method of the engine is touched.
"""

from __future__ import annotations

import pytest

from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as be
from tests.protocols import p
from tests.unit._beartype_engine_support import TestsFlextBeartypeEngine


class TestsFlextCoreBeartypeEngine(TestsFlextBeartypeEngine):
    """Contract of the engine's placement + attribute-acceptance predicates."""

    def test_defined_inside_true_for_nested_class(self) -> None:
        """A class nested inside another is reported as defined inside it."""

        class Outer:
            class Inner:
                """Nested marker class."""

        assert be.defined_inside(Outer.Inner, Outer.__qualname__) is True

    def test_defined_inside_false_for_unrelated_class(self) -> None:
        """A class defined outside the owner qualname is not defined inside."""

        class Outer:
            """Owner marker class."""

        class Other:
            """Unrelated marker class."""

        assert be.defined_inside(Other, Outer.__qualname__) is False

    def test_defined_in_function_scope_true_for_local_class(self) -> None:
        """A class declared in a function body carries a ``<locals>`` qualname."""

        class Local:
            """Function-scoped marker class."""

        assert be.defined_in_function_scope(Local) is True

    def test_defined_in_function_scope_false_for_module_class(self) -> None:
        """A module-level class is not reported as function-scoped."""
        assert be.defined_in_function_scope(TestsFlextBeartypeEngine) is False

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("run", True),
            ("do_thing", True),
            ("_private", False),
            ("__dunder__", False),
        ],
    )
    def test_attr_accept_public_rejects_underscore_names(
        self, name: str, *, expected: bool
    ) -> None:
        """Only names without a leading underscore are accepted as public."""
        assert be.attr_accept_public(name) is expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("run", True),
            ("do_thing", True),
            ("__init__", False),
            ("__class_getitem__", False),
            ("_private", False),
        ],
    )
    def test_attr_accept_utility_excludes_exempt_and_private(
        self, name: str, *, expected: bool
    ) -> None:
        """Public names pass unless they are dunder-exempt utility methods."""
        assert be.attr_accept_utility(name) is expected

    def test_attr_accept_constants_accepts_public_plain_value(self) -> None:
        """A public, non-callable, non-skipped attribute is accepted."""
        value: p.AttributeProbe = 42
        assert be.attr_accept_constants("MAX_RETRIES", value) is True

    @pytest.mark.parametrize("name", ["_private", "model_fields", "__doc__"])
    def test_attr_accept_constants_rejects_private_and_skip_names(
        self, name: str
    ) -> None:
        """Private names and skip-listed attributes are rejected regardless of value."""
        value: p.AttributeProbe = 1
        assert be.attr_accept_constants(name, value) is False

    def test_attr_accept_constants_rejects_type_value(self) -> None:
        """A nested type is not a constant attribute."""

        class Nested:
            """Marker type used as an attribute value."""

        value: p.AttributeProbe = Nested
        assert be.attr_accept_constants("Nested", value) is False

    def test_attr_accept_constants_rejects_descriptor_values(self) -> None:
        """Descriptor values (staticmethod/classmethod/property) are not constants."""

        def _fn(_self: p.AttributeProbe) -> int:
            return 0

        static_value: p.AttributeProbe = staticmethod(_fn)
        class_value: p.AttributeProbe = classmethod(lambda _cls: 0)
        property_value: p.AttributeProbe = property(_fn)

        assert be.attr_accept_constants("as_static", static_value) is False
        assert be.attr_accept_constants("as_class", class_value) is False
        assert be.attr_accept_constants("as_property", property_value) is False

    def test_attr_accept_constants_rejects_callable_value(self) -> None:
        """A plain callable attribute is a method, not a constant."""

        def _handler() -> None:
            """Callable attribute value."""

        value: p.AttributeProbe = _handler
        assert be.attr_accept_constants("handler", value) is False
