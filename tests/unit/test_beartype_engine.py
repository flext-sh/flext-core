"""Behavioral tests for the FlextUtilitiesBeartypeEngine public facade.

Asserts the OBSERVABLE CONTRACT of the engine's structural predicate methods
that the split domain modules do not exercise: symbol-placement predicates
(``defined_inside`` / ``defined_in_function_scope``) and attribute-acceptance
classification (``attr_accept_public`` / ``attr_accept_utility`` /
``attr_accept_constants``). Every assertion targets a public return value; no
private attribute or method of the engine is touched.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import pytest

import tests.utilities
from flext_core.utilities import FlextUtilitiesBeartypeEngine
from tests import p, tm


class TestsFlextCoreBeartypeEngine(
    tests.utilities.TestsFlextUtilities.TestsFlextBeartypeEngine,
):
    """Contract of the engine's placement + attribute-acceptance predicates."""

    @staticmethod
    def test_defined_inside_true_for_nested_class() -> None:
        """A class nested inside another is reported as defined inside it."""

        class Outer:
            class Inner:
                """Nested marker class."""

        tm.that(
            FlextUtilitiesBeartypeEngine.defined_inside(
                Outer.Inner, Outer.__qualname__
            ),
            is_=bool,
            eq=True,
        )

    @staticmethod
    def test_defined_inside_false_for_unrelated_class() -> None:
        """A class defined outside the owner qualname is not defined inside."""

        class Outer:
            """Owner marker class."""

        class Other:
            """Unrelated marker class."""

        tm.that(
            FlextUtilitiesBeartypeEngine.defined_inside(Other, Outer.__qualname__),
            is_=bool,
            eq=False,
        )

    @staticmethod
    def test_defined_in_function_scope_true_for_local_class() -> None:
        """A class declared in a function body carries a ``<locals>`` qualname."""

        class Local:
            """Function-scoped marker class."""

        tm.that(
            FlextUtilitiesBeartypeEngine.defined_in_function_scope(Local) is True,
            truth=True,
        )

    @staticmethod
    def test_defined_in_function_scope_false_for_module_class() -> None:
        """A module-level class is not reported as function-scoped."""
        tm.that(
            FlextUtilitiesBeartypeEngine.defined_in_function_scope(
                tests.utilities.TestsFlextUtilities.TestsFlextBeartypeEngine,
            ),
            is_=bool,
            eq=False,
        )

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("run", True), ("do_thing", True), ("_private", False), ("__dunder__", False)],
    )
    @staticmethod
    def test_attr_accept_public_rejects_underscore_names(
        name: str,
        *,
        expected: bool,
    ) -> None:
        """Only names without a leading underscore are accepted as public."""
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_public(name) is expected,
            truth=True,
        )

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
    @staticmethod
    def test_attr_accept_utility_excludes_exempt_and_private(
        name: str,
        *,
        expected: bool,
    ) -> None:
        """Public names pass unless they are dunder-exempt utility methods."""
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_utility(
                FlextUtilitiesBeartypeEngine,
                name,
            ),
            is_=bool,
            eq=expected,
        )

    @staticmethod
    def test_attr_accept_constants_accepts_public_plain_value() -> None:
        """A public, non-callable, non-skipped attribute is accepted."""
        value: p.AttributeProbe = 42
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_constants("MAX_RETRIES", value),
            is_=bool,
            eq=True,
        )

    @pytest.mark.parametrize("name", ["_private", "model_fields", "__doc__"])
    @staticmethod
    def test_attr_accept_constants_rejects_private_and_skip_names(
        name: str,
    ) -> None:
        """Private names and skip-listed attributes are rejected regardless of value."""
        value: p.AttributeProbe = 1
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_constants(name, value) is False,
            truth=True,
        )

    @staticmethod
    def test_attr_accept_constants_rejects_type_value() -> None:
        """A nested type is not a constant attribute."""

        class Nested:
            """Marker type used as an attribute value."""

        value: p.AttributeProbe = Nested
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_constants("Nested", value),
            is_=bool,
            eq=False,
        )

    @staticmethod
    def test_attr_accept_constants_rejects_descriptor_values() -> None:
        """Descriptor values (staticmethod/classmethod/property) are not constants."""

        def _fn(_self: p.AttributeProbe) -> int:
            return 0

        def _cls_fn(_cls: type) -> int:
            return 0

        static_value: p.AttributeProbe = staticmethod(_fn)
        class_value = cast("p.AttributeProbe", classmethod(_cls_fn))
        property_value: p.AttributeProbe = property(_fn)

        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_constants(
                "as_static",
                static_value,
            ),
            is_=bool,
            eq=False,
        )
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_constants("as_class", class_value),
            is_=bool,
            eq=False,
        )
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_constants(
                "as_property",
                property_value,
            ),
            is_=bool,
            eq=False,
        )

    @staticmethod
    def test_attr_accept_constants_rejects_callable_value() -> None:
        """A plain callable attribute is a method, not a constant."""

        def _handler() -> None:
            """Callable attribute value."""

        value: p.AttributeProbe = _handler
        tm.that(
            FlextUtilitiesBeartypeEngine.attr_accept_constants("handler", value),
            is_=bool,
            eq=False,
        )
