"""Behavior contract for flext_core.__version__.FlextVersion — public API only.

``FlextVersion`` is a metadata facade whose public surface is the derived
attributes (``__version__``, ``__version_info__``, ``__title__`` …) recomputed
per subclass through MRO from the installed package metadata. Tests assert the
observable attribute contract and its parity with the module-level exports.
"""

from __future__ import annotations

from flext_tests import tm

from flext_core.__version__ import FlextVersion, __version__, __version_info__
from tests.constants import c


class TestsFlextCoreVersion:
    """Behavior contract for FlextVersion public attributes and module exports."""

    def test_version_attribute_is_semver_formatted_string(self) -> None:
        tm.that(
            FlextVersion.__version__, is_=str, empty=False, match=c.PATTERN_SEMVER_RE
        )

    def test_version_info_is_non_empty_tuple_starting_with_major(self) -> None:
        tm.that(FlextVersion.__version_info__, is_=tuple, empty=False)
        tm.that(FlextVersion.__version_info__[0], is_=int, gt=-1)

    def test_version_string_and_info_describe_the_same_version(self) -> None:
        rebuilt = ".".join(str(part) for part in FlextVersion.__version_info__)
        tm.that(rebuilt, eq=FlextVersion.__version__)

    def test_metadata_attributes_are_populated_strings(self) -> None:
        for value in (
            FlextVersion.__title__,
            FlextVersion.__description__,
            FlextVersion.__author__,
            FlextVersion.__license__,
            FlextVersion.__url__,
        ):
            tm.that(value, is_=str)
        tm.that(FlextVersion.__title__, eq=c.Tests.CORE_PACKAGE_NAME)
        tm.that(FlextVersion.__version__, match=c.PATTERN_SEMVER_RE)

    def test_module_level_exports_match_class_accessors(self) -> None:
        tm.that(__version__, eq=FlextVersion.__version__)
        tm.that(__version_info__, eq=FlextVersion.__version_info__)
