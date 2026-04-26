"""Behavior contract for flext_core.__version__.FlextVersion — public API only."""

from __future__ import annotations

import pytest
from flext_tests import tm

from flext_core.__version__ import FlextVersion, __version__, __version_info__
from tests import c


class TestsFlextCoreVersion:
    """Behavior contract for FlextVersion public API and module-level exports."""

    def test_resolve_version_string_returns_semver_formatted_string(self) -> None:
        tm.that(
            FlextVersion.resolve_version_string(),
            is_=str,
            empty=False,
            match=c.Tests.Patterns.SEMVER,
        )

    def test_resolve_version_info_returns_non_empty_tuple_starting_with_major(
        self,
    ) -> None:
        info = FlextVersion.resolve_version_info()
        tm.that(info, is_=tuple, empty=False)
        tm.that(info[0], is_=int, gt=-1)

    def test_resolve_package_info_returns_complete_metadata_dict(self) -> None:
        info = FlextVersion.resolve_package_info()
        tm.that(
            info,
            is_=dict,
            has=list(c.Tests.Version.PACKAGE_INFO_REQUIRED_KEYS),
        )
        for key in c.Tests.Version.PACKAGE_INFO_REQUIRED_KEYS:
            tm.that(info[key], is_=str, none=False)
        tm.that(info["name"], eq=c.Tests.Version.CORE_PACKAGE_NAME)
        tm.that(info["version"], match=c.Tests.Patterns.SEMVER)

    @pytest.mark.parametrize(
        ("major", "minor", "patch", "expected"),
        c.Tests.Version.AT_LEAST_CASES,
        ids=c.Tests.Version.AT_LEAST_CASE_IDS,
    )
    def test_version_at_least_compares_against_current(
        self,
        major: int,
        minor: int,
        patch: int,
        *,
        expected: bool,
    ) -> None:
        tm.that(FlextVersion.version_at_least(major, minor, patch), eq=expected)

    def test_module_level_exports_match_class_accessors(self) -> None:
        tm.that(__version__, eq=FlextVersion.resolve_version_string())
        tm.that(__version_info__, eq=FlextVersion.resolve_version_info())
