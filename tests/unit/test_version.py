"""Behavior contract for flext_core.__version__.FlextVersion — public API only.

``FlextVersion`` is a metadata facade whose public surface is the derived
attributes (``__version__``, ``__version_info__``, ``__title__`` …) recomputed
per subclass through MRO from the installed package metadata. Tests assert the
observable attribute contract and its parity with the module-level exports.
"""

from __future__ import annotations

from importlib.metadata import PackageMetadata, PathDistribution
from pathlib import Path

import pytest

from flext_core.__version__ import FlextVersion, __version__, __version_info__
from flext_tests import tm
from tests import c


class TestsFlextCoreVersion:
    """Behavior contract for FlextVersion public attributes and module exports."""

    def test_version_attribute_is_semver_formatted_string(self) -> None:
        """The public version is a non-empty semantic or PEP 440 version string."""
        tm.that(
            FlextVersion.__version__, is_=str, empty=False, match=c.PATTERN_SEMVER_RE
        )

    def test_version_info_is_non_empty_tuple_starting_with_major(self) -> None:
        """Version info exposes a non-empty tuple beginning with the major number."""
        tm.that(FlextVersion.__version_info__, is_=tuple, len=3)
        for part in FlextVersion.__version_info__:
            tm.that(part, is_=int, gt=-1)

    def test_version_string_and_info_describe_the_same_version(self) -> None:
        """String and tuple representations identify the same published version."""
        release = ".".join(str(part) for part in FlextVersion.__version_info__)
        tm.that(FlextVersion.__version__.startswith(release), eq=True)

    @pytest.mark.parametrize(
        "version",
        [
            "1.2.3",
            "1.2.3rc4",
            "1.2.3.rc4",
            "1.2.3.dev5",
            "1.2.3.post6",
            "1.2.3+local.7",
            "4!1.2.3rc4.post5.dev6+local.7",
        ],
    )
    def test_subclass_version_info_is_the_release_triple(
        self, version: str, tmp_path: Path
    ) -> None:
        """Every supported qualifier preserves the integer release triple."""
        distribution_path = tmp_path / "flext_version_contract.dist-info"
        distribution_path.mkdir()
        (distribution_path / "METADATA").write_text(
            "\n".join((
                "Metadata-Version: 2.4",
                "Name: flext-version-contract",
                f"Version: {version}",
                "",
            )),
            encoding="utf-8",
        )
        package_metadata = PathDistribution(distribution_path).metadata

        class VersionContract(FlextVersion):
            _metadata: PackageMetadata = package_metadata

        tm.that(VersionContract.__version__, eq=version)
        tm.that(VersionContract.__version_info__, eq=(1, 2, 3))

    @pytest.mark.parametrize("version", ["1.2", "1.2.3.4", "1.2.3garbage"])
    def test_subclass_rejects_non_semantic_release_metadata(
        self, version: str, tmp_path: Path
    ) -> None:
        """Metadata without major, minor, and patch components fails loudly."""
        distribution_path = tmp_path / "flext_version_contract.dist-info"
        distribution_path.mkdir()
        (distribution_path / "METADATA").write_text(
            "\n".join((
                "Metadata-Version: 2.4",
                "Name: flext-version-contract",
                f"Version: {version}",
                "",
            )),
            encoding="utf-8",
        )
        package_metadata = PathDistribution(distribution_path).metadata

        with pytest.raises(ValueError, match="three-part semantic version"):

            class InvalidVersionContract(FlextVersion):
                _metadata: PackageMetadata = package_metadata

    def test_metadata_attributes_are_populated_strings(self) -> None:
        """Package metadata fields are populated and use the canonical title."""
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
        """Module exports preserve exact parity with the metadata facade."""
        tm.that(__version__, eq=FlextVersion.__version__)
        tm.that(__version_info__, eq=FlextVersion.__version_info__)
